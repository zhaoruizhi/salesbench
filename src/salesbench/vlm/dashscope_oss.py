"""Account-scoped temporary OSS transport for DashScope multimodal files."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable


UPLOAD_CACHE_VERSION = "dashscope-temporary-oss-v1"
DEFAULT_UPLOAD_API_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"


class HTTPStatusFailure(RuntimeError):
    """HTTP failure whose string representation intentionally omits its body."""

    def __init__(self, status_code: int, operation: str, response_body: str = ""):
        super().__init__(f"HTTP {status_code} during DashScope {operation}")
        self.status_code = int(status_code)
        self.operation = operation
        self.response_body = response_body


class DashScopeUploadError(RuntimeError):
    """Safe, credential-free temporary upload failure."""


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_transport_error(exc: Exception) -> str:
    """Classify an upload/API exception without exposing request credentials."""

    for current in _exception_chain(exc):
        status = getattr(current, "status_code", None)
        if status is None and isinstance(current, urllib.error.HTTPError):
            status = current.code
        if isinstance(status, int):
            if status in {408, 409, 425, 429} or status >= 500:
                return "retryable"
            if 400 <= status < 500:
                return "non_retryable"
        if isinstance(
            current,
            (
                ssl.SSLError,
                TimeoutError,
                BrokenPipeError,
                ConnectionError,
                ConnectionResetError,
                urllib.error.URLError,
            ),
        ):
            return "retryable"
    return "unknown"


Requester = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


def _default_requester(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPStatusFailure(exc.code, method.lower(), response_body) from exc


def _safe_model_segment(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._") or "model"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _multipart_body(
    fields: list[tuple[str, str]],
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"----SalesBench{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class DashScopeTemporaryOSSUploader:
    """Upload local frames to DashScope's 48-hour development object store."""

    def __init__(
        self,
        api_key: str,
        cache_root: str | Path = "outputs/cache/dashscope_temporary_oss",
        *,
        upload_api_url: str = DEFAULT_UPLOAD_API_URL,
        retry_max: int = 5,
        retry_backoff_s: float = 5.0,
        request_timeout_s: float = 60.0,
        object_ttl_s: float = 48 * 60 * 60,
        expiry_margin_s: float = 15 * 60,
        requester: Requester | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.time,
    ):
        if retry_max < 1:
            raise ValueError("retry_max must be at least 1")
        self.api_key = api_key
        self.cache_root = Path(cache_root)
        self.upload_api_url = upload_api_url.rstrip("?")
        self.retry_max = retry_max
        self.retry_backoff_s = retry_backoff_s
        self.request_timeout_s = request_timeout_s
        self.object_ttl_s = object_ttl_s
        self.expiry_margin_s = expiry_margin_s
        self.requester = requester or _default_requester
        self.sleep_fn = sleep_fn
        self.time_fn = time_fn

    def upload_frames(self, frame_paths: list[str], model: str) -> list[str]:
        """Return ordered temporary URLs for every local frame path."""

        urls: list[str] = []
        policy: dict[str, object] | None = None
        for raw_path in frame_paths:
            path = Path(raw_path)
            if not path.is_file():
                raise DashScopeUploadError(f"Frame file is missing: {path}")
            digest = _sha256_file(path)
            cached = self._cached_url(model, digest)
            if cached:
                urls.append(cached)
                continue
            oss_url, policy = self._upload_frame(path, model, digest, policy)
            self._write_cache(model, digest, oss_url)
            urls.append(oss_url)
        return urls

    def _cache_path(self, model: str, digest: str) -> Path:
        return self.cache_root / _safe_model_segment(model) / f"{digest}.json"

    def _cached_url(self, model: str, digest: str) -> str | None:
        path = self._cache_path(model, digest)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            record.get("cache_version") != UPLOAD_CACHE_VERSION
            or record.get("model") != model
            or record.get("sha256") != digest
            or not str(record.get("oss_url") or "").startswith("oss://")
        ):
            return None
        expires_at = float(record.get("expires_at") or 0)
        if self.time_fn() + self.expiry_margin_s >= expires_at:
            return None
        return str(record["oss_url"])

    def _write_cache(self, model: str, digest: str, oss_url: str) -> None:
        path = self._cache_path(model, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = self.time_fn()
        record = {
            "cache_version": UPLOAD_CACHE_VERSION,
            "model": model,
            "sha256": digest,
            "oss_url": oss_url,
            "uploaded_at": now,
            "expires_at": now + self.object_ttl_s,
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _upload_frame(
        self,
        path: Path,
        model: str,
        digest: str,
        policy: dict[str, object] | None,
    ) -> tuple[str, dict[str, object]]:
        last_error: Exception | None = None
        for attempt in range(self.retry_max):
            try:
                if policy is None:
                    policy = self._get_policy(model)
                oss_url = self._post_file(policy, path, digest)
                return oss_url, policy
            except Exception as exc:
                last_error = exc
                policy = None
                classification = classify_transport_error(exc)
                if classification != "retryable" or attempt == self.retry_max - 1:
                    break
                self.sleep_fn(self.retry_backoff_s * (2**attempt))

        kind = type(last_error).__name__ if last_error is not None else "UnknownError"
        status = getattr(last_error, "status_code", None)
        suffix = f" HTTP {status}" if isinstance(status, int) else f" {kind}"
        raise DashScopeUploadError(
            f"DashScope temporary frame upload failed after {attempt + 1} attempt(s):{suffix}"
        ) from last_error

    def _get_policy(self, model: str) -> dict[str, object]:
        query = urllib.parse.urlencode({"action": "getPolicy", "model": model})
        status, body = self.requester(
            "GET",
            f"{self.upload_api_url}?{query}",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            None,
            self.request_timeout_s,
        )
        if status != 200:
            raise HTTPStatusFailure(status, "policy")
        try:
            payload = json.loads(body.decode("utf-8"))
            data = payload["data"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DashScopeUploadError("DashScope upload policy response is malformed") from exc
        required = {
            "policy",
            "signature",
            "upload_dir",
            "upload_host",
            "oss_access_key_id",
            "x_oss_object_acl",
            "x_oss_forbid_overwrite",
        }
        if not isinstance(data, dict) or not required.issubset(data):
            raise DashScopeUploadError("DashScope upload policy is missing required fields")
        return data

    def _post_file(self, policy: dict[str, object], path: Path, digest: str) -> str:
        filename = f"{digest[:16]}-{path.name}"
        key = f"{str(policy['upload_dir']).rstrip('/')}/{filename}"
        fields = [
            ("OSSAccessKeyId", str(policy["oss_access_key_id"])),
            ("Signature", str(policy["signature"])),
            ("policy", str(policy["policy"])),
            ("x-oss-object-acl", str(policy["x_oss_object_acl"])),
            ("x-oss-forbid-overwrite", str(policy["x_oss_forbid_overwrite"])),
            ("key", key),
            ("success_action_status", "200"),
        ]
        body, content_type = _multipart_body(fields, path)
        status, _ = self.requester(
            "POST",
            str(policy["upload_host"]),
            {"Content-Type": content_type},
            body,
            self.request_timeout_s,
        )
        if status != 200:
            raise HTTPStatusFailure(status, "upload")
        return f"oss://{key}"
