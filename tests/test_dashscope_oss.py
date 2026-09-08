from __future__ import annotations

import hashlib
import json
import ssl
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from salesbench.vlm.dashscope_oss import (  # noqa: E402
    DashScopeTemporaryOSSUploader,
    DashScopeUploadError,
    HTTPStatusFailure,
    classify_transport_error,
)


def _policy() -> dict[str, object]:
    return {
        "data": {
            "policy": "policy-secret",
            "signature": "signature-secret",
            "upload_dir": "dashscope-instant/account/session",
            "upload_host": "https://upload.example",
            "expire_in_seconds": 300,
            "oss_access_key_id": "temporary-key-secret",
            "x_oss_object_acl": "private",
            "x_oss_forbid_overwrite": "true",
        }
    }


def _frames(root: Path, count: int = 16) -> list[str]:
    paths: list[str] = []
    for index in range(count):
        path = root / f"frame_{index:03d}.jpg"
        path.write_bytes(f"jpeg-{index}".encode("ascii"))
        paths.append(str(path))
    return paths


def test_upload_frames_preserves_all_sixteen_positions_and_reuses_digest_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _frames(root)
        calls: list[tuple[str, str, bytes | None]] = []

        def request(method, url, headers, body, timeout):
            calls.append((method, url, body))
            if method == "GET":
                return 200, json.dumps(_policy()).encode("utf-8")
            return 200, b""

        uploader = DashScopeTemporaryOSSUploader(
            api_key="not-a-real-key",
            cache_root=root / "cache",
            retry_max=1,
            requester=request,
            time_fn=lambda: 1_000.0,
        )
        first = uploader.upload_frames(paths, "qwen3.7-plus")
        second = uploader.upload_frames(paths, "qwen3.7-plus")

    assert len(first) == 16
    assert first == second
    assert all(value.startswith("oss://dashscope-instant/account/session/") for value in first)
    assert len(set(first)) == 16
    assert [method for method, _, _ in calls].count("GET") == 1
    assert [method for method, _, _ in calls].count("POST") == 16
    for index in range(16):
        digest = hashlib.sha256(f"jpeg-{index}".encode("ascii")).hexdigest()[:16]
        assert digest in first[index]


def test_expired_cached_url_is_uploaded_again() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = _frames(root, 1)[0]
        now = [1_000.0]
        call_count = 0

        def request(method, url, headers, body, timeout):
            nonlocal call_count
            call_count += 1
            if method == "GET":
                return 200, json.dumps(_policy()).encode("utf-8")
            return 200, b""

        uploader = DashScopeTemporaryOSSUploader(
            api_key="not-a-real-key",
            cache_root=root / "cache",
            retry_max=1,
            requester=request,
            time_fn=lambda: now[0],
            object_ttl_s=100,
            expiry_margin_s=10,
        )
        uploader.upload_frames([path], "qwen3.7-plus")
        now[0] = 1_091.0
        uploader.upload_frames([path], "qwen3.7-plus")

    assert call_count == 4


def test_retryable_ssl_error_refreshes_policy_and_retries_without_leaking_credentials() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = _frames(root, 1)[0]
        get_calls = 0
        post_calls = 0
        sleeps: list[float] = []

        def request(method, url, headers, body, timeout):
            nonlocal get_calls, post_calls
            if method == "GET":
                get_calls += 1
                return 200, json.dumps(_policy()).encode("utf-8")
            post_calls += 1
            if post_calls == 1:
                raise ssl.SSLError("signature-secret wire failure")
            return 200, b""

        uploader = DashScopeTemporaryOSSUploader(
            api_key="not-a-real-key",
            cache_root=root / "cache",
            retry_max=2,
            retry_backoff_s=0.25,
            requester=request,
            sleep_fn=sleeps.append,
        )
        values = uploader.upload_frames([path], "qwen3.7-plus")

    assert len(values) == 1
    assert get_calls == 2
    assert post_calls == 2
    assert sleeps == [0.25]


def test_non_retryable_authentication_error_fails_once_with_redacted_message() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = _frames(root, 1)[0]
        calls = 0

        def request(method, url, headers, body, timeout):
            nonlocal calls
            calls += 1
            raise HTTPStatusFailure(401, "policy", "do-not-print-response-body")

        uploader = DashScopeTemporaryOSSUploader(
            api_key="not-a-real-key",
            cache_root=root / "cache",
            retry_max=5,
            requester=request,
            sleep_fn=lambda _: None,
        )
        with pytest.raises(DashScopeUploadError) as caught:
            uploader.upload_frames([path], "qwen3.7-plus")

    assert calls == 1
    assert "401" in str(caught.value)
    assert "do-not-print-response-body" not in str(caught.value)
    assert "not-a-real-key" not in str(caught.value)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ssl.SSLError("EOF"), "retryable"),
        (BrokenPipeError("closed"), "retryable"),
        (HTTPStatusFailure(429, "upload"), "retryable"),
        (HTTPStatusFailure(503, "upload"), "retryable"),
        (HTTPStatusFailure(400, "upload"), "non_retryable"),
        (ValueError("bad local state"), "unknown"),
    ],
)
def test_transport_error_classification(error: Exception, expected: str) -> None:
    assert classify_transport_error(error) == expected
