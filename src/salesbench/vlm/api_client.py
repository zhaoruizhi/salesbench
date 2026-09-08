"""Unified VLM/LLM API client with retry and rate limiting."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class APICallResult:
    raw_response: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    success: bool
    error: str | None = None


# Pricing per 1M tokens (as of 2025-01)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "image_per_token": 85},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "image_per_token": 85},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "image_per_token": 85},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "image_per_token": 85},
}


def _disabled_thinking_body(base_url: str, model: str) -> dict[str, object]:
    normalized_url = base_url.lower()
    normalized_model = model.lower()
    if "api.deepseek.com" in normalized_url and normalized_model.startswith("deepseek-v4"):
        return {"thinking": {"type": "disabled"}}
    if (
        normalized_model.startswith("qwen")
        and ("dashscope" in normalized_url or "maas.aliyuncs.com" in normalized_url)
    ):
        return {"enable_thinking": False}
    return {}


def _exception_chain(exc: Exception) -> str:
    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        label = type(current).__name__
        messages.append(f"{label}: {message}" if message else label)
        current = current.__cause__ or current.__context__
    return " <- ".join(messages)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model, PRICING.get("gpt-4o"))
    cost = (input_tokens / 1_000_000) * pricing["input"]
    cost += (output_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)


class VLMClient:
    """OpenAI-compatible VLM API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry_max: int = 3,
        retry_backoff_s: float = 2.0,
        rate_limit_rpm: int = 60,
        disable_thinking: bool = False,
        request_timeout_s: float = 180.0,
        default_headers: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.default_headers = dict(default_headers or {})
        self.client = None
        try:
            from openai import OpenAI

            client_kwargs = {
                "api_key": api_key,
                "timeout": request_timeout_s,
                "max_retries": 0,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            if self.default_headers:
                client_kwargs["default_headers"] = self.default_headers
            self.client = OpenAI(**client_kwargs)
        except ModuleNotFoundError:
            self.client = None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_max = retry_max
        self.retry_backoff_s = retry_backoff_s
        self.disable_thinking = disable_thinking
        self.request_timeout_s = request_timeout_s
        self._min_interval = 60.0 / rate_limit_rpm if rate_limit_rpm > 0 else 0
        self._last_call_time = 0.0

    def _rate_limit_wait(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def call(
        self,
        system_prompt: str,
        user_content: list[dict],
        response_format: str | None = None,
    ) -> APICallResult:
        """
        Call VLM API with images and text.

        Args:
            system_prompt: System message
            user_content: List of content blocks, e.g.:
                [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                    {"type": "text", "text": "Please analyze..."},
                ]
            response_format: "json_object" to request JSON output, or None

        Returns:
            APICallResult with response details
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            thinking_body = _disabled_thinking_body(self.base_url, self.model)
            if thinking_body:
                kwargs["extra_body"] = thinking_body

        last_error = None
        for attempt in range(self.retry_max):
            self._rate_limit_wait()
            start_time = time.time()
            try:
                if self.client is not None:
                    response = self.client.chat.completions.create(**kwargs)
                    choice = response.choices[0]
                    raw_text = choice.message.content or ""
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                else:
                    payload = self._post_chat_completions(kwargs)
                    choices = payload.get("choices") or []
                    message = (choices[0].get("message") if choices else {}) or {}
                    raw_text = message.get("content") or ""
                    usage = payload.get("usage") or {}
                    input_tokens = int(usage.get("prompt_tokens") or 0)
                    output_tokens = int(usage.get("completion_tokens") or 0)

                latency = time.time() - start_time
                self._last_call_time = time.time()
                return APICallResult(
                    raw_response=raw_text,
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_s=round(latency, 2),
                    cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
                    success=True,
                )
            except Exception as exc:
                last_error = _exception_chain(exc)
                latency = time.time() - start_time
                if attempt < self.retry_max - 1:
                    wait = self.retry_backoff_s * (2 ** attempt)
                    time.sleep(wait)

        return APICallResult(
            raw_response="",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            cost_usd=0.0,
            success=False,
            error=last_error,
        )

    def _post_chat_completions(self, payload: dict) -> dict:
        """Minimal OpenAI-compatible HTTP fallback when the SDK is unavailable."""
        payload = dict(payload)
        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.default_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def call_text_only(
        self,
        system_prompt: str,
        user_text: str,
        response_format: str | None = None,
    ) -> APICallResult:
        """Convenience method for text-only calls (no images)."""
        user_content = [{"type": "text", "text": user_text}]
        return self.call(system_prompt, user_content, response_format)
