from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from salesbench.vlm.api_client import VLMClient  # noqa: E402


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        usage=None,
    )


def test_deepseek_disables_default_thinking_and_owns_retries() -> None:
    with patch("openai.OpenAI") as factory:
        factory.return_value.chat.completions.create.return_value = _response()
        client = VLMClient(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            disable_thinking=True,
            request_timeout_s=180,
            retry_max=1,
            rate_limit_rpm=0,
        )

        result = client.call_text_only("system", "user", response_format="json_object")

    assert result.success
    assert factory.call_args.kwargs["timeout"] == 180
    assert factory.call_args.kwargs["max_retries"] == 0
    request = factory.return_value.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_qwen_disables_default_thinking() -> None:
    with patch("openai.OpenAI") as factory:
        factory.return_value.chat.completions.create.return_value = _response()
        client = VLMClient(
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3-vl-plus",
            disable_thinking=True,
            retry_max=1,
            rate_limit_rpm=0,
        )

        result = client.call_text_only("system", "user")

    assert result.success
    request = factory.return_value.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"enable_thinking": False}


def test_unknown_provider_does_not_receive_provider_specific_thinking_fields() -> None:
    with patch("openai.OpenAI") as factory:
        factory.return_value.chat.completions.create.return_value = _response()
        client = VLMClient(
            api_key="test-key",
            base_url="https://relay.example/v1",
            model="custom-model",
            disable_thinking=True,
            retry_max=1,
            rate_limit_rpm=0,
        )

        result = client.call_text_only("system", "user")

    assert result.success
    request = factory.return_value.chat.completions.create.call_args.kwargs
    assert "extra_body" not in request


def test_default_headers_are_bound_to_openai_compatible_sdk_client() -> None:
    with patch("openai.OpenAI") as factory:
        factory.return_value.chat.completions.create.return_value = _response()
        client = VLMClient(
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.7-plus",
            default_headers={"X-DashScope-OssResourceResolve": "enable"},
            retry_max=1,
            rate_limit_rpm=0,
        )

        result = client.call_text_only("system", "user")

    assert result.success
    assert factory.call_args.kwargs["default_headers"] == {
        "X-DashScope-OssResourceResolve": "enable"
    }


def test_default_headers_are_sent_by_http_fallback() -> None:
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

    def urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        return Response()

    with patch("openai.OpenAI") as factory, patch(
        "urllib.request.urlopen", side_effect=urlopen
    ):
        client = VLMClient(
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.7-plus",
            default_headers={"X-DashScope-OssResourceResolve": "enable"},
            retry_max=1,
            rate_limit_rpm=0,
        )
        client.client = None
        result = client.call_text_only("system", "user")

    assert result.success
    assert captured["headers"]["X-dashscope-ossresourceresolve"] == "enable"
