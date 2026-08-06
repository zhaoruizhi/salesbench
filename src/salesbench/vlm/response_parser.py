"""Small JSON extraction helper for OpenAI-compatible model responses."""

from __future__ import annotations

import json
import re


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code fence
    json_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find nested JSON (with inner braces)
    deep_match = re.search(r"\{.*\}", text, re.DOTALL)
    if deep_match:
        try:
            return json.loads(deep_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
