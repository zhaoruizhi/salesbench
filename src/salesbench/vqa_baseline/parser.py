"""Parsers for closed-source baseline responses."""

from __future__ import annotations

import re

from ..utils import clean_text


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def parse_closed_source_response(raw_response: str) -> dict[str, str]:
    """Read a plain final answer while accepting old `<answer>` wrappers."""
    text = raw_response or ""
    answer_match = ANSWER_RE.search(text)
    warnings: list[str] = []

    if answer_match:
        answer = clean_text(answer_match.group(1))
    else:
        answer = clean_text(text)

    return {
        "thought": "",
        "answer": answer,
        "parse_warning": ",".join(warnings),
    }
