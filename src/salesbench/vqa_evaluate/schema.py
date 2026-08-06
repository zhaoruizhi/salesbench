"""Shared schema constants for SalesBench-QA judge evaluation."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text


ALLOWED_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
TASK_TYPES = ("BP", "CM", "SS", "AE")


def coerce_allowed_score(value: object) -> float:
    """Convert a raw judge score to one of the five allowed SalesBench scores."""
    if isinstance(value, str):
        value = value.strip()
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid judge score: {value!r}") from exc
    for allowed in ALLOWED_SCORES:
        if abs(score - allowed) < 1e-9:
            return allowed
    raise ValueError(f"score must be one of {ALLOWED_SCORES}, got {score}")


def normalize_task_type(value: object) -> str:
    text = clean_text(value).upper()
    return text if text in TASK_TYPES else text


def judge_success(record: dict[str, Any]) -> bool:
    return bool(record.get("judge_success")) and record.get("score") is not None
