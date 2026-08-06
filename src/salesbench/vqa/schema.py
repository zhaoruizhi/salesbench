"""Shared helpers for the four-task VQA prediction contract."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text


PUBLIC_TASKS = ("BP", "CM", "SS", "AE")
GOLD_KEYS = {
    "gold_answer",
    "answer",
    "evidence_refs",
    "evidence_context",
    "quality_status",
    "review_status",
    "source_annotation_ids",
    "private_metadata",
    "private_analysis_metadata",
}


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def answer_field(record: dict[str, Any]) -> object:
    for key in ("answer", "pred_answer", "model_answer", "prediction"):
        if key in record:
            return record.get(key)
    return None


def public_vqa_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in GOLD_KEYS}


def validate_prediction_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not clean_text(record.get("vqa_id")):
        errors.append("missing_vqa_id")
    if answer_field(record) is None:
        errors.append("missing_answer_field")
    return errors
