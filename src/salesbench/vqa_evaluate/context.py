"""Evidence-context assembly for SalesBench-QA judge evaluation."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text
from ..vqa.schema import answer_field
from ..goldbank.validators import PRIVATE_KEYS
from .schema import normalize_task_type


def _strip_private(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _strip_private(value)
            for key, value in payload.items()
            if clean_text(key) not in PRIVATE_KEYS
        }
    if isinstance(payload, list):
        return [_strip_private(value) for value in payload]
    return payload


def build_evidence_context(gold_record: dict[str, Any]) -> dict[str, Any]:
    evidence = gold_record.get("evidence_context")
    if evidence is None:
        evidence = gold_record.get("evidence_refs") or []
    return {
        "evidence_refs": _strip_private(evidence),
        "answer_type": clean_text(gold_record.get("answer_type")) or "open",
    }


def build_judge_payload(gold_record: dict[str, Any], answer_record: dict[str, Any]) -> dict[str, Any]:
    task_type = normalize_task_type(gold_record.get("task_type"))
    payload: dict[str, Any] = {
        "vqa_id": clean_text(gold_record.get("vqa_id")),
        "video_id": clean_text(gold_record.get("video_id")),
        "question": clean_text(gold_record.get("question")),
        "task_type": task_type,
        "reference_answer": gold_record.get("gold_answer"),
        "model_output": answer_field(answer_record),
        "evidence_context": build_evidence_context(gold_record),
    }
    return payload
