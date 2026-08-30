"""Strict semantic verification for realized SalesBench QA candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from ..utils import clean_text
from ..vlm.api_client import VLMClient
from .prompts import build_qa_quality_prompt
from .specs import QuestionSpec


class QASemanticVerdict(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


_QUALITY_FIELDS = (
    "answerable_from_evidence",
    "gold_supported",
    "unique_answer",
    "task_aligned",
    "domain_specific",
)


@dataclass(frozen=True)
class QASemanticVerification:
    spec_id: str
    verdict: QASemanticVerdict
    reason: str
    answerable_from_evidence: bool
    gold_supported: bool
    unique_answer: bool
    task_aligned: bool
    domain_specific: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "verdict": self.verdict.value,
            "reason": self.reason,
            **{field: getattr(self, field) for field in _QUALITY_FIELDS},
        }


def parse_qa_semantic_verification(
    raw_response: str,
    expected_spec_id: str,
) -> QASemanticVerification:
    try:
        payload = json.loads(raw_response or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid QA semantic verifier JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("QA semantic verifier response must be an object")
    spec_id = clean_text(payload.get("spec_id"))
    if spec_id != expected_spec_id:
        raise ValueError("QA semantic verifier changed spec_id")
    reason = clean_text(payload.get("reason"))
    if not reason:
        raise ValueError("QA semantic verifier reason is required")
    values: dict[str, bool] = {}
    for field in _QUALITY_FIELDS:
        value = payload.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"QA semantic verifier requires boolean {field}")
        values[field] = value
    try:
        verdict = QASemanticVerdict(clean_text(payload.get("verdict")).upper())
    except ValueError as exc:
        raise ValueError("invalid QA semantic verifier verdict") from exc
    failed_dimensions = [field for field, value in values.items() if not value]
    if verdict == QASemanticVerdict.PASS and failed_dimensions:
        verdict = QASemanticVerdict.REJECT
        reason = f"{reason} Failed quality dimensions: {', '.join(failed_dimensions)}."
    return QASemanticVerification(
        spec_id=spec_id,
        verdict=verdict,
        reason=reason,
        **values,
    )


def verify_qa_candidate(
    client: VLMClient,
    spec: QuestionSpec,
    question: str,
    evidence_context: list[dict[str, object]],
    cue_context: list[dict[str, object]],
    relation_context: list[dict[str, object]],
) -> QASemanticVerification:
    system, user = build_qa_quality_prompt(
        spec,
        question,
        evidence_context,
        cue_context,
        relation_context,
    )
    call = client.call_text_only(system, user, response_format="json_object")
    if not call.success:
        raise ValueError(call.error or "QA semantic verifier model call failed")
    return parse_qa_semantic_verification(call.raw_response, spec.spec_id)
