"""Deterministic routing for generated quality-control records.

The production pipeline writes engineering failures and automatic rejections to
separate artifacts.  Only records whose content is genuinely ambiguous are
allowed to reach the human review queue.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..utils import clean_text


class QualityDisposition(str, Enum):
    ACCEPT = "ACCEPT"
    REPAIR = "REPAIR"
    REJECT = "REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class LifecycleStatus(str, Enum):
    AUTO_ACCEPTED_CANDIDATE = "auto_accepted_candidate"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    HUMAN_ACCEPTED = "human_accepted"
    REJECTED = "rejected"


_AUTO_REJECT_REASONS = {
    "EVIDENCE_BELOW_MIN_CONFIDENCE",
    "EVIDENCE_VALIDATION_FAILED",
    "BELOW_MIN_CONFIDENCE",
    "COMMERCE_CUE_BELOW_MIN_CONFIDENCE",
    "COMMERCIAL_RELATION_BELOW_MIN_CONFIDENCE",
    "COMMERCE_CUE_VALIDATION_FAILED",
    "COMMERCIAL_RELATION_VALIDATION_FAILED",
    "PROPOSAL_GRAPH_VALIDATION_FAILED",
    "VALIDATION_FAILED",
    "SEMANTIC_DUPLICATE",
    "CONFLICTING_VALUE",
    "CHALLENGER_REJECT",
    "CHALLENGER_REJECTED_OR_HUMAN_REVIEW",
}

_REPAIR_REASONS = {
    "CHALLENGER_REVISE",
}

_HUMAN_REASONS = {
    "CHALLENGER_HUMAN_REVIEW",
    "SEMANTIC_VERIFIER_AMBIGUOUS",
    "MULTIPLE_PLAUSIBLE_ANSWERS",
}


def _reason_code(record: dict[str, Any]) -> str:
    explicit = clean_text(record.get("reason_code")).upper()
    if explicit:
        return explicit
    return clean_text(record.get("reason")).split(":", 1)[0].upper()


def _decision_id(record: dict[str, Any], reason_code: str) -> str:
    payload = json.dumps(
        {
            "record_id": record.get("review_item_id") or record.get("id"),
            "video_id": record.get("video_id"),
            "stage": record.get("stage"),
            "reason_code": reason_code,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"qd_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class QualityDecision:
    decision_id: str
    record_id: str
    video_id: str
    stage: str
    item_type: str
    disposition: QualityDisposition
    output_channel: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "record_id": self.record_id,
            "video_id": self.video_id,
            "stage": self.stage,
            "item_type": self.item_type,
            "disposition": self.disposition.value,
            "output_channel": self.output_channel,
            "reason_codes": list(self.reason_codes),
        }


def route_quality_record(record: dict[str, Any]) -> QualityDecision:
    """Classify a legacy pipeline queue row into one canonical output.

    Unknown content decisions fail safe to human review.  Known engineering,
    schema, confidence, and abstention events never become human work.
    """

    reason_code = _reason_code(record)
    item_type = clean_text(record.get("item_type")).lower()
    stage = clean_text(record.get("stage")).lower()

    is_parse_failure = "PARSE_ERROR" in reason_code or reason_code.endswith("_FAILED") and item_type == "stage_failure"
    is_abstention = item_type == "abstention" or reason_code in {"ABSTENTION", "NO_COMMERCIAL_RECORD"}

    if is_abstention or is_parse_failure or item_type == "stage_failure":
        disposition = QualityDisposition.REJECT
        output_channel = "pipeline_diagnostics"
    elif reason_code in _REPAIR_REASONS:
        disposition = QualityDisposition.REPAIR
        output_channel = "pipeline_diagnostics"
    elif reason_code in _AUTO_REJECT_REASONS or reason_code.endswith("_VALIDATION_FAILED"):
        disposition = QualityDisposition.REJECT
        output_channel = "rejected_candidates"
    elif reason_code in _HUMAN_REASONS or (
        stage == "adjudication" and reason_code not in {"ADJUDICATOR_OMITTED", "ADJUDICATOR_UNKNOWN_SOURCE"}
    ):
        disposition = QualityDisposition.HUMAN_REVIEW
        output_channel = "human_review_queue"
    elif reason_code.startswith("ADJUDICATOR_"):
        disposition = QualityDisposition.REJECT
        output_channel = "pipeline_diagnostics"
    else:
        disposition = QualityDisposition.HUMAN_REVIEW
        output_channel = "human_review_queue"

    return QualityDecision(
        decision_id=_decision_id(record, reason_code),
        record_id=clean_text(record.get("review_item_id") or record.get("id")),
        video_id=clean_text(record.get("video_id")),
        stage=stage,
        item_type=item_type,
        disposition=disposition,
        output_channel=output_channel,
        reason_codes=(reason_code or "REVIEW_REQUIRED",),
    )
