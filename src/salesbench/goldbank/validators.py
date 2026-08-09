"""Deterministic EvidenceDataset validation and public/private firewall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils import clean_text
from .normalizer import semantic_key, semantic_target_key
from .ontology import CM_RELATIONS, TASK_MIN_EVIDENCE, allowed_subtypes
from .schema import EvidenceUnit, GoldItem, GoldProposal, GoldTaskType, GoldTier


PRIVATE_KEYS = {
    "likes",
    "collects",
    "shares",
    "comments",
    "like_response_score",
    "collect_response_score",
    "share_response_score",
    "comment_response_score",
    "raw_value",
    "thresholds",
    "performance_data",
    "private_analysis_metadata",
    "raw_snapshot_counts",
    "log1p_counts",
    "interaction_strata",
    "followers_total",
    "followers_million",
    "fan_segment",
    "publish_context",
    "title",
    "product_title",
}

DIRECT_EVIDENCE_MODALITIES = {"visual", "asr", "ocr"}

INFERENCE_MARKERS = (
    "probably",
    "likely",
    "trust",
    "motivation",
    "audience",
    "strategy",
    "appeal",
    "可能",
    "大概",
    "信任",
    "受众",
    "策略",
    "动机",
)

CAUSAL_MARKERS = ("caused", "because of this", "conversion", "sales", "购买转化", "销售", "归因")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    item_id: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "item_id": self.item_id,
            "message": self.message,
        }


def _item_id(item: GoldItem | GoldProposal | EvidenceUnit) -> str:
    return getattr(item, "gold_id", None) or getattr(item, "proposal_id", None) or getattr(item, "evidence_id", "")


def _contains_key(payload: object, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if clean_text(key) in keys or _contains_key(value, keys):
                return True
    elif isinstance(payload, list):
        return any(_contains_key(value, keys) for value in payload)
    return False


def _without_private(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _without_private(value)
            for key, value in payload.items()
            if clean_text(key) not in PRIVATE_KEYS
        }
    if isinstance(payload, list):
        return [_without_private(value) for value in payload]
    return payload


def _text_blob(payload: object) -> str:
    if isinstance(payload, dict):
        return " ".join(_text_blob(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return " ".join(_text_blob(value) for value in payload)
    return clean_text(payload).lower()


def validate_evidence_unit(unit: EvidenceUnit) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not unit.evidence_id or not unit.video_id:
        issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", unit.evidence_id, "Evidence requires ids"))
    if not unit.subject or not unit.predicate or unit.value in (None, ""):
        issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", unit.evidence_id, "Evidence requires subject, predicate, value"))
    if unit.start_s is not None and unit.end_s is not None and unit.start_s > unit.end_s:
        issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", unit.evidence_id, "start_s exceeds end_s"))
    if unit.modality.value not in DIRECT_EVIDENCE_MODALITIES:
        issues.append(
            ValidationIssue(
                "NON_DIRECT_EVIDENCE",
                "ERROR",
                unit.evidence_id,
                "Direct evidence must come from visual frames, in-frame OCR, or ASR",
            )
        )
    if unit.modality.value == "visual" and not unit.frame_indices:
        issues.append(ValidationIssue("MISSING_FRAME_REFERENCE", "ERROR", unit.evidence_id, "Visual evidence requires frame_indices"))
    if unit.modality.value in {"asr", "ocr"} and not unit.text_span:
        issues.append(ValidationIssue("MISSING_TEXT_SPAN", "ERROR", unit.evidence_id, "ASR/OCR evidence requires text_span"))
    return issues


def _validate_common(
    item: GoldItem | GoldProposal,
    evidence: dict[str, EvidenceUnit],
    evidence_ids: tuple[str, ...],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    item_id = _item_id(item)
    try:
        allowed_subtypes(item.task_type)
        if item.task_subtype not in allowed_subtypes(item.task_type):
            issues.append(ValidationIssue("UNKNOWN_TASK_SUBTYPE", "ERROR", item_id, f"Unknown subtype: {item.task_subtype}"))
    except ValueError:
        issues.append(ValidationIssue("UNKNOWN_TASK_SUBTYPE", "ERROR", item_id, f"Unknown task: {item.task_type}"))

    for evidence_id in evidence_ids:
        unit = evidence.get(evidence_id)
        if unit is None:
            issues.append(ValidationIssue("EVIDENCE_NOT_FOUND", "ERROR", item_id, f"Missing evidence: {evidence_id}"))
            continue
        if unit.video_id != item.video_id:
            issues.append(ValidationIssue("EVIDENCE_VIDEO_MISMATCH", "ERROR", item_id, f"Evidence {evidence_id} belongs to {unit.video_id}"))

    return issues


def validate_gold_proposal(proposal: GoldProposal, evidence: dict[str, EvidenceUnit]) -> list[ValidationIssue]:
    issues = _validate_common(proposal, evidence, proposal.evidence_ids)
    if len(proposal.evidence_ids) < TASK_MIN_EVIDENCE.get(proposal.task_type, 1):
        issues.append(ValidationIssue("INSUFFICIENT_EVIDENCE", "WARNING", proposal.proposal_id, "Proposal has too little evidence"))
    if proposal.task_type == GoldTaskType.CM:
        modalities = {evidence[eid].modality.value for eid in set(proposal.evidence_ids) if eid in evidence}
        if len(modalities) < 2:
            issues.append(
                ValidationIssue(
                    "CM_MODALITY_DIVERSITY",
                    "ERROR",
                    proposal.proposal_id,
                    "CM requires evidence from at least two distinct modalities",
                )
            )
    return issues


def validate_gold_item(item: GoldItem, evidence: dict[str, EvidenceUnit]) -> list[ValidationIssue]:
    issues = _validate_common(item, evidence, item.evidence_ids)
    item_id = item.gold_id
    min_evidence = TASK_MIN_EVIDENCE.get(item.task_type, 1)
    if len(set(item.evidence_ids)) < min_evidence:
        severity = "ERROR" if item.gold_tier in {GoldTier.GOLD_A, GoldTier.GOLD_B} else "WARNING"
        issues.append(ValidationIssue("INSUFFICIENT_EVIDENCE", severity, item_id, f"{item.task_type.value} requires {min_evidence} evidence ids"))
    if item.task_type == GoldTaskType.BP and any(marker in _text_blob(item.gold_value) for marker in INFERENCE_MARKERS):
        issues.append(ValidationIssue("OBSERVATION_INFERENCE_MIXED", "ERROR", item_id, "BP must not contain marketing inference language"))
    if item.task_type == GoldTaskType.CM:
        modalities = {evidence[eid].modality.value for eid in set(item.evidence_ids) if eid in evidence}
        if len(modalities) < 2:
            issues.append(
                ValidationIssue(
                    "CM_MODALITY_DIVERSITY",
                    "ERROR",
                    item_id,
                    "CM requires evidence from at least two distinct modalities",
                )
            )
        relation = clean_text(item.gold_value.get("relation")).upper()
        if relation and relation not in CM_RELATIONS:
            issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", item_id, f"Invalid CM relation: {relation}"))
    if any(marker in _text_blob(item.gold_value) for marker in CAUSAL_MARKERS):
        issues.append(ValidationIssue("UNSUPPORTED_CAUSAL_CLAIM", "ERROR", item_id, "VQA Gold cannot claim sales or interaction causation"))
    if _contains_key(item.to_dict(), PRIVATE_KEYS):
        issues.append(ValidationIssue("PRIVATE_FIELD_LEAK", "ERROR", item_id, "Private performance field appears in public item"))
    if item.gold_tier == GoldTier.GOLD_A and item.task_type in {GoldTaskType.SS, GoldTaskType.AE} and item.review_status != "human_accepted":
        issues.append(ValidationIssue("INVALID_TIER", "ERROR", item_id, "SS/AE automatic items cannot be Gold-A"))
    return issues


def find_duplicate_and_conflicting_items(items: list[GoldItem]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_semantics: dict[tuple[str, str, str, str, str], GoldItem] = {}
    by_target: dict[tuple[str, str, str, str], GoldItem] = {}
    for item in items:
        key = semantic_key(item)
        if key in by_semantics:
            issues.append(ValidationIssue("DUPLICATE_SEMANTICS", "WARNING", item.gold_id, f"Duplicates {by_semantics[key].gold_id}"))
        else:
            by_semantics[key] = item

        target_key = semantic_target_key(item)
        if target_key in by_target and semantic_key(by_target[target_key]) != key:
            issues.append(ValidationIssue("CONFLICTING_VALUE", "ERROR", item.gold_id, f"Conflicts with {by_target[target_key].gold_id}"))
        else:
            by_target[target_key] = item
    return issues


def public_gold_record(item: GoldItem) -> dict[str, object]:
    return _without_private(item.to_dict())


def contains_private_fields(payload: object) -> bool:
    return _contains_key(payload, PRIVATE_KEYS)
