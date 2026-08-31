"""Deterministic EvidenceDataset validation and public/private firewall."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..utils import clean_text, contains_cjk
from .commerce_ontology import DEMONSTRATION_CUES, relation_rule
from .commerce_schema import CommerceCue, CommercialRelation, CueType, RelationType
from .normalizer import semantic_key, semantic_target_key
from .ontology import (
    CM_RELATIONS,
    TASK_MIN_EVIDENCE,
    allowed_subtypes,
    capability_level,
    default_reasoning_operator,
)
from .schema import (
    EvidenceAssertionType,
    EvidenceModality,
    EvidenceTemporalScope,
    EvidenceUnit,
    GoldItem,
    GoldProposal,
    GoldTaskType,
    GoldTier,
)


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

CONSUMER_OUTCOME_MARKERS = (
    "causes purchase",
    "cause a purchase",
    "makes the viewer purchase",
    "makes viewers purchase",
    "will purchase",
    "increases trust",
    "makes the viewer trust",
    "makes viewers trust",
    "improves conversion",
    "reduces uncertainty",
    "became less uncertain",
    "becomes less uncertain",
)

CREATOR_METADATA_MARKERS = (
    "creator handle",
    "account handle",
    "account identifier",
    "username",
    "douyin handle",
    "tiktok handle",
    "抖音号",
    "账号",
)

_CLAIM_FACT_CUE_TYPES = {
    CueType.PRODUCT_ATTRIBUTE,
    CueType.PRODUCT_VARIANT,
    CueType.PRODUCT_DESCRIPTION,
}

_CLAIM_ATTRIBUTION_MARKERS = (
    "speaker",
    "presenter",
    "video presents",
    "video frames",
    "video claims",
    "on-screen text",
    "according to",
    "is described as",
    "is promoted as",
)

_NUMERIC_CUE_TYPES = {
    CueType.QUANTITY,
    CueType.BUNDLE,
    CueType.PRICE,
    CueType.DISCOUNT,
    CueType.OFFER_CONDITION,
}

_RELATION_STATUS_CONTRACT = {
    RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION: "SUPPORTED",
    RelationType.CLAIM_PARTIALLY_SUPPORTED: "PARTIALLY_SUPPORTED",
    RelationType.CLAIM_CONTRADICTED: "CONTRADICTED",
    RelationType.CLAIM_TEMPORALLY_MISALIGNED: "TEMPORALLY_MISALIGNED",
}

_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")


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


def _item_id(item: GoldItem | GoldProposal | EvidenceUnit | CommerceCue | CommercialRelation) -> str:
    return (
        getattr(item, "gold_id", None)
        or getattr(item, "proposal_id", None)
        or getattr(item, "evidence_id", None)
        or getattr(item, "cue_id", None)
        or getattr(item, "relation_id", "")
    )


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


def _normalize_numeric_token(value: str) -> str:
    token = clean_text(value).replace(",", "")
    try:
        number = float(token)
    except ValueError:
        return token
    return str(int(number)) if number.is_integer() else f"{number:g}"


def validate_evidence_unit(unit: EvidenceUnit) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not unit.evidence_id or not unit.video_id:
        issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", unit.evidence_id, "Evidence requires ids"))
    if not unit.subject or not unit.predicate or unit.value in (None, ""):
        issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", unit.evidence_id, "Evidence requires subject, predicate, value"))
    if unit.start_s is not None and unit.end_s is not None and unit.start_s > unit.end_s:
        issues.append(ValidationIssue("INVALID_GOLD_VALUE", "ERROR", unit.evidence_id, "start_s exceeds end_s"))
    if unit.confidence <= 0.0:
        issues.append(
            ValidationIssue(
                "INVALID_CONFIDENCE",
                "ERROR",
                unit.evidence_id,
                "Evidence confidence must be greater than zero",
            )
        )
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
    if unit.modality in {EvidenceModality.ASR, EvidenceModality.OCR} and (
        unit.start_s is None or unit.end_s is None
    ):
        issues.append(
            ValidationIssue(
                "MISSING_TEMPORAL_LOCALIZATION",
                "ERROR",
                unit.evidence_id,
                "ASR/OCR evidence requires both start_s and end_s",
            )
        )
    expected_assertion = {
        EvidenceModality.VISUAL: EvidenceAssertionType.OBSERVED,
        EvidenceModality.ASR: EvidenceAssertionType.SPOKEN_CLAIM,
        EvidenceModality.OCR: EvidenceAssertionType.OCR_TEXT,
    }.get(unit.modality)
    if expected_assertion is not None and unit.assertion_type != expected_assertion:
        issues.append(
            ValidationIssue(
                "ASSERTION_MODALITY_MISMATCH",
                "ERROR",
                unit.evidence_id,
                f"{unit.modality.value} evidence must use {expected_assertion.value}",
            )
        )
    allowed_scopes = {
        EvidenceModality.VISUAL: {EvidenceTemporalScope.FRAME, EvidenceTemporalScope.SHORT_CLIP},
        EvidenceModality.OCR: {EvidenceTemporalScope.FRAME, EvidenceTemporalScope.SHORT_CLIP},
        EvidenceModality.ASR: {
            EvidenceTemporalScope.SHORT_CLIP,
            EvidenceTemporalScope.FULL_VIDEO,
            EvidenceTemporalScope.LONG_TERM_CLAIM,
        },
    }.get(unit.modality, set())
    if allowed_scopes and unit.temporal_scope not in allowed_scopes:
        issues.append(
            ValidationIssue(
                "TEMPORAL_SCOPE_MODALITY_MISMATCH",
                "ERROR",
                unit.evidence_id,
                f"{unit.temporal_scope.value} is invalid for {unit.modality.value} evidence",
            )
        )
    if contains_cjk((unit.subject, unit.predicate, unit.value, unit.content_en, unit.attributes)):
        issues.append(
            ValidationIssue(
                "NON_ENGLISH_NORMALIZED_TEXT",
                "ERROR",
                unit.evidence_id,
                "Evidence normalized semantic fields must use English; native text belongs in source_text_native",
            )
        )
    evidence_text = _text_blob(
        (unit.subject, unit.predicate, unit.value, unit.content_en, unit.source_text_native)
    )
    if any(marker in evidence_text for marker in CREATOR_METADATA_MARKERS):
        issues.append(
            ValidationIssue(
                "CREATOR_METADATA_LEAK",
                "ERROR",
                unit.evidence_id,
                "Creator handles and account identifiers are not public benchmark evidence",
            )
        )
    return issues


def validate_commerce_cue(
    cue: CommerceCue,
    evidence: dict[str, EvidenceUnit],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not cue.cue_id or not cue.video_id or not cue.content_en:
        issues.append(ValidationIssue("INVALID_COMMERCE_CUE", "ERROR", cue.cue_id, "Cue requires ids and content_en"))
    if contains_cjk(cue.content_en):
        issues.append(
            ValidationIssue(
                "NON_ENGLISH_CANONICAL_TEXT",
                "ERROR",
                cue.cue_id,
                "CommerceCue content_en must use English",
            )
        )
    if not cue.evidence_ids:
        issues.append(ValidationIssue("INSUFFICIENT_EVIDENCE", "ERROR", cue.cue_id, "Cue requires evidence"))
    for evidence_id in cue.evidence_ids:
        unit = evidence.get(evidence_id)
        if unit is None:
            issues.append(ValidationIssue("EVIDENCE_NOT_FOUND", "ERROR", cue.cue_id, f"Missing evidence: {evidence_id}"))
        elif unit.video_id != cue.video_id:
            issues.append(
                ValidationIssue(
                    "EVIDENCE_VIDEO_MISMATCH",
                    "ERROR",
                    cue.cue_id,
                    f"Evidence {evidence_id} belongs to {unit.video_id}",
                )
            )
    cited_units = [evidence[evidence_id] for evidence_id in cue.evidence_ids if evidence_id in evidence]
    if cue.cue_type in _CLAIM_FACT_CUE_TYPES and cited_units and all(
        unit.assertion_type == EvidenceAssertionType.SPOKEN_CLAIM for unit in cited_units
    ):
        issues.append(
            ValidationIssue(
                "CLAIM_AS_PRODUCT_ATTRIBUTE",
                "ERROR",
                cue.cue_id,
                "A seller claim cannot be normalized as an observed product attribute",
            )
        )
    if (
        cue.cue_type == CueType.BENEFIT
        and cited_units
        and all(unit.assertion_type == EvidenceAssertionType.SPOKEN_CLAIM for unit in cited_units)
        and not any(marker in cue.content_en.lower() for marker in _CLAIM_ATTRIBUTION_MARKERS)
    ):
        issues.append(
            ValidationIssue(
                "UNATTRIBUTED_PROMOTIONAL_CLAIM",
                "ERROR",
                cue.cue_id,
                "A seller-stated benefit must remain explicitly attributed to the speaker or video",
            )
        )
    if cue.cue_type in _NUMERIC_CUE_TYPES:
        cue_numbers = {_normalize_numeric_token(value) for value in _NUMBER_RE.findall(cue.content_en)}
        evidence_numbers = {
            _normalize_numeric_token(value)
            for value in _NUMBER_RE.findall(_text_blob([unit.to_dict() for unit in cited_units]))
        }
        missing_numbers = sorted(cue_numbers - evidence_numbers)
        if missing_numbers:
            issues.append(
                ValidationIssue(
                    "NUMERIC_VALUE_NOT_IN_EVIDENCE",
                    "ERROR",
                    cue.cue_id,
                    f"Numeric values are not present in cited evidence: {', '.join(missing_numbers)}",
                )
            )
    if cue.cue_type in DEMONSTRATION_CUES and not any(
        evidence_id in evidence and evidence[evidence_id].modality.value == "visual"
        for evidence_id in cue.evidence_ids
    ):
        issues.append(
            ValidationIssue(
                "DEMONSTRATION_REQUIRES_VISUAL_EVIDENCE",
                "ERROR",
                cue.cue_id,
                "Demonstration cues require a localized visual EvidenceUnit",
            )
        )
    if cue.directness not in {"DIRECT", "INFERRED", "NEEDS_REVIEW"}:
        issues.append(ValidationIssue("INVALID_DIRECTNESS", "ERROR", cue.cue_id, cue.directness))
    if contains_private_fields(cue.to_dict()):
        issues.append(ValidationIssue("PRIVATE_FIELD_LEAK", "ERROR", cue.cue_id, "Private field appears in cue"))
    return issues


def validate_commercial_relation(
    relation: CommercialRelation,
    cues: dict[str, CommerceCue],
    evidence: dict[str, EvidenceUnit],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    rule = relation_rule(relation.relation_type)
    source_types = rule["source"]
    target_types = rule["target"]

    if not relation.source_cue_ids or not relation.target_cue_ids:
        issues.append(
            ValidationIssue("MISSING_RELATION_ENDPOINT", "ERROR", relation.relation_id, "Relation requires source and target cues")
        )

    for cue_id in relation.source_cue_ids:
        cue = cues.get(cue_id)
        if cue is None:
            issues.append(ValidationIssue("CUE_NOT_FOUND", "ERROR", relation.relation_id, f"Missing cue: {cue_id}"))
            continue
        if cue.video_id != relation.video_id:
            issues.append(ValidationIssue("CUE_VIDEO_MISMATCH", "ERROR", relation.relation_id, cue_id))
        if cue.cue_type not in source_types:
            issues.append(
                ValidationIssue(
                    "INVALID_RELATION_SOURCE_TYPE",
                    "ERROR",
                    relation.relation_id,
                    f"{cue.cue_type.value} cannot be a source for {relation.relation_type.value}",
                )
            )

    for cue_id in relation.target_cue_ids:
        cue = cues.get(cue_id)
        if cue is None:
            issues.append(ValidationIssue("CUE_NOT_FOUND", "ERROR", relation.relation_id, f"Missing cue: {cue_id}"))
            continue
        if cue.video_id != relation.video_id:
            issues.append(ValidationIssue("CUE_VIDEO_MISMATCH", "ERROR", relation.relation_id, cue_id))
        if cue.cue_type not in target_types:
            issues.append(
                ValidationIssue(
                    "INVALID_RELATION_TARGET_TYPE",
                    "ERROR",
                    relation.relation_id,
                    f"{cue.cue_type.value} cannot be a target for {relation.relation_type.value}",
                )
            )

    if relation.relation_type in {
        RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION,
        RelationType.CLAIM_PARTIALLY_SUPPORTED,
        RelationType.CLAIM_TEMPORALLY_MISALIGNED,
    }:
        target_has_visual = any(
            evidence_id in evidence and evidence[evidence_id].modality.value == "visual"
            for cue_id in relation.target_cue_ids
            if cue_id in cues
            for evidence_id in cues[cue_id].evidence_ids
        )
        if not target_has_visual:
            issues.append(
                ValidationIssue(
                    "DEMONSTRATION_REQUIRES_VISUAL_EVIDENCE",
                    "ERROR",
                    relation.relation_id,
                    "Claim-demonstration relations require visual evidence at the demonstration endpoint",
                )
            )

    expected_status = _RELATION_STATUS_CONTRACT.get(relation.relation_type)
    if expected_status is not None and relation.status != expected_status:
        issues.append(
            ValidationIssue(
                "RELATION_STATUS_MISMATCH",
                "ERROR",
                relation.relation_id,
                f"{relation.relation_type.value} requires status={expected_status}",
            )
        )

    if relation.relation_type == RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION:
        source_units = [
            evidence[evidence_id]
            for cue_id in relation.source_cue_ids
            if cue_id in cues
            for evidence_id in cues[cue_id].evidence_ids
            if evidence_id in evidence
        ]
        target_units = [
            evidence[evidence_id]
            for cue_id in relation.target_cue_ids
            if cue_id in cues
            for evidence_id in cues[cue_id].evidence_ids
            if evidence_id in evidence
        ]
        has_long_term_claim = any(
            unit.temporal_scope == EvidenceTemporalScope.LONG_TERM_CLAIM for unit in source_units
        )
        has_matching_observation_scope = any(
            unit.temporal_scope == EvidenceTemporalScope.FULL_VIDEO for unit in target_units
        )
        if has_long_term_claim and not has_matching_observation_scope:
            issues.append(
                ValidationIssue(
                    "UNSUPPORTED_CLAIM_SCOPE",
                    "ERROR",
                    relation.relation_id,
                    "A short visual demonstration cannot fully support a long-term seller claim",
                )
            )

    minimum = int(rule["min_evidence"])
    if len(set(relation.evidence_ids)) < minimum:
        issues.append(
            ValidationIssue(
                "INSUFFICIENT_EVIDENCE",
                "ERROR",
                relation.relation_id,
                f"Relation requires at least {minimum} distinct evidence units",
            )
        )
    for evidence_id in relation.evidence_ids:
        unit = evidence.get(evidence_id)
        if unit is None:
            issues.append(ValidationIssue("EVIDENCE_NOT_FOUND", "ERROR", relation.relation_id, evidence_id))
        elif unit.video_id != relation.video_id:
            issues.append(ValidationIssue("EVIDENCE_VIDEO_MISMATCH", "ERROR", relation.relation_id, evidence_id))

    if relation.provenance != rule["provenance"]:
        issues.append(
            ValidationIssue(
                "INVALID_RELATION_PROVENANCE",
                "ERROR",
                relation.relation_id,
                f"Expected {rule['provenance'].value}",
            )
        )
    if not relation.rationale_en or contains_cjk(relation.rationale_en):
        issues.append(
            ValidationIssue(
                "NON_ENGLISH_CANONICAL_TEXT",
                "ERROR",
                relation.relation_id,
                "Relation rationale_en must use English",
            )
        )
    rationale = clean_text(relation.rationale_en).lower()
    if any(marker in rationale for marker in CONSUMER_OUTCOME_MARKERS):
        issues.append(
            ValidationIssue(
                "CONSUMER_OUTCOME_LEAK",
                "ERROR",
                relation.relation_id,
                "Relation describes an unobservable consumer outcome",
            )
        )
    if relation.status not in {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "TEMPORALLY_MISALIGNED"}:
        issues.append(ValidationIssue("INVALID_RELATION_STATUS", "ERROR", relation.relation_id, relation.status))
    if contains_private_fields(relation.to_dict()):
        issues.append(ValidationIssue("PRIVATE_FIELD_LEAK", "ERROR", relation.relation_id, "Private field appears in relation"))
    return issues


def _validate_common(
    item: GoldItem | GoldProposal,
    evidence: dict[str, EvidenceUnit],
    evidence_ids: tuple[str, ...],
    commerce_cues: dict[str, CommerceCue] | None = None,
    commercial_relations: dict[str, CommercialRelation] | None = None,
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

    if commerce_cues is not None and commercial_relations is not None:
        capability = clean_text(item.capability or item.task_subtype).upper()
        if capability != item.task_subtype:
            issues.append(
                ValidationIssue(
                    "CAPABILITY_SUBTYPE_MISMATCH",
                    "ERROR",
                    item_id,
                    f"Capability {capability} differs from subtype {item.task_subtype}",
                )
            )
        expected_operator = default_reasoning_operator(item.task_type, item.task_subtype)
        if clean_text(item.reasoning_operator).upper() != expected_operator:
            issues.append(
                ValidationIssue(
                    "INVALID_REASONING_OPERATOR",
                    "ERROR",
                    item_id,
                    f"Expected {expected_operator}",
                )
            )
        for cue_id in item.commerce_cue_ids:
            cue = commerce_cues.get(cue_id)
            if cue is None:
                issues.append(ValidationIssue("CUE_NOT_FOUND", "ERROR", item_id, cue_id))
            elif cue.video_id != item.video_id:
                issues.append(ValidationIssue("CUE_VIDEO_MISMATCH", "ERROR", item_id, cue_id))
        for relation_id in item.commercial_relation_ids:
            relation = commercial_relations.get(relation_id)
            if relation is None:
                issues.append(ValidationIssue("RELATION_NOT_FOUND", "ERROR", item_id, relation_id))
            elif relation.video_id != item.video_id:
                issues.append(ValidationIssue("RELATION_VIDEO_MISMATCH", "ERROR", item_id, relation_id))
        level = capability_level(item.task_type, item.task_subtype)
        if level in {"CUE", "RELATION_OPTIONAL"} and not item.commerce_cue_ids:
            issues.append(ValidationIssue("MISSING_COMMERCE_CUE", "ERROR", item_id, level))
        if level in {"RELATION_REQUIRED", "RELATION_PATH_REQUIRED"} and not item.commercial_relation_ids:
            issues.append(ValidationIssue("MISSING_COMMERCIAL_RELATION", "ERROR", item_id, level))
        if level == "CUE_OR_RELATION" and not (item.commerce_cue_ids or item.commercial_relation_ids):
            issues.append(ValidationIssue("MISSING_COMMERCE_GRAPH_REF", "ERROR", item_id, level))

    return issues


def validate_gold_proposal(
    proposal: GoldProposal,
    evidence: dict[str, EvidenceUnit],
    commerce_cues: dict[str, CommerceCue] | None = None,
    commercial_relations: dict[str, CommercialRelation] | None = None,
) -> list[ValidationIssue]:
    issues = _validate_common(
        proposal,
        evidence,
        proposal.evidence_ids,
        commerce_cues,
        commercial_relations,
    )
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


def validate_gold_item(
    item: GoldItem,
    evidence: dict[str, EvidenceUnit],
    commerce_cues: dict[str, CommerceCue] | None = None,
    commercial_relations: dict[str, CommercialRelation] | None = None,
) -> list[ValidationIssue]:
    issues = _validate_common(
        item,
        evidence,
        item.evidence_ids,
        commerce_cues,
        commercial_relations,
    )
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
