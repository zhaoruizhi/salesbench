"""Stable contracts for the SalesBench commercial argument graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..utils import clean_text
from .schema import ActionRole, AssertionScope, stable_digest


class CueType(str, Enum):
    PRODUCT_IDENTITY = "PRODUCT_IDENTITY"
    PRODUCT_ATTRIBUTE = "PRODUCT_ATTRIBUTE"
    PRODUCT_VARIANT = "PRODUCT_VARIANT"
    QUANTITY = "QUANTITY"
    BUNDLE = "BUNDLE"
    PRICE = "PRICE"
    DISCOUNT = "DISCOUNT"
    GIFT = "GIFT"
    OFFER_CONDITION = "OFFER_CONDITION"
    SERVICE_GUARANTEE = "SERVICE_GUARANTEE"
    PRODUCT_DESCRIPTION = "PRODUCT_DESCRIPTION"
    PROCESS_DEMONSTRATION = "PROCESS_DEMONSTRATION"
    OUTCOME_DISPLAY = "OUTCOME_DISPLAY"
    BEFORE_AFTER = "BEFORE_AFTER"
    VICARIOUS_TRIAL = "VICARIOUS_TRIAL"
    USAGE_SCENARIO = "USAGE_SCENARIO"
    FUNCTION_CLAIM = "FUNCTION_CLAIM"
    EFFECT_CLAIM = "EFFECT_CLAIM"
    PRICE_CLAIM = "PRICE_CLAIM"
    FIT_CLAIM = "FIT_CLAIM"
    EXPERIENCE_REVIEW = "EXPERIENCE_REVIEW"
    PAIN_POINT = "PAIN_POINT"
    NEED = "NEED"
    OBJECTION = "OBJECTION"
    FIT_CONSTRAINT = "FIT_CONSTRAINT"
    USAGE_DIFFICULTY = "USAGE_DIFFICULTY"
    PRICE_CONCERN = "PRICE_CONCERN"
    RISK_CONCERN = "RISK_CONCERN"
    BENEFIT = "BENEFIT"
    COMPARISON_ANCHOR = "COMPARISON_ANCHOR"
    CREDIBILITY_SIGNAL = "CREDIBILITY_SIGNAL"
    SOCIAL_PROOF = "SOCIAL_PROOF"
    SCARCITY = "SCARCITY"
    URGENCY = "URGENCY"
    CTA = "CTA"


class RelationType(str, Enum):
    DESCRIPTION_REFERS_TO_PRODUCT = "DESCRIPTION_REFERS_TO_PRODUCT"
    CLAIM_SUPPORTED_BY_DEMONSTRATION = "CLAIM_SUPPORTED_BY_DEMONSTRATION"
    CLAIM_REPEATED_ACROSS_MODALITIES = "CLAIM_REPEATED_ACROSS_MODALITIES"
    CLAIM_PARTIALLY_SUPPORTED = "CLAIM_PARTIALLY_SUPPORTED"
    CLAIM_CONTRADICTED = "CLAIM_CONTRADICTED"
    CLAIM_TEMPORALLY_MISALIGNED = "CLAIM_TEMPORALLY_MISALIGNED"
    FEATURE_FRAMED_AS_BENEFIT = "FEATURE_FRAMED_AS_BENEFIT"
    DEMONSTRATION_SHOWS_STATE_CHANGE = "DEMONSTRATION_SHOWS_STATE_CHANGE"
    PROBLEM_ADDRESSED_BY_SOLUTION = "PROBLEM_ADDRESSED_BY_SOLUTION"
    OBJECTION_RESPONDED_BY_CUE = "OBJECTION_RESPONDED_BY_CUE"
    CONTENT_ADDRESSES_FIT_UNCERTAINTY = "CONTENT_ADDRESSES_FIT_UNCERTAINTY"
    CONTENT_ADDRESSES_USAGE_UNCERTAINTY = "CONTENT_ADDRESSES_USAGE_UNCERTAINTY"
    CONTENT_ADDRESSES_PRICE_UNCERTAINTY = "CONTENT_ADDRESSES_PRICE_UNCERTAINTY"
    OFFER_REQUIRES_CONDITION = "OFFER_REQUIRES_CONDITION"
    CONTENT_PRECEDES_CTA = "CONTENT_PRECEDES_CTA"


class RelationProvenance(str, Enum):
    THEORY_DIRECT = "THEORY_DIRECT"
    THEORY_OPERATIONALIZED = "THEORY_OPERATIONALIZED"
    BENCHMARK_OPERATIONAL = "BENCHMARK_OPERATIONAL"


def _tuple(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _confidence(value: object) -> float:
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {number}")
    return number


def make_cue_id(
    video_id: str,
    cue_type: CueType | str,
    evidence_ids: tuple[str, ...],
    content_en: str,
) -> str:
    cue = cue_type if isinstance(cue_type, CueType) else CueType(clean_text(cue_type).upper())
    digest = stable_digest(
        {
            "video_id": clean_text(video_id),
            "cue_type": cue.value,
            "evidence_ids": sorted(clean_text(item) for item in evidence_ids),
            "content_en": clean_text(content_en).lower(),
        }
    )
    return f"{clean_text(video_id)}_cue_{cue.value.lower()}_{digest}"


def make_relation_id(
    video_id: str,
    relation_type: RelationType | str,
    source_cue_ids: tuple[str, ...],
    target_cue_ids: tuple[str, ...],
) -> str:
    relation = (
        relation_type
        if isinstance(relation_type, RelationType)
        else RelationType(clean_text(relation_type).upper())
    )
    digest = stable_digest(
        {
            "video_id": clean_text(video_id),
            "relation_type": relation.value,
            "source_cue_ids": sorted(clean_text(item) for item in source_cue_ids),
            "target_cue_ids": sorted(clean_text(item) for item in target_cue_ids),
        }
    )
    return f"{clean_text(video_id)}_relation_{relation.value.lower()}_{digest}"


@dataclass(frozen=True)
class CommerceCue:
    cue_id: str
    video_id: str
    cue_type: CueType
    content_en: str
    source_text_native: str
    evidence_ids: tuple[str, ...]
    attributes: dict[str, object]
    directness: str
    theory_tags: tuple[str, ...]
    extractor: str
    confidence: float
    assertion_scope: AssertionScope = AssertionScope.OBSERVED_FACT
    action_role: ActionRole | None = None

    def __post_init__(self) -> None:
        _confidence(self.confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "video_id": self.video_id,
            "cue_type": self.cue_type.value,
            "content_en": self.content_en,
            "source_text_native": self.source_text_native,
            "evidence_ids": list(self.evidence_ids),
            "attributes": dict(self.attributes),
            "directness": self.directness,
            "theory_tags": list(self.theory_tags),
            "extractor": self.extractor,
            "confidence": self.confidence,
            "assertion_scope": self.assertion_scope.value,
            "action_role": self.action_role.value if self.action_role is not None else None,
        }


@dataclass(frozen=True)
class CommercialRelation:
    relation_id: str
    video_id: str
    relation_type: RelationType
    source_cue_ids: tuple[str, ...]
    target_cue_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    status: str
    rationale_en: str
    provenance: RelationProvenance
    directness: str
    extractor: str
    confidence: float
    assertion_scope: AssertionScope = AssertionScope.OBSERVED_FACT

    def __post_init__(self) -> None:
        _confidence(self.confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "relation_id": self.relation_id,
            "video_id": self.video_id,
            "relation_type": self.relation_type.value,
            "source_cue_ids": list(self.source_cue_ids),
            "target_cue_ids": list(self.target_cue_ids),
            "evidence_ids": list(self.evidence_ids),
            "status": self.status,
            "rationale_en": self.rationale_en,
            "provenance": self.provenance.value,
            "directness": self.directness,
            "extractor": self.extractor,
            "confidence": self.confidence,
            "assertion_scope": self.assertion_scope.value,
        }


def parse_commerce_cue(record: dict[str, object]) -> CommerceCue:
    evidence_ids = tuple(clean_text(item) for item in _tuple(record.get("evidence_ids")))
    cue_type = CueType(clean_text(record.get("cue_type")).upper())
    content_en = clean_text(record.get("content_en"))
    video_id = clean_text(record.get("video_id"))
    cue_id = clean_text(record.get("cue_id")) or make_cue_id(video_id, cue_type, evidence_ids, content_en)
    return CommerceCue(
        cue_id=cue_id,
        video_id=video_id,
        cue_type=cue_type,
        content_en=content_en,
        source_text_native=clean_text(record.get("source_text_native")),
        evidence_ids=evidence_ids,
        attributes=dict(record.get("attributes")) if isinstance(record.get("attributes"), dict) else {},
        directness=clean_text(record.get("directness")).upper(),
        theory_tags=tuple(clean_text(item) for item in _tuple(record.get("theory_tags"))),
        extractor=clean_text(record.get("extractor")),
        confidence=_confidence(record.get("confidence", 0.0)),
        assertion_scope=AssertionScope(
            clean_text(record.get("assertion_scope") or "OBSERVED_FACT").upper()
        ),
        action_role=(
            ActionRole(clean_text(record.get("action_role")).upper())
            if clean_text(record.get("action_role"))
            else None
        ),
    )


def parse_commercial_relation(record: dict[str, object]) -> CommercialRelation:
    relation_type = RelationType(clean_text(record.get("relation_type")).upper())
    video_id = clean_text(record.get("video_id"))
    source_cue_ids = tuple(clean_text(item) for item in _tuple(record.get("source_cue_ids")))
    target_cue_ids = tuple(clean_text(item) for item in _tuple(record.get("target_cue_ids")))
    relation_id = clean_text(record.get("relation_id")) or make_relation_id(
        video_id, relation_type, source_cue_ids, target_cue_ids
    )
    return CommercialRelation(
        relation_id=relation_id,
        video_id=video_id,
        relation_type=relation_type,
        source_cue_ids=source_cue_ids,
        target_cue_ids=target_cue_ids,
        evidence_ids=tuple(clean_text(item) for item in _tuple(record.get("evidence_ids"))),
        status=clean_text(record.get("status")).upper(),
        rationale_en=clean_text(record.get("rationale_en")),
        provenance=RelationProvenance(clean_text(record.get("provenance")).upper()),
        directness=clean_text(record.get("directness")).upper(),
        extractor=clean_text(record.get("extractor")),
        confidence=_confidence(record.get("confidence", 0.0)),
        assertion_scope=AssertionScope(
            clean_text(record.get("assertion_scope") or "OBSERVED_FACT").upper()
        ),
    )
