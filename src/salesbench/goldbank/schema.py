"""Stable data contracts for Evidence-First QA generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..utils import clean_text


SCHEMA_VERSION = "evidence-dataset-schema-v4"


class GoldTaskType(str, Enum):
    BP = "BP"
    CM = "CM"
    SS = "SS"
    AE = "AE"


class EvidenceModality(str, Enum):
    VISUAL = "visual"
    ASR = "asr"
    OCR = "ocr"
    TITLE = "title"
    METADATA = "metadata"
    CROSS_MODAL = "cross_modal"


class EvidenceAssertionType(str, Enum):
    """What kind of assertion an EvidenceUnit actually contains."""

    OBSERVED = "OBSERVED"
    SPOKEN_CLAIM = "SPOKEN_CLAIM"
    OCR_TEXT = "OCR_TEXT"


class EvidenceTemporalScope(str, Enum):
    """The observation window that the evidence can directly support."""

    FRAME = "FRAME"
    SHORT_CLIP = "SHORT_CLIP"
    FULL_VIDEO = "FULL_VIDEO"
    LONG_TERM_CLAIM = "LONG_TERM_CLAIM"
    UNSPECIFIED = "UNSPECIFIED"


_LONG_TERM_CLAIM_MARKERS = (
    "after a day",
    "all day",
    "whole day",
    "for hours",
    "for days",
    "for weeks",
    "long-lasting",
    "long lasting",
    "lasting effect",
    "durable over time",
    "overnight",
    "24/7",
)

_LONG_TERM_DURATION_RE = re.compile(
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    r"[\s-]+(?:hours?|days?|weeks?|months?|years?)\b"
)


def infer_assertion_type(modality: EvidenceModality) -> EvidenceAssertionType:
    if modality == EvidenceModality.ASR:
        return EvidenceAssertionType.SPOKEN_CLAIM
    if modality == EvidenceModality.OCR:
        return EvidenceAssertionType.OCR_TEXT
    return EvidenceAssertionType.OBSERVED


def infer_temporal_scope(modality: EvidenceModality, content: object = "") -> EvidenceTemporalScope:
    if modality in {EvidenceModality.VISUAL, EvidenceModality.OCR}:
        return EvidenceTemporalScope.FRAME
    if modality == EvidenceModality.ASR:
        text = clean_text(content).lower()
        if any(marker in text for marker in _LONG_TERM_CLAIM_MARKERS) or (
            _LONG_TERM_DURATION_RE.search(text)
        ):
            return EvidenceTemporalScope.LONG_TERM_CLAIM
        return EvidenceTemporalScope.SHORT_CLIP
    return EvidenceTemporalScope.UNSPECIFIED


def resolve_temporal_scope(
    modality: EvidenceModality,
    content: object = "",
    declared_scope: object = "",
) -> EvidenceTemporalScope:
    """Resolve scope without allowing a model declaration to understate a detected claim window."""

    inferred = infer_temporal_scope(modality, content)
    if inferred == EvidenceTemporalScope.LONG_TERM_CLAIM:
        return inferred
    declared = clean_text(declared_scope).upper()
    return EvidenceTemporalScope(declared) if declared else inferred


class GoldTier(str, Enum):
    GOLD_A = "Gold-A"
    GOLD_B = "Gold-B"
    SILVER = "Silver"
    REJECTED = "Rejected"


class QualityStatus(str, Enum):
    """Local, deterministic quality decision for a grounded annotation."""

    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


QUALITY_TO_TIER = {
    QualityStatus.DIRECT: GoldTier.GOLD_A,
    QualityStatus.INFERRED: GoldTier.GOLD_B,
    QualityStatus.NEEDS_REVIEW: GoldTier.SILVER,
    QualityStatus.REJECTED: GoldTier.REJECTED,
}
TIER_TO_QUALITY = {tier: quality for quality, tier in QUALITY_TO_TIER.items()}


class ReviewVerdict(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECT = "REJECT"


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return clean_text(value)
    return value


def stable_digest(value: object, length: int = 12) -> str:
    encoded = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _tuple(value: object) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _confidence(value: object) -> float:
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return confidence


def make_evidence_id(video_id: str, modality: EvidenceModality, ordinal: int) -> str:
    video = clean_text(video_id)
    digest = stable_digest({"video_id": video, "modality": modality.value, "ordinal": int(ordinal)})
    return f"{video}_{modality.value}_{int(ordinal):03d}_{digest}"


def make_gold_id(
    video_id: str,
    task_type: GoldTaskType,
    task_subtype: str,
    target: dict[str, object],
    gold_value: dict[str, object],
) -> str:
    task = task_type if isinstance(task_type, GoldTaskType) else GoldTaskType(str(task_type).upper())
    digest = stable_digest(
        {
            "video_id": clean_text(video_id),
            "task_type": task.value,
            "task_subtype": clean_text(task_subtype).upper(),
            "target": target,
            "gold_value": gold_value,
        }
    )
    return f"{clean_text(video_id)}_{task.value.lower()}_{digest}"


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    video_id: str
    modality: EvidenceModality
    start_s: float | None
    end_s: float | None
    frame_indices: tuple[int, ...]
    text_span: str
    subject: str
    predicate: str
    value: object
    attributes: dict[str, object]
    source_domains: tuple[str, ...]
    extractor: str
    confidence: float
    timestamp_status: str
    content_en: str = ""
    source_text_native: str = ""
    assertion_type: EvidenceAssertionType = EvidenceAssertionType.OBSERVED
    temporal_scope: EvidenceTemporalScope = EvidenceTemporalScope.FRAME

    def __post_init__(self) -> None:
        _confidence(self.confidence)
        if not self.source_text_native and self.text_span:
            object.__setattr__(self, "source_text_native", clean_text(self.text_span))
        if not self.text_span and self.source_text_native:
            object.__setattr__(self, "text_span", clean_text(self.source_text_native))
        if not self.content_en:
            value = clean_text(self.value)
            derived = " ".join(part for part in (clean_text(self.subject), clean_text(self.predicate), value) if part)
            object.__setattr__(self, "content_en", derived)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "video_id": self.video_id,
            "modality": self.modality.value,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "frame_indices": list(self.frame_indices),
            "content_en": self.content_en,
            "source_text_native": self.source_text_native,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "attributes": dict(self.attributes),
            "source_domains": list(self.source_domains),
            "extractor": self.extractor,
            "confidence": self.confidence,
            "timestamp_status": self.timestamp_status,
            "assertion_type": self.assertion_type.value,
            "temporal_scope": self.temporal_scope.value,
        }


@dataclass(frozen=True)
class GoldProposal:
    proposal_id: str
    video_id: str
    source_agent: str
    task_type: GoldTaskType
    task_subtype: str
    target: dict[str, object]
    proposed_gold: dict[str, object]
    evidence_ids: tuple[str, ...]
    reasoning_edges: tuple[tuple[str, str, str], ...]
    proposal_confidence: float
    capability: str = ""
    reasoning_operator: str = ""
    commerce_cue_ids: tuple[str, ...] = field(default_factory=tuple)
    commercial_relation_ids: tuple[str, ...] = field(default_factory=tuple)
    question_intent: str = ""
    forbidden_inferences: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _confidence(self.proposal_confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "video_id": self.video_id,
            "source_agent": self.source_agent,
            "task_type": self.task_type.value,
            "task_subtype": self.task_subtype,
            "target": dict(self.target),
            "proposed_gold": dict(self.proposed_gold),
            "evidence_ids": list(self.evidence_ids),
            "reasoning_edges": [list(edge) for edge in self.reasoning_edges],
            "proposal_confidence": self.proposal_confidence,
            "capability": self.capability or self.task_subtype,
            "reasoning_operator": self.reasoning_operator,
            "commerce_cue_ids": list(self.commerce_cue_ids),
            "commercial_relation_ids": list(self.commercial_relation_ids),
            "question_intent": self.question_intent,
            "forbidden_inferences": list(self.forbidden_inferences),
        }


@dataclass(frozen=True)
class GoldReview:
    review_id: str
    proposal_id: str
    video_id: str
    reviewer: str
    verdict: ReviewVerdict
    checks: dict[str, object]
    issues: tuple[str, ...]
    suggested_revision: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "proposal_id": self.proposal_id,
            "video_id": self.video_id,
            "reviewer": self.reviewer,
            "verdict": self.verdict.value,
            "checks": dict(self.checks),
            "issues": list(self.issues),
            "suggested_revision": self.suggested_revision,
        }


@dataclass(frozen=True)
class GoldItem:
    gold_id: str
    video_id: str
    task_type: GoldTaskType
    task_subtype: str
    target: dict[str, object]
    gold_value: dict[str, object]
    evidence_ids: tuple[str, ...]
    reasoning_edges: tuple[tuple[str, str, str], ...]
    eligible_question_formats: tuple[str, ...]
    source_proposal_ids: tuple[str, ...]
    gold_tier: GoldTier
    review_status: str
    confidence: float
    capability: str = ""
    reasoning_operator: str = ""
    commerce_cue_ids: tuple[str, ...] = field(default_factory=tuple)
    commercial_relation_ids: tuple[str, ...] = field(default_factory=tuple)
    question_intent: str = ""
    forbidden_inferences: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _confidence(self.confidence)

    @property
    def annotation_id(self) -> str:
        return self.gold_id

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return self.evidence_ids

    @property
    def quality_status(self) -> QualityStatus:
        return TIER_TO_QUALITY[self.gold_tier]

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "task_type": self.task_type.value,
            "task_subtype": self.task_subtype,
            "target": dict(self.target),
            "gold_value": dict(self.gold_value),
            "evidence_refs": list(self.evidence_refs),
            "reasoning_edges": [list(edge) for edge in self.reasoning_edges],
            "eligible_question_formats": list(self.eligible_question_formats),
            "source_proposal_ids": list(self.source_proposal_ids),
            "quality_status": self.quality_status.value,
            "review_status": self.review_status,
            "confidence": self.confidence,
            "capability": self.capability or self.task_subtype,
            "reasoning_operator": self.reasoning_operator,
            "commerce_cue_ids": list(self.commerce_cue_ids),
            "commercial_relation_ids": list(self.commercial_relation_ids),
            "question_intent": self.question_intent,
            "forbidden_inferences": list(self.forbidden_inferences),
        }


@dataclass(frozen=True)
class VideoGoldRecord:
    video_id: str
    schema_version: str
    evidence_unit_ids: tuple[str, ...]
    gold_items: tuple[GoldItem, ...]
    private_interaction_ref: str
    coverage: dict[str, object]
    quality_summary: dict[str, int]
    observation_scope: dict[str, object]
    commerce_cue_ids: tuple[str, ...] = field(default_factory=tuple)
    commercial_relation_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "schema_version": self.schema_version,
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "commerce_cue_ids": list(self.commerce_cue_ids),
            "commercial_relation_ids": list(self.commercial_relation_ids),
            "grounded_annotations": [item.to_dict() for item in self.gold_items],
            "coverage": dict(self.coverage),
            "quality_summary": dict(self.quality_summary),
            "observation_scope": dict(self.observation_scope),
        }


def parse_evidence_unit(record: dict[str, object]) -> EvidenceUnit:
    source_text_native = clean_text(record.get("source_text_native")) or clean_text(record.get("text_span"))
    modality = EvidenceModality(clean_text(record.get("modality")))
    assertion_raw = clean_text(record.get("assertion_type")).upper()
    return EvidenceUnit(
        evidence_id=clean_text(record.get("evidence_id")),
        video_id=clean_text(record.get("video_id")),
        modality=modality,
        start_s=None if record.get("start_s") is None else float(record.get("start_s")),
        end_s=None if record.get("end_s") is None else float(record.get("end_s")),
        frame_indices=tuple(int(value) for value in _tuple(record.get("frame_indices"))),
        text_span=source_text_native,
        subject=clean_text(record.get("subject")),
        predicate=clean_text(record.get("predicate")),
        value=record.get("value"),
        attributes=_dict(record.get("attributes")),
        source_domains=tuple(clean_text(value) for value in _tuple(record.get("source_domains"))),
        extractor=clean_text(record.get("extractor")),
        confidence=_confidence(record.get("confidence", 0.0)),
        timestamp_status=clean_text(record.get("timestamp_status")),
        content_en=clean_text(record.get("content_en")),
        source_text_native=source_text_native,
        assertion_type=(
            EvidenceAssertionType(assertion_raw)
            if assertion_raw
            else infer_assertion_type(modality)
        ),
        temporal_scope=resolve_temporal_scope(
            modality,
            clean_text(record.get("content_en")) or record.get("value"),
            record.get("temporal_scope"),
        ),
    )


def parse_gold_proposal(record: dict[str, object]) -> GoldProposal:
    return GoldProposal(
        proposal_id=clean_text(record.get("proposal_id")),
        video_id=clean_text(record.get("video_id")),
        source_agent=clean_text(record.get("source_agent")),
        task_type=GoldTaskType(clean_text(record.get("task_type")).upper()),
        task_subtype=clean_text(record.get("task_subtype")).upper(),
        target=_dict(record.get("target")),
        proposed_gold=_dict(record.get("proposed_gold")),
        evidence_ids=tuple(clean_text(value) for value in _tuple(record.get("evidence_ids"))),
        reasoning_edges=tuple(tuple(clean_text(part) for part in edge[:3]) for edge in _tuple(record.get("reasoning_edges"))),
        proposal_confidence=_confidence(record.get("proposal_confidence", 0.0)),
        capability=clean_text(record.get("capability")) or clean_text(record.get("task_subtype")).upper(),
        reasoning_operator=clean_text(record.get("reasoning_operator")).upper(),
        commerce_cue_ids=tuple(clean_text(value) for value in _tuple(record.get("commerce_cue_ids"))),
        commercial_relation_ids=tuple(
            clean_text(value) for value in _tuple(record.get("commercial_relation_ids"))
        ),
        question_intent=clean_text(record.get("question_intent")),
        forbidden_inferences=tuple(
            clean_text(value) for value in _tuple(record.get("forbidden_inferences"))
        ),
    )


def parse_gold_review(record: dict[str, object]) -> GoldReview:
    return GoldReview(
        review_id=clean_text(record.get("review_id")),
        proposal_id=clean_text(record.get("proposal_id")),
        video_id=clean_text(record.get("video_id")),
        reviewer=clean_text(record.get("reviewer")),
        verdict=ReviewVerdict(clean_text(record.get("verdict")).upper()),
        checks=_dict(record.get("checks")),
        issues=tuple(clean_text(value) for value in _tuple(record.get("issues"))),
        suggested_revision=record.get("suggested_revision") if isinstance(record.get("suggested_revision"), dict) else None,
    )


def parse_gold_item(record: dict[str, object]) -> GoldItem:
    raw_quality = clean_text(record.get("quality_status")).upper()
    if raw_quality:
        gold_tier = QUALITY_TO_TIER[QualityStatus(raw_quality)]
    else:
        gold_tier = GoldTier(clean_text(record.get("gold_tier")))
    return GoldItem(
        gold_id=clean_text(record.get("annotation_id")) or clean_text(record.get("gold_id")),
        video_id=clean_text(record.get("video_id")),
        task_type=GoldTaskType(clean_text(record.get("task_type")).upper()),
        task_subtype=clean_text(record.get("task_subtype")).upper(),
        target=_dict(record.get("target")),
        gold_value=_dict(record.get("gold_value")),
        evidence_ids=tuple(
            clean_text(value)
            for value in _tuple(record.get("evidence_refs") if record.get("evidence_refs") is not None else record.get("evidence_ids"))
        ),
        reasoning_edges=tuple(tuple(clean_text(part) for part in edge[:3]) for edge in _tuple(record.get("reasoning_edges"))),
        eligible_question_formats=tuple(clean_text(value) for value in _tuple(record.get("eligible_question_formats"))),
        source_proposal_ids=tuple(clean_text(value) for value in _tuple(record.get("source_proposal_ids"))),
        gold_tier=gold_tier,
        review_status=clean_text(record.get("review_status")),
        confidence=_confidence(record.get("confidence", 0.0)),
        capability=clean_text(record.get("capability")) or clean_text(record.get("task_subtype")).upper(),
        reasoning_operator=clean_text(record.get("reasoning_operator")).upper(),
        commerce_cue_ids=tuple(clean_text(value) for value in _tuple(record.get("commerce_cue_ids"))),
        commercial_relation_ids=tuple(
            clean_text(value) for value in _tuple(record.get("commercial_relation_ids"))
        ),
        question_intent=clean_text(record.get("question_intent")),
        forbidden_inferences=tuple(
            clean_text(value) for value in _tuple(record.get("forbidden_inferences"))
        ),
    )


def parse_video_gold_record(record: dict[str, object]) -> VideoGoldRecord:
    annotations = record.get("grounded_annotations")
    if annotations is None:
        annotations = record.get("gold_items")
    return VideoGoldRecord(
        video_id=clean_text(record.get("video_id")),
        schema_version=clean_text(record.get("schema_version")),
        evidence_unit_ids=tuple(clean_text(value) for value in _tuple(record.get("evidence_unit_ids"))),
        gold_items=tuple(parse_gold_item(item) for item in _tuple(annotations) if isinstance(item, dict)),
        private_interaction_ref=clean_text(record.get("private_interaction_ref")),
        coverage=_dict(record.get("coverage")),
        quality_summary={str(key): int(value) for key, value in _dict(record.get("quality_summary")).items()},
        observation_scope=_dict(record.get("observation_scope")),
        commerce_cue_ids=tuple(clean_text(value) for value in _tuple(record.get("commerce_cue_ids"))),
        commercial_relation_ids=tuple(
            clean_text(value) for value in _tuple(record.get("commercial_relation_ids"))
        ),
    )


# Canonical names for the Evidence-First public contract. The legacy class names
# remain import aliases so existing internal call sites can migrate incrementally.
GroundedAnnotation = GoldItem
VideoEvidenceRecord = VideoGoldRecord
