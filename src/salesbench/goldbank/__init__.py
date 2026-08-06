"""Evidence-First pipeline for SalesBench multimodal VQA."""

from .schema import (
    EvidenceModality,
    EvidenceUnit,
    GoldItem,
    GoldProposal,
    GoldReview,
    GoldTaskType,
    GoldTier,
    GroundedAnnotation,
    QualityStatus,
    ReviewVerdict,
    VideoGoldRecord,
    VideoEvidenceRecord,
)

__all__ = [
    "EvidenceModality",
    "EvidenceUnit",
    "GoldItem",
    "GoldProposal",
    "GoldReview",
    "GoldTaskType",
    "GoldTier",
    "GroundedAnnotation",
    "QualityStatus",
    "ReviewVerdict",
    "VideoGoldRecord",
    "VideoEvidenceRecord",
]
