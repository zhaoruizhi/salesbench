"""Controlled ontology for the four public SalesBench VQA tasks."""

from __future__ import annotations

from .schema import GoldTaskType


BP_SUBTYPES = {
    "ENTITY_ATTRIBUTE",
    "COUNT_SPATIAL",
    "ACTION",
    "STATE_CHANGE",
    "TEMPORAL_ORDER",
    "OCR_FACT",
    "ASR_FACT",
}

CM_RELATIONS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "NOT_SHOWN",
    "TEMPORALLY_MISALIGNED",
}

CM_SUBTYPES = {
    "CLAIM_EVIDENCE_RELATION",
    "CLAIM_PARTIAL_SUPPORT",
    "TEXT_VISUAL_CONSISTENCY",
}

SS_ONTOLOGY = {
    "hook": {"pain_problem", "result_first", "question", "contrast", "surprise"},
    "value": {"function", "price_value", "convenience", "health", "aesthetics", "education"},
    "trust": {"demonstration", "comparison", "authority", "social_proof", "process", "guarantee"},
    "objection": {"price", "authenticity", "effectiveness", "difficulty", "risk"},
    "urgency": {"time_limit", "stock_limit", "price_window"},
    "cta": {"purchase", "comment", "collect", "share"},
    "funnel": {"attention", "comprehension", "trust", "action"},
}

SS_SUBTYPES = {
    "HOOK_MECHANISM",
    "VALUE_PROPOSITION",
    "TRUST_MECHANISM",
    "OBJECTION_HANDLING",
    "URGENCY_CTA",
    "FUNNEL_ROLE",
}

AE_FIELDS = {
    "audience_need",
    "usage_context",
    "decision_state",
    "content_motivation",
    "supporting_evidence",
    "uncertainty",
}

AE_SUBTYPES = {
    "AUDIENCE_NEED_FIT",
    "USAGE_CONTEXT",
    "DECISION_STATE",
    "CONTENT_MOTIVATION",
}

TASK_SUBTYPES = {
    GoldTaskType.BP: BP_SUBTYPES,
    GoldTaskType.CM: CM_SUBTYPES,
    GoldTaskType.SS: SS_SUBTYPES,
    GoldTaskType.AE: AE_SUBTYPES,
}

TASK_MIN_EVIDENCE = {
    GoldTaskType.BP: 1,
    GoldTaskType.CM: 2,
    GoldTaskType.SS: 2,
    GoldTaskType.AE: 2,
}

SUBTYPE_QUESTION_FORMATS = {
    (GoldTaskType.BP, "ENTITY_ATTRIBUTE"): ("direct_question",),
    (GoldTaskType.BP, "COUNT_SPATIAL"): ("direct_question",),
    (GoldTaskType.BP, "ACTION"): ("direct_question",),
    (GoldTaskType.BP, "STATE_CHANGE"): ("direct_question",),
    (GoldTaskType.BP, "TEMPORAL_ORDER"): ("direct_question",),
    (GoldTaskType.BP, "OCR_FACT"): ("direct_question",),
    (GoldTaskType.BP, "ASR_FACT"): ("direct_question",),
    (GoldTaskType.CM, "CLAIM_EVIDENCE_RELATION"): ("relation_choice",),
    (GoldTaskType.CM, "CLAIM_PARTIAL_SUPPORT"): ("supported_missing",),
    (GoldTaskType.CM, "TEXT_VISUAL_CONSISTENCY"): ("relation_choice",),
    (GoldTaskType.SS, "HOOK_MECHANISM"): ("mechanism_with_evidence",),
    (GoldTaskType.SS, "VALUE_PROPOSITION"): ("mechanism_with_evidence",),
    (GoldTaskType.SS, "TRUST_MECHANISM"): ("mechanism_with_evidence",),
    (GoldTaskType.SS, "OBJECTION_HANDLING"): ("mechanism_with_evidence",),
    (GoldTaskType.SS, "URGENCY_CTA"): ("mechanism_with_evidence",),
    (GoldTaskType.SS, "FUNNEL_ROLE"): ("mechanism_with_evidence",),
    (GoldTaskType.AE, "AUDIENCE_NEED_FIT"): ("need_with_evidence",),
    (GoldTaskType.AE, "USAGE_CONTEXT"): ("need_with_evidence",),
    (GoldTaskType.AE, "DECISION_STATE"): ("need_with_evidence",),
    (GoldTaskType.AE, "CONTENT_MOTIVATION"): ("need_with_evidence",),
}


def allowed_subtypes(task_type: GoldTaskType | str) -> set[str]:
    task = task_type if isinstance(task_type, GoldTaskType) else GoldTaskType(str(task_type).upper())
    return set(TASK_SUBTYPES[task])


def eligible_question_formats(task_type: GoldTaskType | str, subtype: str) -> tuple[str, ...]:
    task = task_type if isinstance(task_type, GoldTaskType) else GoldTaskType(str(task_type).upper())
    return SUBTYPE_QUESTION_FORMATS.get((task, str(subtype).upper()), ())


def task_description(task_type: GoldTaskType | str) -> str:
    task = task_type if isinstance(task_type, GoldTaskType) else GoldTaskType(str(task_type).upper())
    if task == GoldTaskType.BP:
        return "BP captures directly observable facts in the video."
    if task == GoldTaskType.CM:
        return "CM captures relations between spoken/text claims and visible or textual evidence."
    if task == GoldTaskType.SS:
        return "SS captures controlled marketing strategy mechanisms grounded in evidence."
    return "AE captures audience need and scenario fit grounded in evidence."
