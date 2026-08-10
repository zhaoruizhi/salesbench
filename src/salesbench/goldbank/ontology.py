"""Controlled capability ontology for the four public SalesBench tasks."""

from __future__ import annotations

from .schema import GoldTaskType


BP_SUBTYPES = {
    "PRODUCT_IDENTITY",
    "ATTRIBUTE_AND_VARIANT",
    "QUANTITY_AND_BUNDLE",
    "PRICE_AND_DISCOUNT",
    "OFFER_CONDITION",
    "USAGE_STEP",
    "DEMONSTRATED_STATE_CHANGE",
    "USAGE_SCENARIO",
}

CM_RELATIONS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "NOT_SHOWN",
    "TEMPORALLY_MISALIGNED",
}

CM_SUBTYPES = {
    "SPEECH_VISUAL_COREFERENCE",
    "OCR_SPEECH_OFFER_ALIGNMENT",
    "CLAIM_DEMONSTRATION_STATUS",
    "REPETITION_VS_INDEPENDENT_EVIDENCE",
    "PARTIAL_SUPPORT",
    "CONTRADICTION",
    "TEMPORAL_MISALIGNMENT",
    "NOT_DEMONSTRATED",
}

SS_SUBTYPES = {
    "PROBLEM_SOLUTION",
    "FEATURE_BENEFIT",
    "PROCESS_DEMONSTRATION",
    "OUTCOME_DISPLAY",
    "BEFORE_AFTER_COMPARISON",
    "VICARIOUS_TRIAL",
    "PRICE_VALUE_FRAMING",
    "REFERENCE_PRICE_ANCHORING",
    "CREDIBILITY_SIGNAL",
    "SOCIAL_PROOF",
    "LIMITATION_DISCLOSURE",
    "OBJECTION_HANDLING",
    "SCARCITY_AND_URGENCY",
    "CTA_SEQUENCE",
}

AE_SUBTYPES = {
    "CONTENT_IMPLIED_NEED",
    "USAGE_CONTEXT",
    "FIT_CONSTRAINT",
    "QUALITY_UNCERTAINTY",
    "USAGE_UNCERTAINTY",
    "PRICE_UNCERTAINTY",
    "SERVICE_OR_RISK_CONCERN",
    "DECISION_BARRIER",
    "OFFER_NEED_ALIGNMENT",
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

CAPABILITY_LEVELS = {
    **{(GoldTaskType.BP, subtype): "CUE" for subtype in BP_SUBTYPES},
    **{(GoldTaskType.CM, subtype): "RELATION_REQUIRED" for subtype in CM_SUBTYPES},
    **{(GoldTaskType.SS, subtype): "RELATION_PATH_REQUIRED" for subtype in SS_SUBTYPES},
    **{(GoldTaskType.AE, subtype): "CUE_OR_RELATION" for subtype in AE_SUBTYPES},
    (GoldTaskType.BP, "OFFER_CONDITION"): "RELATION_OPTIONAL",
    (GoldTaskType.BP, "DEMONSTRATED_STATE_CHANGE"): "RELATION_OPTIONAL",
}

REASONING_OPERATORS = {
    (GoldTaskType.BP, "PRODUCT_IDENTITY"): "IDENTIFY_PRODUCT",
    (GoldTaskType.BP, "ATTRIBUTE_AND_VARIANT"): "EXTRACT_ATTRIBUTE",
    (GoldTaskType.BP, "QUANTITY_AND_BUNDLE"): "AGGREGATE_OFFER",
    (GoldTaskType.BP, "PRICE_AND_DISCOUNT"): "EXTRACT_OFFER",
    (GoldTaskType.BP, "OFFER_CONDITION"): "EXTRACT_CONDITION",
    (GoldTaskType.BP, "USAGE_STEP"): "SEQUENCE_ACTION",
    (GoldTaskType.BP, "DEMONSTRATED_STATE_CHANGE"): "COMPARE_STATE",
    (GoldTaskType.BP, "USAGE_SCENARIO"): "LOCALIZE_SCENARIO",
    (GoldTaskType.CM, "SPEECH_VISUAL_COREFERENCE"): "ALIGN_COREFERENCE",
    (GoldTaskType.CM, "OCR_SPEECH_OFFER_ALIGNMENT"): "COMPARE_OFFER",
    (GoldTaskType.CM, "CLAIM_DEMONSTRATION_STATUS"): "CLASSIFY_CLAIM_SUPPORT",
    (GoldTaskType.CM, "REPETITION_VS_INDEPENDENT_EVIDENCE"): "DISTINGUISH_REPETITION",
    (GoldTaskType.CM, "PARTIAL_SUPPORT"): "DECOMPOSE_CLAIM",
    (GoldTaskType.CM, "CONTRADICTION"): "DETECT_CONTRADICTION",
    (GoldTaskType.CM, "TEMPORAL_MISALIGNMENT"): "ALIGN_TEMPORAL_STATE",
    (GoldTaskType.CM, "NOT_DEMONSTRATED"): "VERIFY_ABSENCE_WITH_SCOPE",
    (GoldTaskType.SS, "PROBLEM_SOLUTION"): "RECONSTRUCT_PROBLEM_SOLUTION",
    (GoldTaskType.SS, "FEATURE_BENEFIT"): "MAP_FEATURE_TO_BENEFIT",
    (GoldTaskType.SS, "PROCESS_DEMONSTRATION"): "INTERPRET_PROCESS_ROLE",
    (GoldTaskType.SS, "OUTCOME_DISPLAY"): "INTERPRET_OUTCOME_ROLE",
    (GoldTaskType.SS, "BEFORE_AFTER_COMPARISON"): "COMPARE_PRESENTED_STATES",
    (GoldTaskType.SS, "VICARIOUS_TRIAL"): "RECONSTRUCT_VICARIOUS_TRIAL",
    (GoldTaskType.SS, "PRICE_VALUE_FRAMING"): "LINK_PRICE_TO_VALUE",
    (GoldTaskType.SS, "REFERENCE_PRICE_ANCHORING"): "COMPARE_PRICE_ANCHOR",
    (GoldTaskType.SS, "CREDIBILITY_SIGNAL"): "IDENTIFY_CREDIBILITY_CUE",
    (GoldTaskType.SS, "SOCIAL_PROOF"): "IDENTIFY_INTERNAL_SOCIAL_PROOF",
    (GoldTaskType.SS, "LIMITATION_DISCLOSURE"): "IDENTIFY_LIMITATION",
    (GoldTaskType.SS, "OBJECTION_HANDLING"): "TRACE_OBJECTION_RESPONSE",
    (GoldTaskType.SS, "SCARCITY_AND_URGENCY"): "EXTRACT_URGENCY_CONDITION",
    (GoldTaskType.SS, "CTA_SEQUENCE"): "ORDER_CONTENT_BEFORE_CTA",
    (GoldTaskType.AE, "CONTENT_IMPLIED_NEED"): "INFER_BOUNDED_NEED",
    (GoldTaskType.AE, "USAGE_CONTEXT"): "LOCALIZE_USAGE_CONTEXT",
    (GoldTaskType.AE, "FIT_CONSTRAINT"): "MATCH_FIT_CONSTRAINT",
    (GoldTaskType.AE, "QUALITY_UNCERTAINTY"): "TRACE_QUALITY_CONCERN",
    (GoldTaskType.AE, "USAGE_UNCERTAINTY"): "TRACE_USAGE_CONCERN",
    (GoldTaskType.AE, "PRICE_UNCERTAINTY"): "TRACE_PRICE_CONCERN",
    (GoldTaskType.AE, "SERVICE_OR_RISK_CONCERN"): "TRACE_RISK_RESPONSE",
    (GoldTaskType.AE, "DECISION_BARRIER"): "IDENTIFY_DECISION_BARRIER",
    (GoldTaskType.AE, "OFFER_NEED_ALIGNMENT"): "ALIGN_OFFER_TO_NEED",
}

SUBTYPE_QUESTION_FORMATS = {
    (task, subtype): ("grounded_question",)
    for task, subtypes in TASK_SUBTYPES.items()
    for subtype in subtypes
}


def _task(task_type: GoldTaskType | str) -> GoldTaskType:
    return task_type if isinstance(task_type, GoldTaskType) else GoldTaskType(str(task_type).upper())


def allowed_subtypes(task_type: GoldTaskType | str) -> set[str]:
    return set(TASK_SUBTYPES[_task(task_type)])


def capability_level(task_type: GoldTaskType | str, subtype: str) -> str:
    task = _task(task_type)
    normalized = str(subtype).strip().upper()
    if normalized not in TASK_SUBTYPES[task]:
        raise ValueError(f"Unknown capability for {task.value}: {normalized}")
    return CAPABILITY_LEVELS[(task, normalized)]


def default_reasoning_operator(task_type: GoldTaskType | str, subtype: str) -> str:
    task = _task(task_type)
    normalized = str(subtype).strip().upper()
    try:
        return REASONING_OPERATORS[(task, normalized)]
    except KeyError as exc:
        raise ValueError(f"Unknown capability for {task.value}: {normalized}") from exc


def eligible_question_formats(task_type: GoldTaskType | str, subtype: str) -> tuple[str, ...]:
    return SUBTYPE_QUESTION_FORMATS.get((_task(task_type), str(subtype).upper()), ())


def task_description(task_type: GoldTaskType | str) -> str:
    task = _task(task_type)
    if task == GoldTaskType.BP:
        return "BP grounds products, offers, usage steps, state changes, and usage scenarios."
    if task == GoldTaskType.CM:
        return "CM verifies whether claims are demonstrated, repeated, partial, contradictory, or misaligned."
    if task == GoldTaskType.SS:
        return "SS reconstructs observable persuasion and sales logic from commercial relation paths."
    return "AE aligns represented needs, constraints, objections, and offer responses without profiling viewers."
