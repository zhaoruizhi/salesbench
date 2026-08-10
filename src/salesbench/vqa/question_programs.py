"""Controlled question programs for EvidenceDataset QA compilation."""

from __future__ import annotations

from dataclasses import dataclass

from ..goldbank.schema import GoldItem


class UnsupportedQuestionProgramError(ValueError):
    """Raised when a grounded annotation has no explicit question program."""


QUESTION_PROGRAMS = {
    ("BP", "PRODUCT_IDENTITY", "grounded_question"): "What product is presented in the video?",
    ("BP", "ATTRIBUTE_AND_VARIANT", "grounded_question"): "What product attribute or variant is presented?",
    ("BP", "QUANTITY_AND_BUNDLE", "grounded_question"): "What quantity or bundle is included in the presented offer?",
    ("BP", "PRICE_AND_DISCOUNT", "grounded_question"): "What price or discount is explicitly presented?",
    ("BP", "OFFER_CONDITION", "grounded_question"): "What condition is required for the presented offer?",
    ("BP", "USAGE_STEP", "grounded_question"): "What concrete product-use step is shown?",
    ("BP", "DEMONSTRATED_STATE_CHANGE", "grounded_question"): "What visible state change is demonstrated?",
    ("BP", "USAGE_SCENARIO", "grounded_question"): "What usage scenario is presented for the product?",
    ("CM", "CLAIM_DEMONSTRATION_STATUS", "grounded_question"): "How does the cited visual content relate to the spoken product claim?",
    ("SS", "PROCESS_DEMONSTRATION", "grounded_question"): "What does the product demonstration show, and how is it used in the sales presentation?",
    ("AE", "USAGE_CONTEXT", "grounded_question"): "What usage context represented by the content is connected to the product?",
    ("BP", "COUNT_SPATIAL", "direct_question"): "How many {subject} are visible in the video?",
    ("BP", "ACTION", "direct_question"): "What observable action involving {subject} occurs in the video?",
    ("BP", "ENTITY_ATTRIBUTE", "direct_question"): "What is the {predicate} of the {subject} shown in the video?",
    ("BP", "OCR_FACT", "direct_question"): "According to the on-screen text, what is the {subject}'s {predicate}?",
    ("BP", "ASR_FACT", "direct_question"): "According to the speech, what is stated about the {subject}'s {predicate}?",
    ("BP", "STATE_CHANGE", "direct_question"): "What observable state change happens to the {subject}?",
    ("BP", "TEMPORAL_ORDER", "direct_question"): "In what order do the observable events involving the {subject} occur?",
    ("CM", "CLAIM_EVIDENCE_RELATION", "relation_choice"): "How does the visual or textual evidence relate to the claim that {claim}?",
    ("CM", "CLAIM_PARTIAL_SUPPORT", "supported_missing"): "Which visible evidence supports or fails to support the claim that {claim}?",
    ("CM", "TEXT_VISUAL_CONSISTENCY", "relation_choice"): "Are the speech or on-screen text consistent with the visual content?",
    ("SS", "HOOK_MECHANISM", "mechanism_with_evidence"): "What attention hook is used at the beginning of the video? Support the answer with observable evidence.",
    ("SS", "VALUE_PROPOSITION", "mechanism_with_evidence"): "What main value proposition does the video emphasize? Support the answer with observable evidence.",
    ("SS", "TRUST_MECHANISM", "mechanism_with_evidence"): "What trust-building mechanism does the video use? Support the answer with observable evidence.",
    ("SS", "OBJECTION_HANDLING", "mechanism_with_evidence"): "How does the video address a potential buyer concern? Support the answer with observable evidence.",
    ("SS", "URGENCY_CTA", "mechanism_with_evidence"): "What call to action or urgency mechanism does the video use? Support the answer with observable evidence.",
    ("SS", "FUNNEL_ROLE", "mechanism_with_evidence"): "What role does this segment play in gaining attention, explaining value, building trust, or prompting action? Support the answer with observable evidence.",
    ("AE", "AUDIENCE_NEED_FIT", "need_with_evidence"): "What audience need represented by the content does the video address? Support the answer with observable evidence.",
    ("AE", "USAGE_CONTEXT", "need_with_evidence"): "What primary usage context does the content present? Support the answer with observable evidence.",
    ("AE", "DECISION_STATE", "need_with_evidence"): "What decision barrier represented by the content does the video address? Support the answer with observable evidence.",
    ("AE", "CONTENT_MOTIVATION", "need_with_evidence"): "How does the observable content address a need or encourage further consideration? Support the answer with observable evidence.",
}


@dataclass(frozen=True)
class RenderedQuestion:
    question_program_id: str
    question: str
    answer: str


def _first_value(payload: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return str(value)
    if payload:
        first_key = sorted(payload)[0]
        return str(payload[first_key])
    return ""


def _format_context(item: GoldItem) -> dict[str, object]:
    context = dict(item.target)
    context.update(item.gold_value)
    context.setdefault("subject", item.target.get("subject") or item.gold_value.get("subject") or "target object")
    context.setdefault("predicate", item.target.get("predicate") or item.gold_value.get("predicate") or "observable attribute")
    context.setdefault("claim", item.target.get("claim") or item.gold_value.get("claim") or "the stated claim")
    context.setdefault("mechanism", item.target.get("mechanism") or item.gold_value.get("label") or "the mechanism")
    label_map = {
        "product": "product",
        "video": "video",
        "brand": "brand",
        "method": "method",
        "platform": "platform",
        "color": "color",
        "type": "type",
        "feature": "feature",
        "price": "price",
        "title": "title",
        "name": "name",
        "count": "count",
        "action": "action",
    }
    for key in ("subject", "predicate"):
        value = str(context.get(key) or "")
        context[key] = label_map.get(value.lower(), value)
    return context


def derive_answer(item: GoldItem) -> str:
    if item.task_type.value == "CM":
        return _first_value(item.gold_value, ("answer", "relation"))
    if item.task_type.value == "BP":
        return _first_value(item.gold_value, ("action", "count", "value", "answer"))
    if item.task_type.value == "SS":
        return _first_value(item.gold_value, ("answer", "label", "strategy", "mechanism"))
    return _first_value(item.gold_value, ("answer", "audience_need", "usage_context", "decision_state", "content_motivation"))


def render_question(item: GoldItem, question_format: str) -> RenderedQuestion:
    key = (item.task_type.value, item.task_subtype, question_format)
    template = QUESTION_PROGRAMS.get(key)
    if template is None:
        raise UnsupportedQuestionProgramError(f"No question program for {key}")
    question = template.format(**_format_context(item))
    return RenderedQuestion(
        question_program_id="|".join(key),
        question=question,
        answer=derive_answer(item),
    )
