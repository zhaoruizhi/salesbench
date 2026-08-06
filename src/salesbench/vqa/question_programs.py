"""Controlled question programs for EvidenceDataset QA compilation."""

from __future__ import annotations

from dataclasses import dataclass

from ..goldbank.schema import GoldItem


class UnsupportedQuestionProgramError(ValueError):
    """Raised when a grounded annotation has no explicit question program."""


QUESTION_PROGRAMS = {
    ("BP", "COUNT_SPATIAL", "direct_question"): "视频中出现了多少个{subject}？",
    ("BP", "ACTION", "direct_question"): "视频中人物或画面对{subject}做了什么？",
    ("BP", "ENTITY_ATTRIBUTE", "direct_question"): "视频中{subject}呈现出什么可观察特征？",
    ("BP", "OCR_FACT", "direct_question"): "视频画面文字显示了关于{subject}的什么信息？",
    ("BP", "ASR_FACT", "direct_question"): "视频口播提到了关于{subject}的什么信息？",
    ("BP", "STATE_CHANGE", "direct_question"): "视频中{subject}发生了什么可观察的状态变化？",
    ("BP", "TEMPORAL_ORDER", "direct_question"): "视频中与{subject}有关的事件按什么顺序发生？",
    ("CM", "CLAIM_EVIDENCE_RELATION", "relation_choice"): "画面或文本证据与口播/文案中关于{claim}的说法是什么关系？",
    ("CM", "CLAIM_PARTIAL_SUPPORT", "supported_missing"): "画面中的哪些证据支持或不支持口播关于{claim}的说法？",
    ("CM", "TEXT_VISUAL_CONSISTENCY", "relation_choice"): "口播或画面文字与画面呈现是否一致？",
    ("SS", "HOOK_MECHANISM", "mechanism_with_evidence"): "视频开场使用了什么吸引注意的机制？请结合证据回答。",
    ("SS", "VALUE_PROPOSITION", "mechanism_with_evidence"): "该视频主要强调了什么价值主张？请结合证据回答。",
    ("SS", "TRUST_MECHANISM", "mechanism_with_evidence"): "该视频使用了什么建立信任的机制？请结合证据回答。",
    ("SS", "OBJECTION_HANDLING", "mechanism_with_evidence"): "该视频如何回应潜在用户的顾虑？请结合证据回答。",
    ("SS", "URGENCY_CTA", "mechanism_with_evidence"): "该视频使用了什么行动提示或紧迫性机制？请结合证据回答。",
    ("SS", "FUNNEL_ROLE", "mechanism_with_evidence"): "该内容片段在吸引注意、传递信息、建立信任或促进行动中承担什么作用？请结合证据回答。",
    ("AE", "AUDIENCE_NEED_FIT", "need_with_evidence"): "该内容主要满足哪类受众的什么需求？请说明视频证据。",
    ("AE", "USAGE_CONTEXT", "need_with_evidence"): "该内容对应的主要使用场景是什么？请说明视频证据。",
    ("AE", "DECISION_STATE", "need_with_evidence"): "视频内容主要回应了受众处于什么决策状态时的障碍？请说明视频证据。",
    ("AE", "CONTENT_MOTIVATION", "need_with_evidence"): "视频通过什么可观察内容回应受众需求或促使其继续了解？请说明视频证据。",
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
    context.setdefault("subject", item.target.get("subject") or item.gold_value.get("subject") or "目标对象")
    context.setdefault("claim", item.target.get("claim") or item.gold_value.get("claim") or "该说法")
    context.setdefault("mechanism", item.target.get("mechanism") or item.gold_value.get("label") or "该机制")
    return context


def derive_answer(item: GoldItem) -> str:
    if item.task_type.value == "CM":
        return _first_value(item.gold_value, ("relation", "answer"))
    if item.task_type.value == "BP":
        return _first_value(item.gold_value, ("action", "count", "value", "answer"))
    if item.task_type.value == "SS":
        return _first_value(item.gold_value, ("label", "strategy", "mechanism", "answer"))
    return _first_value(item.gold_value, ("audience_need", "usage_context", "decision_state", "answer"))


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
