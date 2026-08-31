"""Fail-closed local validity gates for QuestionSpecs and compiled QA items."""

from __future__ import annotations

import re
from typing import Any

from ..goldbank.commerce_schema import RelationType
from ..goldbank.schema import GoldTaskType
from ..utils import clean_text, contains_cjk


FORMAL_LIFECYCLE_STATUSES = {"human_accepted"}
CANDIDATE_LIFECYCLE_STATUSES = {"auto_accepted_candidate", "verified"}

_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
_GENERIC_QUESTIONS = {
    "what is shown in the video",
    "what happens in the video",
    "what does the video show",
    "what is happening",
    "what is the main point",
}
_BENCHMARK_META_MARKERS = (
    "evidenceunit",
    "evidence unit",
    "commercecue",
    "commerce cue",
    "commercialrelation",
    "commercial relation",
    "evidence_id",
    "evidence_refs",
    "cue_id",
    "relation_id",
    "spec_id",
    "task_type",
    "task_subtype",
)
_RELATION_ENUM_TOKENS = tuple(relation.value.lower() for relation in RelationType)


def _task(value: Any) -> GoldTaskType:
    if isinstance(value, GoldTaskType):
        return value
    return GoldTaskType(clean_text(value).upper())


def _contains_benchmark_meta(value: object) -> bool:
    lowered = clean_text(value).lower()
    return any(marker in lowered for marker in _BENCHMARK_META_MARKERS) or any(
        re.search(rf"\b{re.escape(token)}\b", lowered) for token in _RELATION_ENUM_TOKENS
    )


def answer_type_for_task(task_type: GoldTaskType | str, answer: str) -> str:
    task = _task(task_type)
    if task == GoldTaskType.BP:
        return "numeric_fact" if _NUMBER_RE.search(clean_text(answer)) else "short_fact"
    if task == GoldTaskType.CM:
        return "relation_explanation"
    if task == GoldTaskType.SS:
        return "commercial_reasoning"
    return "bounded_need_reasoning"


def validate_question_spec(
    spec: Any,
    *,
    allow_auto_candidates: bool = False,
) -> list[str]:
    issues: list[str] = []
    lifecycle = clean_text(getattr(spec, "lifecycle_status", "")).lower()
    allowed = set(FORMAL_LIFECYCLE_STATUSES)
    if allow_auto_candidates:
        allowed.update(CANDIDATE_LIFECYCLE_STATUSES)
    if lifecycle not in allowed:
        issues.append("LIFECYCLE_NOT_COMPILABLE")

    task = _task(getattr(spec, "task_type", ""))
    evidence_refs = tuple(getattr(spec, "evidence_refs", ()) or ())
    minimum = 1 if task == GoldTaskType.BP else 2
    if len(set(evidence_refs)) < minimum:
        issues.append("INSUFFICIENT_TASK_EVIDENCE")
    if task == GoldTaskType.CM and not tuple(
        getattr(spec, "commercial_relation_ids", ()) or ()
    ):
        issues.append("MISSING_COMMERCIAL_RELATION")
    if task == GoldTaskType.SS and not tuple(
        getattr(spec, "commercial_relation_ids", ()) or ()
    ):
        issues.append("MISSING_COMMERCIAL_RELATION")
    if task == GoldTaskType.AE and not (
        tuple(getattr(spec, "commerce_cue_ids", ()) or ())
        or tuple(getattr(spec, "commercial_relation_ids", ()) or ())
    ):
        issues.append("MISSING_COMMERCE_GRAPH_REF")
    if not getattr(spec, "target", None):
        issues.append("MISSING_TARGET")
    gold_answer = clean_text(getattr(spec, "gold_answer", ""))
    if not gold_answer:
        issues.append("MISSING_GOLD_ANSWER")
    if _contains_benchmark_meta(gold_answer):
        issues.append("BENCHMARK_META_LEAKAGE")
    if not clean_text(getattr(spec, "question_intent", "")):
        issues.append("MISSING_QUESTION_INTENT")
    expected_answer_type = answer_type_for_task(
        task, gold_answer
    )
    if clean_text(getattr(spec, "answer_type", "")) != expected_answer_type:
        issues.append("ANSWER_TYPE_MISMATCH")
    return list(dict.fromkeys(issues))


def validate_qa_candidate(
    spec_or_item: Any,
    question: str,
    gold_answer: str,
) -> list[str]:
    issues: list[str] = []
    normalized = clean_text(question)
    lowered = normalized.lower()
    normalized_stem = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    if not normalized or contains_cjk(normalized):
        issues.append("NON_ENGLISH_QUESTION")
    if normalized and not normalized.endswith("?"):
        issues.append("MISSING_QUESTION_MARK")
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", normalized))
    if word_count < 5 or word_count > 45:
        issues.append("QUESTION_LENGTH_OUT_OF_RANGE")
    if normalized_stem in _GENERIC_QUESTIONS:
        issues.append("GENERIC_QUESTION")
    answer = clean_text(gold_answer)
    normalized_answer = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    if answer and len(answer) > 3 and normalized_answer and normalized_answer in normalized_stem:
        issues.append("ANSWER_LEAKAGE")
    if contains_cjk(answer):
        issues.append("NON_ENGLISH_GOLD_ANSWER")
    if _contains_benchmark_meta(normalized) or _contains_benchmark_meta(answer):
        issues.append("BENCHMARK_META_LEAKAGE")

    task = _task(getattr(spec_or_item, "task_type", ""))
    answer_type = answer_type_for_task(task, answer)
    if answer_type == "short_fact" and len(answer.split()) > 20:
        issues.append("BP_ANSWER_TOO_LONG")
    if answer_type == "numeric_fact" and not _NUMBER_RE.search(answer):
        issues.append("INVALID_NUMERIC_ANSWER")
    return list(dict.fromkeys(issues))
