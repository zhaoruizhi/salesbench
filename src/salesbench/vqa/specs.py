"""Semantic QA specifications and surface-realization contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ..goldbank.schema import GoldTaskType, stable_digest
from ..utils import clean_text, normalize_speaker_attribution
from .item_validator import answer_type_for_task, validate_question_spec


def _tuple(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def make_question_spec_id(video_id: str, annotation_id: str, capability: str) -> str:
    digest = stable_digest(
        {
            "video_id": clean_text(video_id),
            "annotation_id": clean_text(annotation_id),
            "capability": clean_text(capability).upper(),
        }
    )
    return f"{clean_text(video_id)}_qs_{clean_text(capability).lower()}_{digest}"


@dataclass(frozen=True)
class QuestionSpec:
    spec_id: str
    video_id: str
    annotation_id: str
    task_type: GoldTaskType
    capability: str
    reasoning_operator: str
    question_intent: str
    target: dict[str, object]
    gold_answer: str
    evidence_refs: tuple[str, ...]
    commerce_cue_ids: tuple[str, ...]
    commercial_relation_ids: tuple[str, ...]
    forbidden_inferences: tuple[str, ...]
    answer_type: str = "open"
    lifecycle_status: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "video_id": self.video_id,
            "annotation_id": self.annotation_id,
            "task_type": self.task_type.value,
            "capability": self.capability,
            "reasoning_operator": self.reasoning_operator,
            "question_intent": self.question_intent,
            "target": dict(self.target),
            "gold_answer": self.gold_answer,
            "evidence_refs": list(self.evidence_refs),
            "commerce_cue_ids": list(self.commerce_cue_ids),
            "commercial_relation_ids": list(self.commercial_relation_ids),
            "forbidden_inferences": list(self.forbidden_inferences),
            "answer_type": self.answer_type,
            "lifecycle_status": self.lifecycle_status,
        }


@dataclass(frozen=True)
class QuestionRealization:
    spec_id: str
    video_id: str
    annotation_id: str
    question: str
    model: str
    prompt_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "video_id": self.video_id,
            "annotation_id": self.annotation_id,
            "question": self.question,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


def parse_question_spec(record: dict[str, object]) -> QuestionSpec:
    return QuestionSpec(
        spec_id=clean_text(record.get("spec_id")),
        video_id=clean_text(record.get("video_id")),
        annotation_id=clean_text(record.get("annotation_id")),
        task_type=GoldTaskType(clean_text(record.get("task_type")).upper()),
        capability=clean_text(record.get("capability")).upper(),
        reasoning_operator=clean_text(record.get("reasoning_operator")).upper(),
        question_intent=clean_text(record.get("question_intent")),
        target=dict(record.get("target")) if isinstance(record.get("target"), dict) else {},
        gold_answer=clean_text(record.get("gold_answer")),
        evidence_refs=tuple(clean_text(value) for value in _tuple(record.get("evidence_refs"))),
        commerce_cue_ids=tuple(clean_text(value) for value in _tuple(record.get("commerce_cue_ids"))),
        commercial_relation_ids=tuple(
            clean_text(value) for value in _tuple(record.get("commercial_relation_ids"))
        ),
        forbidden_inferences=tuple(
            clean_text(value) for value in _tuple(record.get("forbidden_inferences"))
        ),
        answer_type=clean_text(record.get("answer_type") or "open"),
        lifecycle_status=clean_text(record.get("lifecycle_status")),
    )


def parse_question_realization(record: dict[str, object]) -> QuestionRealization:
    return QuestionRealization(
        spec_id=clean_text(record.get("spec_id")),
        video_id=clean_text(record.get("video_id")),
        annotation_id=clean_text(record.get("annotation_id")),
        question=clean_text(record.get("question")),
        model=clean_text(record.get("model")),
        prompt_version=clean_text(record.get("prompt_version")),
    )


def _answer(gold_value: dict[str, object]) -> str:
    for key in (
        "answer",
        "value",
        "action",
        "relation",
        "usage_context",
        "label",
    ):
        value = gold_value.get(key)
        if value not in (None, ""):
            return normalize_speaker_attribution(value)
    if gold_value:
        return normalize_speaker_attribution(gold_value[sorted(gold_value)[0]])
    return ""


def build_question_specs(
    video_records: list[dict[str, object]],
    *,
    allow_auto_candidates: bool = False,
    include_invalid: bool = False,
) -> list[QuestionSpec]:
    specs: list[QuestionSpec] = []
    for record in video_records:
        video_id = clean_text(record.get("video_id"))
        for raw in record.get("grounded_annotations", []) or []:
            if not isinstance(raw, dict):
                continue
            if clean_text(raw.get("quality_status")).upper() not in {"DIRECT", "INFERRED"}:
                continue
            annotation_id = clean_text(raw.get("annotation_id") or raw.get("gold_id"))
            task_type = GoldTaskType(clean_text(raw.get("task_type")).upper())
            capability = clean_text(raw.get("capability") or raw.get("task_subtype")).upper()
            gold_value = dict(raw.get("gold_value")) if isinstance(raw.get("gold_value"), dict) else {}
            gold_answer = _answer(gold_value)
            spec = QuestionSpec(
                    spec_id=make_question_spec_id(video_id, annotation_id, capability),
                    video_id=video_id,
                    annotation_id=annotation_id,
                    task_type=task_type,
                    capability=capability,
                    reasoning_operator=clean_text(raw.get("reasoning_operator")).upper(),
                    question_intent=clean_text(raw.get("question_intent")),
                    target=dict(raw.get("target")) if isinstance(raw.get("target"), dict) else {},
                    gold_answer=gold_answer,
                    evidence_refs=tuple(
                        clean_text(value) for value in _tuple(raw.get("evidence_refs"))
                    ),
                    commerce_cue_ids=tuple(
                        clean_text(value) for value in _tuple(raw.get("commerce_cue_ids"))
                    ),
                    commercial_relation_ids=tuple(
                        clean_text(value)
                        for value in _tuple(raw.get("commercial_relation_ids"))
                    ),
                    forbidden_inferences=tuple(
                        clean_text(value) for value in _tuple(raw.get("forbidden_inferences"))
                    ),
                    answer_type=answer_type_for_task(task_type, gold_answer),
                    lifecycle_status=clean_text(raw.get("review_status")).lower(),
                )
            issues = validate_question_spec(
                spec,
                allow_auto_candidates=allow_auto_candidates,
            )
            if include_invalid or not issues:
                specs.append(spec)
    return sorted(specs, key=lambda item: (item.video_id, item.task_type.value, item.spec_id))
