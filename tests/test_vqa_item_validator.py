from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, "src")

from salesbench.goldbank.schema import GoldTaskType  # noqa: E402
from salesbench.vqa.item_validator import (  # noqa: E402
    answer_type_for_task,
    validate_qa_candidate,
    validate_question_spec,
)


def spec(**overrides):
    payload = {
        "spec_id": "qs1",
        "video_id": "v1",
        "annotation_id": "a1",
        "task_type": GoldTaskType.CM,
        "capability": "CLAIM_DEMONSTRATION_STATUS",
        "reasoning_operator": "CLASSIFY_CLAIM_SUPPORT",
        "question_intent": "Ask which part of the spoken claim the visible demonstration supports.",
        "target": {"claim": "the cleaner removes the mark"},
        "gold_answer": "The mark becomes lighter, but long-term removal is not demonstrated.",
        "evidence_refs": ("e1", "e2"),
        "commerce_cue_ids": ("c1", "c2"),
        "commercial_relation_ids": ("r1",),
        "forbidden_inferences": ("Do not infer long-term removal.",),
        "answer_type": "relation_explanation",
        "lifecycle_status": "human_accepted",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_answer_schema_is_task_specific_without_replacing_llm_judge():
    assert answer_type_for_task(GoldTaskType.BP, "9.9") == "numeric_fact"
    assert answer_type_for_task(GoldTaskType.BP, "red package") == "short_fact"
    assert answer_type_for_task(GoldTaskType.CM, "supported") == "relation_explanation"
    assert answer_type_for_task(GoldTaskType.SS, "problem and solution") == "commercial_reasoning"
    assert answer_type_for_task(GoldTaskType.AE, "usage need") == "bounded_need_reasoning"


def test_non_formal_lifecycle_is_rejected_unless_candidate_mode_is_explicit():
    candidate = spec(lifecycle_status="auto_accepted_candidate")

    assert "LIFECYCLE_NOT_COMPILABLE" in validate_question_spec(candidate)
    assert "LIFECYCLE_NOT_COMPILABLE" not in validate_question_spec(
        candidate, allow_auto_candidates=True
    )


def test_cross_modal_spec_requires_two_evidence_units_and_a_relation():
    invalid = spec(evidence_refs=("e1",), commercial_relation_ids=())

    issues = validate_question_spec(invalid)

    assert "INSUFFICIENT_TASK_EVIDENCE" in issues
    assert "MISSING_COMMERCIAL_RELATION" in issues


def test_generic_or_answer_leaking_surface_is_rejected_before_compilation():
    generic = validate_qa_candidate(spec(), "What is shown in the video?", "A red package.")
    leaking = validate_qa_candidate(
        spec(),
        "How does the demonstration show that the mark becomes lighter?",
        "The mark becomes lighter.",
    )

    assert "GENERIC_QUESTION" in generic
    assert "ANSWER_LEAKAGE" in leaking


def test_internal_benchmark_ontology_is_rejected_from_gold_and_question_surfaces():
    leaked_gold = spec(
        gold_answer=(
            "The visual description refers to the product, as supported by the commercial "
            "relation DESCRIPTION_REFERS_TO_PRODUCT."
        )
    )

    assert "BENCHMARK_META_LEAKAGE" in validate_question_spec(leaked_gold)
    assert "BENCHMARK_META_LEAKAGE" in validate_qa_candidate(
        spec(),
        "What does CommercialRelation r1 establish about the product?",
        "The spoken description refers to the featured product.",
    )
