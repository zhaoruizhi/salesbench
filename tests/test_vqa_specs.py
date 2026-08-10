from __future__ import annotations

import sys

sys.path.insert(0, "src")

from salesbench.vqa.specs import (  # noqa: E402
    QuestionRealization,
    QuestionSpec,
    build_question_specs,
    make_question_spec_id,
    parse_question_realization,
    parse_question_spec,
)


def _record() -> dict[str, object]:
    return {
        "video_id": "v1",
        "schema_version": "evidence-dataset-schema-v3",
        "grounded_annotations": [
            {
                "annotation_id": "a1",
                "video_id": "v1",
                "task_type": "CM",
                "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                "capability": "CLAIM_DEMONSTRATION_STATUS",
                "reasoning_operator": "CLASSIFY_CLAIM_SUPPORT",
                "target": {"claim": "the cleaner removes the visible mark"},
                "gold_value": {
                    "answer": "The wiping demonstration shows the visible mark becoming lighter, but it does not establish long-term removal."
                },
                "evidence_refs": ["e1", "e2"],
                "commerce_cue_ids": ["c1", "c2"],
                "commercial_relation_ids": ["r1"],
                "question_intent": "Ask what the demonstration shows and what part of the claim remains unverified.",
                "forbidden_inferences": ["Do not treat the stated long-term effect as demonstrated."],
                "quality_status": "DIRECT",
                "review_status": "verified",
                "confidence": 0.9,
            }
        ],
    }


def test_question_spec_round_trips_without_audit_translation_fields():
    specs = build_question_specs([_record()])
    spec = specs[0]

    assert spec.spec_id == make_question_spec_id("v1", "a1", "CLAIM_DEMONSTRATION_STATUS")
    assert parse_question_spec(spec.to_dict()) == spec
    assert spec.gold_answer.startswith("The wiping demonstration")
    assert "question_zh" not in spec.to_dict()
    assert "translated_text" not in spec.to_dict()


def test_question_realization_round_trips_as_english_surface_form():
    realization = QuestionRealization(
        spec_id="qs1",
        video_id="v1",
        annotation_id="a1",
        question="What does the wiping demonstration show, and what claimed effect remains unverified?",
        model="gpt-4o",
        prompt_version="question-realizer-prompt-v1",
    )

    assert parse_question_realization(realization.to_dict()) == realization
    assert realization.question.endswith("?")
