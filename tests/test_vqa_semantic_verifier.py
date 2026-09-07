from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

from salesbench.vlm.api_client import APICallResult  # noqa: E402
from salesbench.vqa.semantic_verifier import (  # noqa: E402
    QASemanticVerdict,
    parse_qa_semantic_verification,
    verify_qa_candidate,
)
from salesbench.vqa.specs import parse_question_spec  # noqa: E402


def question_spec():
    return parse_question_spec(
        {
            "spec_id": "qs1",
            "video_id": "v1",
            "annotation_id": "a1",
            "task_type": "CM",
            "capability": "CLAIM_DEMONSTRATION_STATUS",
            "reasoning_operator": "CLASSIFY_CLAIM_SUPPORT",
            "question_intent": "Ask what the demonstration supports.",
            "target": {"claim": "the cleaner removes the mark"},
            "gold_answer": "The mark becomes lighter, but long-term removal is not demonstrated.",
            "evidence_refs": ["e1", "e2"],
            "commerce_cue_ids": ["c1", "c2"],
            "commercial_relation_ids": ["r1"],
            "forbidden_inferences": ["Do not infer long-term removal."],
            "answer_type": "relation_explanation",
            "lifecycle_status": "auto_accepted_candidate",
        }
    )


class FakeClient:
    model = "gpt-4o"

    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[dict[str, object]] = []

    def call_text_only(self, system_prompt, user_text, response_format=None):
        self.calls.append({"system": system_prompt, "user": json.loads(user_text)})
        return APICallResult(
            raw_response=json.dumps(self.result),
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            latency_s=0.01,
            cost_usd=0.0,
            success=True,
        )


def result(verdict: str = "PASS", **overrides):
    payload = {
        "spec_id": "qs1",
        "verdict": verdict,
        "reason": "The question and answer are grounded in the supplied relation.",
        "answerable_from_evidence": True,
        "gold_supported": True,
        "unique_answer": True,
        "task_aligned": True,
        "content_specific": True,
        "commerce_relevant": True,
        "natural_question": True,
        "non_trivial": True,
        "commercially_diagnostic": True,
        "claim_scope_preserved": True,
        "intended_modality_required": True,
        "reference_closed": True,
    }
    payload.update(overrides)
    return payload


def test_parser_rejects_invented_spec_id():
    with pytest.raises(ValueError, match="changed spec_id"):
        parse_qa_semantic_verification(json.dumps(result(spec_id="invented")), "qs1")


def test_parser_cannot_pass_when_a_quality_dimension_fails():
    parsed = parse_qa_semantic_verification(
        json.dumps(result(gold_supported=False)),
        "qs1",
    )

    assert parsed.verdict == QASemanticVerdict.REJECT
    assert "gold_supported" in parsed.reason


def test_parser_requires_every_production_quality_dimension():
    incomplete = result()
    del incomplete["non_trivial"]

    with pytest.raises(ValueError, match="non_trivial"):
        parse_qa_semantic_verification(json.dumps(incomplete), "qs1")


def test_verifier_receives_only_cited_public_context():
    client = FakeClient(result())
    spec = question_spec()

    verification = verify_qa_candidate(
        client,
        spec,
        "What does the wiping demonstration support about the spoken removal claim?",
        [{"evidence_id": "e1", "content_en": "The seller states the removal claim."}],
        [{"cue_id": "c1", "content_en": "A removal claim is stated."}],
        [{"relation_id": "r1", "status": "PARTIALLY_SUPPORTED"}],
    )

    assert verification.verdict == QASemanticVerdict.PASS
    assert client.calls[0]["user"]["question_spec"]["spec_id"] == "qs1"
    serialized = json.dumps(client.calls[0], ensure_ascii=False)
    assert "followers" not in serialized
    assert "likes" not in serialized
