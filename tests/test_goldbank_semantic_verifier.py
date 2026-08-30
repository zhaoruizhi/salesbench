from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CommerceCue,
    CommercialRelation,
    CueType,
    RelationProvenance,
    RelationType,
)
from salesbench.goldbank.semantic_verifier import (  # noqa: E402
    SemanticVerdict,
    cited_frame_indices,
    parse_semantic_verifications,
    verify_relations,
)
from salesbench.goldbank.schema import (  # noqa: E402
    EvidenceModality,
    EvidenceUnit,
)
from salesbench.vlm.api_client import APICallResult  # noqa: E402


def visual_evidence() -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id="e_visual",
        video_id="v1",
        modality=EvidenceModality.VISUAL,
        start_s=1.0,
        end_s=1.0,
        frame_indices=(7,),
        text_span="",
        subject="host",
        predicate="bends",
        value="while wearing the pants",
        attributes={},
        source_domains=("C6_raw_video",),
        extractor="test",
        confidence=0.95,
        timestamp_status="available",
    )


def relation() -> CommercialRelation:
    return CommercialRelation(
        relation_id="r1",
        video_id="v1",
        relation_type=RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION,
        source_cue_ids=("c_claim",),
        target_cue_ids=("c_demo",),
        evidence_ids=("e_asr", "e_visual"),
        status="SUPPORTED",
        rationale_en="The visible action is presented as support for the spoken claim.",
        provenance=RelationProvenance.THEORY_OPERATIONALIZED,
        directness="INFERRED",
        extractor="test",
        confidence=0.9,
    )


class FakeVerifierClient:
    model = "fake-verifier"

    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def call(self, system_prompt, user_content, response_format=None):
        self.calls.append({"system": system_prompt, "user": user_content})
        return APICallResult(
            raw_response=json.dumps(self.payload),
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            latency_s=0.01,
            cost_usd=0.0,
            success=True,
        )


def test_parser_rejects_invented_relation_ids():
    with pytest.raises(ValueError, match="unknown relation_id"):
        parse_semantic_verifications(
            json.dumps(
                {
                    "verifications": [
                        {"relation_id": "invented", "verdict": "PASS", "reason": "Looks supported."}
                    ]
                }
            ),
            {"r1"},
        )


def test_cited_frame_selection_uses_only_relation_evidence():
    evidence = {"e_visual": visual_evidence()}

    assert cited_frame_indices(relation(), evidence) == (7,)


def test_frame_aware_verifier_returns_ambiguous_without_promoting_it():
    client = FakeVerifierClient(
        {
            "verifications": [
                {
                    "relation_id": "r1",
                    "verdict": "AMBIGUOUS",
                    "reason": "The single frame does not reveal the full state transition.",
                }
            ]
        }
    )
    visual = visual_evidence()
    claim = EvidenceUnit(
        **{
            **visual.__dict__,
            "evidence_id": "e_asr",
            "modality": EvidenceModality.ASR,
            "frame_indices": (),
            "text_span": "穿着不鼓包",
            "source_text_native": "穿着不鼓包",
        }
    )
    cues = {
        "c_claim": CommerceCue(
            "c_claim", "v1", CueType.FUNCTION_CLAIM, "The speaker makes a fit claim.", "",
            ("e_asr",), {}, "DIRECT", (), "test", 0.9,
        ),
        "c_demo": CommerceCue(
            "c_demo", "v1", CueType.PROCESS_DEMONSTRATION, "The host bends while wearing the pants.", "",
            ("e_visual",), {}, "DIRECT", (), "test", 0.9,
        ),
    }

    batch = verify_relations(
        client,
        "v1",
        [relation()],
        cues,
        {"e_visual": visual, "e_asr": claim},
        {7: "AAA"},
    )

    assert batch.verifications[0].verdict == SemanticVerdict.AMBIGUOUS
    assert any(block.get("type") == "image_url" for block in client.calls[0]["user"])
    assert "invent" in client.calls[0]["system"].lower()
