from __future__ import annotations

import json
from copy import deepcopy
import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankPipeline, build_bp_proposals_from_graph  # noqa: E402
from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CommerceCue,
    CueType,
    RelationType,
    make_cue_id,
    make_relation_id,
)
from salesbench.goldbank.schema import EvidenceModality, EvidenceUnit  # noqa: E402
from salesbench.multiagent.context import build_context_bundle  # noqa: E402
from salesbench.vlm.api_client import APICallResult  # noqa: E402


class FakeGoldClient:
    def __init__(self, responses: list[dict[str, object] | str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.model = "fake-model"

    def _result(self) -> APICallResult:
        response = self.responses.pop(0)
        if isinstance(response, str):
            raw = response
        else:
            raw = json.dumps(response, ensure_ascii=False)
        return APICallResult(
            raw_response=raw,
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            latency_s=0.01,
            cost_usd=0.0,
            success=True,
        )

    def call(self, system_prompt: str, user_content: list[dict], response_format: str | None = None) -> APICallResult:
        self.calls.append({"method": "call", "system_prompt": system_prompt, "user_content": user_content})
        return self._result()

    def call_text_only(self, system_prompt: str, user_text: str, response_format: str | None = None) -> APICallResult:
        self.calls.append({"method": "call_text_only", "system_prompt": system_prompt, "user_text": user_text})
        return self._result()


def bundle():
    return build_context_bundle(
        "v1",
        raw_video={"video_id": "v1", "title": "T", "video_text": "", "likes": 99},
        text_language={"title": "T"},
    )


def successful_responses() -> list[dict[str, object]]:
    evidence_id = "v1_visual_000_abc"
    return [
        {
            "evidence_units": [
                {
                    "evidence_id": evidence_id,
                    "video_id": "v1",
                    "modality": "visual",
                    "start_s": 0,
                    "end_s": 1,
                    "frame_indices": [0],
                    "text_span": "",
                    "subject": "product",
                    "predicate": "visible",
                    "value": True,
                    "attributes": {},
                    "source_domains": ["C6_raw_video"],
                    "extractor": "fake",
                    "confidence": 0.95,
                    "timestamp_status": "available",
                }
            ]
        },
        {
            "commerce_cues": [
                {
                    "cue_type": "PRODUCT_IDENTITY",
                    "content_en": "A product is visible.",
                    "source_text_native": "",
                    "evidence_ids": [evidence_id],
                    "attributes": {},
                    "directness": "DIRECT",
                    "theory_tags": ["product_description"],
                    "confidence": 0.95,
                }
            ],
            "abstentions": [],
        },
        {"commercial_relations": [], "abstentions": []},
        {
            "proposals": [
                {
                    "proposal_id": "p_cm",
                    "video_id": "v1",
                    "source_agent": "cm_proposer",
                    "task_type": "CM",
                    "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                    "target": {"claim": "claim"},
                    "proposed_gold": {"relation": "SUPPORTED"},
                    "evidence_ids": [evidence_id, evidence_id],
                    "reasoning_edges": [],
                    "proposal_confidence": 0.9,
                }
            ],
            "abstentions": [],
        },
        {"proposals": [], "abstentions": []},
        {"proposals": [], "abstentions": []},
        {
            "reviews": [
                {
                    "review_id": "r_cm",
                    "proposal_id": "p_cm",
                    "video_id": "v1",
                    "reviewer": "gold_challenger",
                    "verdict": "PASS",
                    "checks": {"evidence_exists": True},
                    "issues": [],
                    "suggested_revision": None,
                }
            ]
        },
        {
            "grounded_annotations": [
                {
                    "gold_id": "g_cm",
                    "video_id": "v1",
                    "task_type": "CM",
                    "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                    "target": {"claim": "claim"},
                    "gold_value": {"relation": "SUPPORTED"},
                    "evidence_ids": [evidence_id, evidence_id],
                    "reasoning_edges": [],
                    "eligible_question_formats": ["relation_choice"],
                    "source_proposal_ids": ["p_cm"],
                    "gold_tier": "Gold-B",
                    "review_status": "verified",
                    "confidence": 0.9,
                }
            ],
            "human_review_queue": [],
        },
    ]


def successful_responses_with_two_evidence() -> list[dict[str, object]]:
    responses = deepcopy(successful_responses())
    second_id = "v1_asr_001_def"
    responses[0]["evidence_units"].append(
        {
            "evidence_id": second_id,
            "video_id": "v1",
            "modality": "asr",
            "start_s": 1,
            "end_s": 2,
            "frame_indices": [],
            "text_span": "这款产品采用编织材质",
            "subject": "speaker",
            "predicate": "claims",
            "value": "the product uses braided material",
            "attributes": {},
            "source_domains": ["C2_audio_speech"],
            "extractor": "fake",
            "confidence": 0.95,
            "timestamp_status": "available",
        }
    )
    process_content = "The product is demonstrated in use."
    claim_content = "The speaker claims that the product uses braided material."
    process_cue_id = make_cue_id(
        "v1", CueType.PROCESS_DEMONSTRATION, ("v1_visual_000_abc",), process_content
    )
    claim_cue_id = make_cue_id("v1", CueType.FUNCTION_CLAIM, (second_id,), claim_content)
    relation_id = make_relation_id(
        "v1",
        RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION,
        (claim_cue_id,),
        (process_cue_id,),
    )
    responses[1] = {
        "commerce_cues": [
            {
                "cue_type": "PROCESS_DEMONSTRATION",
                "content_en": process_content,
                "source_text_native": "",
                "evidence_ids": ["v1_visual_000_abc"],
                "attributes": {},
                "directness": "DIRECT",
                "theory_tags": ["product_demonstration"],
                "confidence": 0.95,
            },
            {
                "cue_type": "FUNCTION_CLAIM",
                "content_en": claim_content,
                "source_text_native": "这款产品采用编织材质",
                "evidence_ids": [second_id],
                "attributes": {},
                "directness": "DIRECT",
                "theory_tags": ["product_claim"],
                "confidence": 0.95,
            },
        ],
        "abstentions": [],
    }
    responses[2] = {
        "commercial_relations": [
            {
                "relation_type": "CLAIM_SUPPORTED_BY_DEMONSTRATION",
                "source_cue_ids": [claim_cue_id],
                "target_cue_ids": [process_cue_id],
                "evidence_ids": [second_id, "v1_visual_000_abc"],
                "status": "SUPPORTED",
                "rationale_en": "The spoken product claim is paired with a distinct visual demonstration.",
                "directness": "DIRECT",
                "confidence": 0.9,
            }
        ],
        "abstentions": [],
    }
    responses[3]["proposals"][0]["evidence_ids"] = ["v1_visual_000_abc", second_id]
    responses[3]["proposals"][0].update(
        {
            "capability": "CLAIM_DEMONSTRATION_STATUS",
            "reasoning_operator": "CLASSIFY_CLAIM_SUPPORT",
            "commerce_cue_ids": [claim_cue_id, process_cue_id],
            "commercial_relation_ids": [relation_id],
            "question_intent": "Ask how the visible demonstration relates to the spoken product claim.",
            "forbidden_inferences": ["Do not treat repeated wording as independent proof."],
        }
    )
    responses[7]["grounded_annotations"][0]["evidence_ids"] = ["v1_visual_000_abc", second_id]
    return responses


def successful_v7_responses() -> list[dict[str, object]]:
    responses = successful_responses_with_two_evidence()
    responses[7] = {
        "accepted_groups": [{"source_proposal_ids": ["p_cm"], "reason": "The two modalities agree."}],
        "human_review_queue": [],
    }
    return responses


class GoldBankPipelineTest(unittest.TestCase):
    def test_pipeline_extracts_language_and_visual_evidence_in_separate_stages(self):
        responses = successful_responses_with_two_evidence()
        visual_raw, asr_raw = responses[0]["evidence_units"]
        vlm = FakeGoldClient([{"evidence_units": [visual_raw]}])
        llm = FakeGoldClient([{"evidence_units": [asr_raw]}, *responses[1:]])
        split_bundle = build_context_bundle(
            "v1",
            raw_video={"video_id": "v1", "video_text": "这款产品采用编织材质"},
            frames=[{"frame_index": 0, "timestamp_s": 0.0, "path": "/tmp/f0.jpg"}],
        )

        result = GoldBankPipeline(vlm, llm).run_video(split_bundle, frames_b64=["AAA"])

        stages = [trace["stage"] for trace in result.agent_traces]
        self.assertEqual(stages[:2], ["language_evidence_extraction", "visual_evidence_extraction"])
        self.assertLess(stages.index("visual_evidence_extraction"), stages.index("commerce_cue_extraction"))
        self.assertEqual({unit["modality"] for unit in result.evidence_units}, {"asr", "visual"})
        self.assertEqual(vlm.calls[0]["user_content"][0]["text"], "[FRAME frame_index=0 timestamp_s=0.0]")

    def test_pipeline_repairs_missing_visual_commerce_cues_before_relations(self):
        responses = successful_responses_with_two_evidence()
        visual_raw, asr_raw = responses[0]["evidence_units"]
        initial_cues = {
            "commerce_cues": [responses[1]["commerce_cues"][1]],
            "abstentions": [],
        }
        visual_cues = {
            "commerce_cues": [responses[1]["commerce_cues"][0]],
            "abstentions": [],
        }
        vlm = FakeGoldClient([{"evidence_units": [visual_raw]}])
        llm = FakeGoldClient(
            [
                {"evidence_units": [asr_raw]},
                initial_cues,
                visual_cues,
                *responses[2:],
            ]
        )
        split_bundle = build_context_bundle(
            "v1",
            raw_video={"video_id": "v1", "video_text": "这款产品采用编织材质"},
            frames=[{"frame_index": 0, "timestamp_s": 0.0, "path": "/tmp/f0.jpg"}],
        )

        result = GoldBankPipeline(vlm, llm).run_video(split_bundle, frames_b64=["AAA"])

        stages = [trace["stage"] for trace in result.agent_traces]
        self.assertIn("visual_commerce_cue_repair", stages)
        process = next(cue for cue in result.commerce_cues if cue["cue_type"] == "PROCESS_DEMONSTRATION")
        self.assertEqual(process["evidence_ids"], ["v1_visual_000_abc"])

    def test_bp_preserves_spoken_claim_boundary_and_requires_visual_demonstration(self):
        evidence = EvidenceUnit(
            evidence_id="v1_asr_000_claim",
            video_id="v1",
            modality=EvidenceModality.ASR,
            start_s=0.0,
            end_s=1.0,
            frame_indices=(),
            text_span="支持六十瓦快充",
            subject="speaker",
            predicate="claims",
            value="the cable supports 60W fast charging",
            attributes={},
            source_domains=("C2_audio_speech",),
            extractor="test",
            confidence=0.95,
            timestamp_status="available",
            content_en="The speaker claims that the cable supports 60W fast charging.",
            source_text_native="支持六十瓦快充",
        )
        attribute = CommerceCue(
            cue_id="cue_attribute",
            video_id="v1",
            cue_type=CueType.PRODUCT_ATTRIBUTE,
            content_en="The cable supports 60W fast charging.",
            source_text_native="支持六十瓦快充",
            evidence_ids=(evidence.evidence_id,),
            attributes={},
            directness="DIRECT",
            theory_tags=("product_claim",),
            extractor="test",
            confidence=0.95,
        )
        claimed_demo = CommerceCue(
            cue_id="cue_demo",
            video_id="v1",
            cue_type=CueType.PROCESS_DEMONSTRATION,
            content_en="The cable is tested for fast charging.",
            source_text_native="支持六十瓦快充",
            evidence_ids=(evidence.evidence_id,),
            attributes={},
            directness="DIRECT",
            theory_tags=("product_demonstration",),
            extractor="test",
            confidence=0.95,
        )

        proposals = build_bp_proposals_from_graph("v1", [evidence], [attribute, claimed_demo], [])

        self.assertEqual(len(proposals), 1)
        self.assertEqual(
            proposals[0].proposed_gold["answer"],
            "The speaker states: The cable supports 60W fast charging.",
        )
        self.assertIn("what the speaker states", proposals[0].question_intent.lower())

    def test_v7_adjudicator_decision_reconstructs_annotation_locally(self):
        responses = successful_v7_responses()
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        cm_item = next(item for item in result.video_gold_record["grounded_annotations"] if item["task_type"] == "CM")
        self.assertEqual(cm_item["source_proposal_ids"], ["p_cm"])
        self.assertEqual(cm_item["evidence_refs"], ["v1_visual_000_abc", "v1_asr_001_def"])
        self.assertEqual(cm_item["quality_status"], "DIRECT")
        adjudication_trace = next(trace for trace in result.agent_traces if trace["stage"] == "adjudication")
        self.assertIn("accepted_groups", adjudication_trace["parsed_output"])

    def test_evidence_images_keep_frame_labels_and_order(self):
        vlm = FakeGoldClient(successful_responses()[:1])
        llm = FakeGoldClient(successful_responses()[1:])
        frame_bundle = build_context_bundle(
            "v1",
            raw_video={"video_id": "v1", "video_text": ""},
            frames=[
                {"frame_index": 4, "timestamp_s": 1.5, "path": "/tmp/f4.jpg"},
                {"frame_index": 9, "timestamp_s": 4.0, "path": "/tmp/f9.jpg"},
            ],
        )

        GoldBankPipeline(vlm, llm).run_video(frame_bundle, frames_b64=["AAA", "BBB"])

        content = vlm.calls[0]["user_content"]
        self.assertEqual(content[0]["text"], "[FRAME frame_index=4 timestamp_s=1.5]")
        self.assertTrue(content[1]["image_url"]["url"].endswith("AAA"))
        self.assertEqual(content[2]["text"], "[FRAME frame_index=9 timestamp_s=4.0]")
        self.assertTrue(content[3]["image_url"]["url"].endswith("BBB"))

    def test_pipeline_builds_evidence_before_proposals(self):
        vlm = FakeGoldClient(successful_responses()[:1])
        llm = FakeGoldClient(successful_responses()[1:])
        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        self.assertEqual(result.status, "ok")
        self.assertEqual([trace["stage"] for trace in result.agent_traces], [
            "evidence_extraction",
            "commerce_cue_extraction",
            "commercial_relation_building",
            "task_proposal",
            "task_proposal",
            "task_proposal",
            "challenge",
            "adjudication",
        ])
        self.assertTrue(result.commerce_cues)
        self.assertEqual(result.commercial_relations, [])

    def test_bp_builder_does_not_call_proposer(self):
        vlm = FakeGoldClient(successful_responses()[:1])
        llm = FakeGoldClient(successful_responses()[1:])
        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        proposer_text = " ".join(str(call) for call in llm.calls[:3])
        self.assertNotIn("Propose only BP", proposer_text)
        bp = next(proposal for proposal in result.gold_proposals if proposal["task_type"] == "BP")
        self.assertEqual(bp["source_agent"], "bp_compiler")

    def test_rejected_proposal_never_enters_gold_items(self):
        responses = successful_responses()
        responses[6] = {
            "reviews": [
                {
                    "review_id": "r_cm",
                    "proposal_id": "p_cm",
                    "video_id": "v1",
                    "reviewer": "gold_challenger",
                    "verdict": "REJECT",
                    "checks": {},
                    "issues": ["unsupported"],
                    "suggested_revision": None,
                }
            ]
        }
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        gold_ids = [item["annotation_id"] for item in result.video_gold_record["grounded_annotations"]]
        self.assertNotIn("g_cm", gold_ids)

    def test_low_confidence_proposal_enters_review_and_cannot_be_passed(self):
        responses = successful_responses()
        responses[3]["proposals"][0]["proposal_confidence"] = 0.6
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm, min_confidence=0.7).run_video(bundle())

        self.assertTrue(
            any(
                item["proposal_id"] == "p_cm" and item["reason"] == "below_min_confidence"
                for item in result.human_review_queue
            )
        )
        self.assertNotIn(
            "g_cm",
            [item["annotation_id"] for item in result.video_gold_record["grounded_annotations"]],
        )

        queue_item = next(item for item in result.human_review_queue if item["proposal_id"] == "p_cm")
        self.assertEqual(queue_item["stage"], "proposal")
        self.assertEqual(queue_item["item_type"], "candidate")
        self.assertEqual(queue_item["task_type"], "CM")
        self.assertEqual(queue_item["candidate_gold"], {"relation": "SUPPORTED"})
        self.assertEqual(queue_item["evidence_refs"], ["v1_visual_000_abc"])

    def test_abstention_uses_canonical_review_queue_shape(self):
        responses = successful_responses()
        responses[4]["abstentions"] = [
            {
                "task_type": "SS",
                "task_subtype": "PROCESS_DEMONSTRATION",
                "reason": "Fewer than two evidence units support this mechanism.",
            }
        ]
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        abstention = next(item for item in result.human_review_queue if item["item_type"] == "abstention")
        self.assertEqual(abstention["stage"], "proposal")
        self.assertEqual(abstention["task_type"], "SS")
        self.assertEqual(abstention["candidate_gold"], {})
        self.assertEqual(abstention["reason"], "Fewer than two evidence units support this mechanism.")

    def test_rejected_bp_proposal_never_enters_grounded_annotations(self):
        responses = successful_responses()
        local_bp_id = "v1_local_bp_000"
        responses[6] = {
            "reviews": [
                {
                    "review_id": "r_bp",
                    "proposal_id": local_bp_id,
                    "video_id": "v1",
                    "reviewer": "gold_challenger",
                    "verdict": "REJECT",
                    "checks": {},
                    "issues": ["unsupported"],
                    "suggested_revision": None,
                }
            ]
        }
        responses[7]["grounded_annotations"] = [
            {
                "annotation_id": "g_bp_injected",
                "video_id": "v1",
                "task_type": "BP",
                "task_subtype": "PRODUCT_IDENTITY",
                "target": {"subject": "product"},
                "gold_value": {"value": True},
                "evidence_refs": ["v1_visual_000_abc"],
                "reasoning_edges": [],
                "eligible_question_formats": ["direct_question"],
                "source_proposal_ids": [local_bp_id],
                "quality_status": "DIRECT",
                "review_status": "verified",
                "confidence": 0.95,
            }
        ]
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        self.assertFalse(
            any(item["task_type"] == "BP" for item in result.video_gold_record["grounded_annotations"])
        )

    def test_adjudicator_quality_status_is_ignored(self):
        responses = successful_responses_with_two_evidence()
        annotation = responses[7]["grounded_annotations"][0]
        annotation["quality_status"] = "REJECTED"
        annotation["gold_tier"] = "Rejected"
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        cm_item = next(item for item in result.video_gold_record["grounded_annotations"] if item["task_type"] == "CM")
        self.assertEqual(cm_item["quality_status"], "DIRECT")

    def test_semantic_duplicates_are_removed(self):
        responses = successful_responses_with_two_evidence()
        duplicate_proposal = deepcopy(responses[3]["proposals"][0])
        duplicate_proposal["proposal_id"] = "p_cm_duplicate"
        responses[3]["proposals"].append(duplicate_proposal)
        responses[6]["reviews"].append(
            {
                "review_id": "r_cm_duplicate",
                "proposal_id": "p_cm_duplicate",
                "video_id": "v1",
                "reviewer": "gold_challenger",
                "verdict": "PASS",
                "checks": {"evidence_exists": True},
                "issues": [],
                "suggested_revision": None,
            }
        )
        duplicate_annotation = deepcopy(responses[7]["grounded_annotations"][0])
        duplicate_annotation["gold_id"] = "g_cm_duplicate"
        duplicate_annotation["source_proposal_ids"] = ["p_cm_duplicate"]
        responses[7]["grounded_annotations"].append(duplicate_annotation)
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        cm_items = [
            item for item in result.video_gold_record["grounded_annotations"] if item["task_type"] == "CM"
        ]
        self.assertEqual(len(cm_items), 1)
        self.assertTrue(any(item["reason"] == "semantic_duplicate" for item in result.human_review_queue))

    def test_invalid_proposal_is_isolated_instead_of_dropping_agent_batch(self):
        responses = successful_responses_with_two_evidence()
        responses[4]["proposals"] = [
            {
                "proposal_id": "bad_cm",
                "task_type": "CM",
                "task_subtype": "OFFER_NEED_ALIGNMENT",
                "target": {"claim": "claim"},
                "proposed_gold": {"answer": "bad task pairing"},
                "evidence_ids": ["v1_visual_000_abc", "v1_asr_001_def"],
                "reasoning_edges": [],
                "proposal_confidence": 0.9,
            }
        ]
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        self.assertEqual(result.status, "partial")
        self.assertTrue(any(item["task_type"] == "CM" for item in result.video_gold_record["grounded_annotations"]))
        self.assertTrue(
            any("ss_proposer_proposal_parse_error" in item["reason"] for item in result.human_review_queue)
        )

    def test_parse_failure_is_visible_and_not_passed(self):
        vlm = FakeGoldClient(["not json"])
        llm = FakeGoldClient([])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.agent_traces[0])

    def test_agent_prompt_never_receives_performance_data(self):
        vlm = FakeGoldClient(successful_responses()[:1])
        llm = FakeGoldClient(successful_responses()[1:])

        GoldBankPipeline(vlm, llm).run_video(bundle())
        prompt_text = str(vlm.calls) + str(llm.calls)

        self.assertNotIn("likes", prompt_text)
        self.assertNotIn("performance_data", prompt_text)

    def test_adjudicator_receives_only_non_bp_passed_proposals(self):
        responses = successful_responses_with_two_evidence()
        responses[6]["reviews"].append(
            {
                "review_id": "r_bp",
                "proposal_id": "v1_local_bp_000",
                "video_id": "v1",
                "reviewer": "gold_challenger",
                "verdict": "PASS",
                "checks": {"evidence_exists": True},
                "issues": [],
                "suggested_revision": None,
            }
        )
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        GoldBankPipeline(vlm, llm).run_video(bundle())

        adjudicator_payload = llm.calls[-1]["user_text"]
        self.assertIn('"task_type": "CM"', adjudicator_payload)
        self.assertNotIn('"task_type": "BP"', adjudicator_payload)
        self.assertNotIn("v1_local_bp_000", adjudicator_payload)


if __name__ == "__main__":
    unittest.main()
