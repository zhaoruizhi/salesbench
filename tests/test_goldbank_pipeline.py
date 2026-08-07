from __future__ import annotations

import json
from copy import deepcopy
import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankPipeline  # noqa: E402
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
        raw_video={"video_id": "v1", "title": "T", "video_text": "claim", "likes": 99},
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
        {"proposals": [], "abstentions": []},
        {
            "proposals": [
                {
                    "proposal_id": "p_cm",
                    "video_id": "v1",
                    "source_agent": "operator",
                    "task_type": "CM",
                    "task_subtype": "CLAIM_EVIDENCE_RELATION",
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
                    "task_subtype": "CLAIM_EVIDENCE_RELATION",
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
    second_id = "v1_visual_001_def"
    responses[0]["evidence_units"].append(
        {
            "evidence_id": second_id,
            "video_id": "v1",
            "modality": "visual",
            "start_s": 1,
            "end_s": 2,
            "frame_indices": [1],
            "text_span": "",
            "subject": "product detail",
            "predicate": "visible",
            "value": True,
            "attributes": {},
            "source_domains": ["C6_raw_video"],
            "extractor": "fake",
            "confidence": 0.95,
            "timestamp_status": "available",
        }
    )
    responses[2]["proposals"][0]["evidence_ids"] = ["v1_visual_000_abc", second_id]
    responses[5]["grounded_annotations"][0]["evidence_ids"] = ["v1_visual_000_abc", second_id]
    return responses


class GoldBankPipelineTest(unittest.TestCase):
    def test_evidence_images_keep_frame_labels_and_order(self):
        vlm = FakeGoldClient(successful_responses()[:1])
        llm = FakeGoldClient(successful_responses()[1:])
        frame_bundle = build_context_bundle(
            "v1",
            raw_video={"video_id": "v1", "video_text": "claim"},
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
            "consumer_proposal",
            "operator_proposal",
            "strategist_proposal",
            "challenge",
            "adjudication",
        ])

    def test_bp_builder_does_not_call_proposer(self):
        vlm = FakeGoldClient(successful_responses()[:1])
        llm = FakeGoldClient(successful_responses()[1:])
        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        proposer_text = " ".join(str(call) for call in llm.calls[:3])
        self.assertNotIn("Propose only BP", proposer_text)
        self.assertTrue(any(proposal["task_type"] == "BP" for proposal in result.gold_proposals))

    def test_rejected_proposal_never_enters_gold_items(self):
        responses = successful_responses()
        responses[4] = {
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
        responses[2]["proposals"][0]["proposal_confidence"] = 0.6
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

    def test_rejected_bp_proposal_never_enters_grounded_annotations(self):
        responses = successful_responses()
        local_bp_id = "v1_local_bp_000"
        responses[4] = {
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
        responses[5]["grounded_annotations"] = [
            {
                "annotation_id": "g_bp_injected",
                "video_id": "v1",
                "task_type": "BP",
                "task_subtype": "ENTITY_ATTRIBUTE",
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
        annotation = responses[5]["grounded_annotations"][0]
        annotation["quality_status"] = "REJECTED"
        annotation["gold_tier"] = "Rejected"
        vlm = FakeGoldClient(responses[:1])
        llm = FakeGoldClient(responses[1:])

        result = GoldBankPipeline(vlm, llm).run_video(bundle())

        cm_item = next(item for item in result.video_gold_record["grounded_annotations"] if item["task_type"] == "CM")
        self.assertEqual(cm_item["quality_status"], "DIRECT")

    def test_semantic_duplicates_are_removed(self):
        responses = successful_responses_with_two_evidence()
        duplicate_proposal = deepcopy(responses[2]["proposals"][0])
        duplicate_proposal["proposal_id"] = "p_cm_duplicate"
        responses[2]["proposals"].append(duplicate_proposal)
        responses[4]["reviews"].append(
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
        duplicate_annotation = deepcopy(responses[5]["grounded_annotations"][0])
        duplicate_annotation["gold_id"] = "g_cm_duplicate"
        duplicate_annotation["source_proposal_ids"] = ["p_cm_duplicate"]
        responses[5]["grounded_annotations"].append(duplicate_annotation)
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
        responses[1]["proposals"] = [
            {
                "proposal_id": "bad_consumer",
                "task_type": "CM",
                "task_subtype": "CONTENT_MOTIVATION",
                "target": {"claim": "claim"},
                "proposed_gold": {"answer": "bad task pairing"},
                "evidence_ids": ["v1_visual_000_abc", "v1_visual_001_def"],
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
            any("consumer_proposal_parse_error" in item["reason"] for item in result.human_review_queue)
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


if __name__ == "__main__":
    unittest.main()
