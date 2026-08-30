from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankPipeline  # noqa: E402
from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CueType,
    RelationType,
    make_cue_id,
    make_relation_id,
)
from salesbench.goldbank.runner import run_gold_bank_records  # noqa: E402
from salesbench.io_utils import read_records  # noqa: E402
from salesbench.vlm.api_client import APICallResult  # noqa: E402
from salesbench.vqa.compiler import CompilePolicy, compile_vqa_from_gold  # noqa: E402
from salesbench.vqa.specs import build_question_specs  # noqa: E402
from salesbench.vqa_baseline.runner import run_salesbench_qa_baseline_records  # noqa: E402
from salesbench.vqa_evaluate.runner import evaluate_salesbench_qa_files  # noqa: E402


class SequenceClient:
    def __init__(self, responses: list[dict[str, object] | str], model: str):
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict[str, object]] = []

    def _next(self) -> APICallResult:
        if not self.responses:
            raise AssertionError("response queue exhausted")
        response = self.responses.pop(0)
        raw = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        return APICallResult(raw, self.model, 1, 1, 0.01, 0.0, True)

    def call(self, system_prompt, user_content, response_format=None):
        self.calls.append({"system": system_prompt, "user": user_content})
        return self._next()

    def call_text_only(self, system_prompt, user_text, response_format=None):
        self.calls.append({"system": system_prompt, "user": user_text})
        return self._next()


class RepeatingClient:
    def __init__(self, response: str, model: str):
        self.response = response
        self.model = model
        self.calls: list[dict[str, object]] = []

    def call(self, system_prompt, user_content, response_format=None):
        self.calls.append({"system": system_prompt, "user": user_content})
        return APICallResult(self.response, self.model, 1, 1, 0.01, 0.0, True)

    def call_text_only(self, system_prompt, user_text, response_format=None):
        self.calls.append({"system": system_prompt, "user": user_text})
        return APICallResult(self.response, self.model, 1, 1, 0.01, 0.0, True)


def evidence_responses() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    e1 = "v1_visual_000"
    e2 = "v1_asr_001"
    product_content = "A yellow product package is shown."
    usage_content = "The speaker presents commuting as a usage scenario."
    process_content = "The product is demonstrated in use."
    claim_content = "The speaker claims that the product is suitable for commuting."
    product_cue_id = make_cue_id("v1", CueType.PRODUCT_IDENTITY, (e1,), product_content)
    usage_cue_id = make_cue_id("v1", CueType.USAGE_SCENARIO, (e2,), usage_content)
    process_cue_id = make_cue_id("v1", CueType.PROCESS_DEMONSTRATION, (e1,), process_content)
    claim_cue_id = make_cue_id("v1", CueType.FIT_CLAIM, (e2,), claim_content)
    claim_relation_id = make_relation_id(
        "v1",
        RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION,
        (claim_cue_id,),
        (process_cue_id,),
    )
    vision = [
        {
            "evidence_units": [
                {
                    "evidence_id": e1,
                    "modality": "visual",
                    "start_s": 0,
                    "end_s": 1,
                    "frame_indices": [0],
                    "text_span": "",
                    "subject": "product package",
                    "predicate": "has color",
                    "value": "yellow",
                    "attributes": {},
                    "source_domains": ["C1_visual"],
                    "extractor": "fake",
                    "confidence": 0.95,
                    "timestamp_status": "available",
                },
                {
                    "evidence_id": e2,
                    "modality": "asr",
                    "start_s": 1,
                    "end_s": 2,
                    "frame_indices": [],
                    "text_span": "这款产品适合通勤",
                    "subject": "speaker",
                    "predicate": "claims",
                    "value": "the product is suitable for commuting",
                    "attributes": {},
                    "source_domains": ["C2_audio_speech"],
                    "extractor": "fake",
                    "confidence": 0.95,
                    "timestamp_status": "available",
                },
            ]
        }
    ]
    proposals = {
        "p_ae": {
            "proposal_id": "p_ae",
            "task_type": "AE",
            "task_subtype": "USAGE_CONTEXT",
            "target": {"scenario": "commuting"},
            "proposed_gold": {"usage_context": "commuting"},
            "evidence_ids": [e1, e2],
            "reasoning_edges": [],
            "proposal_confidence": 0.9,
            "capability": "USAGE_CONTEXT",
            "reasoning_operator": "LOCALIZE_USAGE_CONTEXT",
            "commerce_cue_ids": [usage_cue_id],
            "commercial_relation_ids": [],
            "question_intent": "Ask which represented usage context is connected to the product.",
            "forbidden_inferences": ["Do not infer a real viewer profile."],
        },
        "p_cm": {
            "proposal_id": "p_cm",
            "task_type": "CM",
            "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
            "target": {"claim": "the product is portable"},
            "proposed_gold": {"relation": "SUPPORTED"},
            "evidence_ids": [e1, e2],
            "reasoning_edges": [],
            "proposal_confidence": 0.9,
            "capability": "CLAIM_DEMONSTRATION_STATUS",
            "reasoning_operator": "CLASSIFY_CLAIM_SUPPORT",
            "commerce_cue_ids": [claim_cue_id, process_cue_id],
            "commercial_relation_ids": [claim_relation_id],
            "question_intent": "Ask how the visual demonstration relates to the spoken claim.",
            "forbidden_inferences": ["Do not treat repeated wording as visual proof."],
        },
        "p_ss": {
            "proposal_id": "p_ss",
            "task_type": "SS",
            "task_subtype": "PROCESS_DEMONSTRATION",
            "target": {"segment": "demonstration", "mechanism": "product demonstration"},
            "proposed_gold": {"label": "demonstration", "answer": "The video demonstrates the product in use."},
            "evidence_ids": [e1, e2],
            "reasoning_edges": [],
            "proposal_confidence": 0.9,
            "capability": "PROCESS_DEMONSTRATION",
            "reasoning_operator": "INTERPRET_PROCESS_ROLE",
            "commerce_cue_ids": [process_cue_id, product_cue_id],
            "commercial_relation_ids": [claim_relation_id],
            "question_intent": "Ask what the product demonstration shows in the sales presentation.",
            "forbidden_inferences": ["Do not claim that the demonstration caused purchase."],
        },
    }
    proposal_ids = ["v1_local_bp_000", "v1_local_bp_001", "p_ae", "p_cm", "p_ss"]
    reviews = [
        {
            "review_id": f"r_{proposal_id}",
            "proposal_id": proposal_id,
            "video_id": "v1",
            "reviewer": "fake_challenger",
            "verdict": "PASS",
            "checks": {"evidence_exists": True},
            "issues": [],
            "suggested_revision": None,
        }
        for proposal_id in proposal_ids
    ]
    annotations = []
    annotation_values = {
        "p_ae": ("g_ae", "AE", "USAGE_CONTEXT", {"usage_context": "commuting", "answer": "The content presents commuting as a usage context."}, "grounded_question"),
        "p_cm": ("g_cm", "CM", "CLAIM_DEMONSTRATION_STATUS", {"relation": "SUPPORTED", "answer": "The visible product supports the spoken claim."}, "grounded_question"),
        "p_ss": ("g_ss", "SS", "PROCESS_DEMONSTRATION", {"label": "demonstration", "answer": "The video demonstrates the product in use."}, "grounded_question"),
    }
    for proposal_id, (annotation_id, task, subtype, value, question_format) in annotation_values.items():
        annotations.append(
            {
                "annotation_id": annotation_id,
                "video_id": "v1",
                "task_type": task,
                "task_subtype": subtype,
                "target": proposals[proposal_id]["target"],
                "gold_value": value,
                "evidence_refs": [e1, e2],
                "reasoning_edges": [],
                "eligible_question_formats": [question_format],
                "source_proposal_ids": [proposal_id],
                "review_status": "verified",
                "confidence": 0.9,
            }
        )
    text = [
        {
            "commerce_cues": [
                {
                    "cue_type": "PRODUCT_IDENTITY",
                    "content_en": product_content,
                    "source_text_native": "",
                    "evidence_ids": [e1],
                    "attributes": {"color": "yellow"},
                    "directness": "DIRECT",
                    "theory_tags": ["product_description"],
                    "confidence": 0.95,
                },
                {
                    "cue_type": "USAGE_SCENARIO",
                    "content_en": usage_content,
                    "source_text_native": "这款产品适合通勤",
                    "evidence_ids": [e2],
                    "attributes": {"scenario": "commuting"},
                    "directness": "DIRECT",
                    "theory_tags": ["usage_scenario"],
                    "confidence": 0.95,
                },
                {
                    "cue_type": "PROCESS_DEMONSTRATION",
                    "content_en": process_content,
                    "source_text_native": "",
                    "evidence_ids": [e1],
                    "attributes": {},
                    "directness": "DIRECT",
                    "theory_tags": ["product_demonstration"],
                    "confidence": 0.95,
                },
                {
                    "cue_type": "FIT_CLAIM",
                    "content_en": claim_content,
                    "source_text_native": "这款产品适合通勤",
                    "evidence_ids": [e2],
                    "attributes": {},
                    "directness": "DIRECT",
                    "theory_tags": ["product_claim"],
                    "confidence": 0.95,
                },
            ],
            "abstentions": [],
        },
        {
            "commercial_relations": [
                {
                    "relation_type": "CLAIM_SUPPORTED_BY_DEMONSTRATION",
                    "source_cue_ids": [claim_cue_id],
                    "target_cue_ids": [process_cue_id],
                    "evidence_ids": [e2, e1],
                    "status": "SUPPORTED",
                    "rationale_en": "The spoken commuting claim is paired with a distinct visual product demonstration.",
                    "directness": "DIRECT",
                    "confidence": 0.9,
                }
            ],
            "abstentions": [],
        },
        {"proposals": [proposals["p_cm"]], "abstentions": []},
        {"proposals": [proposals["p_ss"]], "abstentions": []},
        {"proposals": [proposals["p_ae"]], "abstentions": []},
        {"reviews": reviews},
        {"grounded_annotations": annotations, "human_review_queue": []},
    ]
    return vision, text


class EvidenceVQAE2ETest(unittest.TestCase):
    def test_evidence_dataset_to_public_predictions_and_judge(self):
        vision_responses, text_responses = evidence_responses()
        pipeline = GoldBankPipeline(
            SequenceClient(vision_responses, "fake-vision"),
            SequenceClient(text_responses, "fake-text"),
        )
        cohort = {
            "version": "evidence-e2e",
            "video_ids": ["v1"],
            "frame_strategy": "hook_plus_uniform",
            "frames_per_video": 16,
            "prompt_version": "evidence-prompt-v9",
            "schema_version": "evidence-dataset-schema-v3",
            "min_confidence": 0.7,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            qa_dir = root / "qa"
            run_dir = root / "run"
            eval_dir = root / "evaluation"
            run_gold_bank_records(
                [{"video_id": "v1", "video_text": ""}],
                cohort,
                evidence_dir,
                pipeline,
                resume=False,
            )
            evidence_records = read_records(evidence_dir / "video_evidence_dataset.jsonl")
            questions = {
                "BP": "What product detail or use is directly presented in this video?",
                "CM": "How does the visible product demonstration relate to the spoken commuting claim?",
                "SS": "What does the product demonstration show within the sales presentation?",
                "AE": "What usage context does the content connect to the product?",
            }
            realizations_path = root / "qa_realizations_reviewed.jsonl"
            with realizations_path.open("w", encoding="utf-8") as handle:
                for spec in build_question_specs(evidence_records):
                    handle.write(
                        json.dumps(
                            {"spec_id": spec.spec_id, "question": questions[spec.task_type.value]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            compile_meta = compile_vqa_from_gold(
                evidence_dir,
                qa_dir,
                CompilePolicy(allow_auto_candidates=True),
                bank_filename="video_evidence_dataset.jsonl",
                realizations_path=realizations_path,
            )
            public_items = read_records(qa_dir / "vqa_public.jsonl")
            answer_client = RepeatingClient("The answer is based on the frames and speech.", "fake-model")
            run_meta = run_salesbench_qa_baseline_records(
                public_items,
                {"v1": {"video_id": "v1", "video_text": "适合通勤，画面现场演示。"}},
                run_dir,
                "fake-model",
                answer_client,
                max_workers=1,
            )
            judge_client = RepeatingClient(
                '{"score": 1.0, "reason": "The answer is correct.", "evidence_alignment": "The evidence aligns."}',
                "fake-judge",
            )
            report = evaluate_salesbench_qa_files(
                qa_dir / "vqa_gold_private.jsonl",
                Path(run_meta["outputs"]["answers"]),
                eval_dir,
                client=judge_client,
                max_workers=1,
            )

            self.assertTrue((evidence_dir / "video_evidence_dataset.jsonl").exists())
            self.assertEqual(set(compile_meta["public_tasks"]), {"BP", "CM", "SS", "AE"})
            self.assertTrue(all(compile_meta["counts"]["per_task"][task] > 0 for task in compile_meta["public_tasks"]))
            self.assertTrue((run_dir / "predictions.jsonl").exists())
            self.assertEqual(report["summary"]["matched_answer_count"], len(public_items))
            self.assertEqual(report["metrics"]["macro_average"]["relaxed_accuracy"], 1.0)
            runner_prompts = json.dumps(answer_client.calls, ensure_ascii=False)
            self.assertNotIn("gold_answer", runner_prompts)
            self.assertNotIn("likes", runner_prompts)


if __name__ == "__main__":
    unittest.main()
