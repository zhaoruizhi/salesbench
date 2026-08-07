from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankPipeline  # noqa: E402
from salesbench.goldbank.runner import run_gold_bank_records  # noqa: E402
from salesbench.io_utils import read_records  # noqa: E402
from salesbench.vlm.api_client import APICallResult  # noqa: E402
from salesbench.vqa.compiler import CompilePolicy, compile_vqa_from_gold  # noqa: E402
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
    e2 = "v1_visual_001"
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
                    "subject": "产品包装",
                    "predicate": "颜色",
                    "value": "黄色",
                    "attributes": {},
                    "source_domains": ["C1_visual"],
                    "extractor": "fake",
                    "confidence": 0.95,
                    "timestamp_status": "available",
                },
                {
                    "evidence_id": e2,
                    "modality": "visual",
                    "start_s": 1,
                    "end_s": 2,
                    "frame_indices": [1],
                    "text_span": "",
                    "subject": "产品",
                    "predicate": "action",
                    "value": "现场演示",
                    "attributes": {},
                    "source_domains": ["C1_visual"],
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
            "target": {"subject": "产品"},
            "proposed_gold": {"usage_context": "通勤场景"},
            "evidence_ids": [e1, e2],
            "reasoning_edges": [],
            "proposal_confidence": 0.9,
        },
        "p_cm": {
            "proposal_id": "p_cm",
            "task_type": "CM",
            "task_subtype": "CLAIM_EVIDENCE_RELATION",
            "target": {"claim": "产品便携"},
            "proposed_gold": {"relation": "SUPPORTED"},
            "evidence_ids": [e1, e2],
            "reasoning_edges": [],
            "proposal_confidence": 0.9,
        },
        "p_ss": {
            "proposal_id": "p_ss",
            "task_type": "SS",
            "task_subtype": "TRUST_MECHANISM",
            "target": {"subject": "产品"},
            "proposed_gold": {"mechanism": "现场演示"},
            "evidence_ids": [e1, e2],
            "reasoning_edges": [],
            "proposal_confidence": 0.9,
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
        "p_ae": ("g_ae", "AE", "USAGE_CONTEXT", {"usage_context": "通勤场景"}, "need_with_evidence"),
        "p_cm": ("g_cm", "CM", "CLAIM_EVIDENCE_RELATION", {"relation": "SUPPORTED"}, "relation_choice"),
        "p_ss": ("g_ss", "SS", "TRUST_MECHANISM", {"mechanism": "现场演示"}, "mechanism_with_evidence"),
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
        {"proposals": [proposals["p_ae"]], "abstentions": []},
        {"proposals": [proposals["p_cm"]], "abstentions": []},
        {"proposals": [proposals["p_ss"]], "abstentions": []},
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
            "prompt_version": "evidence-prompt-v5",
            "schema_version": "evidence-dataset-schema-v2",
            "min_confidence": 0.7,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            qa_dir = root / "qa"
            run_dir = root / "run"
            eval_dir = root / "evaluation"
            run_gold_bank_records(
                [{"video_id": "v1", "video_text": "适合通勤，画面现场演示。"}],
                cohort,
                evidence_dir,
                pipeline,
                resume=False,
            )

            compile_meta = compile_vqa_from_gold(
                evidence_dir,
                qa_dir,
                CompilePolicy(),
                bank_filename="video_evidence_dataset.jsonl",
            )
            public_items = read_records(qa_dir / "vqa_public.jsonl")
            answer_client = RepeatingClient("基于画面和口播作答。", "fake-model")
            run_meta = run_salesbench_qa_baseline_records(
                public_items,
                {"v1": {"video_id": "v1", "video_text": "适合通勤，画面现场演示。"}},
                run_dir,
                "fake-model",
                answer_client,
                max_workers=1,
            )
            judge_client = RepeatingClient(
                '{"score": 1.0, "reason": "正确", "evidence_alignment": "一致"}',
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
