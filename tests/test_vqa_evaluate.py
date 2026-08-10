"""Tests for SalesBench-QA LLM-as-Judge evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.vlm.api_client import APICallResult  # noqa: E402


class FakeJudgeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.model = "fake-judge"

    def call_text_only(self, system_prompt: str, user_text: str, response_format: str | None = None) -> APICallResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_text": user_text,
                "response_format": response_format,
            }
        )
        if not self.responses:
            raise AssertionError("FakeJudgeClient response queue exhausted")
        return APICallResult(
            raw_response=self.responses.pop(0),
            model=self.model,
            input_tokens=17,
            output_tokens=5,
            latency_s=0.01,
            cost_usd=0.001,
            success=True,
        )


def _bp_gold() -> dict[str, object]:
    return {
        "vqa_id": "v1_bp_001",
        "video_id": "v1",
        "task_layer": "salesbench_qa",
        "task_type": "BP",
        "question": "What color is the product package shown in the video?",
        "gold_answer": "The package is yellow.",
        "answer_type": "open",
        "evidence_context": [{"evidence_id": "e1", "modality": "visual", "text_span": "画面展示黄色包装。"}],
    }


def _ss_gold() -> dict[str, object]:
    return {
        "vqa_id": "v1_ss_001",
        "video_id": "v1",
        "task_layer": "salesbench_qa",
        "task_type": "SS",
        "question": "How does the video build trust?",
        "gold_answer": "It builds trust through an in-use demonstration.",
        "answer_type": "open",
        "evidence_context": [{"evidence_id": "e2", "modality": "visual", "text_span": "现场演示产品使用。"}],
        "performance_metadata": {"likes": 100, "collects": 80, "shares": 20, "comments": 5},
    }


class JudgeParsingTest(unittest.TestCase):
    def test_parse_judge_response_accepts_json_and_answer_line(self) -> None:
        from salesbench.vqa_evaluate.judge import parse_judge_response

        parsed_json = parse_judge_response(
            '{"score": 0.75, "reason": "The core answer is correct but vague.", "evidence_alignment": "The evidence aligns."}'
        )
        parsed_line = parse_judge_response("Answer: 0.25\nReason: evidence mismatch")

        self.assertEqual(parsed_json["score"], 0.75)
        self.assertEqual(parsed_json["reason"], "The core answer is correct but vague.")
        self.assertEqual(parsed_line["score"], 0.25)

        parsed_v7 = parse_judge_response(
            '{"score":0.75,"correctness":1.0,"grounding":0.75,"completeness":0.75,'
            '"reason":"The conclusion is correct but incomplete.","evidence_alignment":"The evidence supports the core conclusion."}'
        )
        self.assertEqual(parsed_v7["correctness"], 1.0)
        self.assertEqual(parsed_v7["grounding"], 0.75)
        self.assertEqual(parsed_v7["completeness"], 0.75)

        locally_scored = parse_judge_response(
            '{"score":1.0,"correctness":0.25,"grounding":0.25,"completeness":1.0,'
            '"reason":"A material error remains.","evidence_alignment":"The evidence provides only partial support."}'
        )
        self.assertEqual(locally_scored["reported_score"], 1.0)
        self.assertEqual(locally_scored["score"], 0.5)

    def test_parse_judge_response_rejects_invalid_scores(self) -> None:
        from salesbench.vqa_evaluate.judge import parse_judge_response

        with self.assertRaises(ValueError):
            parse_judge_response('{"score": 0.8, "reason": "not allowed"}')

    def test_parse_judge_response_rejects_non_english_explanations(self) -> None:
        from salesbench.vqa_evaluate.judge import parse_judge_response

        with self.assertRaisesRegex(ValueError, "English"):
            parse_judge_response(
                '{"score":1.0,"correctness":1.0,"grounding":1.0,"completeness":1.0,'
                '"reason":"完全正确","evidence_alignment":"证据一致"}'
            )


class JudgePromptAndContextTest(unittest.TestCase):
    def test_prompt_covers_salesbench_scores_and_four_tasks(self) -> None:
        from salesbench.vqa_evaluate.prompts import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

        self.assertEqual(JUDGE_PROMPT_VERSION, "judge-prompt-v3")
        for text in ("1.0", "0.75", "0.5", "0.25", "0"):
            self.assertIn(text, JUDGE_SYSTEM_PROMPT)
        for task_type in ("BP", "CM", "SS", "AE"):
            self.assertIn(task_type, JUDGE_SYSTEM_PROMPT)
        self.assertIn("natural-language fields must use English", JUDGE_SYSTEM_PROMPT)
        self.assertIn("directly observable facts", JUDGE_SYSTEM_PROMPT)
        self.assertIn("cross-modal relationship", JUDGE_SYSTEM_PROMPT)
        self.assertIn("persuasion mechanism", JUDGE_SYSTEM_PROMPT)
        self.assertIn("bounded interpretation", JUDGE_SYSTEM_PROMPT)
        self.assertNotRegex(JUDGE_SYSTEM_PROMPT, r"[\u4e00-\u9fff]")
        for dimension in ("correctness", "grounding", "completeness"):
            self.assertIn(dimension, JUDGE_SYSTEM_PROMPT)
        self.assertNotIn("Performance Metadata", JUDGE_SYSTEM_PROMPT)
        self.assertNotIn("Performance Metadata", JUDGE_SYSTEM_PROMPT)

        user_prompt = build_judge_user_prompt(
            {
                "question": "问题",
                "task_type": "SS",
                "reference_answer": "参考答案",
                "model_output": "模型答案",
                "evidence_context": {"evidence": "证据"},
            }
        )
        for label in ("Question", "Task Type", "Reference Answer", "Model Answer", "Evidence Context"):
            self.assertIn(label, user_prompt)
        self.assertNotIn("Performance Metadata", user_prompt)

    def test_context_includes_evidence_but_strips_performance_metadata(self) -> None:
        from salesbench.vqa_evaluate.context import build_judge_payload

        ss_payload = build_judge_payload(_ss_gold(), {"answer": "It builds trust through a demonstration."})
        bp_payload = build_judge_payload(_bp_gold(), {"answer": "Yellow."})

        ss_text = json.dumps(ss_payload, ensure_ascii=False)
        self.assertIn("现场演示产品使用", ss_text)
        for key in ("likes", "collects", "shares", "comments"):
            self.assertNotIn(key, ss_text)
        self.assertIn("画面展示黄色包装", json.dumps(bp_payload, ensure_ascii=False))


class JudgeMetricsTest(unittest.TestCase):
    def test_aggregate_metrics_reports_macro_and_micro_scores(self) -> None:
        from salesbench.vqa_evaluate.metrics import aggregate_judge_metrics

        details = [
            {"vqa_id": "bp1", "task_type": "BP", "score": 1.0, "judge_success": True},
            {"vqa_id": "cm1", "task_type": "CM", "score": 0.75, "judge_success": True},
            {"vqa_id": "ae1", "task_type": "AE", "score": 0.25, "judge_success": True},
            {"vqa_id": "ss1", "task_type": "SS", "score": None, "judge_success": False},
        ]

        report = aggregate_judge_metrics(details, gold_count=4, answer_count=4, matched_answer_count=4, skipped_gold_count=0)

        self.assertEqual(report["summary"]["judge_failed_count"], 1)
        self.assertEqual(report["metrics"]["overall"]["strict_accuracy"], 1 / 3)
        self.assertAlmostEqual(report["metrics"]["overall"]["relaxed_accuracy"], (1.0 + 0.75 + 0.25) / 3)
        self.assertAlmostEqual(report["metrics"]["macro_average"]["relaxed_accuracy"], (1.0 + 0.75 + 0.25) / 3)
        self.assertEqual(report["metrics"]["per_task"]["CM"]["count"], 1)
        self.assertEqual(report["metrics"]["per_task"]["AE"]["strict_accuracy"], 0.0)


class JudgeRunnerTest(unittest.TestCase):
    def test_mock_runner_writes_details_and_summary(self) -> None:
        from salesbench.vqa_evaluate.runner import evaluate_salesbench_qa_files

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gold_path = tmp_path / "gold.jsonl"
            answers_path = tmp_path / "answers.jsonl"
            output_dir = tmp_path / "evaluation"
            gold_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in [_bp_gold(), _ss_gold()]) + "\n",
                encoding="utf-8",
            )
            answers_path.write_text(
                "\n".join(
                    json.dumps(item, ensure_ascii=False)
                    for item in [
                        {"vqa_id": "v1_bp_001", "answer": "The package is yellow."},
                        {"vqa_id": "v1_ss_001", "answer": "It builds trust through an in-use demonstration."},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            fake = FakeJudgeClient(
                [
                    '{"score": 1.0, "reason": "Fully correct.", "evidence_alignment": "Aligned."}',
                    '{"score": 0.75, "reason": "The core answer is correct.", "evidence_alignment": "Aligned."}',
                ]
            )

            report = evaluate_salesbench_qa_files(
                gold_path=gold_path,
                answers_path=answers_path,
                output_dir=output_dir,
                client=fake,
            )

            self.assertEqual(report["summary"]["matched_answer_count"], 2)
            self.assertEqual(report["summary"]["judge_failed_count"], 0)
            self.assertAlmostEqual(report["metrics"]["overall"]["relaxed_accuracy"], 0.875)
            self.assertTrue((output_dir / "answers_judge_details.jsonl").exists())
            self.assertTrue((output_dir / "answers_salesbench_qa_eval.json").exists())
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(fake.calls[0]["response_format"], "json_object")
            self.assertEqual(report["judge_prompt_version"], "judge-prompt-v3")


if __name__ == "__main__":
    unittest.main()
