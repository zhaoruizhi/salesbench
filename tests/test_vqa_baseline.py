"""Tests for SalesBench-QA closed-source VLM baseline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.vlm.api_client import APICallResult  # noqa: E402


class FakeVLMClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.model = "fake-vlm"

    def call(self, system_prompt: str, user_content: list[dict], response_format: str | None = None) -> APICallResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response_format": response_format,
            }
        )
        if not self.responses:
            raise AssertionError("FakeVLMClient response queue exhausted")
        return APICallResult(
            raw_response=self.responses.pop(0),
            model=self.model,
            input_tokens=23,
            output_tokens=11,
            latency_s=0.01,
            cost_usd=0.002,
            success=True,
        )


def _gold_item(task_type: str = "BP") -> dict[str, object]:
    return {
        "vqa_id": f"v1_{task_type.lower()}_001",
        "video_id": "v1",
        "task_layer": "salesbench_qa",
        "task_type": task_type,
        "question": "视频中展示的产品包装颜色是什么？" if task_type == "BP" else "视频通过什么机制引出用户顾虑？",
        "answer_type": "open",
        "gold_answer": "黄色。" if task_type == "BP" else "通过提问引出用户顾虑。",
        "answer": "黄色。" if task_type == "BP" else "通过提问引出用户顾虑。",
        "performance_metadata": {"likes": 100, "collects": 80, "shares": 20, "comments": 5},
        "calibration_metadata": {"status": "ACCEPT"},
    }


def _video_lookup() -> dict[str, dict[str, object]]:
    return {
        "v1": {
            "video_id": "v1",
            "title": "黄芪霜促销",
            "video_text": "三瓶四块九，喜欢可以收藏评论。",
            "product_title": "黄芪霜",
            "has_video_asset": 0,
            "primary_video_path": "",
        }
    }


class BaselinePromptTest(unittest.TestCase):
    def test_prompt_uses_closed_source_template_and_hides_gold_fields(self) -> None:
        from salesbench.vqa_baseline.prompts import (
            BASELINE_PROMPT_VERSION,
            CLOSED_SOURCE_SYSTEM_PROMPT,
            build_closed_source_user_prompt,
        )

        prompt = build_closed_source_user_prompt(_gold_item(), _video_lookup()["v1"])

        for token in ("Video:", "Voicer:", "Question:"):
            self.assertIn(token, prompt)
        self.assertNotIn("<think>", prompt)
        self.assertNotIn("<answer>", prompt)
        self.assertEqual(BASELINE_PROMPT_VERSION, "closed-source-prompt-v2")
        self.assertIn("final answer in English", CLOSED_SOURCE_SYSTEM_PROMPT)
        self.assertIn("final answer in English", prompt)
        self.assertNotIn("answer in Chinese", CLOSED_SOURCE_SYSTEM_PROMPT + prompt)
        for forbidden in (
            "Title",
            "Product",
            "Public VQA item",
            "video_id",
            "task_type",
            "answer_type",
            "gold_answer",
            "performance_metadata",
            "calibration_metadata",
        ):
            self.assertNotIn(forbidden, prompt)


class BaselineParserTest(unittest.TestCase):
    def test_parse_closed_source_response_extracts_tags(self) -> None:
        from salesbench.vqa_baseline.parser import parse_closed_source_response

        parsed = parse_closed_source_response("<think>看包装颜色。</think><answer>包装是黄色。</answer>")

        self.assertEqual(parsed["thought"], "")
        self.assertEqual(parsed["answer"], "包装是黄色。")
        self.assertEqual(parsed["parse_warning"], "")

    def test_parse_closed_source_response_falls_back_without_tags(self) -> None:
        from salesbench.vqa_baseline.parser import parse_closed_source_response

        parsed = parse_closed_source_response("包装是黄色。")

        self.assertEqual(parsed["answer"], "包装是黄色。")
        self.assertEqual(parsed["thought"], "")
        self.assertEqual(parsed["parse_warning"], "")


class BaselineRunnerTest(unittest.TestCase):
    def test_mock_runner_writes_answers_and_hides_private_fields(self) -> None:
        from salesbench.vqa_baseline.runner import run_salesbench_qa_baseline_records

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            fake = FakeVLMClient(
                [
                    "包装是黄色。",
                    "通过提问引出用户顾虑。",
                ]
            )
            summary = run_salesbench_qa_baseline_records(
                items=[_gold_item("BP"), _gold_item("SS")],
                video_lookup=_video_lookup(),
                output_dir=output_dir,
                model="gpt-4o",
                client=fake,
                max_workers=1,
            )

            answer_path = output_dir / "predictions.jsonl"
            meta_path = output_dir / "model_answers" / "gpt-4o_closed_source_run_meta.json"
            answers = [json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["answer_count"], 2)
            self.assertEqual(answers[0]["answer"], "包装是黄色。")
            self.assertEqual(answers[0]["thought"], "")
            self.assertTrue(meta_path.exists())
            joined_prompts = json.dumps(fake.calls, ensure_ascii=False)
            for forbidden in (
                "Title",
                "Product",
                "Public VQA item",
                "video_id",
                "task_type",
                "answer_type",
                "gold_answer",
                "performance_metadata",
                "calibration_metadata",
            ):
                self.assertNotIn(forbidden, joined_prompts)


class BaselineAnalysisTest(unittest.TestCase):
    def test_analysis_markdown_contains_metrics_and_low_score_examples(self) -> None:
        from salesbench.vqa_baseline.analysis import build_analysis_markdown

        report = {
            "metrics": {
                "overall": {"count": 2, "strict_accuracy": 0.5, "relaxed_accuracy": 0.75},
                "per_task": {
                    "BP": {"count": 1, "strict_accuracy": 1.0, "relaxed_accuracy": 1.0},
                    "SS": {"count": 1, "strict_accuracy": 0.0, "relaxed_accuracy": 0.5},
                },
            },
            "summary": {"judge_failed_count": 0, "total_cost_usd": 0.03},
        }
        details = [
            {"vqa_id": "v1_ss_001", "task_type": "SS", "score": 0.5, "question": "如何建立信任？", "model_output": "通过演示", "reason": "证据不完整。"}
        ]

        markdown = build_analysis_markdown(report, details, model="gpt-4o")

        self.assertIn("Overall", markdown)
        self.assertIn("Per-Task", markdown)
        self.assertIn("Low-Score", markdown)
        self.assertIn("v1_ss_001", markdown)


if __name__ == "__main__":
    unittest.main()
