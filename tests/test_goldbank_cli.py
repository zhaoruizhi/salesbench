from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "src")

from salesbench.cli import (  # noqa: E402
    build_audit_translations_command,
    build_evidence_dataset_command,
    build_parser,
    evaluate_vqa_benchmark_command,
    realize_qa_command,
    run_vqa_benchmark_command,
)


class GoldBankCLITest(unittest.TestCase):
    def test_build_evidence_dataset_command_defaults(self):
        args = build_parser().parse_args(["build-evidence-dataset"])

        self.assertEqual(args.output_dir, "outputs/evidence/v2")
        self.assertEqual(args.model, "gpt-4o")
        self.assertTrue(args.resume)

    def test_apply_gold_reviews_command_parses(self):
        args = build_parser().parse_args(
            [
                "apply-evidence-reviews",
                "--evidence-dir",
                "out",
                "--decisions",
                "decisions.jsonl",
                "--output",
                "reviewed.jsonl",
            ]
        )

        self.assertEqual(args.evidence_dir, "out")
        self.assertEqual(args.decisions, "decisions.jsonl")

    def test_compile_and_audit_commands_parse(self):
        compile_args = build_parser().parse_args(
            [
                "compile-vqa",
                "--evidence-dir",
                "gold",
                "--realizations",
                "qa_realizations_reviewed.jsonl",
                "--output-dir",
                "qa",
            ]
        )
        audit_args = build_parser().parse_args(
            ["audit-evidence-dataset", "--dataset", "gold/video_evidence_dataset.jsonl", "--evidence", "gold/evidence_units.jsonl"]
        )

        self.assertEqual(compile_args.dataset_file, "video_evidence_dataset_reviewed.jsonl")
        self.assertEqual(audit_args.output, None)

    def test_realize_qa_command_leaves_provider_model_to_environment(self):
        args = build_parser().parse_args(
            ["realize-qa", "--evidence-dir", "evidence", "--output-dir", "qa"]
        )

        self.assertIsNone(args.text_model)
        self.assertEqual(args.dataset_file, "video_evidence_dataset.jsonl")
        self.assertTrue(args.resume)

    def test_build_audit_translations_command_parses(self):
        args = build_parser().parse_args(
            [
                "build-audit-translations",
                "--manifest",
                "configs/pilot64_gpt4o_v9_delivery.json",
                "--output",
                "outputs/audit/translations.jsonl",
            ]
        )

        self.assertIsNone(args.model)
        self.assertEqual(args.batch_size, 20)

    @patch.dict(
        "os.environ",
        {
            "QWEN_API_KEY": "qwen-key",
            "QWEN_BASE_URL": "https://qwen.example/v1",
            "QWEN_VISION_MODEL": "qwen-vision",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-text",
        },
        clear=True,
    )
    @patch("salesbench.goldbank.runner.build_gold_bank_dataset")
    @patch("salesbench.cli.ensure_output_dirs")
    @patch("salesbench.cli.load_config")
    def test_evidence_command_routes_qwen_vision_and_deepseek_text(
        self,
        load_config,
        _ensure_output_dirs,
        build_dataset,
    ):
        load_config.return_value = SimpleNamespace(repo_root=Path("/repo"))
        build_dataset.return_value = {"ok": True}
        args = build_parser().parse_args(["build-evidence-dataset"])

        self.assertEqual(build_evidence_dataset_command(args), 0)
        kwargs = build_dataset.call_args.kwargs
        self.assertEqual(kwargs["vision_api_key"], "qwen-key")
        self.assertEqual(kwargs["vision_base_url"], "https://qwen.example/v1")
        self.assertEqual(kwargs["vision_model"], "qwen-vision")
        self.assertEqual(kwargs["text_api_key"], "deepseek-key")
        self.assertEqual(kwargs["text_base_url"], "https://deepseek.example/v1")
        self.assertEqual(kwargs["text_model"], "deepseek-text")

    @patch.dict(
        "os.environ",
        {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-text",
        },
        clear=True,
    )
    @patch("salesbench.vqa.realizer.run_qa_realizer")
    @patch("salesbench.vlm.api_client.VLMClient")
    def test_realize_command_uses_deepseek_environment(self, client, run_realizer):
        run_realizer.return_value = {"ok": True}
        args = build_parser().parse_args(
            ["realize-qa", "--evidence-dir", "evidence", "--output-dir", "qa"]
        )

        self.assertEqual(realize_qa_command(args), 0)
        self.assertEqual(client.call_args.kwargs["api_key"], "deepseek-key")
        self.assertEqual(client.call_args.kwargs["base_url"], "https://deepseek.example/v1")
        self.assertEqual(client.call_args.kwargs["model"], "deepseek-text")

    @patch.dict(
        "os.environ",
        {
            "QWEN_API_KEY": "qwen-key",
            "QWEN_BASE_URL": "https://qwen.example/v1",
            "QWEN_VISION_MODEL": "qwen-vision",
        },
        clear=True,
    )
    @patch("salesbench.vqa_baseline.runner.run_salesbench_qa_baseline")
    @patch("salesbench.cli.load_config")
    def test_benchmark_runner_uses_qwen_environment(self, load_config, run_baseline):
        load_config.return_value = SimpleNamespace(repo_root=Path("/repo"))
        run_baseline.return_value = {"ok": True}
        args = build_parser().parse_args(
            ["run-vqa-benchmark", "--vqa", "qa.jsonl"]
        )

        self.assertEqual(run_vqa_benchmark_command(args), 0)
        self.assertEqual(run_baseline.call_args.kwargs["api_key"], "qwen-key")
        self.assertEqual(run_baseline.call_args.kwargs["base_url"], "https://qwen.example/v1")
        self.assertEqual(run_baseline.call_args.kwargs["model"], "qwen-vision")

    @patch.dict(
        "os.environ",
        {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-text",
        },
        clear=True,
    )
    @patch("salesbench.vqa_evaluate.runner.evaluate_salesbench_qa_files")
    @patch("salesbench.cli.load_config")
    def test_judge_uses_deepseek_environment(self, load_config, evaluate):
        load_config.return_value = SimpleNamespace(repo_root=Path("/repo"))
        evaluate.return_value = {"ok": True}
        args = build_parser().parse_args(
            [
                "evaluate-vqa-benchmark",
                "--gold",
                "gold.jsonl",
                "--predictions",
                "predictions.jsonl",
            ]
        )

        self.assertEqual(evaluate_vqa_benchmark_command(args), 0)
        self.assertEqual(evaluate.call_args.kwargs["api_key"], "deepseek-key")
        self.assertEqual(evaluate.call_args.kwargs["base_url"], "https://deepseek.example/v1")
        self.assertEqual(evaluate.call_args.kwargs["judge_model"], "deepseek-text")

    @patch.dict(
        "os.environ",
        {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-text",
        },
        clear=True,
    )
    @patch("salesbench.audit_translation.run_audit_translations")
    @patch("salesbench.audit_translation.collect_audit_translation_jobs")
    @patch("salesbench.vlm.api_client.VLMClient")
    def test_audit_translation_uses_deepseek_environment(
        self,
        client,
        collect_jobs,
        run_translations,
    ):
        collect_jobs.return_value = []
        run_translations.return_value = {"ok": True}
        args = build_parser().parse_args(
            [
                "build-audit-translations",
                "--manifest",
                "configs/pilot64_gpt4o_v9_delivery.json",
                "--output",
                "translations.jsonl",
            ]
        )

        self.assertEqual(build_audit_translations_command(args), 0)
        self.assertEqual(client.call_args.kwargs["api_key"], "deepseek-key")
        self.assertEqual(client.call_args.kwargs["base_url"], "https://deepseek.example/v1")
        self.assertEqual(client.call_args.kwargs["model"], "deepseek-text")

    @patch.dict(
        "os.environ",
        {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-text",
            "QWEN_API_KEY": "qwen-key",
            "QWEN_BASE_URL": "https://qwen.example/v1",
            "QWEN_VISION_MODEL": "qwen3-vl-plus",
        },
        clear=True,
    )
    @patch("salesbench.audit_translation.run_audit_translations")
    @patch("salesbench.audit_translation.collect_audit_translation_jobs")
    @patch("salesbench.vlm.api_client.VLMClient")
    def test_audit_translation_routes_explicit_qwen_model_to_qwen_provider(
        self,
        client,
        collect_jobs,
        run_translations,
    ):
        collect_jobs.return_value = []
        run_translations.return_value = {"ok": True}
        args = build_parser().parse_args(
            [
                "build-audit-translations",
                "--manifest",
                "configs/pilot64_qwen_deepseek_v10_delivery.json",
                "--output",
                "translations.jsonl",
                "--model",
                "qwen3-vl-plus",
            ]
        )

        self.assertEqual(build_audit_translations_command(args), 0)
        self.assertEqual(client.call_args.kwargs["api_key"], "qwen-key")
        self.assertEqual(client.call_args.kwargs["base_url"], "https://qwen.example/v1")
        self.assertEqual(client.call_args.kwargs["model"], "qwen3-vl-plus")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("salesbench.goldbank.runner.build_gold_bank_dataset")
    @patch("salesbench.cli.ensure_output_dirs")
    @patch("salesbench.cli.load_config")
    def test_evidence_command_only_passes_supported_runner_arguments(
        self,
        load_config,
        _ensure_output_dirs,
        build_dataset,
    ):
        load_config.return_value = SimpleNamespace(repo_root=Path("/repo"))
        build_dataset.return_value = {"ok": True}
        args = build_parser().parse_args(["build-evidence-dataset"])

        self.assertEqual(build_evidence_dataset_command(args), 0)
        self.assertNotIn("allow_auto_candidates", build_dataset.call_args.kwargs)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("salesbench.vqa.realizer.run_qa_realizer")
    @patch("salesbench.vlm.api_client.VLMClient")
    def test_realize_command_forwards_explicit_candidate_mode(
        self,
        _client,
        run_realizer,
    ):
        run_realizer.return_value = {"ok": True}
        args = build_parser().parse_args(
            [
                "realize-qa",
                "--evidence-dir",
                "evidence",
                "--output-dir",
                "qa",
                "--allow-auto-candidates",
                "--strict-semantic-verification",
            ]
        )

        self.assertEqual(realize_qa_command(args), 0)
        self.assertTrue(run_realizer.call_args.kwargs["allow_auto_candidates"])
        self.assertTrue(run_realizer.call_args.kwargs["strict_semantic_verification"])


if __name__ == "__main__":
    unittest.main()
