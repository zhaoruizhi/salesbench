from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "src")

from salesbench.cli import (  # noqa: E402
    build_evidence_dataset_command,
    build_parser,
    realize_qa_command,
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

    def test_realize_qa_command_parses_gpt4o_defaults(self):
        args = build_parser().parse_args(
            ["realize-qa", "--evidence-dir", "evidence", "--output-dir", "qa"]
        )

        self.assertEqual(args.text_model, "gpt-4o")
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

        self.assertEqual(args.model, "gpt-4o")
        self.assertEqual(args.batch_size, 20)

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
            ]
        )

        self.assertEqual(realize_qa_command(args), 0)
        self.assertTrue(run_realizer.call_args.kwargs["allow_auto_candidates"])


if __name__ == "__main__":
    unittest.main()
