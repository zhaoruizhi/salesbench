from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.cli import build_parser  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
