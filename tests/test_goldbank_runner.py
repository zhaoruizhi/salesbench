from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankResult  # noqa: E402
from salesbench.goldbank.runner import GOLD_BANK_OUTPUT_FILES, run_gold_bank_records  # noqa: E402


class FakePipeline:
    def __init__(self):
        self.calls: list[str] = []

    def run_video(self, bundle, frames_b64=None):
        self.calls.append(bundle.video_id)
        return GoldBankResult(
            video_id=bundle.video_id,
            evidence_units=[
                {
                    "evidence_id": f"{bundle.video_id}_visual_000_abc",
                    "video_id": bundle.video_id,
                    "modality": "visual",
                    "subject": "product",
                }
            ],
            commerce_cues=[
                {
                    "cue_id": f"{bundle.video_id}_cue_product_identity_abc",
                    "video_id": bundle.video_id,
                    "cue_type": "PRODUCT_IDENTITY",
                }
            ],
            commercial_relations=[],
            gold_proposals=[{"proposal_id": f"p_{bundle.video_id}", "video_id": bundle.video_id}],
            gold_reviews=[{"review_id": f"r_{bundle.video_id}", "video_id": bundle.video_id}],
            video_gold_record={
                "video_id": bundle.video_id,
                "schema_version": "evidence-dataset-schema-v3",
                "evidence_unit_ids": [f"{bundle.video_id}_visual_000_abc"],
                "commerce_cue_ids": [f"{bundle.video_id}_cue_product_identity_abc"],
                "commercial_relation_ids": [],
                "grounded_annotations": [],
                "coverage": {},
                "quality_summary": {},
                "observation_scope": {},
            },
            human_review_queue=[],
            rejected_candidates=[
                {
                    "review_item_id": f"reject_{bundle.video_id}",
                    "video_id": bundle.video_id,
                    "reason_code": "BELOW_MIN_CONFIDENCE",
                }
            ],
            pipeline_diagnostics=[
                {
                    "review_item_id": f"diag_{bundle.video_id}",
                    "video_id": bundle.video_id,
                    "reason_code": "ABSTENTION",
                }
            ],
            quality_decisions=[
                {
                    "decision_id": f"decision_{bundle.video_id}",
                    "video_id": bundle.video_id,
                    "disposition": "REJECT",
                }
            ],
            agent_traces=[{"stage": "evidence_extraction", "video_id": bundle.video_id}],
            status="ok",
        )


class GoldBankRunnerTest(unittest.TestCase):
    def records(self):
        return [
            {"video_id": "v2", "douyin_handle": "b"},
            {"video_id": "v1", "douyin_handle": "a"},
        ]

    def pilot_config(self):
        return {
            "version": "test",
            "video_ids": ["v2", "v1"],
            "frame_strategy": "hook_plus_uniform",
            "frames_per_video": 16,
            "prompt_version": "evidence-prompt-v9",
            "schema_version": "evidence-dataset-schema-v3",
            "min_confidence": 0.7,
        }

    def test_explicit_video_ids_preserve_requested_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_gold_bank_records(self.records(), self.pilot_config(), Path(tmp), FakePipeline())

        self.assertEqual(summary["video_ids"], ["v2", "v1"])

    def test_runner_writes_all_contract_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, FakePipeline())

            visible = {path.name for path in output_dir.iterdir() if path.is_file()}
            self.assertTrue((output_dir / "commerce_cues.jsonl").is_file())
            self.assertTrue((output_dir / "commercial_relations.jsonl").is_file())
            self.assertTrue((output_dir / "rejected_candidates.jsonl").is_file())
            self.assertTrue((output_dir / "pipeline_diagnostics.jsonl").is_file())
            self.assertTrue((output_dir / "quality_decisions.jsonl").is_file())

        self.assertEqual(visible, set(GOLD_BANK_OUTPUT_FILES))

    def test_records_are_deterministically_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, FakePipeline())
            lines = (output_dir / "evidence_units.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertIn('"video_id": "v1"', lines[0])
        self.assertIn('"video_id": "v2"', lines[1])

    def test_resume_skips_completed_video_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = FakePipeline()
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, first)
            second = FakePipeline()
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, second)

        self.assertEqual(first.calls, ["v2", "v1"])
        self.assertEqual(second.calls, [])

    def test_resume_reuses_terminal_partial_candidate_with_no_failed_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, FakePipeline())
            part_path = output_dir / ".parts" / "v2" / "result.json"
            payload = json.loads(part_path.read_text(encoding="utf-8"))
            payload["status"] = "partial"
            part_path.write_text(json.dumps(payload), encoding="utf-8")
            resumed = FakePipeline()

            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, resumed)

        self.assertEqual(resumed.calls, [])

    def test_resume_retries_partial_video_with_a_failed_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, FakePipeline())
            part_path = output_dir / ".parts" / "v2" / "result.json"
            payload = json.loads(part_path.read_text(encoding="utf-8"))
            payload["status"] = "partial"
            payload["agent_traces"].append(
                {"stage": "language_evidence_extraction", "success": False, "error": "bad output"}
            )
            part_path.write_text(json.dumps(payload), encoding="utf-8")
            retry = FakePipeline()

            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, retry)

        self.assertEqual(retry.calls, ["v2"])

    def test_generation_meta_records_prompt_and_schema_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, FakePipeline())
            meta = (output_dir / "generation_meta.json").read_text(encoding="utf-8")

        self.assertIn("evidence-prompt-v9", meta)
        self.assertIn("evidence-dataset-schema-v3", meta)


if __name__ == "__main__":
    unittest.main()
