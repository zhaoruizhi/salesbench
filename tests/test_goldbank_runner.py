from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankResult  # noqa: E402
from salesbench.goldbank.runner import (  # noqa: E402
    GOLD_BANK_OUTPUT_FILES,
    VisionPreflightError,
    _pipeline_fingerprint,
    run_gold_bank_records,
)
from salesbench.vlm.api_client import APICallResult  # noqa: E402
from salesbench.vlm.frame_sampler import Frame  # noqa: E402


class FakePipeline:
    def __init__(self):
        self.calls: list[str] = []

    def run_video(self, bundle, frames_b64=None, frame_urls=None, resume_result=None):
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


class PartialPipeline:
    def __init__(self, result: GoldBankResult):
        self.result = result
        self.calls: list[str] = []
        self.resume_results: list[GoldBankResult | None] = []

    def run_video(self, bundle, frames_b64=None, frame_urls=None, resume_result=None):
        self.calls.append(bundle.video_id)
        self.resume_results.append(resume_result)
        return GoldBankResult(**{**self.result.__dict__, "video_id": bundle.video_id})


class FailedPreflightClient:
    model = "qwen3.7-plus"

    def __init__(self):
        self.calls = 0

    def call(self, system_prompt, user_content, response_format=None):
        self.calls += 1
        return APICallResult(
            raw_response="",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.01,
            cost_usd=0.0,
            success=False,
            error="SSL EOF",
            error_kind="retryable",
        )


class PreflightPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.vlm_client = FailedPreflightClient()


class FakeFrameUploader:
    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    def upload_frames(self, paths, model):
        self.calls.append((list(paths), model))
        return [f"oss://temporary/frame-{index}.jpg" for index in range(len(paths))]


class FailedFrameUploader(FakeFrameUploader):
    def upload_frames(self, paths, model):
        from salesbench.vlm.dashscope_oss import DashScopeUploadError

        self.calls.append((list(paths), model))
        raise DashScopeUploadError("temporary upload failed")


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

    def test_resume_passes_matching_partial_as_stage_retry_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = PartialPipeline(
                GoldBankResult(
                    video_id="v2",
                    evidence_units=[
                        {
                            "evidence_id": "v2_asr_000_seed",
                            "video_id": "v2",
                            "modality": "asr",
                            "content_en": "The speaker names the product.",
                        }
                    ],
                    gold_proposals=[],
                    gold_reviews=[],
                    video_gold_record=None,
                    human_review_queue=[],
                    agent_traces=[
                        {"stage": "language_evidence_extraction", "success": True},
                        {"stage": "visual_evidence_extraction", "success": False},
                    ],
                    status="partial",
                )
            )
            config = {**self.pilot_config(), "video_ids": ["v2"]}
            run_gold_bank_records(self.records(), config, output_dir, first)
            retry = PartialPipeline(first.result)

            run_gold_bank_records(self.records(), config, output_dir, retry)

        self.assertEqual(len(retry.resume_results), 1)
        self.assertIsNotNone(retry.resume_results[0])
        self.assertEqual(retry.resume_results[0].evidence_units[0]["modality"], "asr")

    def test_resume_never_overwrites_a_better_partial_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = {**self.pilot_config(), "video_ids": ["v2"]}
            better = GoldBankResult(
                video_id="v2",
                evidence_units=[
                    {
                        "evidence_id": "v2_asr_000_seed",
                        "video_id": "v2",
                        "modality": "asr",
                        "content_en": "The speaker names the product.",
                    }
                ],
                gold_proposals=[],
                gold_reviews=[],
                video_gold_record=None,
                human_review_queue=[],
                agent_traces=[
                    {"stage": "language_evidence_extraction", "success": True},
                    {"stage": "visual_evidence_extraction", "success": False},
                ],
                status="partial",
            )
            worse = GoldBankResult(
                video_id="v2",
                evidence_units=[],
                gold_proposals=[],
                gold_reviews=[],
                video_gold_record=None,
                human_review_queue=[],
                agent_traces=[{"stage": "visual_evidence_extraction", "success": False}],
                status="failed",
            )
            run_gold_bank_records(
                self.records(), config, output_dir, PartialPipeline(better)
            )

            summary = run_gold_bank_records(
                self.records(), config, output_dir, PartialPipeline(worse)
            )
            payload = json.loads(
                (output_dir / ".parts" / "v2" / "result.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["evidence_units"][0]["modality"], "asr")
        self.assertEqual(summary["counts"]["evidence_units"], 1)

    def test_generation_meta_records_prompt_and_schema_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_gold_bank_records(self.records(), self.pilot_config(), output_dir, FakePipeline())
            meta = (output_dir / "generation_meta.json").read_text(encoding="utf-8")

        self.assertIn("evidence-prompt-v9", meta)
        self.assertIn("evidence-dataset-schema-v3", meta)

    def test_generation_meta_reports_human_ambiguity_release_gate(self):
        config = self.pilot_config()
        config.update(
            {
                "human_ambiguity_target_rate": 0.05,
                "human_ambiguity_block_rate": 0.10,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_gold_bank_records(
                self.records(), config, Path(tmp), FakePipeline()
            )

        self.assertIn("human_ambiguity_rate", summary["quality_gate"])
        self.assertEqual(summary["quality_gate"]["target_rate"], 0.05)
        self.assertEqual(summary["quality_gate"]["block_rate"], 0.10)

    def test_run_metadata_binds_evidence_to_release_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run-001"
            summary = run_gold_bank_records(
                self.records(),
                self.pilot_config(),
                run_root / "evidence",
                FakePipeline(),
                run_id="run-001",
                benchmark_release="salesbench-v10-candidate.2",
            )
            manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["run_id"], "run-001")
        self.assertEqual(summary["benchmark_release"], "salesbench-v10-candidate.2")
        self.assertEqual(summary["release_status"], "CANDIDATE")
        self.assertEqual(len(summary["source_fingerprint"]), 64)
        self.assertEqual(len(summary["evidence_fingerprint"]), 64)
        self.assertEqual(manifest["fingerprints"]["evidence"], summary["evidence_fingerprint"])
        self.assertEqual(manifest["artifacts"]["evidence"], "evidence")

    def test_completed_output_rejects_a_different_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-001" / "evidence"
            run_gold_bank_records(
                self.records(), self.pilot_config(), output_dir, FakePipeline(), run_id="run-001"
            )

            with self.assertRaisesRegex(ValueError, "immutable Evidence output"):
                run_gold_bank_records(
                    self.records(), self.pilot_config(), output_dir, FakePipeline(), run_id="run-002"
                )

    def test_completed_output_rejects_changed_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run-001" / "evidence"
            run_gold_bank_records(
                self.records(), self.pilot_config(), output_dir, FakePipeline(), run_id="run-001"
            )
            changed = self.pilot_config()
            changed["min_confidence"] = 0.91

            with self.assertRaisesRegex(ValueError, "source fingerprint"):
                run_gold_bank_records(
                    self.records(), changed, output_dir, FakePipeline(), run_id="run-001"
                )

    def test_runner_rejects_incomplete_frame_sampling_instead_of_reducing_input(self):
        record = {
            "video_id": "v2",
            "video_path": "/tmp/video.mp4",
            "has_video_asset": True,
        }
        config = {
            **self.pilot_config(),
            "video_ids": ["v2"],
            "require_exact_frame_count": True,
        }
        fifteen = [
            Frame("AAA", float(index), index, f"/tmp/f{index}.jpg")
            for index in range(15)
        ]

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            VisionPreflightError, "expected 16 sampled frames, got 15"
        ):
            run_gold_bank_records(
                [record],
                config,
                Path(tmp),
                FakePipeline(),
                frame_sampler=lambda **_: fifteen,
            )

    def test_failed_full_frame_preflight_aborts_before_any_video_part_is_written(self):
        record = {
            "video_id": "v2",
            "video_path": "/tmp/video.mp4",
            "has_video_asset": True,
        }
        config = {
            **self.pilot_config(),
            "video_ids": ["v2"],
            "vision_transport": "dashscope_temporary_oss",
            "vision_preflight": True,
            "vision_preflight_attempts": 2,
            "vision_preflight_required_successes": 2,
        }
        frames = [
            Frame("AAA", float(index), index, f"/tmp/f{index}.jpg")
            for index in range(16)
        ]
        pipeline = PreflightPipeline()
        uploader = FakeFrameUploader()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaisesRegex(VisionPreflightError, "full-frame preflight"):
                run_gold_bank_records(
                    [record],
                    config,
                    output_dir,
                    pipeline,
                    frame_sampler=lambda **_: frames,
                    frame_uploader=uploader,
                )

            self.assertFalse((output_dir / ".parts" / "v2" / "result.json").exists())
            report = json.loads(
                (output_dir / ".parts" / "vision_preflight.json").read_text(encoding="utf-8")
            )

        self.assertEqual(pipeline.vlm_client.calls, 2)
        self.assertEqual(report["requested_frame_count"], 16)
        self.assertEqual(report["success_count"], 0)
        self.assertNotIn("oss://", json.dumps(report))

    def test_upload_failure_is_written_as_redacted_preflight_diagnostic(self):
        record = {
            "video_id": "v2",
            "video_path": "/tmp/video.mp4",
            "has_video_asset": True,
        }
        config = {
            **self.pilot_config(),
            "video_ids": ["v2"],
            "vision_transport": "dashscope_temporary_oss",
            "vision_preflight": True,
        }
        frames = [
            Frame("AAA", float(index), index, f"/tmp/f{index}.jpg")
            for index in range(16)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaisesRegex(VisionPreflightError, "preparation failed"):
                run_gold_bank_records(
                    [record],
                    config,
                    output_dir,
                    PreflightPipeline(),
                    frame_sampler=lambda **_: frames,
                    frame_uploader=FailedFrameUploader(),
                )
            report_text = (output_dir / ".parts" / "vision_preflight.json").read_text(
                encoding="utf-8"
            )

        self.assertIn('"phase": "frame_preparation"', report_text)
        self.assertNotIn("temporary upload failed", report_text)

    def test_part_fingerprint_changes_when_visual_transport_changes(self):
        record = {"video_id": "v2"}
        pipeline = FakePipeline()
        base = self.pilot_config()
        base["vision_transport"] = "base64"
        urls = {**base, "vision_transport": "dashscope_temporary_oss"}

        assert _pipeline_fingerprint(record, base, pipeline) != _pipeline_fingerprint(
            record, urls, pipeline
        )


if __name__ == "__main__":
    unittest.main()
