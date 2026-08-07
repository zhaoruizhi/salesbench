from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "src")

from salesbench.goldbank.pipeline import GoldBankResult  # noqa: E402
from salesbench.goldbank.runner import run_gold_bank_records  # noqa: E402
from salesbench.goldbank.schema import (  # noqa: E402
    GoldTaskType,
    QualityStatus,
    parse_gold_item,
)
from salesbench.goldbank.splits import build_group_split_manifest  # noqa: E402
from salesbench.interaction_analysis import (  # noqa: E402
    build_private_analysis_metadata,
    interaction_diagnostic_slices,
)
from salesbench.vqa.compiler import CompilePolicy, compile_qa_records  # noqa: E402
from salesbench.vqa_evaluate.context import build_judge_payload  # noqa: E402
from salesbench.vqa_evaluate.runner import evaluate_salesbench_qa_records  # noqa: E402
from salesbench.vqa_baseline.runner import run_salesbench_qa_baseline_records  # noqa: E402


def grounded_item(task_type: str = "BP"):
    return parse_gold_item(
        {
            "annotation_id": f"g_{task_type.lower()}",
            "video_id": "v1",
            "task_type": task_type,
            "task_subtype": "ACTION" if task_type == "BP" else "HOOK_MECHANISM",
            "target": {"subject": "product"},
            "gold_value": {"action": "opened"} if task_type == "BP" else {"mechanism": "question"},
            "evidence_refs": ["e1"],
            "reasoning_edges": [],
            "eligible_question_formats": ["direct_question"] if task_type == "BP" else ["mechanism_with_evidence"],
            "source_proposal_ids": ["p1"],
            "quality_status": "DIRECT" if task_type == "BP" else "INFERRED",
            "review_status": "verified",
            "confidence": 0.9,
        }
    )


class NoCallClient:
    model = "unused"

    def call_text_only(self, *args, **kwargs):
        raise AssertionError("missing answers must be scored locally without calling the judge")


class RecordingPipeline:
    min_confidence = 0.7

    def __init__(self):
        self.bundles = []
        self.frames = []

    def run_video(self, bundle, frames_b64=None):
        self.bundles.append(bundle)
        self.frames.append(list(frames_b64 or []))
        return GoldBankResult(
            video_id=bundle.video_id,
            evidence_units=[],
            gold_proposals=[],
            gold_reviews=[],
            video_gold_record={
                "video_id": bundle.video_id,
                "schema_version": "evidence-dataset-schema-v2",
                "evidence_unit_ids": [],
                "grounded_annotations": [],
                "coverage": {},
                "quality_summary": {},
                "observation_scope": {},
            },
            human_review_queue=[],
            agent_traces=[],
            status="ok",
        )


class RecordingStore:
    def __init__(self):
        self.calls = []

    def bundle_for_video(self, video_id, frames=None):
        from salesbench.multiagent.context import build_context_bundle

        self.calls.append((video_id, list(frames or [])))
        return build_context_bundle(
            video_id,
            raw_video={"video_id": video_id, "video_text": "ASR"},
            visual_features={"video_id": video_id, "visual": 1},
            audio_speech={"video_id": video_id, "video_text": "ASR"},
            text_language={"video_id": video_id, "title": "private title"},
            publish_context={"video_id": video_id, "followers_total": 9},
            cross_modal={"video_id": video_id, "consistency": 1},
            frames=list(frames or []),
        )


class BenchmarkConvergenceTest(unittest.TestCase):
    def test_public_task_contract_has_exactly_four_tasks(self):
        self.assertEqual([task.value for task in GoldTaskType], ["BP", "CM", "SS", "AE"])

    def test_compiler_emits_unified_contract(self):
        records, validation = compile_qa_records([grounded_item()], CompilePolicy())

        self.assertEqual(validation[0]["status"], "accepted")
        self.assertEqual(records[0]["vqa_id"], "v1_bp_00001")
        self.assertEqual(records[0]["task_layer"], "salesbench_qa")
        self.assertEqual(records[0]["evidence_refs"], ["e1"])
        self.assertEqual(records[0]["quality_status"], QualityStatus.DIRECT.value)
        self.assertNotIn("qa_id", records[0])

    def test_judge_payload_never_contains_private_metadata(self):
        payload = build_judge_payload(
            {
                "vqa_id": "q1",
                "video_id": "v1",
                "task_type": "SS",
                "question": "视频如何建立信任？",
                "gold_answer": "通过演示",
                "evidence_refs": [{"evidence_id": "e1", "text_span": "现场演示"}],
                "private_analysis_metadata": {"likes": 100},
                "followers_total": 999,
            },
            {"vqa_id": "q1", "answer": "通过演示"},
        )

        text = str(payload)
        self.assertNotIn("likes", text)
        self.assertNotIn("followers", text)
        self.assertIn("现场演示", text)

    def test_missing_answer_is_scored_zero(self):
        report, details = evaluate_salesbench_qa_records(
            [{"vqa_id": "q1", "task_layer": "salesbench_qa", "task_type": "BP", "video_id": "v1"}],
            [],
            NoCallClient(),
        )

        self.assertEqual(details[0]["score"], 0.0)
        self.assertEqual(details[0]["error"], "missing_model_answer")
        self.assertEqual(report["summary"]["missing_answer_count"], 1)
        self.assertEqual(report["metrics"]["macro_average"]["relaxed_accuracy"], 0.0)

    def test_unknown_task_is_rejected_by_runner_and_evaluator(self):
        invalid = [
            {
                "vqa_id": "q1",
                "video_id": "v1",
                "task_layer": "salesbench_qa",
                "task_type": "UNKNOWN",
                "question": "invalid",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported public VQA task types"):
                run_salesbench_qa_baseline_records(
                    invalid,
                    {"v1": {"video_id": "v1"}},
                    Path(tmp),
                    "unused",
                    NoCallClient(),
                )
        with self.assertRaisesRegex(ValueError, "Unsupported public VQA task types"):
            evaluate_salesbench_qa_records(invalid, [], NoCallClient())

    def test_creator_disjoint_split_is_approximately_80_10_10(self):
        records = [
            {"video_id": f"v{i:03d}", "douyin_handle": f"creator{i:03d}"}
            for i in range(100)
        ]
        split = build_group_split_manifest(records, seed=7)

        self.assertEqual((len(split.train_ids), len(split.val_ids), len(split.test_ids)), (80, 10, 10))

    def test_runner_uses_context_store_and_passes_sampled_frames(self):
        records = [{"video_id": "v1", "primary_video_path": "/tmp/v1.mp4", "has_video_asset": True}]
        pilot = {
            "version": "test",
            "video_ids": ["v1"],
            "frame_strategy": "hook_plus_uniform",
            "frames_per_video": 16,
            "prompt_version": "evidence-prompt-v5",
            "schema_version": "evidence-dataset-schema-v2",
            "min_confidence": 0.7,
        }
        store = RecordingStore()
        pipeline = RecordingPipeline()

        def fake_sampler(**kwargs):
            from salesbench.vlm.frame_sampler import Frame

            return [Frame(image_base64="abc", timestamp_s=0.5, frame_index=0, path="/tmp/f.jpg")]

        with tempfile.TemporaryDirectory() as tmp:
            run_gold_bank_records(
                records,
                pilot,
                Path(tmp),
                pipeline,
                context_store=store,
                frame_sampler=fake_sampler,
                resume=False,
            )

        self.assertEqual(store.calls[0][0], "v1")
        self.assertEqual(store.calls[0][1][0]["timestamp_s"], 0.5)
        self.assertEqual(pipeline.frames, [["abc"]])
        self.assertTrue(pipeline.bundles[0].content_context["C1_visual"])

    def test_interaction_analysis_is_private_and_diagnostic_only(self):
        records = []
        for i in range(20):
            records.append(
                {
                    "video_id": f"v{i}",
                    "product_bucket": "食品",
                    "followers_total": 1000 + i,
                    "publish_date": "2024-04-05",
                    "video_duration_s": 20,
                    "likes": i + 1,
                    "comments": i + 2,
                    "shares": i + 3,
                    "collects": i + 4,
                }
            )
        private, meta = build_private_analysis_metadata(records)
        details = [
            {"video_id": row["video_id"], "task_type": "BP", "score": 1.0, "judge_success": True}
            for row in private
        ]
        diagnostics = interaction_diagnostic_slices(details, private, bootstrap_samples=20)

        self.assertEqual(meta["analysis_type"], "snapshot_interaction_profile")
        self.assertIn("interaction_strata", private[0])
        self.assertNotIn("gold_answer", private[0])
        self.assertEqual(diagnostics["role"], "diagnostic_only")
        self.assertFalse(diagnostics["included_in_leaderboard"])

    def test_frame_cache_invalidates_on_sampling_config_and_video_hash(self):
        import salesbench.vlm.frame_sampler as sampler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"video-v1")

            def extract(_video_path, _timestamp, output_path):
                Path(output_path).write_bytes(b"jpeg")
                return True

            with (
                patch.object(sampler, "CACHE_ROOT", root / "cache"),
                patch.object(sampler, "_get_video_duration", return_value=10.0),
                patch.object(sampler, "_extract_frame_at", side_effect=extract) as mocked_extract,
                patch.object(sampler, "_is_black_frame", return_value=False),
            ):
                sampler.sample_frames(str(video), "v1", total_frames=2, hook_frames=1)
                first_calls = mocked_extract.call_count
                sampler.sample_frames(str(video), "v1", total_frames=2, hook_frames=1)
                self.assertEqual(mocked_extract.call_count, first_calls)
                sampler.sample_frames(str(video), "v1", total_frames=2, hook_frames=0)
                self.assertGreater(mocked_extract.call_count, first_calls)
                before_hash_change = mocked_extract.call_count
                video.write_bytes(b"video-v2")
                sampler.sample_frames(str(video), "v1", total_frames=2, hook_frames=0)
                self.assertGreater(mocked_extract.call_count, before_hash_change)


if __name__ == "__main__":
    unittest.main()
