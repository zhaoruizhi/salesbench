from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.vqa.compiler import CompilePolicy, compile_qa_records, compile_vqa_from_gold  # noqa: E402
from salesbench.vqa.goldbank_loader import load_compilable_gold  # noqa: E402


def gold_record():
    return {
        "video_id": "v1",
        "schema_version": "evidence-dataset-schema-v2",
        "evidence_unit_ids": ["e1", "e2"],
        "grounded_annotations": [
            {
                "annotation_id": "g_bp",
                "video_id": "v1",
                "task_type": "BP",
                "task_subtype": "ACTION",
                "target": {"subject": "product"},
                "gold_value": {"action": "opened"},
                "evidence_refs": ["e1"],
                "reasoning_edges": [],
                "eligible_question_formats": ["direct_question"],
                "source_proposal_ids": ["p_bp"],
                "quality_status": "DIRECT",
                "review_status": "verified",
                "confidence": 0.9,
            },
            {
                "annotation_id": "g_silver",
                "video_id": "v1",
                "task_type": "SS",
                "task_subtype": "HOOK_MECHANISM",
                "target": {"mechanism": "question"},
                "gold_value": {"label": "question"},
                "evidence_refs": ["e1", "e2"],
                "reasoning_edges": [],
                "eligible_question_formats": ["mechanism_with_evidence"],
                "source_proposal_ids": ["p_ss"],
                "quality_status": "NEEDS_REVIEW",
                "review_status": "needs_review",
                "confidence": 0.7,
            },
        ],
        "coverage": {},
        "quality_summary": {},
        "observation_scope": {},
    }


class GoldBankQACompilerTest(unittest.TestCase):
    def test_only_gold_a_and_gold_b_are_compiled(self):
        item = load_compilable_gold_from_records([gold_record()])

        self.assertEqual([gold.gold_id for gold in item], ["g_bp"])

    def test_each_qa_has_source_annotation_ids_and_evidence_refs(self):
        qa, validation = compile_qa_records(load_compilable_gold_from_records([gold_record()]), CompilePolicy())

        self.assertEqual(validation[0]["status"], "accepted")
        self.assertEqual(qa[0]["source_annotation_ids"], ["g_bp"])
        self.assertEqual(qa[0]["evidence_refs"], ["e1"])

    def test_same_gold_compiles_deterministically(self):
        items = load_compilable_gold_from_records([gold_record()])

        first, _ = compile_qa_records(items, CompilePolicy())
        second, _ = compile_qa_records(items, CompilePolicy())

        self.assertEqual(first, second)

    def test_answer_is_derived_from_value_not_generated_by_llm(self):
        qa, _ = compile_qa_records(load_compilable_gold_from_records([gold_record()]), CompilePolicy())

        self.assertEqual(qa[0]["gold_answer"], "opened")

    def test_bp_fact_question_includes_subject_and_predicate(self):
        record = gold_record()
        bp = record["grounded_annotations"][0]
        bp["task_subtype"] = "OCR_FACT"
        bp["target"] = {"subject": "product", "predicate": "name"}
        bp["gold_value"] = {"value": "test product"}

        qa, _ = compile_qa_records(load_compilable_gold_from_records([record]), CompilePolicy())

        self.assertIn("product's name", qa[0]["question"])

    def test_ss_answer_prefers_complete_answer_over_short_label(self):
        record = gold_record()
        ss = record["grounded_annotations"][1]
        ss["quality_status"] = "INFERRED"
        ss["review_status"] = "verified"
        ss["gold_value"] = {
            "label": "result first",
            "answer": "The opening states the desired result before showing the corresponding product.",
        }

        qa, _ = compile_qa_records(load_compilable_gold_from_records([record]), CompilePolicy())
        ss_qa = next(item for item in qa if item["task_type"] == "SS")

        self.assertEqual(ss_qa["gold_answer"], ss["gold_value"]["answer"])

    def test_compiler_rejects_non_english_public_question_or_answer(self):
        record = gold_record()
        record["grounded_annotations"][0]["gold_value"] = {"action": "打开包装"}

        qa, validation = compile_qa_records(load_compilable_gold_from_records([record]), CompilePolicy())

        self.assertEqual(qa, [])
        self.assertEqual(validation[0]["reason"], "non_english_public_text")

    def test_public_qa_contains_no_private_interaction_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold_dir = Path(tmp) / "gold"
            out_dir = Path(tmp) / "qa"
            gold_dir.mkdir()
            write_jsonl(gold_dir / "video_evidence_dataset.jsonl", [gold_record()])
            write_jsonl(gold_dir / "evidence_units.jsonl", [{"evidence_id": "e1", "video_id": "v1"}])
            summary = compile_vqa_from_gold(
                gold_dir,
                out_dir,
                CompilePolicy(require_all_tasks=False),
                bank_filename="video_evidence_dataset.jsonl",
            )
            public_text = (out_dir / "vqa_public.jsonl").read_text(encoding="utf-8")

        self.assertEqual(summary["counts"]["vqa_gold_private"], 1)
        self.assertNotIn("gold_answer", public_text)
        self.assertNotIn("thresholds", public_text)

    def test_question_does_not_leak_answer(self):
        qa, validation = compile_qa_records(load_compilable_gold_from_records([gold_record()]), CompilePolicy())

        self.assertNotIn(qa[0]["gold_answer"], qa[0]["question"])
        self.assertEqual(validation[0]["status"], "accepted")

    def test_per_video_task_quota_is_enforced(self):
        record = gold_record()
        record["grounded_annotations"].append({**record["grounded_annotations"][0], "annotation_id": "g_bp_2", "gold_value": {"action": "shown"}})

        qa, _ = compile_qa_records(load_compilable_gold_from_records([record]), CompilePolicy(max_per_task=1))

        self.assertEqual(len(qa), 1)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_compilable_gold_from_records(records: list[dict[str, object]]):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bank.jsonl"
        write_jsonl(path, records)
        return load_compilable_gold(path)


if __name__ == "__main__":
    unittest.main()
