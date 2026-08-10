"""Compile deterministic VQA records from an EvidenceDataset."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..goldbank.schema import GoldItem
from ..goldbank.validators import PRIVATE_KEYS
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..utils import clean_text, contains_cjk
from .goldbank_loader import load_compilable_gold
from .question_programs import UnsupportedQuestionProgramError, render_question


COMPILER_VERSION = "evidence-qa-compiler-v4"


@dataclass(frozen=True)
class CompilePolicy:
    max_questions_per_video: int = 8
    task_priority: tuple[str, ...] = ("BP", "CM", "SS", "AE")
    max_per_task: int = 2
    include_tiers: tuple[str, ...] = ("Gold-A", "Gold-B")
    require_all_tasks: bool = True


def _contains_private(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(clean_text(key) in PRIVATE_KEYS or _contains_private(value) for key, value in payload.items())
    if isinstance(payload, list):
        return any(_contains_private(value) for value in payload)
    return False


def _public_qa(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {
            "gold_answer",
            "answer",
            "evidence_refs",
            "evidence_context",
            "gold_value",
            "source_annotation_ids",
            "quality_status",
            "review_status",
            "private_metadata",
            "private_analysis_metadata",
        }
    }


def _question_leaks_answer(question: str, answer: str) -> bool:
    answer = clean_text(answer)
    if not answer or len(answer) <= 1:
        return False
    return answer in question


def compile_qa_records(
    gold_items: list[GoldItem],
    policy: CompilePolicy,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_video: dict[str, list[GoldItem]] = defaultdict(list)
    for item in gold_items:
        if item.gold_tier.value in policy.include_tiers:
            by_video[item.video_id].append(item)

    qa_records: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    for video_id in sorted(by_video):
        video_count = 0
        per_task_count = {task: 0 for task in policy.task_priority}
        ordered_items = sorted(
            by_video[video_id],
            key=lambda item: (
                policy.task_priority.index(item.task_type.value)
                if item.task_type.value in policy.task_priority
                else len(policy.task_priority),
                item.gold_id,
            ),
        )
        for item in ordered_items:
            if video_count >= policy.max_questions_per_video:
                validation.append({"gold_id": item.gold_id, "status": "skipped", "reason": "video_quota"})
                continue
            if per_task_count.get(item.task_type.value, 0) >= policy.max_per_task:
                validation.append({"gold_id": item.gold_id, "status": "skipped", "reason": "task_quota"})
                continue
            compiled = None
            error = None
            for question_format in item.eligible_question_formats:
                try:
                    compiled = render_question(item, question_format)
                    break
                except UnsupportedQuestionProgramError as exc:
                    error = str(exc)
            if compiled is None:
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": error or "unsupported_question_program"})
                continue
            if _question_leaks_answer(compiled.question, compiled.answer):
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "question_leaks_answer"})
                continue
            if contains_cjk((compiled.question, compiled.answer)):
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "non_english_public_text"})
                continue
            vqa_id = f"{video_id}_{item.task_type.value.lower()}_{len(qa_records) + 1:05d}"
            record = {
                "vqa_id": vqa_id,
                "video_id": video_id,
                "task_layer": "salesbench_qa",
                "task_type": item.task_type.value,
                "task_subtype": item.task_subtype,
                "question": compiled.question,
                "answer_type": "open",
                "gold_answer": compiled.answer,
                "source_annotation_ids": [item.annotation_id],
                "evidence_refs": list(item.evidence_refs),
                "question_program_id": compiled.question_program_id,
                "quality_status": item.quality_status.value,
                "review_status": item.review_status,
                "compiler_version": COMPILER_VERSION,
            }
            if _contains_private(record):
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "private_field_leak"})
                continue
            qa_records.append(record)
            video_count += 1
            per_task_count[item.task_type.value] = per_task_count.get(item.task_type.value, 0) + 1
            validation.append({"annotation_id": item.annotation_id, "vqa_id": vqa_id, "status": "accepted", "reason": ""})
    return qa_records, validation


def compile_vqa_from_gold(
    gold_bank_dir: Path,
    output_dir: Path,
    policy: CompilePolicy,
    bank_filename: str = "video_evidence_dataset_reviewed.jsonl",
) -> dict[str, object]:
    bank_path = gold_bank_dir / bank_filename
    if not bank_path.exists():
        raise FileNotFoundError(f"EvidenceDataset file not found: {bank_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_items = load_compilable_gold(bank_path)
    qa_records, validation = compile_qa_records(gold_items, policy)
    task_counts = {
        task: sum(1 for record in qa_records if record.get("task_type") == task)
        for task in policy.task_priority
    }
    missing_tasks = [task for task, count in task_counts.items() if count == 0]
    if policy.require_all_tasks and missing_tasks:
        raise ValueError(f"EvidenceDataset cannot compile a public benchmark with empty tasks: {missing_tasks}")
    evidence_lookup = {
        clean_text(record.get("evidence_id")): record
        for record in read_jsonl(gold_bank_dir / "evidence_units.jsonl")
        if clean_text(record.get("evidence_id"))
    } if (gold_bank_dir / "evidence_units.jsonl").exists() else {}
    for record in qa_records:
        record["evidence_context"] = [
            evidence_lookup[evidence_id]
            for evidence_id in record.get("evidence_refs", [])
            if evidence_id in evidence_lookup
        ]

    qa_plan = [
        {
            "video_id": item.video_id,
            "annotation_id": item.annotation_id,
            "task_type": item.task_type.value,
            "task_subtype": item.task_subtype,
            "eligible_question_formats": list(item.eligible_question_formats),
        }
        for item in gold_items
    ]

    write_jsonl(output_dir / "qa_plan.jsonl", qa_plan)
    write_jsonl(output_dir / "qa_candidates.jsonl", qa_records)
    write_jsonl(output_dir / "qa_validation.jsonl", validation)
    write_jsonl(output_dir / "vqa_gold_private.jsonl", qa_records)
    public_records = [_public_qa(record) for record in qa_records]
    write_jsonl(output_dir / "vqa_public.jsonl", public_records)
    for task in policy.task_priority:
        write_jsonl(
            output_dir / "public" / f"{task.lower()}.jsonl",
            [record for record in public_records if record.get("task_type") == task],
        )
    meta = {
        "compiler_version": COMPILER_VERSION,
        "public_tasks": list(policy.task_priority),
        "bank_file": str(bank_path),
        "counts": {
            "qa_plan": len(qa_plan),
            "qa_candidates": len(qa_records),
            "qa_validation": len(validation),
            "vqa_gold_private": len(qa_records),
            "vqa_public": len(qa_records),
            "per_task": task_counts,
        },
        "validation": {"missing_tasks": missing_tasks, "all_tasks_nonempty": not missing_tasks},
        "inputs": {
            "evidence_units": len(read_jsonl(gold_bank_dir / "evidence_units.jsonl")) if (gold_bank_dir / "evidence_units.jsonl").exists() else 0,
        },
    }
    write_json(output_dir / "generation_meta.json", meta)
    return meta
