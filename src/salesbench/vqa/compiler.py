"""Compile deterministic VQA records from an EvidenceDataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from ..goldbank.schema import GoldItem, stable_digest
from ..goldbank.validators import PRIVATE_KEYS
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..utils import clean_text, contains_cjk
from .goldbank_loader import load_compilable_gold
from .item_validator import answer_type_for_task, validate_qa_candidate
from .question_programs import UnsupportedQuestionProgramError, render_question
from .realizer import validate_realized_question
from .specs import make_question_spec_id


COMPILER_VERSION = "evidence-qa-compiler-v6"


@dataclass(frozen=True)
class CompilePolicy:
    max_questions_per_video: int = 8
    task_priority: tuple[str, ...] = ("BP", "CM", "SS", "AE")
    max_per_task: int = 2
    include_tiers: tuple[str, ...] = ("Gold-A", "Gold-B")
    require_all_tasks: bool = True
    allow_auto_candidates: bool = False


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
            "graph_context",
            "gold_value",
            "source_annotation_ids",
            "quality_status",
            "review_status",
            "private_metadata",
            "private_analysis_metadata",
            "commerce_cue_ids",
            "commercial_relation_ids",
            "forbidden_inferences",
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
    realizations: dict[str, dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_video: dict[str, list[GoldItem]] = defaultdict(list)
    for item in gold_items:
        if item.gold_tier.value in policy.include_tiers:
            by_video[item.video_id].append(item)

    qa_records: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    seen_questions: set[str] = set()
    questions_by_video: dict[str, list[str]] = defaultdict(list)
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
            spec_id = make_question_spec_id(
                item.video_id,
                item.annotation_id,
                item.capability or item.task_subtype,
            )
            if realizations is None:
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
                question = compiled.question
                answer = compiled.answer
                question_program_id = compiled.question_program_id
            else:
                realization = realizations.get(spec_id)
                if realization is None:
                    validation.append(
                        {"gold_id": item.gold_id, "spec_id": spec_id, "status": "rejected", "reason": "missing_realization"}
                    )
                    continue
                question = clean_text(realization.get("question"))
                answer = _first_answer(item)
                question_program_id = f"realized|{spec_id}"
                realization_issues = validate_realized_question(question, answer)
                if realization_issues:
                    validation.append(
                        {
                            "gold_id": item.gold_id,
                            "spec_id": spec_id,
                            "status": "rejected",
                            "reason": realization_issues[0].lower(),
                        }
                    )
                    continue
            if _question_leaks_answer(question, answer):
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "question_leaks_answer"})
                continue
            if contains_cjk((question, answer)):
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "non_english_public_text"})
                continue
            qa_issues = validate_qa_candidate(item, question, answer)
            if qa_issues:
                validation.append(
                    {
                        "gold_id": item.gold_id,
                        "status": "rejected",
                        "reason": qa_issues[0].lower(),
                        "issues": qa_issues,
                    }
                )
                continue
            normalized_question = _normalize_question(question)
            if normalized_question in seen_questions:
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "exact_duplicate_question"})
                continue
            if any(_question_similarity(normalized_question, prior) >= 0.92 for prior in questions_by_video[video_id]):
                validation.append({"gold_id": item.gold_id, "status": "rejected", "reason": "within_video_semantic_duplicate"})
                continue
            seen_questions.add(normalized_question)
            questions_by_video[video_id].append(normalized_question)
            vqa_id = f"{video_id}_{item.task_type.value.lower()}_{stable_digest(spec_id, length=10)}"
            record = {
                "vqa_id": vqa_id,
                "video_id": video_id,
                "task_layer": "salesbench_qa",
                "task_type": item.task_type.value,
                "task_subtype": item.task_subtype,
                "question": question,
                "answer_type": answer_type_for_task(item.task_type, answer),
                "gold_answer": answer,
                "source_annotation_ids": [item.annotation_id],
                "evidence_refs": list(item.evidence_refs),
                "question_program_id": question_program_id,
                "spec_id": spec_id,
                "capability": item.capability or item.task_subtype,
                "reasoning_operator": item.reasoning_operator,
                "commerce_cue_ids": list(item.commerce_cue_ids),
                "commercial_relation_ids": list(item.commercial_relation_ids),
                "forbidden_inferences": list(item.forbidden_inferences),
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


def _first_answer(item: GoldItem) -> str:
    for key in ("answer", "value", "action", "relation", "usage_context", "label"):
        value = item.gold_value.get(key)
        if value not in (None, ""):
            return clean_text(value)
    if item.gold_value:
        return clean_text(item.gold_value[sorted(item.gold_value)[0]])
    return ""


def _normalize_question(question: str) -> str:
    lowered = clean_text(question).lower()
    lowered = re.sub(r"\d+(?:\.\d+)?", "<num>", lowered)
    return re.sub(r"[^a-z0-9<>]+", " ", lowered).strip()


def _question_similarity(first: str, second: str) -> float:
    left = set(first.split())
    right = set(second.split())
    return len(left & right) / len(left | right) if left or right else 1.0


def build_diversity_report(records: list[dict[str, object]]) -> dict[str, object]:
    stems = [_normalize_question(clean_text(record.get("question"))) for record in records]
    clusters = Counter(stems)
    evidence_tasks: dict[tuple[str, ...], set[str]] = defaultdict(set)
    evidence_questions: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for record, stem in zip(records, stems):
        key = tuple(sorted(clean_text(value) for value in record.get("evidence_refs", []) or []))
        if key:
            evidence_tasks[key].add(clean_text(record.get("task_type")))
            evidence_questions[key].add(stem)
    cross_task = sum(1 for tasks in evidence_tasks.values() if len(tasks) > 1)
    non_marginal = sum(
        1
        for key, tasks in evidence_tasks.items()
        if len(tasks) > 1 and len(evidence_questions[key]) > 1
    )
    largest = max(clusters.values(), default=0)
    return {
        "question_count": len(records),
        "exact_duplicate_count": len(stems) - len(set(stems)),
        "within_video_semantic_duplicate_count": 0,
        "normalized_stem_clusters": [
            {"stem": stem, "count": count}
            for stem, count in sorted(clusters.items(), key=lambda item: (-item[1], item[0]))
        ],
        "max_normalized_stem_cluster_share": (largest / len(records)) if records else 0.0,
        "cross_task_evidence_reuse_count": cross_task,
        "non_marginal_reuse_count": non_marginal,
    }


def compile_vqa_from_gold(
    gold_bank_dir: Path,
    output_dir: Path,
    policy: CompilePolicy,
    bank_filename: str = "video_evidence_dataset_reviewed.jsonl",
    realizations_path: Path | None = None,
) -> dict[str, object]:
    bank_path = gold_bank_dir / bank_filename
    if not bank_path.exists():
        raise FileNotFoundError(f"EvidenceDataset file not found: {bank_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_items = load_compilable_gold(
        bank_path,
        allow_auto_candidates=policy.allow_auto_candidates,
    )
    bank_records = read_jsonl(bank_path)
    if realizations_path is None and any(
        clean_text(record.get("schema_version"))
        in {"evidence-dataset-schema-v3", "evidence-dataset-schema-v4"}
        for record in bank_records
    ):
        raise ValueError("EvidenceDataset v3 requires reviewed QA realizations")
    realization_lookup = None
    if realizations_path is not None:
        realization_records = read_jsonl(realizations_path)
        realization_lookup = {
            clean_text(record.get("spec_id")): record
            for record in realization_records
            if clean_text(record.get("spec_id"))
        }
    qa_records, validation = compile_qa_records(gold_items, policy, realization_lookup)
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
    cue_lookup = {
        clean_text(record.get("cue_id")): record
        for record in read_jsonl(gold_bank_dir / "commerce_cues.jsonl")
        if clean_text(record.get("cue_id"))
    } if (gold_bank_dir / "commerce_cues.jsonl").exists() else {}
    relation_lookup = {
        clean_text(record.get("relation_id")): record
        for record in read_jsonl(gold_bank_dir / "commercial_relations.jsonl")
        if clean_text(record.get("relation_id"))
    } if (gold_bank_dir / "commercial_relations.jsonl").exists() else {}
    for record in qa_records:
        record["evidence_context"] = [
            evidence_lookup[evidence_id]
            for evidence_id in record.get("evidence_refs", [])
            if evidence_id in evidence_lookup
        ]
        record["graph_context"] = {
            "commerce_cues": [
                {
                    key: cue_lookup[cue_id][key]
                    for key in ("cue_id", "cue_type", "content_en", "evidence_ids", "directness")
                    if key in cue_lookup[cue_id]
                }
                for cue_id in record.get("commerce_cue_ids", [])
                if cue_id in cue_lookup
            ],
            "commercial_relations": [
                {
                    key: relation_lookup[relation_id][key]
                    for key in (
                        "relation_id",
                        "relation_type",
                        "source_cue_ids",
                        "target_cue_ids",
                        "evidence_ids",
                        "status",
                        "rationale_en",
                        "provenance",
                    )
                    if key in relation_lookup[relation_id]
                }
                for relation_id in record.get("commercial_relation_ids", [])
                if relation_id in relation_lookup
            ],
        }

    qa_plan = [
        {
            "video_id": item.video_id,
            "annotation_id": item.annotation_id,
            "task_type": item.task_type.value,
            "task_subtype": item.task_subtype,
            "eligible_question_formats": list(item.eligible_question_formats),
            "spec_id": make_question_spec_id(
                item.video_id,
                item.annotation_id,
                item.capability or item.task_subtype,
            ),
            "capability": item.capability or item.task_subtype,
            "reasoning_operator": item.reasoning_operator,
        }
        for item in gold_items
    ]

    write_jsonl(output_dir / "qa_plan.jsonl", qa_plan)
    write_jsonl(output_dir / "qa_candidates.jsonl", qa_records)
    write_jsonl(output_dir / "qa_validation.jsonl", validation)
    write_jsonl(output_dir / "vqa_gold_private.jsonl", qa_records)
    public_records = [_public_qa(record) for record in qa_records]
    write_jsonl(output_dir / "vqa_public.jsonl", public_records)
    diversity = build_diversity_report(qa_records)
    write_json(output_dir / "qa_diversity.json", diversity)
    for task in policy.task_priority:
        write_jsonl(
            output_dir / "public" / f"{task.lower()}.jsonl",
            [record for record in public_records if record.get("task_type") == task],
        )
    meta = {
        "compiler_version": COMPILER_VERSION,
        "public_tasks": list(policy.task_priority),
        "bank_file": str(bank_path),
        "realizations_file": str(realizations_path) if realizations_path is not None else "legacy_question_programs",
        "counts": {
            "qa_plan": len(qa_plan),
            "qa_candidates": len(qa_records),
            "qa_validation": len(validation),
            "vqa_gold_private": len(qa_records),
            "vqa_public": len(qa_records),
            "per_task": task_counts,
        },
        "validation": {"missing_tasks": missing_tasks, "all_tasks_nonempty": not missing_tasks},
        "diversity": diversity,
        "inputs": {
            "evidence_units": len(read_jsonl(gold_bank_dir / "evidence_units.jsonl")) if (gold_bank_dir / "evidence_units.jsonl").exists() else 0,
        },
    }
    write_json(output_dir / "generation_meta.json", meta)
    return meta
