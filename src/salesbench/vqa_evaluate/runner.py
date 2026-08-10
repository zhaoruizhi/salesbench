"""Runner for SalesBench-QA LLM-as-Judge evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
from typing import Any

from ..io_utils import read_records, write_json, write_jsonl
from ..utils import clean_text
from ..vlm.api_client import VLMClient
from .judge import judge_qa_item
from .metrics import aggregate_judge_metrics
from .prompts import JUDGE_PROMPT_VERSION
from .schema import TASK_TYPES


def _salesbench_gold(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if clean_text(record.get("task_layer")) == "salesbench_qa"
    ]
    invalid = sorted(
        {
            clean_text(record.get("task_type")).upper()
            for record in selected
            if clean_text(record.get("task_type")).upper() not in TASK_TYPES
        }
    )
    if invalid:
        raise ValueError(f"Unsupported public VQA task types: {invalid}")
    return selected


def _answer_lookup(answer_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in answer_records:
        vqa_id = clean_text(record.get("vqa_id"))
        if not vqa_id:
            continue
        if vqa_id in lookup:
            raise ValueError(f"Duplicate prediction vqa_id: {vqa_id}")
        lookup[vqa_id] = record
    return lookup


def evaluate_salesbench_qa_records(
    gold_records: list[dict[str, Any]],
    answer_records: list[dict[str, Any]],
    client: VLMClient,
    max_workers: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_items = _salesbench_gold(gold_records)
    answers = _answer_lookup(answer_records)

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_details: list[dict[str, Any]] = []
    skipped_gold_count = 0
    for gold in gold_items:
        vqa_id = clean_text(gold.get("vqa_id"))
        if not vqa_id:
            skipped_gold_count += 1
            continue
        if vqa_id not in answers:
            missing_details.append(
                {
                    "vqa_id": vqa_id,
                    "video_id": clean_text(gold.get("video_id")),
                    "task_type": clean_text(gold.get("task_type")).upper(),
                    "question": clean_text(gold.get("question")),
                    "reference_answer": gold.get("gold_answer"),
                    "model_output": "",
                    "score": 0.0,
                    "reported_score": 0.0,
                    "correctness": 0.0,
                    "grounding": 0.0,
                    "completeness": 0.0,
                    "error_tags": ["UNANSWERED"],
                    "reason": "missing_model_answer",
                    "evidence_alignment": "",
                    "judge_success": True,
                    "raw_response": "",
                    "error": "missing_model_answer",
                    "model": "local_missing_answer_rule",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0.0,
                    "cost_usd": 0.0,
                }
            )
            continue
        matched.append((gold, answers[vqa_id]))

    details: list[dict[str, Any]] = list(missing_details)
    completed = 0
    lock = threading.Lock()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(judge_qa_item, gold, answer, client): clean_text(gold.get("vqa_id"))
            for gold, answer in matched
        }
        for future in as_completed(futures):
            vqa_id = futures[future]
            try:
                detail = future.result()
            except Exception as exc:
                detail = {
                    "vqa_id": vqa_id,
                    "task_type": "",
                    "score": None,
                    "judge_success": False,
                    "error_tags": [],
                    "error": str(exc),
                    "cost_usd": 0.0,
                }
            details.append(detail)
            with lock:
                completed += 1
                if completed % 20 == 0 or completed == len(matched):
                    elapsed = time.time() - start_time
                    print(f"  [{completed}/{len(matched)}] judged={completed} elapsed={elapsed/60:.1f}min")

    details.sort(key=lambda record: clean_text(record.get("vqa_id")))
    report = aggregate_judge_metrics(
        details,
        gold_count=len(gold_items),
        answer_count=len(answer_records),
        matched_answer_count=len(matched),
        skipped_gold_count=skipped_gold_count,
        missing_answer_count=len(missing_details),
    )
    return report, details


def evaluate_salesbench_qa_files(
    gold_path: Path,
    answers_path: Path,
    output_dir: Path,
    api_key: str | None = None,
    judge_model: str = "gpt-4o",
    base_url: str | None = None,
    max_workers: int = 4,
    client: VLMClient | None = None,
) -> dict[str, Any]:
    if client is None:
        if not api_key:
            raise ValueError("api_key is required when client is not provided")
        client = VLMClient(
            api_key=api_key,
            model=judge_model,
            base_url=base_url,
            temperature=0.0,
            max_tokens=700,
            rate_limit_rpm=0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report, details = evaluate_salesbench_qa_records(
        gold_records=read_records(gold_path),
        answer_records=read_records(answers_path),
        client=client,
        max_workers=max_workers,
    )

    details_path = output_dir / f"{answers_path.stem}_judge_details.jsonl"
    report_path = output_dir / f"{answers_path.stem}_salesbench_qa_eval.json"
    write_jsonl(details_path, details)
    report["sources"] = {
        "gold": str(gold_path),
        "answers": str(answers_path),
    }
    report["outputs"] = {
        "judge_details": str(details_path),
        "summary": str(report_path),
    }
    report["judge_model"] = getattr(client, "model", judge_model)
    report["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    write_json(report_path, report)
    return report
