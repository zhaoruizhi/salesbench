"""Metric aggregation for SalesBench-QA judge scores."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text, mean
from .schema import TASK_TYPES, judge_success


def _score_values(records: list[dict[str, Any]]) -> list[float]:
    return [float(record["score"]) for record in records if judge_success(record)]


def _metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = _score_values(records)
    if not scores:
        return {
            "count": 0,
            "strict_accuracy": None,
            "relaxed_accuracy": None,
        }
    return {
        "count": len(scores),
        "strict_accuracy": sum(1 for score in scores if score == 1.0) / len(scores),
        "relaxed_accuracy": mean(scores),
    }


def aggregate_judge_metrics(
    details: list[dict[str, Any]],
    gold_count: int,
    answer_count: int,
    matched_answer_count: int,
    skipped_gold_count: int,
    missing_answer_count: int = 0,
) -> dict[str, Any]:
    scored = [record for record in details if judge_success(record)]
    failed = [record for record in details if not judge_success(record)]

    per_task = {
        task_type: _metric_block([record for record in details if clean_text(record.get("task_type")).upper() == task_type])
        for task_type in TASK_TYPES
    }
    per_task = {
        task_type: block
        for task_type, block in per_task.items()
        if block["count"] or any(clean_text(record.get("task_type")).upper() == task_type for record in details)
    }

    task_blocks = [per_task[task] for task in TASK_TYPES if task in per_task and per_task[task]["count"]]
    macro_average = {
        "task_count": len(task_blocks),
        "strict_accuracy": mean([block["strict_accuracy"] for block in task_blocks]) if task_blocks else None,
        "relaxed_accuracy": mean([block["relaxed_accuracy"] for block in task_blocks]) if task_blocks else None,
    }

    return {
        "summary": {
            "gold_count": gold_count,
            "answer_count": answer_count,
            "matched_answer_count": matched_answer_count,
            "skipped_gold_count": skipped_gold_count,
            "missing_answer_count": missing_answer_count,
            "scored_count": len(scored),
            "judge_failed_count": len(failed),
            "total_cost_usd": round(sum(float(record.get("cost_usd") or 0.0) for record in details), 6),
        },
        "metrics": {
            "macro_average": macro_average,
            "overall_micro_auxiliary": _metric_block(details),
            "overall": _metric_block(details),
            "per_task": per_task,
        },
    }
