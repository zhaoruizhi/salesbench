"""Markdown analysis helpers for SalesBench-QA baseline evaluations."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return clean_text(value)


def build_analysis_markdown(report: dict[str, Any], details: list[dict[str, Any]], model: str) -> str:
    metrics = report.get("metrics") or {}
    overall = metrics.get("macro_average") or metrics.get("overall") or {}
    per_task = metrics.get("per_task") or {}
    summary = report.get("summary") or {}
    low_score = [
        item for item in details
        if item.get("score") is not None and float(item.get("score")) <= 0.5
    ][:10]

    lines = [
        f"# SalesBench-QA Closed-Source Baseline Analysis: {model}",
        "",
        "## Overall Macro-Average",
        "",
        f"- Count: {overall.get('count', 0)}",
        f"- Strict Accuracy: {_fmt(overall.get('strict_accuracy'))}",
        f"- Relaxed Accuracy: {_fmt(overall.get('relaxed_accuracy'))}",
        f"- Judge failures: {summary.get('judge_failed_count', 0)}",
        f"- Total judge cost USD: {_fmt(summary.get('total_cost_usd'))}",
        "",
        "## Per-Task",
        "",
        "| Task | Count | Strict | Relaxed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task_type, block in sorted(per_task.items()):
        lines.append(
            f"| {task_type} | {block.get('count', 0)} | {_fmt(block.get('strict_accuracy'))} | {_fmt(block.get('relaxed_accuracy'))} |"
        )

    lines.extend(["", "## Low-Score Examples", ""])
    if not low_score:
        lines.append("- No examples with score <= 0.5.")
    for item in low_score:
        lines.append(f"- `{item.get('vqa_id')}` ({item.get('task_type')}, score={_fmt(item.get('score'))})")
        lines.append(f"  - Question: {clean_text(item.get('question'))}")
        lines.append(f"  - Model: {clean_text(item.get('model_output'))}")
        lines.append(f"  - Judge: {clean_text(item.get('reason'))}")

    return "\n".join(lines) + "\n"
