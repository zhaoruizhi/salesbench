"""LLM judge invocation and response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from ..utils import clean_text, contains_cjk
from ..vlm.api_client import APICallResult, VLMClient
from ..vlm.response_parser import _try_parse_json
from .context import build_judge_payload
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from .schema import coerce_allowed_score, normalize_task_type


ANSWER_LINE_RE = re.compile(r"Answer\s*:\s*(1(?:\.0)?|0\.75|0\.5(?:0)?|0\.25|0(?:\.0)?)\b", re.IGNORECASE)


def _score_from_dimensions(correctness: float, grounding: float, completeness: float) -> float:
    weighted = 0.4 * correctness + 0.4 * grounding + 0.2 * completeness
    allowed = (0.0, 0.25, 0.5, 0.75, 1.0)
    score = min(allowed, key=lambda value: (abs(value - weighted), value))
    weakest_core = min(correctness, grounding)
    if weakest_core == 0.0:
        return min(score, 0.25)
    if weakest_core == 0.25:
        return min(score, 0.5)
    return score


def parse_judge_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON judge output, with E-VAds-style `Answer: 0.75` fallback."""
    text = raw_text.strip()
    parsed = _try_parse_json(text)
    if parsed:
        reported_score = coerce_allowed_score(parsed.get("score", parsed.get("Answer", parsed.get("answer"))))
        dimension_values: dict[str, float | None] = {}
        for name in ("correctness", "grounding", "completeness"):
            raw_value = parsed.get(name)
            dimension_values[name] = None if raw_value is None else coerce_allowed_score(raw_value)
        if all(value is not None for value in dimension_values.values()):
            score = _score_from_dimensions(
                dimension_values["correctness"],  # type: ignore[arg-type]
                dimension_values["grounding"],  # type: ignore[arg-type]
                dimension_values["completeness"],  # type: ignore[arg-type]
            )
        else:
            score = reported_score
        reason = clean_text(parsed.get("reason"))
        evidence_alignment = clean_text(parsed.get("evidence_alignment"))
        if contains_cjk((reason, evidence_alignment)):
            raise ValueError("Judge natural-language output must use English")
        return {
            "score": score,
            "reported_score": reported_score,
            **dimension_values,
            "reason": reason,
            "evidence_alignment": evidence_alignment,
        }

    match = ANSWER_LINE_RE.search(text)
    if not match:
        raise ValueError("judge response does not contain a valid score")
    score = coerce_allowed_score(match.group(1))
    reason = ""
    reason_match = re.search(r"Reason\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = clean_text(reason_match.group(1))
    if contains_cjk(reason):
        raise ValueError("Judge natural-language output must use English")
    return {
        "score": score,
        "reported_score": score,
        "correctness": None,
        "grounding": None,
        "completeness": None,
        "reason": reason,
        "evidence_alignment": "",
    }


def _failure_detail(gold_record: dict[str, Any], answer_record: dict[str, Any], result: APICallResult | None, error: str) -> dict[str, Any]:
    task_type = normalize_task_type(gold_record.get("task_type"))
    return {
        "vqa_id": clean_text(gold_record.get("vqa_id")),
        "video_id": clean_text(gold_record.get("video_id")),
        "task_type": task_type,
        "question": clean_text(gold_record.get("question")),
        "reference_answer": gold_record.get("gold_answer"),
        "model_output": answer_record.get("answer"),
        "score": None,
        "reported_score": None,
        "correctness": None,
        "grounding": None,
        "completeness": None,
        "reason": "",
        "evidence_alignment": "",
        "judge_success": False,
        "raw_response": result.raw_response if result else "",
        "error": error,
        "model": result.model if result else "",
        "input_tokens": result.input_tokens if result else 0,
        "output_tokens": result.output_tokens if result else 0,
        "latency_s": result.latency_s if result else 0.0,
        "cost_usd": result.cost_usd if result else 0.0,
    }


def judge_qa_item(
    gold_record: dict[str, Any],
    answer_record: dict[str, Any],
    client: VLMClient,
) -> dict[str, Any]:
    payload = build_judge_payload(gold_record, answer_record)
    result = client.call_text_only(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_text=build_judge_user_prompt(payload),
        response_format="json_object",
    )
    if not result.success:
        return _failure_detail(gold_record, answer_record, result, result.error or "judge_call_failed")
    try:
        parsed = parse_judge_response(result.raw_response)
    except Exception as exc:
        return _failure_detail(gold_record, answer_record, result, str(exc))

    task_type = normalize_task_type(gold_record.get("task_type"))
    return {
        "vqa_id": clean_text(gold_record.get("vqa_id")),
        "video_id": clean_text(gold_record.get("video_id")),
        "task_type": task_type,
        "question": clean_text(gold_record.get("question")),
        "reference_answer": gold_record.get("gold_answer"),
        "model_output": payload.get("model_output"),
        "score": parsed["score"],
        "reported_score": parsed.get("reported_score"),
        "correctness": parsed.get("correctness"),
        "grounding": parsed.get("grounding"),
        "completeness": parsed.get("completeness"),
        "reason": parsed.get("reason", ""),
        "evidence_alignment": parsed.get("evidence_alignment", ""),
        "judge_success": True,
        "raw_response": result.raw_response,
        "error": None,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": result.latency_s,
        "cost_usd": result.cost_usd,
    }


def dumps_for_debug(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
