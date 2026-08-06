"""LLM judge invocation and response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from ..utils import clean_text
from ..vlm.api_client import APICallResult, VLMClient
from ..vlm.response_parser import _try_parse_json
from .context import build_judge_payload
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from .schema import coerce_allowed_score, normalize_task_type


ANSWER_LINE_RE = re.compile(r"Answer\s*:\s*(1(?:\.0)?|0\.75|0\.5(?:0)?|0\.25|0(?:\.0)?)\b", re.IGNORECASE)


def parse_judge_response(raw_text: str) -> dict[str, Any]:
    """Parse JSON judge output, with E-VAds-style `Answer: 0.75` fallback."""
    text = raw_text.strip()
    parsed = _try_parse_json(text)
    if parsed:
        score = coerce_allowed_score(parsed.get("score", parsed.get("Answer", parsed.get("answer"))))
        return {
            "score": score,
            "reason": clean_text(parsed.get("reason")),
            "evidence_alignment": clean_text(parsed.get("evidence_alignment")),
        }

    match = ANSWER_LINE_RE.search(text)
    if not match:
        raise ValueError("judge response does not contain a valid score")
    score = coerce_allowed_score(match.group(1))
    reason = ""
    reason_match = re.search(r"Reason\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = clean_text(reason_match.group(1))
    return {
        "score": score,
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
