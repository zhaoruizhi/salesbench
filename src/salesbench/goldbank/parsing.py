"""Strict JSON parsing for Evidence-First model calls."""

from __future__ import annotations

import json


class ModelOutputError(ValueError):
    """Raised when a model response violates the JSON contract."""


def parse_json_object(raw_response: str, required_key: str) -> dict[str, object]:
    raw = (raw_response or "").strip()
    if not raw:
        raise ModelOutputError("empty model response")
    if raw.startswith("```") or raw.endswith("```"):
        raise ModelOutputError("markdown fenced JSON is not accepted")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("top-level JSON must be an object")
    if required_key not in payload:
        raise ModelOutputError(f"missing top-level key: {required_key}")
    return payload


def parse_evidence_response(raw_response: str) -> list[dict[str, object]]:
    payload = parse_json_object(raw_response, "evidence_units")
    units = payload["evidence_units"]
    if not isinstance(units, list):
        raise ModelOutputError("evidence_units must be a list")
    return [unit for unit in units if isinstance(unit, dict)]


def parse_proposal_response(raw_response: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = parse_json_object(raw_response, "proposals")
    proposals = payload["proposals"]
    abstentions = payload.get("abstentions", [])
    if not isinstance(proposals, list) or not isinstance(abstentions, list):
        raise ModelOutputError("proposals and abstentions must be lists")
    return [item for item in proposals if isinstance(item, dict)], [item for item in abstentions if isinstance(item, dict)]


def parse_review_response(raw_response: str) -> list[dict[str, object]]:
    payload = parse_json_object(raw_response, "reviews")
    reviews = payload["reviews"]
    if not isinstance(reviews, list):
        raise ModelOutputError("reviews must be a list")
    return [review for review in reviews if isinstance(review, dict)]


def parse_adjudication_response(raw_response: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw = (raw_response or "").strip()
    if not raw:
        raise ModelOutputError("empty model response")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ModelOutputError("top-level JSON must be an object")
    if "accepted_groups" in decoded:
        groups = decoded["accepted_groups"]
        queue = decoded.get("human_review_queue", [])
        if not isinstance(groups, list) or not isinstance(queue, list):
            raise ModelOutputError("accepted_groups and human_review_queue must be lists")
        normalized_groups = []
        for item in groups:
            if isinstance(item, dict):
                normalized_groups.append({**item, "_decision_only": True})
        return normalized_groups, [item for item in queue if isinstance(item, dict)]

    # Keep legacy v6 replay readable. New prompts never request this format.
    payload = parse_json_object(raw_response, "grounded_annotations")
    queue = payload.get("human_review_queue", [])
    if not isinstance(payload["grounded_annotations"], list) or not isinstance(queue, list):
        raise ModelOutputError("grounded_annotations and human_review_queue must be lists")
    return [item for item in payload["grounded_annotations"] if isinstance(item, dict)], [item for item in queue if isinstance(item, dict)]
