"""Canonical review-queue adapter for v8 and historical EvidenceDataset runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


ProposalIndex = Mapping[tuple[str, str], dict[str, Any]]


def _strings(values: object) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_dict(*values: object) -> dict[str, Any]:
    for value in values:
        payload = _dict(value)
        if payload:
            return payload
    return {}


def _first_list(*values: object) -> list[Any]:
    for value in values:
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    return []


def _reason_code(reason: str, explicit: object = "") -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip().upper()
    prefix = reason.split(":", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]*", prefix or ""):
        return prefix.replace(" ", "_").upper()
    return "REVIEW_REQUIRED"


def _infer_stage(row: dict[str, Any], reason_code: str, item_type: str) -> str:
    if str(row.get("stage") or "").strip():
        return str(row["stage"])
    review_id = str(row.get("review_item_id") or "").lower()
    if "adjudicator" in review_id or reason_code.startswith("ADJUDICATOR"):
        return "adjudication"
    if reason_code.startswith("CHALLENGE") or reason_code.startswith("CHALLENGER"):
        return "challenge"
    if reason_code in {"VALIDATION_FAILED", "CONFLICTING_VALUE", "SEMANTIC_DUPLICATE"}:
        return "validation"
    if item_type == "stage_failure" and "adjudication" in reason_code.lower():
        return "adjudication"
    return "proposal"


def _infer_item_type(row: dict[str, Any], reason_code: str) -> str:
    explicit = str(row.get("item_type") or "").strip()
    if explicit:
        return explicit
    if isinstance(row.get("abstention"), dict) or reason_code == "ABSTENTION":
        return "abstention"
    issue_codes = {
        str(issue.get("code") or "")
        for issue in row.get("issues") or []
        if isinstance(issue, dict)
    }
    if reason_code in {"CONFLICTING_VALUE", "SEMANTIC_DUPLICATE"} or issue_codes & {
        "CONFLICTING_VALUE",
        "DUPLICATE_SEMANTICS",
    }:
        return "conflict"
    if reason_code.endswith("_FAILED") and not row.get("proposal") and not row.get("candidate_gold"):
        return "stage_failure"
    return "candidate"


def normalize_review_queue_row(
    row: dict[str, Any],
    proposal_index: ProposalIndex,
) -> dict[str, Any]:
    """Return one readable canonical queue item from any v6-v8 queue shape."""
    video_id = str(row.get("video_id") or "")
    candidate_snapshot = _first_dict(row.get("candidate_snapshot"), row.get("proposal"))
    nested = _first_dict(row.get("proposal"), candidate_snapshot)
    abstention = _dict(row.get("abstention"))

    source_ids = _strings(row.get("source_proposal_ids"))
    if not source_ids:
        source_ids = _strings(nested.get("source_proposal_ids"))
    nested_proposal_id = str(nested.get("proposal_id") or "").strip()
    row_proposal_id = str(row.get("proposal_id") or "").strip()
    if not source_ids and nested_proposal_id:
        source_ids = [nested_proposal_id]
    if not source_ids and row_proposal_id and row_proposal_id != "stage":
        source_ids = [row_proposal_id]

    linked = [proposal_index[(video_id, proposal_id)] for proposal_id in source_ids if (video_id, proposal_id) in proposal_index]
    primary = _first_dict(nested, linked[0] if linked else {})
    task_source = _first_dict(
        {"task_type": row.get("task_type"), "task_subtype": row.get("task_subtype")}
        if row.get("task_type") or row.get("task_subtype")
        else {},
        abstention,
        nested,
        linked[0] if linked else {},
    )

    target = _first_dict(row.get("target"), nested.get("target"), primary.get("target"))
    candidate_gold = _first_dict(
        row.get("candidate_gold"),
        row.get("gold_value"),
        row.get("proposed_gold"),
        nested.get("gold_value"),
        nested.get("proposed_gold"),
        primary.get("gold_value"),
        primary.get("proposed_gold"),
    )
    evidence_refs = _strings(
        _first_list(
            row.get("evidence_refs"),
            row.get("evidence_ids"),
            nested.get("evidence_refs"),
            nested.get("evidence_ids"),
            primary.get("evidence_refs"),
            primary.get("evidence_ids"),
        )
    )
    if not evidence_refs and linked:
        evidence_refs = _strings(
            [evidence_id for proposal in linked for evidence_id in proposal.get("evidence_ids") or []]
        )
    commerce_cue_ids = _strings(
        _first_list(
            row.get("commerce_cue_ids"),
            nested.get("commerce_cue_ids"),
            primary.get("commerce_cue_ids"),
        )
    )
    commercial_relation_ids = _strings(
        _first_list(
            row.get("commercial_relation_ids"),
            nested.get("commercial_relation_ids"),
            primary.get("commercial_relation_ids"),
        )
    )
    if not commerce_cue_ids and candidate_snapshot:
        commerce_cue_ids = _strings(
            [
                *(candidate_snapshot.get("source_cue_ids") or []),
                *(candidate_snapshot.get("target_cue_ids") or []),
            ]
        )
    if not commercial_relation_ids and candidate_snapshot.get("relation_id"):
        commercial_relation_ids = [str(candidate_snapshot["relation_id"])]
    if linked:
        if not commerce_cue_ids:
            commerce_cue_ids = _strings(
                [cue_id for proposal in linked for cue_id in proposal.get("commerce_cue_ids") or []]
            )
        if not commercial_relation_ids:
            commercial_relation_ids = _strings(
                [
                    relation_id
                    for proposal in linked
                    for relation_id in proposal.get("commercial_relation_ids") or []
                ]
            )

    raw_reason = str(row.get("reason") or "").strip()
    reason = str(abstention.get("reason") or raw_reason).strip()
    reason_code = _reason_code(raw_reason, row.get("reason_code"))
    item_type = _infer_item_type(row, reason_code)
    stage = _infer_stage(row, reason_code, item_type)

    unresolved = [proposal_id for proposal_id in source_ids if (video_id, proposal_id) not in proposal_index]
    has_embedded_candidate = bool(nested or target or candidate_gold or evidence_refs)
    if item_type == "abstention":
        resolution_status = "not_applicable"
        display_summary = "No candidate generated"
    elif unresolved and not has_embedded_candidate:
        resolution_status = "unresolved"
        display_summary = "Candidate source could not be resolved"
    elif not target and not candidate_gold and not primary:
        resolution_status = "diagnostic"
        display_summary = "Pipeline diagnostic record"
    else:
        resolution_status = "resolved"
        display_summary = "Candidate requires review"

    source_candidates = []
    candidates = linked or ([nested] if nested else [])
    for candidate in candidates:
        source_candidates.append(
            {
                "proposal_id": candidate.get("proposal_id") or candidate.get("annotation_id") or candidate.get("gold_id"),
                "task_type": candidate.get("task_type"),
                "task_subtype": candidate.get("task_subtype"),
                "target": candidate.get("target") or {},
                "candidate_gold": candidate.get("gold_value") or candidate.get("proposed_gold") or {},
                "evidence_refs": candidate.get("evidence_refs") or candidate.get("evidence_ids") or [],
                "commerce_cue_ids": candidate.get("commerce_cue_ids") or [],
                "commercial_relation_ids": candidate.get("commercial_relation_ids") or [],
                "capability": candidate.get("capability") or candidate.get("task_subtype"),
                "reasoning_operator": candidate.get("reasoning_operator") or "",
                "question_intent": candidate.get("question_intent") or "",
                "confidence": candidate.get("confidence", candidate.get("proposal_confidence")),
            }
        )

    return {
        "review_item_id": str(row.get("review_item_id") or row.get("id") or ""),
        "video_id": video_id,
        "stage": stage,
        "item_type": item_type,
        "task_type": str(task_source.get("task_type") or ""),
        "task_subtype": str(task_source.get("task_subtype") or ""),
        "reason_code": reason_code,
        "reason": reason,
        "target": target,
        "candidate_gold": candidate_gold,
        "evidence_refs": evidence_refs,
        "commerce_cue_ids": commerce_cue_ids,
        "commercial_relation_ids": commercial_relation_ids,
        "capability": str(primary.get("capability") or task_source.get("task_subtype") or ""),
        "reasoning_operator": str(primary.get("reasoning_operator") or ""),
        "question_intent": str(primary.get("question_intent") or ""),
        "source_proposal_ids": source_ids,
        "issues": list(row.get("issues") or []),
        "source_candidates": source_candidates,
        "resolution_status": resolution_status,
        "unresolved_proposal_ids": unresolved,
        "display_summary": display_summary,
        "proposal_id": row_proposal_id or nested_proposal_id or (source_ids[0] if len(source_ids) == 1 else ""),
        "candidate_snapshot": candidate_snapshot,
        "quality_disposition": str(row.get("quality_disposition") or ""),
        "semantic_verifier": _dict(candidate_snapshot.get("semantic_verifier")),
    }
