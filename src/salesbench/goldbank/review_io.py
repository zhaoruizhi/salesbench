"""Human review decision validation and EvidenceDataset merge helpers."""

from __future__ import annotations

from copy import deepcopy

from ..utils import clean_text
from .ontology import eligible_question_formats
from .schema import GoldTaskType, make_gold_id


VALID_DECISIONS = {"ACCEPT_INFERRED", "ACCEPT_GOLD_B", "REVISE", "REJECT"}
REQUIRED_FIELDS = {"review_item_id", "proposal_id", "decision", "reviewer_id", "reviewed_at", "reason_code"}


def validate_human_review_decisions(records: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for idx, record in enumerate(records):
        missing = sorted(field for field in REQUIRED_FIELDS if not clean_text(record.get(field)))
        if missing:
            errors.append(f"record {idx} missing fields: {missing}")
        decision = clean_text(record.get("decision")).upper()
        if decision not in VALID_DECISIONS:
            errors.append(f"record {idx} invalid decision: {decision}")
        if decision == "REVISE":
            if not isinstance(record.get("revised_value"), dict):
                errors.append(f"record {idx} REVISE requires revised_value")
            if not isinstance(record.get("revised_evidence_ids"), list):
                errors.append(f"record {idx} REVISE requires revised_evidence_ids")
        review_item_id = clean_text(record.get("review_item_id"))
        if review_item_id in seen:
            errors.append(f"duplicate review_item_id: {review_item_id}")
        seen.add(review_item_id)
    return errors


def _proposal_to_gold_item(queue_item: dict[str, object], decision: dict[str, object]) -> dict[str, object]:
    proposal = dict(queue_item.get("proposal") or {})
    if not proposal:
        raise ValueError(f"review item {queue_item.get('review_item_id')} has no proposal payload")
    task_type = GoldTaskType(clean_text(proposal.get("task_type")).upper())
    task_subtype = clean_text(proposal.get("task_subtype")).upper()
    target = dict(proposal.get("target") or {})
    gold_value = (
        dict(decision.get("revised_value") or {})
        if clean_text(decision.get("decision")).upper() == "REVISE"
        else dict(proposal.get("proposed_gold") or {})
    )
    evidence_ids = (
        [clean_text(value) for value in decision.get("revised_evidence_ids") or []]
        if clean_text(decision.get("decision")).upper() == "REVISE"
        else [clean_text(value) for value in proposal.get("evidence_ids") or []]
    )
    return {
        "annotation_id": make_gold_id(clean_text(proposal.get("video_id")), task_type, task_subtype, target, gold_value),
        "video_id": clean_text(proposal.get("video_id")),
        "task_type": task_type.value,
        "task_subtype": task_subtype,
        "target": target,
        "gold_value": gold_value,
        "evidence_refs": evidence_ids,
        "reasoning_edges": list(proposal.get("reasoning_edges") or []),
        "eligible_question_formats": list(eligible_question_formats(task_type, task_subtype)),
        "source_proposal_ids": [clean_text(proposal.get("proposal_id"))],
        "quality_status": "INFERRED",
        "review_status": "human_accepted",
        "confidence": float(proposal.get("proposal_confidence", 0.8) or 0.8),
        "human_review": {
            "review_item_id": clean_text(decision.get("review_item_id")),
            "reviewer_id": clean_text(decision.get("reviewer_id")),
            "reviewed_at": clean_text(decision.get("reviewed_at")),
            "reason_code": clean_text(decision.get("reason_code")),
        },
    }


def apply_human_reviews(
    video_gold_records: list[dict[str, object]],
    review_queue: list[dict[str, object]],
    decisions: list[dict[str, object]],
) -> list[dict[str, object]]:
    errors = validate_human_review_decisions(decisions)
    if errors:
        raise ValueError("; ".join(errors))

    queue_by_id = {clean_text(item.get("review_item_id")): item for item in review_queue}
    records_by_video = {clean_text(record.get("video_id")): deepcopy(record) for record in video_gold_records}
    for decision in decisions:
        review_item_id = clean_text(decision.get("review_item_id"))
        if review_item_id not in queue_by_id:
            raise ValueError(f"Unknown review_item_id: {review_item_id}")
        decision_type = clean_text(decision.get("decision")).upper()
        if decision_type == "REJECT":
            continue
        item = _proposal_to_gold_item(queue_by_id[review_item_id], decision)
        video_id = clean_text(item.get("video_id"))
        if video_id not in records_by_video:
            records_by_video[video_id] = {
                "video_id": video_id,
                "schema_version": "evidence-dataset-schema-v3",
                "evidence_unit_ids": [],
                "commerce_cue_ids": [],
                "commercial_relation_ids": [],
                "grounded_annotations": [],
                "coverage": {},
                "quality_summary": {},
                "observation_scope": {},
            }
        annotations = records_by_video[video_id].get("grounded_annotations")
        if annotations is None:
            annotations = records_by_video[video_id].pop("gold_items", [])
            records_by_video[video_id]["grounded_annotations"] = annotations
        annotations.append(item)

    for record in records_by_video.values():
        annotations = record.get("grounded_annotations")
        if annotations is None:
            annotations = record.pop("gold_items", [])
        record.pop("private_interaction_ref", None)
        record["grounded_annotations"] = sorted(
            annotations,
            key=lambda item: clean_text(item.get("annotation_id")) or clean_text(item.get("gold_id")),
        )
    return [records_by_video[video_id] for video_id in sorted(records_by_video)]
