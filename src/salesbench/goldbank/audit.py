"""EvidenceDataset quality audit helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..utils import clean_text
from .schema import GoldItem, GoldTier, parse_gold_item
from .validators import PRIVATE_KEYS, find_duplicate_and_conflicting_items


def _contains_private(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(clean_text(key) in PRIVATE_KEYS or _contains_private(value) for key, value in payload.items())
    if isinstance(payload, list):
        return any(_contains_private(value) for value in payload)
    return False


def _evidence_lookup(evidence: list[dict[str, object]]) -> dict[str, str]:
    return {
        clean_text(record.get("evidence_id")): clean_text(record.get("video_id"))
        for record in evidence
        if clean_text(record.get("evidence_id"))
    }


def _parse_items(records: list[dict[str, object]]) -> list[GoldItem]:
    items: list[GoldItem] = []
    for record in records:
        annotations = record.get("grounded_annotations")
        if annotations is None:
            annotations = record.get("gold_items", [])
        for raw_item in annotations or []:
            if isinstance(raw_item, dict) and raw_item.get("video_id"):
                try:
                    items.append(parse_gold_item(raw_item))
                except ValueError:
                    continue
    return items


def audit_gold_bank(records: list[dict[str, object]], evidence: list[dict[str, object]]) -> dict[str, object]:
    items = _parse_items(records)
    evidence_by_id = _evidence_lookup(evidence)
    task_counts = Counter(item.task_type.value for item in items)
    tier_counts = Counter(item.gold_tier.value for item in items)
    valid_evidence_items = 0
    cross_modal_items = 0
    for item in items:
        refs = list(item.evidence_ids)
        if refs and all(ref in evidence_by_id and evidence_by_id[ref] == item.video_id for ref in refs):
            valid_evidence_items += 1
        if item.task_type.value == "CM" and len(set(refs)) >= 2:
            cross_modal_items += 1

    duplicate_conflict = find_duplicate_and_conflicting_items(items)
    duplicate_count = sum(1 for issue in duplicate_conflict if issue.code == "DUPLICATE_SEMANTICS")
    conflict_count = sum(1 for issue in duplicate_conflict if issue.code == "CONFLICTING_VALUE")
    video_ids = [clean_text(record.get("video_id")) for record in records if clean_text(record.get("video_id"))]
    by_video_task = {(item.video_id, item.task_type.value) for item in items}
    missing_required = [
        video_id
        for video_id in video_ids
        if (video_id, "BP") not in by_video_task or (video_id, "CM") not in by_video_task
    ]
    private_leakage = 0
    for record in records:
        if _contains_private(record):
            private_leakage += 1
    item_count = len(items)
    return {
        "video_count": len(video_ids),
        "item_count": item_count,
        "item_count_by_task": dict(task_counts),
        "item_count_by_quality": {
            "DIRECT": tier_counts.get(GoldTier.GOLD_A.value, 0),
            "INFERRED": tier_counts.get(GoldTier.GOLD_B.value, 0),
            "NEEDS_REVIEW": tier_counts.get(GoldTier.SILVER.value, 0),
            "REJECTED": tier_counts.get(GoldTier.REJECTED.value, 0),
        },
        "evidence_coverage": round(valid_evidence_items / item_count, 6) if item_count else 0.0,
        "cross_modal_coverage": round(cross_modal_items / max(task_counts.get("CM", 0), 1), 6) if task_counts.get("CM", 0) else 0.0,
        "duplicate_rate": round(duplicate_count / item_count, 6) if item_count else 0.0,
        "conflict_rate": round(conflict_count / item_count, 6) if item_count else 0.0,
        "needs_review_rate": round(tier_counts.get(GoldTier.SILVER.value, 0) / item_count, 6) if item_count else 0.0,
        "human_review_rate": round(sum(1 for item in items if item.review_status == "human_accepted") / item_count, 6) if item_count else 0.0,
        "abstention_count": sum(len(record.get("abstentions", []) or []) for record in records),
        "videos_missing_required_tasks": missing_required,
        "private_field_leakage": private_leakage,
    }
