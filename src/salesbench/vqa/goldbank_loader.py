"""Load compilable grounded annotations for deterministic QA compilation."""

from __future__ import annotations

from pathlib import Path

from ..goldbank.schema import GoldItem, GoldTier, parse_gold_item
from ..io_utils import read_jsonl


COMPILABLE_TIERS = {GoldTier.GOLD_A, GoldTier.GOLD_B}
FORMAL_COMPILABLE_STATUSES = {"human_accepted"}
CANDIDATE_COMPILABLE_STATUSES = {"auto_accepted_candidate", "verified"}


def load_compilable_gold(path: Path, *, allow_auto_candidates: bool = False) -> list[GoldItem]:
    statuses = set(FORMAL_COMPILABLE_STATUSES)
    if allow_auto_candidates:
        statuses.update(CANDIDATE_COMPILABLE_STATUSES)
    items: list[GoldItem] = []
    for record in read_jsonl(path):
        annotations = record.get("grounded_annotations")
        if annotations is None:
            annotations = record.get("gold_items", [])
        for raw_item in annotations or []:
            if not isinstance(raw_item, dict):
                continue
            item = parse_gold_item(raw_item)
            if item.gold_tier in {GoldTier.SILVER, GoldTier.REJECTED}:
                continue
            if item.gold_tier not in COMPILABLE_TIERS:
                raise ValueError(f"Unsupported quality tier in EvidenceDataset: {item.gold_tier}")
            if item.review_status not in statuses:
                continue
            items.append(item)
    return sorted(items, key=lambda item: (item.video_id, item.task_type.value, item.gold_id))
