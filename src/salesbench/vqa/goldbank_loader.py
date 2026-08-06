"""Load compilable grounded annotations for deterministic QA compilation."""

from __future__ import annotations

from pathlib import Path

from ..goldbank.schema import GoldItem, GoldTier, parse_gold_item
from ..io_utils import read_jsonl


COMPILABLE_TIERS = {GoldTier.GOLD_A, GoldTier.GOLD_B}
COMPILABLE_STATUSES = {"verified", "human_accepted"}


def load_compilable_gold(path: Path) -> list[GoldItem]:
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
            if item.review_status not in COMPILABLE_STATUSES:
                continue
            items.append(item)
    return sorted(items, key=lambda item: (item.video_id, item.task_type.value, item.gold_id))
