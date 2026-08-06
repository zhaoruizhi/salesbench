"""Creator-disjoint split helpers for private interaction labels."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..utils import clean_text


@dataclass(frozen=True)
class SplitManifest:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    grouping_key: str
    seed: int
    warnings: tuple[str, ...] = ()

    def split_for(self, video_id: str) -> str:
        if video_id in self.train_ids:
            return "train"
        if video_id in self.val_ids:
            return "val"
        if video_id in self.test_ids:
            return "test"
        return "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "test_ids": list(self.test_ids),
            "grouping_key": self.grouping_key,
            "seed": self.seed,
            "warnings": list(self.warnings),
        }


def _creator_key(record: dict[str, Any], group_key: str) -> tuple[str, str]:
    preferred = clean_text(record.get(group_key))
    if preferred:
        return preferred, group_key
    handle = clean_text(record.get("douyin_handle"))
    if handle:
        return handle, "douyin_handle"
    author = clean_text(record.get("author_name"))
    if author:
        return author, "author_name"
    return clean_text(record.get("video_id")), "video_id_fallback"


def build_group_split_manifest(
    records: list[dict[str, object]],
    group_key: str = "douyin_handle",
    seed: int = 42,
) -> SplitManifest:
    groups: dict[str, list[str]] = {}
    used_keys: set[str] = set()
    warnings: list[str] = []
    for record in records:
        video_id = clean_text(record.get("video_id"))
        if not video_id:
            continue
        key, used = _creator_key(record, group_key)
        used_keys.add(used)
        if used == "video_id_fallback":
            warnings.append(f"video_id_fallback:{video_id}")
        groups.setdefault(key, []).append(video_id)

    group_items = sorted((key, tuple(sorted(ids))) for key, ids in groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    n = len(group_items)
    if n >= 3:
        train_count = max(1, min(n - 2, int(round(n * 0.8))))
        val_count = max(1, min(n - train_count - 1, int(round(n * 0.1))))
    elif n == 2:
        train_count, val_count = 1, 0
    elif n == 1:
        train_count, val_count = 1, 0
    else:
        train_count, val_count = 0, 0
    train_cut = train_count
    val_cut = train_cut + val_count
    train_groups = group_items[:train_cut]
    val_groups = group_items[train_cut:val_cut]
    test_groups = group_items[val_cut:]

    def flatten(items: list[tuple[str, tuple[str, ...]]]) -> tuple[str, ...]:
        return tuple(sorted(video_id for _, ids in items for video_id in ids))

    grouping_name = group_key if used_keys <= {group_key} else "+".join(sorted(used_keys))
    return SplitManifest(
        train_ids=flatten(train_groups),
        val_ids=flatten(val_groups),
        test_ids=flatten(test_groups),
        grouping_key=grouping_name,
        seed=seed,
        warnings=tuple(sorted(set(warnings))),
    )
