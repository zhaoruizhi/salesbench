"""Cohort selection for EvidenceDataset pilot and alpha runs."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from ..utils import clean_text


DEFAULT_ALPHA_ANCHOR_IDS = (
    "7356138818094910761",
    "7356450539795746075",
    "7356924628301843727",
    "7357194715717963043",
    "7358612170411658550",
    "7359005328937159986",
    "7359538830657064192",
    "7359920990538779930",
    "7360291671562521891",
    "7360565413341613348",
    "7360594186715909416",
    "7362087837640019215",
    "7362426540593892642",
    "7362525521537781027",
    "7362856271365606683",
    "7363104048272149775",
    "7363168161673399592",
    "7363930497476611355",
    "7364201585741712667",
    "7365054535036849420",
)

PRODUCT_ORDER = ("食品", "家居家电", "服饰", "个护美妆", "其他")
FAN_ORDER = ("头部", "头腰", "腰部", "尾部")


def _creator(record: dict[str, Any]) -> str:
    return clean_text(record.get("douyin_handle")) or clean_text(record.get("author_name")) or clean_text(record.get("video_id"))


def _eligible(record: dict[str, Any], require_video_asset: bool) -> bool:
    if not clean_text(record.get("video_id")):
        return False
    if not clean_text(record.get("product_bucket")) or not clean_text(record.get("fan_segment")):
        return False
    if require_video_asset and not bool(record.get("has_video_asset")):
        return False
    return bool(clean_text(record.get("title")) or clean_text(record.get("video_text")))


def _cell(record: dict[str, Any]) -> tuple[str, str]:
    return clean_text(record.get("product_bucket")), clean_text(record.get("fan_segment"))


def _append_candidate(
    selected: list[str],
    selected_set: set[str],
    used_creators: set[str],
    lookup: dict[str, dict[str, Any]],
    video_id: str,
    unique_creators: bool,
) -> bool:
    if video_id in selected_set:
        return False
    creator = _creator(lookup[video_id])
    if unique_creators and creator in used_creators:
        return False
    selected.append(video_id)
    selected_set.add(video_id)
    used_creators.add(creator)
    return True


def select_goldbank_cohort(
    records: list[dict[str, object]],
    total: int = 128,
    anchor_ids: list[str] | tuple[str, ...] = DEFAULT_ALPHA_ANCHOR_IDS,
    seed: int = 42,
    require_video_asset: bool = True,
    unique_creators: bool = True,
) -> dict[str, object]:
    eligible = [record for record in records if _eligible(record, require_video_asset)]
    lookup = {clean_text(record.get("video_id")): record for record in eligible}
    selected: list[str] = []
    selected_set: set[str] = set()
    used_creators: set[str] = set()
    warnings: list[str] = []

    for anchor_id in anchor_ids:
        video_id = clean_text(anchor_id)
        if video_id not in lookup:
            warnings.append(f"missing_anchor:{video_id}")
            continue
        _append_candidate(selected, selected_set, used_creators, lookup, video_id, unique_creators=False)
        if len(selected) >= total:
            break

    by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in eligible:
        by_cell[_cell(record)].append(clean_text(record.get("video_id")))
    rng = random.Random(seed)
    for ids in by_cell.values():
        ids.sort()
        rng.shuffle(ids)

    cells = [(product, fan) for product in PRODUCT_ORDER for fan in FAN_ORDER if (product, fan) in by_cell]
    fallback_cells = sorted(cell for cell in by_cell if cell not in cells)
    cells.extend(fallback_cells)

    passes = (True, False) if unique_creators else (False,)
    for enforce_unique in passes:
        made_progress = True
        while len(selected) < total and made_progress:
            made_progress = False
            for cell in cells:
                for video_id in list(by_cell[cell]):
                    if _append_candidate(selected, selected_set, used_creators, lookup, video_id, enforce_unique):
                        made_progress = True
                        break
                if len(selected) >= total:
                    break

    selected_records = [lookup[video_id] for video_id in selected]
    cell_counts = Counter("|".join(_cell(record)) for record in selected_records)
    return {
        "version": "evidence-alpha-v2",
        "video_ids": selected,
        "frame_strategy": "hook_plus_uniform",
        "frames_per_video": 16,
        "prompt_version": "evidence-prompt-v2",
        "schema_version": "evidence-dataset-schema-v2",
        "min_confidence": 0.70,
        "seed": seed,
        "selection": {
            "target_total": total,
            "actual_total": len(selected),
            "anchor_count": sum(1 for video_id in selected if video_id in set(anchor_ids)),
            "require_video_asset": require_video_asset,
            "unique_creators_preferred": unique_creators,
            "cell_counts": dict(sorted(cell_counts.items())),
            "warnings": warnings,
        },
    }
