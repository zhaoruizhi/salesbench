from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _numeric_stem(path: Path) -> str | None:
    stem = path.stem.strip()
    return stem if stem.isdigit() else None


def build_asset_index(video_dir: Path, sales_dir: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, object]]:
    asset_index: defaultdict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"video_paths": [], "sales_paths": []}
    )
    unmatched_video_files: list[str] = []
    unmatched_sales_files: list[str] = []

    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        video_id = _numeric_stem(path)
        if video_id is None:
            unmatched_video_files.append(str(path))
            continue
        asset_index[video_id]["video_paths"].append(str(path))

    for path in sorted(sales_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        video_id = _numeric_stem(path)
        if video_id is None:
            unmatched_sales_files.append(str(path))
            continue
        asset_index[video_id]["sales_paths"].append(str(path))

    normalized_index: dict[str, dict[str, list[str]]] = {}
    asset_rows: list[dict[str, object]] = []
    for video_id in sorted(asset_index):
        entry = asset_index[video_id]
        entry["video_paths"].sort()
        entry["sales_paths"].sort()
        normalized_index[video_id] = {
            "video_paths": list(entry["video_paths"]),
            "sales_paths": list(entry["sales_paths"]),
        }
        asset_rows.append(
            {
                "video_id": video_id,
                "video_paths": list(entry["video_paths"]),
                "video_asset_count": len(entry["video_paths"]),
                "sales_paths": list(entry["sales_paths"]),
                "sales_asset_count": len(entry["sales_paths"]),
            }
        )

    manifest = {
        "summary": {
            "video_ids_with_any_asset": len(normalized_index),
            "video_file_count": sum(len(entry["video_paths"]) for entry in normalized_index.values()),
            "sales_image_count": sum(len(entry["sales_paths"]) for entry in normalized_index.values()),
            "unmatched_video_file_count": len(unmatched_video_files),
            "unmatched_sales_file_count": len(unmatched_sales_files),
        },
        "assets": asset_rows,
        "unmatched": {
            "video_files": unmatched_video_files,
            "sales_files": unmatched_sales_files,
        },
    }
    return normalized_index, manifest
