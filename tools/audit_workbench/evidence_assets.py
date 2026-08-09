from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _resolve_frame_path(raw_path: object, repo_root: Path, manifest_dir: Path) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (repo_root / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (manifest_dir / path.name).resolve()


def load_frame_manifests(
    frame_cache_root: Path,
    video_ids: set[str],
    *,
    repo_root: Path,
) -> dict[str, dict[int, dict[str, Any]]]:
    manifests: dict[str, dict[int, dict[str, Any]]] = {}
    for video_id in sorted(video_ids):
        manifest_path = frame_cache_root / video_id / "manifest.json"
        if not manifest_path.exists():
            manifests[video_id] = {}
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames: dict[int, dict[str, Any]] = {}
        for raw in payload.get("frames") or []:
            if not isinstance(raw, dict) or raw.get("frame_index") is None:
                continue
            index = int(raw["frame_index"])
            frames[index] = {
                "frame_index": index,
                "timestamp_s": raw.get("timestamp_s"),
                "source_path": str(_resolve_frame_path(raw.get("path"), repo_root, manifest_path.parent)),
            }
        manifests[video_id] = frames
    return manifests


def _frame_descriptor(frame: dict[str, Any], relation: str) -> dict[str, Any]:
    return {
        "frame_index": frame["frame_index"],
        "timestamp_s": frame.get("timestamp_s"),
        "source_path": frame.get("source_path", ""),
        "relation": relation,
    }


def _representative_frames(frames: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = [frames[index] for index in sorted(frames)]
    if not ordered:
        return []
    positions = sorted({0, len(ordered) // 2, len(ordered) - 1})
    return [_frame_descriptor(ordered[position], "representative") for position in positions]


def _temporal_frames(
    frames: dict[int, dict[str, Any]], start_s: float, end_s: float
) -> list[dict[str, Any]]:
    timed = [frame for frame in frames.values() if frame.get("timestamp_s") is not None]
    if not timed:
        return []
    within = [frame for frame in timed if start_s <= float(frame["timestamp_s"]) <= end_s]
    if within:
        return [_frame_descriptor(frame, "temporal_nearest") for frame in sorted(within, key=lambda row: row["frame_index"])]
    midpoint = (start_s + end_s) / 2
    nearest = min(timed, key=lambda frame: abs(float(frame["timestamp_s"]) - midpoint))
    return [_frame_descriptor(nearest, "temporal_nearest")]


def enrich_evidence_refs(
    video_id: str,
    evidence_refs: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
    frame_manifests: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    video_frames = frame_manifests.get(video_id, {})
    if not evidence_refs:
        return [
            {
                "evidence_id": "",
                "video_id": video_id,
                "modality": "context",
                "semantic_text": "视频代表上下文",
                "text_span": "",
                "localization_note": "该审计项没有直接 Evidence 引用，以下为视频代表帧。",
                "frames": _representative_frames(video_frames),
            }
        ]
    for evidence_id in evidence_refs:
        unit = evidence_by_id.get(str(evidence_id))
        if unit is None:
            items.append(
                {
                    "evidence_id": str(evidence_id),
                    "missing": True,
                    "semantic_text": "EvidenceUnit 不存在",
                    "localization_note": "无法解析该 evidence_ref。",
                    "frames": [],
                }
            )
            continue
        modality = str(unit.get("modality") or "")
        frames: list[dict[str, Any]] = []
        note = ""
        if modality in {"visual", "ocr"}:
            for frame_index in unit.get("frame_indices") or []:
                frame = video_frames.get(int(frame_index))
                if frame:
                    frames.append(_frame_descriptor(frame, "direct"))
            if not frames:
                note = "该视觉/OCR Evidence 没有可读取的关联帧。"
        elif modality == "asr":
            start_s = unit.get("start_s")
            end_s = unit.get("end_s")
            if start_s is not None and end_s is not None:
                frames = _temporal_frames(video_frames, float(start_s), float(end_s))
                note = "以下帧按 ASR 时间范围关联。" if frames else "ASR 有时间戳，但帧缓存中没有可关联帧。"
            else:
                frames = _representative_frames(video_frames)
                note = "ASR 缺少时间戳，以下为视频代表帧，不能精确定位到该语音片段。"
        semantic_values = [unit.get("subject"), unit.get("predicate"), unit.get("value")]
        items.append(
            {
                "evidence_id": str(unit.get("evidence_id") or evidence_id),
                "video_id": str(unit.get("video_id") or video_id),
                "modality": modality,
                "subject": unit.get("subject"),
                "predicate": unit.get("predicate"),
                "value": unit.get("value"),
                "semantic_text": "｜".join(str(value) for value in semantic_values if value not in (None, "")),
                "text_span": unit.get("text_span") or "",
                "start_s": unit.get("start_s"),
                "end_s": unit.get("end_s"),
                "frame_indices": list(unit.get("frame_indices") or []),
                "confidence": unit.get("confidence"),
                "timestamp_status": unit.get("timestamp_status") or "",
                "localization_note": note,
                "frames": frames,
            }
        )
    return items
