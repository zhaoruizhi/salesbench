"""Dataset runner and output writer for Evidence-First annotation generation."""

from __future__ import annotations

import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ..config import BenchmarkConfig
from ..io_utils import read_json, write_json, write_jsonl
from ..multiagent.context import SalesBenchContextStore, build_context_bundle
from ..utils import clean_text
from ..vlm.api_client import VLMClient
from ..vlm.frame_sampler import Frame, sample_frames
from .pipeline import GoldBankPipeline, GoldBankResult
from .schema import stable_digest


PIPELINE_VERSION = "evidence-first-pipeline-v6"


GOLD_BANK_OUTPUT_FILES = (
    "video_samples.jsonl",
    "evidence_units.jsonl",
    "gold_proposals.jsonl",
    "gold_reviews.jsonl",
    "video_evidence_dataset.jsonl",
    "human_review_queue.jsonl",
    "agent_traces.jsonl",
    "generation_meta.json",
)


def _part_dir(output_dir: Path, video_id: str) -> Path:
    return output_dir / ".parts" / video_id


def _part_result_path(output_dir: Path, video_id: str) -> Path:
    return _part_dir(output_dir, video_id) / "result.json"


def _result_to_payload(result: GoldBankResult, pipeline_fingerprint: str) -> dict[str, object]:
    return {
        "pipeline_fingerprint": pipeline_fingerprint,
        "video_id": result.video_id,
        "evidence_units": result.evidence_units,
        "gold_proposals": result.gold_proposals,
        "gold_reviews": result.gold_reviews,
        "video_gold_record": result.video_gold_record,
        "human_review_queue": result.human_review_queue,
        "agent_traces": result.agent_traces,
        "status": result.status,
    }


def _payload_to_result(payload: dict[str, object]) -> GoldBankResult:
    return GoldBankResult(
        video_id=clean_text(payload.get("video_id")),
        evidence_units=list(payload.get("evidence_units") or []),
        gold_proposals=list(payload.get("gold_proposals") or []),
        gold_reviews=list(payload.get("gold_reviews") or []),
        video_gold_record=payload.get("video_gold_record") if isinstance(payload.get("video_gold_record"), dict) else None,
        human_review_queue=list(payload.get("human_review_queue") or []),
        agent_traces=list(payload.get("agent_traces") or []),
        status=clean_text(payload.get("status")),
    )


def _write_part(output_dir: Path, result: GoldBankResult, pipeline_fingerprint: str) -> None:
    path = _part_result_path(output_dir, result.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, _result_to_payload(result, pipeline_fingerprint))


def _load_completed(
    output_dir: Path,
    video_ids: list[str],
    fingerprints: dict[str, str],
) -> dict[str, GoldBankResult]:
    completed: dict[str, GoldBankResult] = {}
    for video_id in video_ids:
        path = _part_result_path(output_dir, video_id)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if clean_text(payload.get("pipeline_fingerprint")) != fingerprints.get(video_id):
                continue
            if clean_text(payload.get("status")) != "ok":
                continue
            completed[video_id] = _payload_to_result(payload)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return completed


def _sample_record(record: dict[str, object], pilot_config: dict[str, object]) -> dict[str, object]:
    video_id = clean_text(record.get("video_id"))
    return {
        "video_id": video_id,
        "requested_order": int(pilot_config["video_ids"].index(video_id)) if video_id in pilot_config.get("video_ids", []) else -1,
        "frame_strategy": pilot_config.get("frame_strategy"),
        "frames_per_video": pilot_config.get("frames_per_video"),
        "has_video_asset": bool(record.get("has_video_asset") or record.get("video_path") or record.get("asset_path")),
        "video_path": _video_path(record),
    }


def _video_path(record: dict[str, object]) -> str:
    candidates = (
        record.get("primary_video_path"),
        record.get("video_path"),
        record.get("asset_path"),
        record.get("local_video_path"),
    )
    for candidate in candidates:
        value = clean_text(candidate)
        if value:
            return value
    paths = record.get("video_paths")
    if isinstance(paths, list) and paths:
        return clean_text(paths[0])
    return ""


def _file_digest(path: str) -> str:
    candidate = Path(path)
    if not path or not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_fingerprint(
    record: dict[str, object],
    pilot_config: dict[str, object],
    pipeline: GoldBankPipeline,
) -> str:
    return stable_digest(
        {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": pilot_config.get("schema_version"),
            "prompt_version": pilot_config.get("prompt_version"),
            "frame_strategy": pilot_config.get("frame_strategy"),
            "frames_per_video": pilot_config.get("frames_per_video"),
            "min_confidence": pilot_config.get("min_confidence"),
            "video_id": clean_text(record.get("video_id")),
            "video_sha256": _file_digest(_video_path(record)),
            "vision_model": getattr(getattr(pipeline, "vlm_client", None), "model", ""),
            "text_model": getattr(getattr(pipeline, "llm_client", None), "model", ""),
        },
        length=24,
    )


def _sort_records(records: list[dict[str, object]], *keys: str) -> list[dict[str, object]]:
    return sorted(records, key=lambda item: tuple(clean_text(item.get(key)) for key in keys))


def _merge_outputs(
    records: list[dict[str, object]],
    pilot_config: dict[str, object],
    results: list[GoldBankResult],
    output_dir: Path,
    started_at: float,
    fingerprints: dict[str, str],
) -> dict[str, object]:
    evidence_units: list[dict[str, object]] = []
    proposals: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    gold_records: list[dict[str, object]] = []
    queue: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for result in results:
        evidence_units.extend(result.evidence_units)
        proposals.extend(result.gold_proposals)
        reviews.extend(result.gold_reviews)
        if result.video_gold_record is not None:
            gold_records.append(result.video_gold_record)
        queue.extend(result.human_review_queue)
        for trace in result.agent_traces:
            trace_record = dict(trace)
            trace_record.setdefault("video_id", result.video_id)
            traces.append(trace_record)

    video_samples = [_sample_record(record, pilot_config) for record in records]

    write_jsonl(output_dir / "video_samples.jsonl", _sort_records(video_samples, "video_id"))
    write_jsonl(output_dir / "evidence_units.jsonl", _sort_records(evidence_units, "video_id", "evidence_id"))
    write_jsonl(output_dir / "gold_proposals.jsonl", _sort_records(proposals, "video_id", "proposal_id"))
    write_jsonl(output_dir / "gold_reviews.jsonl", _sort_records(reviews, "video_id", "review_id"))
    write_jsonl(output_dir / "video_evidence_dataset.jsonl", _sort_records(gold_records, "video_id"))
    write_jsonl(output_dir / "human_review_queue.jsonl", _sort_records(queue, "video_id", "review_item_id"))
    write_jsonl(output_dir / "agent_traces.jsonl", _sort_records(traces, "video_id", "stage"))

    summary = {
        "version": pilot_config.get("version"),
        "prompt_version": pilot_config.get("prompt_version"),
        "schema_version": pilot_config.get("schema_version"),
        "video_ids": list(pilot_config.get("video_ids") or []),
        "counts": {
            "video_samples": len(video_samples),
            "evidence_units": len(evidence_units),
            "gold_proposals": len(proposals),
            "gold_reviews": len(reviews),
            "video_gold_records": len(gold_records),
            "human_review_queue": len(queue),
            "agent_traces": len(traces),
        },
        "statuses": {result.video_id: result.status for result in results},
        "pipeline_version": PIPELINE_VERSION,
        "config_fingerprint": stable_digest(
            {"pilot_config": pilot_config, "video_fingerprints": fingerprints},
            length=24,
        ),
        "video_fingerprints": dict(sorted(fingerprints.items())),
        "elapsed_s": round(time.time() - started_at, 2),
        "outputs": {name: str(output_dir / name) for name in GOLD_BANK_OUTPUT_FILES},
    }
    write_json(output_dir / "generation_meta.json", summary)
    return summary


def run_gold_bank_records(
    records: list[dict[str, object]],
    pilot_config: dict[str, object],
    output_dir: Path,
    pipeline: GoldBankPipeline,
    resume: bool = True,
    context_store: SalesBenchContextStore | None = None,
    frame_sampler: Callable[..., list[Frame]] = sample_frames,
    max_workers: int = 1,
) -> dict[str, object]:
    started_at = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_ids = [clean_text(video_id) for video_id in pilot_config.get("video_ids", [])]
    lookup = {clean_text(record.get("video_id")): record for record in records}
    missing = [video_id for video_id in requested_ids if video_id not in lookup]
    if missing:
        raise ValueError(f"Unknown video_ids: {missing}")
    selected_records = [lookup[video_id] for video_id in requested_ids]
    fingerprints = {
        video_id: _pipeline_fingerprint(lookup[video_id], pilot_config, pipeline)
        for video_id in requested_ids
    }
    completed = _load_completed(output_dir, requested_ids, fingerprints) if resume else {}

    results_by_id = dict(completed)

    def process_record(record: dict[str, object]) -> GoldBankResult:
        video_id = clean_text(record.get("video_id"))
        frames: list[Frame] = []
        path = _video_path(record)
        if path and bool(record.get("has_video_asset", True)):
            frames = frame_sampler(
                video_path=path,
                video_id=video_id,
                strategy=clean_text(pilot_config.get("frame_strategy")) or "hook_plus_uniform",
                total_frames=int(pilot_config.get("frames_per_video", 16)),
            )
        frame_metadata = [
            {
                "frame_index": frame.frame_index,
                "timestamp_s": frame.timestamp_s,
                "path": frame.path,
            }
            for frame in frames
        ]
        if context_store is not None:
            bundle = context_store.bundle_for_video(video_id, frames=frame_metadata)
        else:
            bundle = build_context_bundle(video_id, raw_video=record, frames=frame_metadata)
        return pipeline.run_video(bundle, frames_b64=[frame.image_base64 for frame in frames])

    pending = [record for record in selected_records if clean_text(record.get("video_id")) not in completed]
    if max_workers <= 1:
        for record in pending:
            result = process_record(record)
            _write_part(output_dir, result, fingerprints[result.video_id])
            results_by_id[result.video_id] = result
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_record, record): clean_text(record.get("video_id")) for record in pending}
            for future in as_completed(futures):
                result = future.result()
                _write_part(output_dir, result, fingerprints[result.video_id])
                results_by_id[result.video_id] = result

    ordered_results = [results_by_id[video_id] for video_id in requested_ids if video_id in results_by_id]
    return _merge_outputs(selected_records, pilot_config, ordered_results, output_dir, started_at, fingerprints)


def build_gold_bank_dataset(
    config: BenchmarkConfig,
    pilot_config_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    base_url: str | None = None,
    max_workers: int = 1,
    resume: bool = True,
    vision_api_key: str | None = None,
    vision_model: str | None = None,
    vision_base_url: str | None = None,
    text_api_key: str | None = None,
    text_model: str | None = None,
    text_base_url: str | None = None,
) -> dict[str, object]:
    pilot_config = read_json(pilot_config_path)
    if not isinstance(pilot_config, dict):
        raise ValueError(f"{pilot_config_path} must contain a JSON object")
    store = SalesBenchContextStore.from_config(config)
    records = store.raw_records_for_ids([clean_text(video_id) for video_id in pilot_config.get("video_ids", [])])
    vlm_client = VLMClient(
        api_key=vision_api_key or api_key,
        model=vision_model or model,
        base_url=vision_base_url if vision_base_url is not None else base_url,
        max_tokens=4096,
    )
    llm_client = VLMClient(
        api_key=text_api_key or api_key,
        model=text_model or model,
        base_url=text_base_url if text_base_url is not None else base_url,
        max_tokens=4096,
    )
    pipeline = GoldBankPipeline(
        vlm_client=vlm_client,
        llm_client=llm_client,
        min_confidence=float(pilot_config.get("min_confidence", 0.7)),
    )
    return run_gold_bank_records(
        records,
        pilot_config,
        output_dir,
        pipeline,
        resume=resume,
        context_store=store,
        max_workers=max_workers,
    )
