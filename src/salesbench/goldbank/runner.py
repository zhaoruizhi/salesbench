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
from ..run_integrity import canonical_sha256, compute_evidence_fingerprint
from ..utils import clean_text
from ..vlm.api_client import VLMClient
from ..vlm.frame_sampler import Frame, sample_frames
from .pipeline import GoldBankPipeline, GoldBankResult
from .quality_prompts import QUALITY_PROMPT_VERSION
from .schema import stable_digest


PIPELINE_VERSION = "evidence-first-pipeline-v10.5"


GOLD_BANK_OUTPUT_FILES = (
    "video_samples.jsonl",
    "evidence_units.jsonl",
    "commerce_cues.jsonl",
    "commercial_relations.jsonl",
    "gold_proposals.jsonl",
    "gold_reviews.jsonl",
    "video_evidence_dataset.jsonl",
    "human_review_queue.jsonl",
    "repaired_candidates.jsonl",
    "rejected_candidates.jsonl",
    "pipeline_diagnostics.jsonl",
    "quality_decisions.jsonl",
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
        "commerce_cues": result.commerce_cues,
        "commercial_relations": result.commercial_relations,
        "gold_proposals": result.gold_proposals,
        "gold_reviews": result.gold_reviews,
        "video_gold_record": result.video_gold_record,
        "human_review_queue": result.human_review_queue,
        "repaired_candidates": result.repaired_candidates,
        "rejected_candidates": result.rejected_candidates,
        "pipeline_diagnostics": result.pipeline_diagnostics,
        "quality_decisions": result.quality_decisions,
        "agent_traces": result.agent_traces,
        "status": result.status,
    }


def _payload_to_result(payload: dict[str, object]) -> GoldBankResult:
    return GoldBankResult(
        video_id=clean_text(payload.get("video_id")),
        evidence_units=list(payload.get("evidence_units") or []),
        commerce_cues=list(payload.get("commerce_cues") or []),
        commercial_relations=list(payload.get("commercial_relations") or []),
        gold_proposals=list(payload.get("gold_proposals") or []),
        gold_reviews=list(payload.get("gold_reviews") or []),
        video_gold_record=payload.get("video_gold_record") if isinstance(payload.get("video_gold_record"), dict) else None,
        human_review_queue=list(payload.get("human_review_queue") or []),
        repaired_candidates=list(payload.get("repaired_candidates") or []),
        rejected_candidates=list(payload.get("rejected_candidates") or []),
        pipeline_diagnostics=list(payload.get("pipeline_diagnostics") or []),
        quality_decisions=list(payload.get("quality_decisions") or []),
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
            status = clean_text(payload.get("status"))
            traces = list(payload.get("agent_traces") or [])
            terminal_partial = (
                status == "partial"
                and isinstance(payload.get("video_gold_record"), dict)
                and not any(isinstance(trace, dict) and trace.get("success") is False for trace in traces)
            )
            if status != "ok" and not terminal_partial:
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
            "strict_semantic_verification": bool(
                pilot_config.get("strict_semantic_verification", False)
            ),
            "quality_prompt_version": QUALITY_PROMPT_VERSION,
            "video_id": clean_text(record.get("video_id")),
            "video_sha256": _file_digest(_video_path(record)),
            "vision_model": getattr(getattr(pipeline, "vlm_client", None), "model", ""),
            "text_model": getattr(getattr(pipeline, "llm_client", None), "model", ""),
            "semantic_verifier_model": getattr(
                getattr(pipeline, "semantic_verifier_client", None), "model", ""
            ),
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
    *,
    run_id: str = "",
    benchmark_release: str = "salesbench-v10-candidate.2",
    source_fingerprint: str = "",
    pipeline: GoldBankPipeline | None = None,
) -> dict[str, object]:
    evidence_units: list[dict[str, object]] = []
    commerce_cues: list[dict[str, object]] = []
    commercial_relations: list[dict[str, object]] = []
    proposals: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    gold_records: list[dict[str, object]] = []
    queue: list[dict[str, object]] = []
    repaired: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for result in results:
        evidence_units.extend(result.evidence_units)
        commerce_cues.extend(result.commerce_cues)
        commercial_relations.extend(result.commercial_relations)
        proposals.extend(result.gold_proposals)
        reviews.extend(result.gold_reviews)
        if result.video_gold_record is not None:
            gold_records.append(result.video_gold_record)
        queue.extend(result.human_review_queue)
        repaired.extend(result.repaired_candidates)
        rejected.extend(result.rejected_candidates)
        diagnostics.extend(result.pipeline_diagnostics)
        decisions.extend(result.quality_decisions)
        for trace in result.agent_traces:
            trace_record = dict(trace)
            trace_record.setdefault("video_id", result.video_id)
            traces.append(trace_record)

    video_samples = [_sample_record(record, pilot_config) for record in records]
    accepted_annotation_count = sum(
        len(record.get("grounded_annotations") or []) for record in gold_records
    )
    quality_candidate_count = accepted_annotation_count + len(queue) + len(rejected)
    human_ambiguity_rate = (
        len(queue) / quality_candidate_count if quality_candidate_count else 0.0
    )
    ambiguity_target_rate = float(pilot_config.get("human_ambiguity_target_rate", 0.05))
    ambiguity_block_rate = float(pilot_config.get("human_ambiguity_block_rate", 0.10))

    write_jsonl(output_dir / "video_samples.jsonl", _sort_records(video_samples, "video_id"))
    write_jsonl(output_dir / "evidence_units.jsonl", _sort_records(evidence_units, "video_id", "evidence_id"))
    write_jsonl(output_dir / "commerce_cues.jsonl", _sort_records(commerce_cues, "video_id", "cue_id"))
    write_jsonl(
        output_dir / "commercial_relations.jsonl",
        _sort_records(commercial_relations, "video_id", "relation_id"),
    )
    write_jsonl(output_dir / "gold_proposals.jsonl", _sort_records(proposals, "video_id", "proposal_id"))
    write_jsonl(output_dir / "gold_reviews.jsonl", _sort_records(reviews, "video_id", "review_id"))
    write_jsonl(output_dir / "video_evidence_dataset.jsonl", _sort_records(gold_records, "video_id"))
    write_jsonl(output_dir / "human_review_queue.jsonl", _sort_records(queue, "video_id", "review_item_id"))
    write_jsonl(
        output_dir / "repaired_candidates.jsonl",
        _sort_records(repaired, "video_id", "review_item_id"),
    )
    write_jsonl(
        output_dir / "rejected_candidates.jsonl",
        _sort_records(rejected, "video_id", "review_item_id"),
    )
    write_jsonl(
        output_dir / "pipeline_diagnostics.jsonl",
        _sort_records(diagnostics, "video_id", "review_item_id"),
    )
    write_jsonl(
        output_dir / "quality_decisions.jsonl",
        _sort_records(decisions, "video_id", "decision_id"),
    )
    write_jsonl(output_dir / "agent_traces.jsonl", _sort_records(traces, "video_id", "stage"))

    component_versions = {
        "evidence_schema": clean_text(pilot_config.get("schema_version")),
        "evidence_prompt": clean_text(pilot_config.get("prompt_version")),
        "evidence_pipeline": PIPELINE_VERSION,
        "quality_prompt": QUALITY_PROMPT_VERSION,
    }
    models = {
        "vision": getattr(getattr(pipeline, "vlm_client", None), "model", ""),
        "text": getattr(getattr(pipeline, "llm_client", None), "model", ""),
        "semantic_verifier": getattr(
            getattr(pipeline, "semantic_verifier_client", None), "model", ""
        ),
    }
    evidence_fingerprint = compute_evidence_fingerprint(
        source={
            "source_fingerprint": source_fingerprint,
            "video_fingerprints": dict(sorted(fingerprints.items())),
        },
        components=component_versions,
        models=models,
        policy=pilot_config,
        evidence_units=_sort_records(evidence_units, "video_id", "evidence_id"),
        commerce_cues=_sort_records(commerce_cues, "video_id", "cue_id"),
        commercial_relations=_sort_records(
            commercial_relations, "video_id", "relation_id"
        ),
        dataset_rows=_sort_records(gold_records, "video_id"),
    )
    summary = {
        "benchmark_release": benchmark_release,
        "run_id": run_id,
        "release_status": "CANDIDATE",
        "component_versions": component_versions,
        "source_fingerprint": source_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "version": pilot_config.get("version"),
        "prompt_version": pilot_config.get("prompt_version"),
        "schema_version": pilot_config.get("schema_version"),
        "quality_prompt_version": QUALITY_PROMPT_VERSION,
        "strict_semantic_verification": bool(
            pilot_config.get("strict_semantic_verification", False)
        ),
        "video_ids": list(pilot_config.get("video_ids") or []),
        "counts": {
            "video_samples": len(video_samples),
            "evidence_units": len(evidence_units),
            "commerce_cues": len(commerce_cues),
            "commercial_relations": len(commercial_relations),
            "gold_proposals": len(proposals),
            "gold_reviews": len(reviews),
            "video_gold_records": len(gold_records),
            "human_review_queue": len(queue),
            "repaired_candidates": len(repaired),
            "rejected_candidates": len(rejected),
            "pipeline_diagnostics": len(diagnostics),
            "quality_decisions": len(decisions),
            "agent_traces": len(traces),
        },
        "statuses": {result.video_id: result.status for result in results},
        "quality_gate": {
            "accepted_annotation_count": accepted_annotation_count,
            "quality_candidate_count": quality_candidate_count,
            "human_ambiguity_count": len(queue),
            "human_ambiguity_rate": round(human_ambiguity_rate, 6),
            "target_rate": ambiguity_target_rate,
            "block_rate": ambiguity_block_rate,
            "target_met": human_ambiguity_rate <= ambiguity_target_rate,
            "release_blocked": human_ambiguity_rate > ambiguity_block_rate,
        },
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
    if run_id:
        run_root = output_dir.parent
        manifest_path = run_root / "run_manifest.json"
        existing: dict[str, object] = {}
        if manifest_path.exists():
            loaded = read_json(manifest_path)
            if isinstance(loaded, dict):
                existing = loaded
        manifest = {
            **existing,
            "benchmark_release": benchmark_release,
            "run_id": run_id,
            "release_status": "CANDIDATE",
            "components": {
                **(existing.get("components") if isinstance(existing.get("components"), dict) else {}),
                **component_versions,
            },
            "artifacts": {
                **(existing.get("artifacts") if isinstance(existing.get("artifacts"), dict) else {}),
                "evidence": output_dir.name,
            },
            "fingerprints": {
                **(existing.get("fingerprints") if isinstance(existing.get("fingerprints"), dict) else {}),
                "source": source_fingerprint,
                "evidence": evidence_fingerprint,
            },
        }
        write_json(manifest_path, manifest)
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
    run_id: str = "",
    benchmark_release: str = "salesbench-v10-candidate.2",
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
    source_fingerprint = canonical_sha256(
        {
            "pipeline_version": PIPELINE_VERSION,
            "pilot_config": pilot_config,
            "video_fingerprints": dict(sorted(fingerprints.items())),
        }
    )
    prior_meta_path = output_dir / "generation_meta.json"
    if run_id and prior_meta_path.exists():
        prior_meta = read_json(prior_meta_path)
        if not isinstance(prior_meta, dict) or clean_text(prior_meta.get("run_id")) != run_id:
            raise ValueError("immutable Evidence output belongs to a different run identity")
        if clean_text(prior_meta.get("source_fingerprint")) != source_fingerprint:
            raise ValueError("immutable Evidence output source fingerprint changed")
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
    return _merge_outputs(
        selected_records,
        pilot_config,
        ordered_results,
        output_dir,
        started_at,
        fingerprints,
        run_id=run_id,
        benchmark_release=benchmark_release,
        source_fingerprint=source_fingerprint,
        pipeline=pipeline,
    )


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
    run_id: str = "",
    benchmark_release: str = "salesbench-v10-candidate.2",
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
        disable_thinking=True,
    )
    llm_client = VLMClient(
        api_key=text_api_key or api_key,
        model=text_model or model,
        base_url=text_base_url if text_base_url is not None else base_url,
        max_tokens=4096,
        disable_thinking=True,
    )
    pipeline = GoldBankPipeline(
        vlm_client=vlm_client,
        llm_client=llm_client,
        min_confidence=float(pilot_config.get("min_confidence", 0.7)),
        strict_semantic_verification=bool(
            pilot_config.get("strict_semantic_verification", False)
        ),
    )
    return run_gold_bank_records(
        records,
        pilot_config,
        output_dir,
        pipeline,
        resume=resume,
        context_store=store,
        max_workers=max_workers,
        run_id=run_id,
        benchmark_release=benchmark_release,
    )
