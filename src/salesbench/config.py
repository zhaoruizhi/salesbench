from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark_v1.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path
    dataset_name: str
    research_workbook: Path
    raw_video_dir: Path
    raw_sales_dir: Path
    output_dir: Path
    input_dir: Path
    processed_main: Path
    asset_manifest: Path
    data_profile: Path
    input_raw_video: Path
    input_visual_features: Path
    input_audio_speech_features: Path
    input_text_language_features: Path
    input_publish_context_features: Path
    input_cross_modal_consistency_features: Path
    input_summary: Path


def _resolve_repo_path(repo_root: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _resolve_output_path(output_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return output_dir / candidate


def _resolve_child_path(base_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def load_config(config_path: str | Path | None = None) -> BenchmarkConfig:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file

    payload = json.loads(config_file.read_text(encoding="utf-8"))
    repo_root = REPO_ROOT
    output_dir = _resolve_repo_path(repo_root, payload["output_dir"])
    input_dir = _resolve_repo_path(repo_root, payload["input_dir"])
    return BenchmarkConfig(
        repo_root=repo_root,
        dataset_name=payload["dataset_name"],
        research_workbook=_resolve_repo_path(repo_root, payload["research_workbook"]),
        raw_video_dir=_resolve_repo_path(repo_root, payload["raw_video_dir"]),
        raw_sales_dir=_resolve_repo_path(repo_root, payload["raw_sales_dir"]),
        output_dir=output_dir,
        input_dir=input_dir,
        processed_main=_resolve_output_path(output_dir, payload["processed_main"]),
        asset_manifest=_resolve_output_path(output_dir, payload["asset_manifest"]),
        data_profile=_resolve_output_path(output_dir, payload["data_profile"]),
        input_raw_video=_resolve_child_path(input_dir, payload["input_raw_video"]),
        input_visual_features=_resolve_child_path(input_dir, payload["input_visual_features"]),
        input_audio_speech_features=_resolve_child_path(input_dir, payload["input_audio_speech_features"]),
        input_text_language_features=_resolve_child_path(input_dir, payload["input_text_language_features"]),
        input_publish_context_features=_resolve_child_path(input_dir, payload["input_publish_context_features"]),
        input_cross_modal_consistency_features=_resolve_child_path(input_dir, payload["input_cross_modal_consistency_features"]),
        input_summary=_resolve_child_path(input_dir, payload["input_summary"]),
    )


def ensure_output_dirs(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.input_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        config.processed_main,
        config.asset_manifest,
        config.data_profile,
        config.input_raw_video,
        config.input_visual_features,
        config.input_audio_speech_features,
        config.input_text_language_features,
        config.input_publish_context_features,
        config.input_cross_modal_consistency_features,
        config.input_summary,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def config_summary(config: BenchmarkConfig) -> dict[str, object]:
    return {
        "dataset_name": config.dataset_name,
        "research_workbook": str(config.research_workbook),
        "raw_video_dir": str(config.raw_video_dir),
        "raw_sales_dir": str(config.raw_sales_dir),
        "processed_main": str(config.processed_main),
        "input_dir": str(config.input_dir),
        "input_raw_video": str(config.input_raw_video),
        "input_visual_features": str(config.input_visual_features),
        "input_audio_speech_features": str(config.input_audio_speech_features),
        "input_text_language_features": str(config.input_text_language_features),
        "input_publish_context_features": str(config.input_publish_context_features),
        "input_cross_modal_consistency_features": str(config.input_cross_modal_consistency_features),
        "input_summary": str(config.input_summary),
        "asset_manifest": str(config.asset_manifest),
        "data_profile": str(config.data_profile),
    }
