"""C1-C6 asset loading and public observation assembly for SalesBench QA."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..config import BenchmarkConfig
from ..io_utils import read_records
from ..utils import clean_text
from .schema import VideoContextBundle


C1_VISUAL = "C1_visual"
C2_AUDIO = "C2_audio_speech"
C3_TEXT = "C3_text_language"
C4_PUBLISH = "C4_publish_context"
C5_CROSS = "C5_cross_modal"
C6_VIDEO = "C6_raw_video"

PERFORMANCE_KEYS = ("likes", "collects", "shares", "comments")


def _without_performance_keys(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    return {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in PERFORMANCE_KEYS and key != "performance_data"
    }


def build_context_bundle(
    video_id: str,
    raw_video: dict[str, Any] | None = None,
    visual_features: dict[str, Any] | None = None,
    audio_speech: dict[str, Any] | None = None,
    text_language: dict[str, Any] | None = None,
    publish_context: dict[str, Any] | None = None,
    cross_modal: dict[str, Any] | None = None,
    frames: list[dict[str, Any]] | None = None,
) -> VideoContextBundle:
    """Build a six-domain internal asset bundle without interaction data."""
    raw = _without_performance_keys(raw_video)
    text = _without_performance_keys(text_language)
    audio = _without_performance_keys(audio_speech)

    if "video_text" not in audio:
        audio["video_text"] = raw.get("video_text") or text.get("video_text")
    if "title" not in text:
        text["title"] = raw.get("title")
    if "product_title" not in text:
        text["product_title"] = raw.get("product_title")
    if "video_text" not in text:
        text["video_text"] = raw.get("video_text")

    video_domain = {
        **raw,
        "frames": list(frames or []),
    }

    return VideoContextBundle(
        video_id=clean_text(video_id),
        content_context={
            C1_VISUAL: _without_performance_keys(visual_features),
            C2_AUDIO: audio,
            C3_TEXT: text,
            C4_PUBLISH: _without_performance_keys(publish_context),
            C5_CROSS: _without_performance_keys(cross_modal),
            C6_VIDEO: video_domain,
        },
    )


def public_observation_context(bundle: VideoContextBundle) -> dict[str, object]:
    """Return the exact non-image evidence visible to annotators and models."""
    audio = bundle.content_context.get(C2_AUDIO) or {}
    video = bundle.content_context.get(C6_VIDEO) or {}
    allowed_audio_keys = ("video_text", "asr", "asr_segments", "subtitle", "subtitles")
    return {
        "asr_subtitles": {
            key: deepcopy(audio[key])
            for key in allowed_audio_keys
            if key in audio and audio.get(key) not in (None, "", [])
        },
        "sampled_frames": deepcopy(video.get("frames") or []),
    }


def _lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        clean_text(record.get("video_id")): record
        for record in records
        if clean_text(record.get("video_id"))
    }


class SalesBenchContextStore:
    """Load and join SalesBench input artifacts by video_id."""

    def __init__(
        self,
        raw_video_records: list[dict[str, Any]],
        visual_records: list[dict[str, Any]],
        audio_records: list[dict[str, Any]],
        text_records: list[dict[str, Any]],
        publish_records: list[dict[str, Any]],
        cross_records: list[dict[str, Any]],
    ):
        self.raw_video_records = raw_video_records
        self.raw_lookup = _lookup(raw_video_records)
        self.visual_lookup = _lookup(visual_records)
        self.audio_lookup = _lookup(audio_records)
        self.text_lookup = _lookup(text_records)
        self.publish_lookup = _lookup(publish_records)
        self.cross_lookup = _lookup(cross_records)

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> "SalesBenchContextStore":
        return cls(
            raw_video_records=read_records(config.input_raw_video),
            visual_records=read_records(config.input_visual_features),
            audio_records=read_records(config.input_audio_speech_features),
            text_records=read_records(config.input_text_language_features),
            publish_records=read_records(config.input_publish_context_features),
            cross_records=read_records(config.input_cross_modal_consistency_features),
        )

    def raw_records_for_ids(self, video_ids: list[str]) -> list[dict[str, Any]]:
        requested_ids = [clean_text(video_id) for video_id in video_ids]
        missing = [video_id for video_id in requested_ids if video_id not in self.raw_lookup]
        if missing:
            raise ValueError(f"Unknown video_ids: {missing}")
        return [self.raw_lookup[video_id] for video_id in requested_ids]

    def bundle_for_video(self, video_id: str, frames: list[dict[str, Any]] | None = None) -> VideoContextBundle:
        video_id = clean_text(video_id)
        return build_context_bundle(
            video_id=video_id,
            raw_video=self.raw_lookup.get(video_id),
            visual_features=self.visual_lookup.get(video_id),
            audio_speech=self.audio_lookup.get(video_id),
            text_language=self.text_lookup.get(video_id),
            publish_context=self.publish_lookup.get(video_id),
            cross_modal=self.cross_lookup.get(video_id),
            frames=frames,
        )
