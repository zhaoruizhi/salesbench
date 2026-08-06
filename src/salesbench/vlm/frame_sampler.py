"""Video frame sampling using ffmpeg."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Frame:
    image_base64: str
    timestamp_s: float
    frame_index: int
    path: str  # cached file path


CACHE_ROOT = Path("outputs/cache/frames")
CACHE_VERSION = "frame-cache-v2"


def _frame_cache_dir(video_id: str) -> Path:
    return CACHE_ROOT / video_id


def _cache_manifest_path(video_id: str) -> Path:
    return _frame_cache_dir(video_id) / "manifest.json"


def _video_digest(video_path: str) -> str:
    path = Path(video_path)
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_cached(
    video_id: str,
    strategy: str,
    total_frames: int,
    hook_window_s: float,
    hook_frames: int,
    video_sha256: str,
) -> bool:
    manifest_path = _cache_manifest_path(video_id)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            manifest.get("cache_version") == CACHE_VERSION
            and
            manifest.get("strategy") == strategy
            and manifest.get("total_frames") == total_frames
            and manifest.get("hook_window_s") == hook_window_s
            and manifest.get("hook_frames") == hook_frames
            and manifest.get("video_sha256") == video_sha256
            and all(
                Path(f["path"]).exists()
                for f in manifest.get("frames", [])
            )
        )
    except Exception:
        return False


def _load_cached_frames(video_id: str) -> list[Frame]:
    manifest = json.loads(_cache_manifest_path(video_id).read_text(encoding="utf-8"))
    frames = []
    for f in manifest["frames"]:
        img_path = Path(f["path"])
        img_bytes = img_path.read_bytes()
        frames.append(Frame(
            image_base64=base64.b64encode(img_bytes).decode("utf-8"),
            timestamp_s=f["timestamp_s"],
            frame_index=f["frame_index"],
            path=str(img_path),
        ))
    return frames


def _get_video_duration(video_path: str) -> float | None:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return None


def _compute_timestamps(
    duration: float,
    strategy: str,
    total_frames: int,
    hook_window_s: float = 3.0,
    hook_frames: int = 3,
) -> list[float]:
    """Compute frame timestamps based on strategy."""
    if duration <= 0:
        return [0.0]

    if strategy == "hook_plus_uniform":
        timestamps = []
        # Hook frames: evenly spaced in first hook_window_s seconds
        actual_hook_window = min(hook_window_s, duration)
        if hook_frames > 0 and actual_hook_window > 0:
            for i in range(hook_frames):
                t = actual_hook_window * (i + 0.5) / hook_frames
                timestamps.append(round(t, 2))

        # Uniform frames: evenly spaced in remaining video
        remaining_frames = total_frames - hook_frames
        if remaining_frames > 0 and duration > actual_hook_window:
            remaining_duration = duration - actual_hook_window
            for i in range(remaining_frames):
                t = actual_hook_window + remaining_duration * (i + 0.5) / remaining_frames
                timestamps.append(round(min(t, duration - 0.1), 2))

        return timestamps[:total_frames] if timestamps else [0.0]

    elif strategy == "uniform":
        if total_frames == 1:
            return [duration / 2.0]
        return [
            round(duration * (i + 0.5) / total_frames, 2)
            for i in range(total_frames)
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _extract_frame_at(video_path: str, timestamp_s: float, output_path: str) -> bool:
    """Extract a single frame at given timestamp using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(timestamp_s),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",  # high quality JPEG
                output_path,
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


def _is_black_frame(image_path: str, threshold: int = 20) -> bool:
    """Check if frame is mostly black using Pillow."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")  # grayscale
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels) if pixels else 0
        return avg < threshold
    except Exception:
        return False


def sample_frames(
    video_path: str,
    video_id: str,
    strategy: str = "hook_plus_uniform",
    total_frames: int = 8,
    hook_window_s: float = 3.0,
    hook_frames: int = 3,
) -> list[Frame]:
    """
    Sample frames from a video file.

    Args:
        video_path: Path to mp4 file
        video_id: Video ID for caching
        strategy: "hook_plus_uniform" or "uniform"
        total_frames: Total number of frames to extract
        hook_window_s: Hook window in seconds (for hook_plus_uniform)
        hook_frames: Number of frames in hook window

    Returns:
        List of Frame objects with base64 encoded images
    """
    # Check cache
    video_sha256 = _video_digest(video_path)
    if _is_cached(video_id, strategy, total_frames, hook_window_s, hook_frames, video_sha256):
        return _load_cached_frames(video_id)

    # Get duration
    duration = _get_video_duration(video_path)
    if duration is None or duration <= 0:
        return []

    # Compute timestamps
    timestamps = _compute_timestamps(
        duration, strategy, total_frames, hook_window_s, hook_frames
    )

    # Create cache directory
    cache_dir = _frame_cache_dir(video_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frames: list[Frame] = []
    for idx, ts in enumerate(timestamps):
        output_path = str(cache_dir / f"frame_{idx:03d}.jpg")
        success = _extract_frame_at(video_path, ts, output_path)

        if not success:
            continue

        # Skip black frames - try a slightly later timestamp
        if _is_black_frame(output_path):
            alt_ts = min(ts + 0.5, duration - 0.1)
            _extract_frame_at(video_path, alt_ts, output_path)

        if not Path(output_path).exists():
            continue

        img_bytes = Path(output_path).read_bytes()
        frames.append(Frame(
            image_base64=base64.b64encode(img_bytes).decode("utf-8"),
            timestamp_s=ts,
            frame_index=idx,
            path=output_path,
        ))

    # Save manifest
    manifest = {
        "cache_version": CACHE_VERSION,
        "video_id": video_id,
        "video_path": video_path,
        "video_sha256": video_sha256,
        "strategy": strategy,
        "total_frames": total_frames,
        "hook_window_s": hook_window_s,
        "hook_frames": hook_frames,
        "duration_s": duration,
        "frames": [
            {
                "frame_index": f.frame_index,
                "timestamp_s": f.timestamp_s,
                "path": f.path,
            }
            for f in frames
        ],
    }
    _cache_manifest_path(video_id).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return frames
