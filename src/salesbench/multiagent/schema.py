"""Internal context contracts shared by Evidence-First stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VideoContextBundle:
    video_id: str
    content_context: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "content_context": self.content_context,
        }
