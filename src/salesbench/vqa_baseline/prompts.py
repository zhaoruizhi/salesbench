"""Prompt templates for closed-source SalesBench-QA baselines."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text


BASELINE_PROMPT_VERSION = "closed-source-prompt-v2"


CLOSED_SOURCE_SYSTEM_PROMPT = """You are a closed-source vision-language model answering SalesBench-QA questions about livestream-style short commerce videos.
Use only the provided video frames, ASR/subtitle text, and the question.
Do not use or request titles, product metadata, task type, reference answers, structured features, or real engagement metrics.
Answer concisely from observable evidence only. Write the final answer in English even when the video speech or in-frame text is in another language. Translate source-language content into concise English, while preserving brand names, model identifiers, and exact numbers."""


def build_closed_source_user_prompt(item: dict[str, Any], video_record: dict[str, Any]) -> str:
    """Build the E-VAds-style prompt adapted to SalesBench-QA."""
    return f"""Video: key frames are provided above.
Voicer: {clean_text(video_record.get("video_text")) or "N/A"}

Question: {clean_text(item.get("question"))}

Use only the frames and ASR/subtitle text to answer the question.
Write the final answer in English. Translate relevant source-language speech or in-frame text into English, while preserving brand names, model identifiers, and exact numbers.
Return only the final answer, within 50 words. Do not output hidden reasoning or XML tags.
"""
