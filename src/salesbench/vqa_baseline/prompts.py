"""Prompt templates for closed-source SalesBench-QA baselines."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text


CLOSED_SOURCE_SYSTEM_PROMPT = """You are a closed-source vision-language model answering SalesBench-QA questions about livestream-style short commerce videos.
Use only the provided video frames, ASR/subtitle text, and the question.
Do not use or request titles, product metadata, task type, reference answers, structured features, or real engagement metrics.
Answer concisely from observable evidence only. If the question is Chinese, answer in Chinese."""


def build_closed_source_user_prompt(item: dict[str, Any], video_record: dict[str, Any]) -> str:
    """Build the E-VAds-style prompt adapted to SalesBench-QA."""
    return f"""Video: key frames are provided above.
Voicer: {clean_text(video_record.get("video_text")) or "N/A"}

Question: {clean_text(item.get("question"))}

Use only the frames and ASR/subtitle text to answer the question.
If the question is written in Chinese, write the final answer in Chinese. Do not mix English into the final answer unless the original video text contains the same English term.
Return only the final answer, within 50 Chinese characters or words. Do not output hidden reasoning or XML tags.
"""
