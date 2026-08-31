"""All-English prompt for audit-only Simplified Chinese translations."""

from __future__ import annotations

import json
import re


AUDIT_TRANSLATION_PROMPT_VERSION = "audit-translation-prompt-v2"

_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_CONTROLLED_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

AUDIT_TRANSLATION_SYSTEM_PROMPT = """You are the Simplified Chinese audit translation layer for SalesBench. Translate every supplied English field faithfully for human review. Preserve every number, unit, price, quantity, negation, condition, modality distinction, claim-versus-fact boundary, uncertainty phrase, enum token, JSON key, and evidence or object ID. Every item supplies required_numbers and required_controlled_tokens; copy every listed value verbatim into translated_text. Do not summarize, correct, strengthen, weaken, explain, or add marketing interpretations. Keep controlled uppercase enum tokens and IDs unchanged inside the Chinese translation. If source_text is only an identifier or error code, return a readable Chinese label followed by the unchanged original code. Return strict JSON with exactly one top-level key named translations. Each translation must contain the exact supplied translation_id and translated_text. Return one translation for every input job and no extra items."""


def build_audit_translation_prompt(jobs: list[dict[str, object]]) -> tuple[str, str]:
    payload = {
        "translation_jobs": [
            {
                "translation_id": job["translation_id"],
                "source_text": job["source_text"],
                "source_field": job["source_field"],
                "required_numbers": _NUMBER_RE.findall(str(job["source_text"])),
                "required_controlled_tokens": sorted(
                    set(_CONTROLLED_TOKEN_RE.findall(str(job["source_text"])))
                ),
            }
            for job in jobs
        ]
    }
    return AUDIT_TRANSLATION_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, sort_keys=True)
