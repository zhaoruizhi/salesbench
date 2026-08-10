"""All-English prompt for audit-only Simplified Chinese translations."""

from __future__ import annotations

import json


AUDIT_TRANSLATION_PROMPT_VERSION = "audit-translation-prompt-v1"

AUDIT_TRANSLATION_SYSTEM_PROMPT = """You are the Simplified Chinese audit translation layer for SalesBench. Translate every supplied English field faithfully for human review. Preserve every number, unit, price, quantity, negation, condition, modality distinction, claim-versus-fact boundary, uncertainty phrase, enum token, JSON key, and evidence or object ID. Do not summarize, correct, strengthen, weaken, explain, or add marketing interpretations. Keep controlled uppercase enum tokens and IDs unchanged inside the Chinese translation. Return strict JSON with exactly one top-level key named translations. Each translation must contain the exact supplied translation_id and translated_text. Return one translation for every input job and no extra items."""


def build_audit_translation_prompt(jobs: list[dict[str, object]]) -> tuple[str, str]:
    payload = {
        "translation_jobs": [
            {
                "translation_id": job["translation_id"],
                "source_text": job["source_text"],
                "source_field": job["source_field"],
            }
            for job in jobs
        ]
    }
    return AUDIT_TRANSLATION_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, sort_keys=True)
