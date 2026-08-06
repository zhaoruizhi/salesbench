"""Prompt builders for Evidence-First multi-agent annotation stages."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .validators import PRIVATE_KEYS


PROMPT_VERSION = "evidence-prompt-v2"


def _strip_private(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _strip_private(value)
            for key, value in payload.items()
            if str(key) not in PRIVATE_KEYS and str(key) != "performance_data"
        }
    if isinstance(payload, list):
        return [_strip_private(value) for value in payload]
    return payload


def _json(payload: object) -> str:
    return json.dumps(_strip_private(deepcopy(payload)), ensure_ascii=False, sort_keys=True)


def build_evidence_extractor_prompt(video_id: str, content_context: dict[str, object]) -> tuple[str, list[dict]]:
    system = (
        "You are an objective Evidence Extractor for SalesBench-QA. "
        "Return only localized Evidence Units as strict JSON with top-level key evidence_units. "
        "Do not create questions, marketing strategy labels, audience conclusions, or hidden reasoning. "
        "Each unit must separate observable fact from inference and include modality, subject, predicate, value, evidence_id when known, and confidence."
    )
    user_text = _json({"video_id": video_id, "content_context": content_context})
    return system, [{"type": "text", "text": user_text}]


def build_proposer_prompt(
    perspective: str,
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    perspective = perspective.lower()
    scopes = {
        "consumer": "Propose only AE and need-related SS annotations. Do not propose BP or interaction-effect claims.",
        "operator": "Propose only CM and content-structure or CTA-related SS annotations. Do not propose BP or interaction-effect claims.",
        "strategist": "Propose only SS about hook, value, trust, objection, urgency, and funnel role. Do not propose BP or interaction-effect claims.",
    }
    system = (
        f"You are the {perspective} Gold Proposer. {scopes.get(perspective, scopes['consumer'])} "
        "Return strict JSON with top-level keys proposals and abstentions. "
        "Every proposal must reference existing evidence_id values from the input. "
        "If evidence is insufficient, write an abstentions entry. Do not invent evidence or answer QA questions."
    )
    user = _json({"video_id": video_id, "evidence_units": evidence_units})
    return system, user


def build_challenger_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the Gold Challenger. Return strict JSON with top-level key reviews. "
        "Review each proposal for evidence existence, evidence support, fact/inference separation, alternative interpretation, duplicates, conflicts, and causal overreach. "
        "Verdict must be PASS, REVISE, HUMAN_REVIEW, or REJECT. You must not default PASS when evidence is missing or ambiguous."
    )
    user = _json({"video_id": video_id, "proposals": proposals, "evidence_units": evidence_units})
    return system, user


def build_adjudicator_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    reviews: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the Evidence Adjudicator. Return strict JSON with top-level keys grounded_annotations and human_review_queue. "
        "Merge synonymous supported proposals, preserve non-conflicting alternatives, and send ambiguous or conflicting items to human_review_queue. "
        "Use annotation_id and evidence_refs in each annotation. Do not add new evidence or hidden facts. "
        "Do not assign a tier or quality status; local code makes the final quality decision."
    )
    user = _json({"video_id": video_id, "proposals": proposals, "reviews": reviews, "evidence_units": evidence_units})
    return system, user
