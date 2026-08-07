"""Prompt builders for Evidence-First multi-agent annotation stages."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .validators import PRIVATE_KEYS


PROMPT_VERSION = "evidence-prompt-v4"


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
        "Use exactly one modality per unit: visual for visible non-text facts, ocr for in-frame written text, or asr for speech/subtitles. "
        "Never output image or text as a modality. Visual units require frame_indices. OCR units require frame_indices and an exact text_span. "
        "ASR units require an exact text_span from the supplied ASR/subtitles. confidence must be a JSON number from 0 to 1, never high/medium/low. "
        "evidence_id is optional and may only be a source locator such as frame_006 or asr_subtitles; local code assigns canonical IDs. "
        "Every unit must include subject, predicate, and value. Omit inferred, ambiguous, or unlocalized claims. "
        "Return JSON matching this shape: "
        '{"evidence_units":[{"modality":"visual","frame_indices":[0],"text_span":"","subject":"product",'
        '"predicate":"color","value":"red","confidence":0.9}]}'
    )
    user_text = _json({"video_id": video_id, "content_context": content_context})
    return system, [{"type": "text", "text": user_text}]


def build_proposer_prompt(
    perspective: str,
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    perspective = perspective.lower()
    contracts = {
        "consumer": (
            "Allowed task/subtype pairs: AE with AUDIENCE_NEED_FIT, USAGE_CONTEXT, DECISION_STATE, or CONTENT_MOTIVATION; "
            "SS with VALUE_PROPOSITION or OBJECTION_HANDLING."
        ),
        "operator": (
            "Allowed task/subtype pairs: CM with CLAIM_EVIDENCE_RELATION, CLAIM_PARTIAL_SUPPORT, or TEXT_VISUAL_CONSISTENCY; "
            "SS with HOOK_MECHANISM, URGENCY_CTA, or FUNNEL_ROLE."
        ),
        "strategist": (
            "Allowed task/subtype pairs: SS with HOOK_MECHANISM, VALUE_PROPOSITION, TRUST_MECHANISM, "
            "OBJECTION_HANDLING, URGENCY_CTA, or FUNNEL_ROLE."
        ),
    }
    scope = contracts.get(perspective, contracts["consumer"])
    system = (
        f"You are the {perspective} Gold Proposer for SalesBench-QA. {scope} "
        "Never propose BP, interaction effects, popularity, sales, conversion, causal performance claims, or facts inferred from titles or engagement. "
        "Return one strict JSON object with exactly the top-level keys proposals and abstentions; both values must be arrays. "
        "Return zero to three high-quality proposals. Every proposal must match exactly this schema: "
        '{"proposal_id":"optional string","task_type":"CM|SS|AE","task_subtype":"ALLOWED_SUBTYPE",'
        '"target":{"claim_or_subject":"specific target"},"proposed_gold":{"answer":"concise answer grounded in the cited evidence"},'
        '"evidence_ids":["existing_id_1","existing_id_2"],'
        '"reasoning_edges":[["existing_id_1","specific support statement","SUPPORTED"],'
        '["existing_id_2","specific support statement","SUPPORTED"]],"proposal_confidence":0.85}. '
        "task_type and task_subtype are required and must use the exact uppercase controlled values above. "
        "target and proposed_gold must be non-empty JSON objects, not strings or wrapper objects. "
        "For CM, target must contain claim and proposed_gold must contain relation using exactly SUPPORTED, PARTIALLY_SUPPORTED, "
        "CONTRADICTED, NOT_SHOWN, or TEMPORALLY_MISALIGNED; it may also contain answer. "
        "For SS, target should identify the content segment or mechanism and proposed_gold must contain label or answer describing the strategy with observable support. "
        "For AE, target should identify the need, scenario, or decision barrier and proposed_gold must contain the subtype-specific field "
        "audience_need, usage_context, decision_state, or answer. Treat AE as a bounded content interpretation, never a claim about actual viewers or conversion. "
        "Each proposal must cite at least two distinct evidence_ids copied exactly from the input. Each reasoning edge must start with one of those IDs. "
        "proposal_confidence must be a JSON number from 0 to 1, never high/medium/low. "
        "Do not return informal annotation, proposal_type, hook, value, trust, strategy, content_structure, or other wrapper formats. "
        "If fewer than two distinct evidence units support an allowed proposal, put an object in abstentions with task_type, task_subtype, and reason. "
        "Do not invent evidence, quote IDs not present in the input, or answer a benchmark question."
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
