"""Prompt builders for Evidence-First multi-agent annotation stages."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .validators import PRIVATE_KEYS


PROMPT_VERSION = "evidence-prompt-v6"


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
        "When the source content is primarily Chinese, write subject, predicate, value, and attributes in Chinese while preserving exact OCR/ASR text_span. "
        "Return JSON matching this shape: "
        '{"evidence_units":[{"modality":"visual","frame_indices":[0],"text_span":"","subject":"产品",'
        '"predicate":"颜色","value":"红色","confidence":0.9}]}'
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
    examples = {
        "consumer": (
            '{"proposal_id":"optional","task_type":"AE","task_subtype":"USAGE_CONTEXT",'
            '"target":{"scenario":"家庭收纳"},"proposed_gold":{"usage_context":"厨房和卫生间收纳",'
            '"answer":"内容通过展示置物架并在口播中列出厨房和卫生间用途，对应家庭收纳场景。"},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","shows kitchen use","SUPPORTED"],'
            '["existing_id_2","口播提到卫生间用途","SUPPORTED"]],"proposal_confidence":0.85}'
        ),
        "operator": (
            '{"proposal_id":"optional","task_type":"CM","task_subtype":"CLAIM_EVIDENCE_RELATION",'
            '"target":{"claim":"充电线采用编织材质"},"proposed_gold":{"relation":"SUPPORTED","answer":"画面中的编织外层支持该说法。"},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","states the claim","SUPPORTED"],'
            '["existing_id_2","画面显示编织外层","SUPPORTED"]],"proposal_confidence":0.85}'
        ),
        "strategist": (
            '{"proposal_id":"optional","task_type":"SS","task_subtype":"HOOK_MECHANISM",'
            '"target":{"segment":"开场"},"proposed_gold":{"label":"结果前置",'
            '"answer":"开场先给出预期结果，再展示对应产品，以结果前置方式吸引注意。"},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","states a result in the opening","SUPPORTED"],'
            '["existing_id_2","画面展示对应产品","SUPPORTED"]],"proposal_confidence":0.85}'
        ),
    }
    scope = contracts.get(perspective, contracts["consumer"])
    example = examples.get(perspective, examples["consumer"])
    system = (
        f"You are the {perspective} Gold Proposer for SalesBench-QA. {scope} "
        "Never propose BP, interaction effects, popularity, sales, conversion, causal performance claims, or facts inferred from titles or engagement. "
        "Return one strict JSON object with exactly the top-level keys proposals and abstentions; both values must be arrays. "
        f"Return zero to three high-quality proposals. A valid proposal for this role looks exactly like this: {example}. "
        "task_type and task_subtype are required and must use the exact uppercase controlled values above. "
        "target and proposed_gold must be non-empty JSON objects, not strings or wrapper objects. "
        "When the evidence is primarily Chinese, all free-text values in target, proposed_gold, and reasoning_edges must be Chinese; keep JSON keys and controlled enum values unchanged. "
        "For CM, target must contain claim and proposed_gold must contain relation using exactly SUPPORTED, PARTIALLY_SUPPORTED, "
        "CONTRADICTED, NOT_SHOWN, or TEMPORALLY_MISALIGNED; it may also contain answer. "
        "For SS, target should identify the content segment or mechanism and proposed_gold must contain both a concise label and a complete answer sentence describing the strategy with observable support. "
        "For AE, target should identify the need, scenario, or decision barrier and proposed_gold must contain the subtype-specific field "
        "audience_need, usage_context, decision_state, or content_motivation, plus a complete answer sentence connecting the interpretation to observable evidence. "
        "Treat AE as a bounded content interpretation, never a claim about actual viewers or conversion. "
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
        "Return exactly one review for every input proposal and copy its proposal_id exactly. "
        "Every review must use this schema: "
        '{"review_id":"unique string","proposal_id":"exact input proposal_id","video_id":"exact input video_id",'
        '"reviewer":"gold_challenger","verdict":"PASS|REVISE|HUMAN_REVIEW|REJECT",'
        '"checks":{"evidence_exists":true,"evidence_support":true,"fact_inference_separated":true,'
        '"no_duplicate":true,"no_conflict":true,"no_causal_overreach":true},'
        '"issues":[],"suggested_revision":null}. '
        "checks must be a JSON object, issues must be an array of strings, and suggested_revision must be an object or null. "
        "Verdict must be exactly PASS, REVISE, HUMAN_REVIEW, or REJECT. You must not default PASS when evidence is missing or ambiguous."
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
        "Only proposals supplied in this call have PASS reviews. For each accepted proposal, preserve its task_type, task_subtype, target, proposed_gold, "
        "evidence_ids, reasoning_edges, and proposal_confidence without changing their meaning. "
        "Map proposed_gold to gold_value, evidence_ids to evidence_refs, proposal_confidence to confidence, and include the exact proposal_id in source_proposal_ids. "
        "Every grounded annotation must use exactly this schema: "
        '{"annotation_id":"unique string","video_id":"exact input video_id","task_type":"BP|CM|SS|AE",'
        '"task_subtype":"exact input subtype","target":{},"gold_value":{},"evidence_refs":["existing evidence id"],'
        '"reasoning_edges":[["existing evidence id","support statement","SUPPORTED"]],'
        '"eligible_question_formats":[],"source_proposal_ids":["exact input proposal_id"],'
        '"review_status":"verified","confidence":0.85}. '
        "Do not return abbreviated subject/predicate/value annotations and do not rename proposal IDs. "
        "human_review_queue must be an array of objects containing source_proposal_ids and reason. Do not add new evidence or hidden facts. "
        "Do not assign a tier or quality status; local code makes the final quality decision."
    )
    user = _json({"video_id": video_id, "proposals": proposals, "reviews": reviews, "evidence_units": evidence_units})
    return system, user
