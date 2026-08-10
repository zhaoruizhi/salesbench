"""Prompt builders for Evidence-First multi-agent annotation stages."""

from __future__ import annotations

import json
from copy import deepcopy

from .commerce_schema import CueType, RelationType
from .validators import PRIVATE_KEYS


PROMPT_VERSION = "evidence-prompt-v9"

BP_COMPILER_CONTRACT = (
    "BP is produced by a deterministic local compiler, not by an LLM proposer. "
    "For every validated direct EvidenceUnit, the compiler creates at most one BP candidate, "
    "maps the evidence modality and predicate to a controlled BP subtype, copies only the "
    "English normalized value into proposed_gold, and retains the EvidenceUnit ID as provenance. "
    "Verbatim source-language OCR or ASR remains in EvidenceUnit.source_text_native and is not copied into "
    "the public English Gold answer."
)


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
        "You are the objective Evidence Extractor for SalesBench-QA. Extract only EvidenceUnits "
        "that can be localized in the supplied frames, in-frame text, or ASR/subtitles. Do not "
        "generate questions, selling-strategy interpretations, audience conclusions, interaction "
        "outcomes, or hidden reasoning. Return strict JSON whose only top-level key is evidence_units. "
        "Each unit must use exactly one modality: visual for visible non-text facts, ocr for in-frame "
        "text, or asr for speech/subtitles. Never output image, text, or metadata as a modality. "
        "A visual unit must include frame_indices. An ocr unit must include frame_indices and a "
        "verbatim source_text_native. An asr unit must include the verbatim source_text_native supplied in the input. "
        "OCR source_text_native must contain only the visible source text, never coordinates, bounding boxes, "
        "or values such as [0,50]. If ASR start_s/end_s are supplied, copy them exactly; otherwise use "
        "null and never infer timing from semantics or frame position. Every unit must contain subject, "
        "predicate, and value. All normalized semantic fields, including subject, predicate, value, "
        "content_en, subject, predicate, value, and descriptive attributes, must be in English. "
        "content_en must be a concise factual English sentence. OCR/ASR source_text_native must preserve the original "
        "source language verbatim; brand names and source terms may remain unchanged only there. "
        "A seller's performance, effect, or usage statement is an unverified claim. Encode it as "
        "subject='speaker', predicate='claims', and value=<English normalized claim>, while retaining "
        "the original source_text_native. Do not state that the product has the claimed effect unless visible "
        "evidence directly verifies it. confidence must be a JSON number from 0 to 1, never "
        "high/medium/low. evidence_id may be omitted; if supplied, it is only a source locator such as "
        "frame_006 or asr_subtitles because local code creates the canonical ID. Omit anything that "
        "cannot be localized, is ambiguous, or requires external knowledge. Example: "
        '{"evidence_units":[{"modality":"visual","start_s":null,"end_s":null,'
        '"frame_indices":[0],"content_en":"The product package is red.","source_text_native":"",'
        '"subject":"product package",'
        '"predicate":"has color","value":"red","attributes":{},"confidence":0.9}]}'
    )
    user_text = _json({"video_id": video_id, "content_context": content_context})
    return system, [{"type": "text", "text": user_text}]


def _enum_values(enum_type: type) -> str:
    return ", ".join(item.value for item in enum_type)


_NO_CONSUMER_OUTCOMES = (
    "Never claim that a viewer trusted, purchased, converted, or became less uncertain. "
    "Describe only what the supplied content presents or how cited content units relate."
)


def build_commerce_cue_prompt(
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the SalesBench Commerce Cue Extractor for presenter-led e-commerce short videos. "
        "Convert validated atomic EvidenceUnits into grounded commercial presentation cues without "
        "adding consumer outcomes, external knowledge, creator metadata, or interaction data. "
        f"Allowed cue_type values are: {_enum_values(CueType)}. "
        "Return strict JSON with exactly two top-level arrays: commerce_cues and abstentions. "
        "Each commerce_cues item must contain exactly cue_type, content_en, source_text_native, "
        "evidence_ids, attributes, directness, theory_tags, and confidence. content_en must be a "
        "specific English sentence. source_text_native may copy only verbatim ASR or OCR text from "
        "the cited EvidenceUnits and otherwise must be an empty string. evidence_ids must contain "
        "one or more existing EvidenceUnit IDs copied exactly. directness must be DIRECT, INFERRED, "
        "or NEEDS_REVIEW. confidence must be a JSON number from 0 to 1. Do not output cue_id, "
        "video_id, extractor, translations, questions, or answers; local code adds canonical IDs. "
        "Keep a spoken product-effect statement as FUNCTION_CLAIM, EFFECT_CLAIM, PRICE_CLAIM, "
        "FIT_CLAIM, or EXPERIENCE_REVIEW until separate evidence demonstrates it. Do not relabel a "
        "claim as an observed product fact. Abstain when the cue cannot be localized. "
        f"{_NO_CONSUMER_OUTCOMES}"
    )
    user = _json({"video_id": video_id, "evidence_units": evidence_units})
    return system, user


def build_commercial_relation_prompt(
    video_id: str,
    evidence_units: list[dict[str, object]],
    commerce_cues: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the SalesBench Commercial Relation Builder. Connect existing grounded CommerceCue "
        "nodes only when their cited EvidenceUnits support a controlled commercial argument relation. "
        f"Allowed relation_type values are: {_enum_values(RelationType)}. "
        "Return strict JSON with exactly two top-level arrays: commercial_relations and abstentions. "
        "Each commercial_relations item must contain exactly relation_type, source_cue_ids, "
        "target_cue_ids, evidence_ids, status, rationale_en, directness, and confidence. Copy existing "
        "cue IDs and EvidenceUnit IDs exactly; never invent IDs. status must be SUPPORTED, "
        "PARTIALLY_SUPPORTED, CONTRADICTED, or TEMPORALLY_MISALIGNED. rationale_en must be a concise "
        "English explanation of the cited relationship. directness must be DIRECT, INFERRED, or "
        "NEEDS_REVIEW. Do not output relation_id, provenance, extractor, translations, questions, or "
        "answers; local code generates IDs and overrides provenance from the frozen ontology. "
        "CLAIM_REPEATED_ACROSS_MODALITIES means two modalities repeat equivalent promotional wording; "
        "it is not independent evidence. CLAIM_SUPPORTED_BY_DEMONSTRATION requires a distinct visual "
        "demonstration of the material claim. Abstain when endpoints are ambiguous or evidence is "
        "insufficient. Never claim that a viewer trusted, purchased, converted, or became less uncertain. "
        "Describe only relationships among the supplied content cues."
    )
    user = _json(
        {
            "video_id": video_id,
            "evidence_units": evidence_units,
            "commerce_cues": commerce_cues,
        }
    )
    return system, user


def build_proposer_prompt(
    generator: str,
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    generator = generator.lower()
    contracts = {
        "cm_proposer": (
            "Generate CM candidates only; never generate BP, SS, or AE. Allowed subtypes are "
            "CLAIM_EVIDENCE_RELATION, CLAIM_PARTIAL_SUPPORT, and TEXT_VISUAL_CONSISTENCY. For "
            "CLAIM_EVIDENCE_RELATION or CLAIM_PARTIAL_SUPPORT use target={claim,observation_window} "
            "and proposed_gold={relation,modality_pair,answer}. For TEXT_VISUAL_CONSISTENCY use "
            "target={text_claim,visual_target} and proposed_gold={relation,modality_pair,answer}. "
            "relation must be SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, NOT_SHOWN, or "
            "TEMPORALLY_MISALIGNED. Every CM candidate must cite at least two distinct EvidenceUnits "
            "from two distinct modalities, and modality_pair must list the actual modalities. Use "
            "NOT_SHOWN only when the input defines a complete-video observation window; abstain when "
            "only sampled frames are available."
        ),
        "ss_proposer": (
            "Generate SS candidates only; never generate BP, CM, or AE. Allowed subtypes are "
            "HOOK_MECHANISM, VALUE_PROPOSITION, TRUST_MECHANISM, OBJECTION_HANDLING, URGENCY_CTA, "
            "and FUNNEL_ROLE. target must contain segment and mechanism. proposed_gold must contain "
            "label and answer. The answer must explain how observable content instantiates the "
            "mechanism; it must not predict sales, conversion, or interaction outcomes."
        ),
        "ae_proposer": (
            "Generate AE candidates only; never generate BP, CM, or SS. Allowed subtypes are "
            "AUDIENCE_NEED_FIT, USAGE_CONTEXT, DECISION_STATE, and CONTENT_MOTIVATION. Use these exact "
            "structures: AUDIENCE_NEED_FIT target={need}, proposed_gold={audience_need,answer}; "
            "USAGE_CONTEXT target={scenario}, proposed_gold={usage_context,answer}; DECISION_STATE "
            "target={decision_barrier}, proposed_gold={decision_state,answer}; CONTENT_MOTIVATION "
            "target={motivation_cue}, proposed_gold={content_motivation,answer}. AE must not use claim "
            "or relation fields and must not claim a real audience profile, real conversion result, "
            "or facts about actual viewers."
        ),
    }
    examples = {
        "cm_proposer": (
            '{"task_type":"CM","task_subtype":"CLAIM_EVIDENCE_RELATION",'
            '"target":{"claim":"the cable uses braided material","observation_window":"cited speech and frames"},'
            '"proposed_gold":{"relation":"SUPPORTED","modality_pair":["asr","visual"],'
            '"answer":"The speech claims braided material and the visible outer weave supports the claim."},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","speech states the material claim","SUPPORTED"],'
            '["existing_id_2","the frame shows a woven outer layer","SUPPORTED"]],'
            '"proposal_confidence":0.85}'
        ),
        "ss_proposer": (
            '{"task_type":"SS","task_subtype":"HOOK_MECHANISM",'
            '"target":{"segment":"opening","mechanism":"result first"},'
            '"proposed_gold":{"label":"result-first hook",'
            '"answer":"The opening states the desired result before showing the corresponding product."},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","the opening states the intended result","SUPPORTED"],'
            '["existing_id_2","the frame shows the corresponding product","SUPPORTED"]],'
            '"proposal_confidence":0.85}'
        ),
        "ae_proposer": (
            '{"task_type":"AE","task_subtype":"USAGE_CONTEXT",'
            '"target":{"scenario":"home organization"},'
            '"proposed_gold":{"usage_context":"kitchen and bathroom organization",'
            '"answer":"The content shows a storage rack and names kitchen and bathroom uses."},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","the frame shows the rack in use","SUPPORTED"],'
            '["existing_id_2","speech names the bathroom use","SUPPORTED"]],'
            '"proposal_confidence":0.85}'
        ),
    }
    if generator not in contracts:
        raise ValueError(f"Unknown task generator: {generator}")
    system = (
        f"You are the SalesBench-QA {generator}. {contracts[generator]} "
        "Return strict JSON with exactly two top-level arrays: proposals and abstentions. Output no "
        "more than three high-quality proposals. A valid example for this generator is: "
        f"{examples[generator]}. task_type and task_subtype must use the controlled English values "
        "above. target and proposed_gold must be non-empty objects. Do not output proposal_id; local "
        "code creates the canonical proposal ID. Every proposal must cite at least two distinct "
        "evidence_ids copied exactly from the input, and the first item in each reasoning_edges entry "
        "must be one of those cited IDs. All natural-language values in target, proposed_gold, "
        "reasoning_edges, and abstentions.reason must use English. JSON keys, task/subtype, relation, "
        "and modality enums remain the controlled English tokens. Do not use informal wrappers such as "
        "annotation, proposal_type, hook, value, trust, strategy, or content_structure. "
        "proposal_confidence must be a JSON number from 0 to 1. Never invent evidence or IDs. Never "
        "use titles, follower counts, interaction metrics, private metadata, or external knowledge. "
        "Never discuss popularity, sales, conversion, propagation effects, or causal performance. If "
        "there are fewer than two distinct EvidenceUnits, the schema would be incomplete, the claim "
        "has a reasonable alternative interpretation, or the conclusion exceeds observable evidence, "
        "place task_type, task_subtype, and an English reason in abstentions instead of forcing a proposal."
    )
    user = _json({"video_id": video_id, "evidence_units": evidence_units})
    return system, user


def build_challenger_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the SalesBench-QA Gold Challenger. Return strict JSON whose only top-level key is "
        "reviews. Return exactly one review for every input proposal and copy proposal_id exactly. You "
        "must not default to PASS when evidence is missing, ambiguous, or schema-invalid. Check that "
        "Evidence IDs exist, cited evidence supports each claim, fact and inference are separated, "
        "reasonable alternatives are addressed, candidates do not duplicate or conflict, no title, "
        "interaction metric, private metadata, or external knowledge is used, and no sales or "
        "interaction causation is claimed. Task rubrics: BP must be a directly localizable fact and a "
        "spoken effect claim is not a verified product fact. CM must use two distinct modalities and "
        "must validate claim, relation, modality_pair, observation window, temporal alignment, and the "
        "complete-window requirement for NOT_SHOWN. SS needs at least two observable evidence units for "
        "a specific persuasion mechanism and must not inflate generic description into strategy or "
        "performance prediction. AE must be a bounded interpretation of needs, contexts, or decision "
        "barriers represented by the content, never a real audience profile or conversion conclusion. "
        "Each review must follow this schema: "
        '{"review_id":"unique string","proposal_id":"exact input proposal_id",'
        '"video_id":"exact input video_id","reviewer":"gold_challenger",'
        '"verdict":"PASS|REVISE|HUMAN_REVIEW|REJECT","checks":{"evidence_exists":true,'
        '"evidence_support":true,"fact_inference_separated":true,"schema_valid":true,'
        '"modality_diverse":true,"localizable":true,"no_duplicate":true,"no_conflict":true,'
        '"no_causal_overreach":true},"issues":[],"suggested_revision":null}. '
        "verdict must be PASS, REVISE, HUMAN_REVIEW, or REJECT. All natural language in issues and "
        "suggested_revision must use English. suggested_revision must be an object or null. Identify "
        "problems only; do not add Evidence or generate GroundedAnnotations."
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
        "You are the SalesBench-QA Evidence Adjudicator. The input contains only non-BP proposals that "
        "the Challenger marked PASS. Decide which proposals are acceptable, which strictly synonymous "
        "proposals may be merged, and which still require human review. You must not rewrite proposal "
        "content or generate GroundedAnnotations, gold_value, evidence_refs, tier, quality_status, or "
        "new evidence. Return strict JSON with exactly accepted_groups and human_review_queue. Each "
        "accepted_groups item has the shape "
        '{"source_proposal_ids":["exact input proposal_id"],"reason":"English acceptance or merge reason"}. '
        "A proposal_id may appear only once. Merge only when task_type, task_subtype, target semantics, "
        "and proposed_gold conclusion are equivalent and non-conflicting; otherwise accept separately "
        "or place them in human_review_queue. Each human_review_queue item has the shape "
        '{"source_proposal_ids":["exact input proposal_id"],"reason":"English review reason"}. '
        "Copy every ID exactly, use English for every reason, and do not omit any input proposal. Local "
        "code reconstructs annotations deterministically and applies final quality rules."
    )
    user = _json({"video_id": video_id, "proposals": proposals, "reviews": reviews, "evidence_units": evidence_units})
    return system, user
