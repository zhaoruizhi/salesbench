"""Prompt builders for Evidence-First multi-agent annotation stages."""

from __future__ import annotations

import json
from copy import deepcopy

from .commerce_ontology import RELATION_RULES
from .commerce_schema import CueType, RelationType
from .ontology import allowed_subtypes, default_reasoning_operator
from .schema import GoldTaskType
from .validators import PRIVATE_KEYS


PROMPT_VERSION = "evidence-prompt-v10.2"

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
        "When asr_subtitles is non-empty, output ASR EvidenceUnits before visual or OCR EvidenceUnits. "
        "Cover the transcript with three to ten high-information ASR units unless the supplied transcript "
        "is too short, prioritizing product identity, offer terms, claims, demonstrations described by the "
        "speaker, objections, comparisons, limitations, and action prompts. Do not omit ASR merely because "
        "the same words appear as burned-in subtitle text in frames. Treat burned-in subtitle text that "
        "repeats the supplied ASR as duplicate speech, not as separate OCR evidence. Reserve OCR for "
        "independent product labels, variants, prices, quantities, offer conditions, measurements, and "
        "other commercially material in-frame text. Exclude any creator handle, account identifier, "
        "watermark, platform logo, or engagement counter. "
        "A visual unit must include frame_indices. An ocr unit must include frame_indices and a "
        "verbatim source_text_native. An asr unit must include the verbatim source_text_native supplied in the input. "
        "OCR source_text_native must contain only the visible source text, never coordinates, bounding boxes, "
        "or values such as [0,50]. If ASR start_s/end_s are supplied, copy them exactly; otherwise use "
        "null and never infer timing from semantics or frame position. Every unit must contain subject, "
        "predicate, value, assertion_type, temporal_scope, and confidence. All normalized semantic "
        "fields, including subject, predicate, value, "
        "content_en, and descriptive attributes, must be in English. CJK characters are forbidden in "
        "subject, predicate, value, content_en, and attributes; they are allowed only in source_text_native. "
        "content_en must be a concise factual English sentence. OCR/ASR source_text_native must preserve the original "
        "source language verbatim; brand names and source terms may remain unchanged only there. "
        "A seller's performance, effect, or usage statement is an unverified claim. Encode it as "
        "subject='speaker', predicate='claims', and value=<English normalized claim>, while retaining "
        "the original source_text_native. Use assertion_type=OBSERVED for visual items, OCR_TEXT for OCR "
        "items, and SPOKEN_CLAIM for ASR items. Use temporal_scope=FRAME for a single-frame observation, "
        "SHORT_CLIP for a localized transcript or multi-frame observation, and LONG_TERM_CLAIM only when "
        "speech explicitly claims an effect beyond the observed clip. Do not state that the product has the "
        "claimed effect unless visible evidence directly verifies it. The exact JSON field \"confidence\" "
        "must be a number from 0 to 1, never "
        "high/medium/low. evidence_id may be omitted; if supplied, it is only a source locator such as "
        "frame_006 or asr_subtitles because local code creates the canonical ID. Omit anything that "
        "cannot be localized, is ambiguous, or requires external knowledge. Example: "
        '{"evidence_units":[{"modality":"visual","start_s":null,"end_s":null,'
        '"frame_indices":[0],"content_en":"The product package is red.","source_text_native":"",'
        '"subject":"product package",'
        '"predicate":"has color","value":"red","attributes":{},"assertion_type":"OBSERVED",'
        '"temporal_scope":"FRAME","confidence":0.9}]}'
    )
    user_text = _json({"video_id": video_id, "content_context": content_context})
    return system, [{"type": "text", "text": user_text}]


def build_language_evidence_prompt(
    video_id: str,
    content_context: dict[str, object],
) -> tuple[str, str]:
    system = (
        "You are the SalesBench ASR Evidence Extractor. Convert only the supplied ASR or subtitle "
        "transcript into three to ten high-information atomic EvidenceUnits. Every item must use "
        "modality=asr and contain start_s, end_s, frame_indices as an empty array, content_en, "
        "source_text_native, subject, predicate, value, attributes, \"assertion_type\", "
        "\"temporal_scope\", and \"confidence\". assertion_type must be SPOKEN_CLAIM. "
        "temporal_scope must be SHORT_CLIP for a claim bounded to the supplied transcript interval or "
        "LONG_TERM_CLAIM when the words explicitly claim an effect lasting beyond that interval. "
        "The exact JSON field \"confidence\" must be a number from 0 to 1. "
        "source_text_native must be a verbatim span copied from the supplied transcript. subject, "
        "predicate, value, content_en, and attributes must use English only; CJK characters are "
        "allowed only in source_text_native. Preserve claim-versus-fact boundaries: seller statements "
        "about performance, effects, compatibility, scarcity, popularity, or usage must use "
        "subject='speaker' and predicate='claims' or 'states'. Prioritize product identity, variants, "
        "offer terms, claims, described demonstrations, comparisons, objections, limitations, and CTAs. "
        "Do not extract titles, creator identity, follower or interaction data, or external knowledge. "
        "Return strict JSON with exactly one top-level key, evidence_units, and do not generate questions."
    )
    user = _json(
        {
            "video_id": video_id,
            "asr_subtitles": content_context.get("asr_subtitles") or {},
        }
    )
    return system, user


def build_visual_evidence_prompt(
    video_id: str,
    content_context: dict[str, object],
) -> tuple[str, list[dict]]:
    system = (
        "You are the SalesBench Visual and OCR Evidence Extractor. Inspect only the supplied sampled "
        "video frames and return six to sixteen localized EvidenceUnits. When at least one frame is "
        "supplied, return at least one visual EvidenceUnit describing a concrete visible product, "
        "person, action, state, comparison, demonstration, or usage scene. Use modality=visual for "
        "visible non-text facts and modality=ocr only for independent commercially material in-frame "
        "text such as product labels, variants, prices, quantities, offer conditions, or measurements. "
        "Do not enumerate burned-in speech subtitles as OCR. Exclude creator handles, account IDs, "
        "watermarks, platform logos, and engagement counters. Every item must contain the exact field "
        "\"modality\" set to visual or ocr, plus start_s, end_s, frame_indices, content_en, "
        "source_text_native, subject, predicate, value, attributes, "
        "\"assertion_type\", \"temporal_scope\", and \"confidence\". visual items must use "
        "assertion_type=OBSERVED; OCR items must use assertion_type=OCR_TEXT. Use temporal_scope=FRAME "
        "for one localized frame or SHORT_CLIP only when multiple supplied frames show the same event. "
        "The exact JSON field \"confidence\" must be a number from 0 to 1. visual items require one or "
        "more exact supplied frame indices and an empty "
        "source_text_native. ocr items require exact supplied frame indices and verbatim source_text_native. "
        "All normalized semantic fields must use English; CJK characters are allowed only in OCR "
        "source_text_native. Describe observable content only, preserve claim-versus-proof boundaries, "
        "and do not infer product effects, audience response, sales, or external facts. Return strict "
        "JSON with exactly one top-level key, evidence_units, and do not generate questions."
    )
    user = _json(
        {
            "video_id": video_id,
            "sampled_frames": content_context.get("sampled_frames") or [],
        }
    )
    return system, [{"type": "text", "text": user}]


def build_visual_evidence_repair_prompt(
    video_id: str,
    content_context: dict[str, object],
    rejected_candidates: list[dict[str, object]],
    local_rejection_reasons: list[dict[str, object]],
) -> tuple[str, list[dict]]:
    system = (
        "You are the SalesBench Visual Evidence Repairer. A previous frame-grounded response was "
        "rejected by deterministic local validation. Reinspect the supplied frames and return three to "
        "eight corrected atomic EvidenceUnits. Every item must use modality=\"visual\" exactly and must "
        "describe a concrete visible product, person, action, state, comparison, demonstration, package, "
        "or usage scene. Every item must contain the exact field \"modality\" set to visual, plus "
        "start_s, end_s, frame_indices, content_en, source_text_native, subject, predicate, value, "
        "attributes, \"assertion_type\", "
        "\"temporal_scope\", and \"confidence\". assertion_type must be OBSERVED, temporal_scope must "
        "be FRAME or SHORT_CLIP, and the exact JSON field \"confidence\" must be a number from 0 to 1. "
        "Use one or "
        "more exact supplied frame indices, keep source_text_native empty, and write content_en, subject, "
        "predicate, value, and every descriptive attribute in English only. Translate a visible product "
        "category into English instead of copying CJK characters into normalized fields. Do not output OCR, "
        "ASR, subtitles, creator handles, account IDs, watermarks, platform logos, or engagement counters. "
        "Do not infer product effects, audience response, sales, or external facts. Rejected candidates are "
        "diagnostic hints, not trusted facts: retain a candidate only if the supplied frames independently "
        "show it. Return strict JSON with exactly one top-level key, evidence_units, and do not generate "
        "questions."
    )
    user = _json(
        {
            "video_id": video_id,
            "sampled_frames": content_context.get("sampled_frames") or [],
            "rejected_candidates": rejected_candidates,
            "local_rejection_reasons": local_rejection_reasons,
        }
    )
    return system, [{"type": "text", "text": user}]


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
        "claim as an observed product fact. An ASR-only performance, compatibility, durability, or "
        "effect statement must not become PRODUCT_ATTRIBUTE and must not become PROCESS_DEMONSTRATION; "
        "keep it as an appropriate claim cue unless independently localized visual evidence exists. "
        "A BENEFIT derived only from seller speech must say that the speaker or video presents or frames "
        "the benefit; never state it as a verified effect. Clothing, glasses, a professional-looking room, "
        "or a generic product demonstration alone is not a CREDIBILITY_SIGNAL. "
        "Abstain when the cue cannot be localized. "
        f"{_NO_CONSUMER_OUTCOMES}"
    )
    user = _json({"video_id": video_id, "evidence_units": evidence_units})
    return system, user


def build_visual_commerce_cue_prompt(
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the SalesBench Visual Commerce Cue Extractor. Convert the supplied visual and OCR "
        "EvidenceUnits into grounded visual commercial presentation cues that were missed by the "
        "combined cue pass. Return strict JSON with exactly two top-level arrays: commerce_cues and "
        "abstentions. Each cue must contain exactly cue_type, content_en, source_text_native, "
        "evidence_ids, attributes, directness, theory_tags, and confidence. Use only existing EvidenceUnit "
        "IDs, and every cue must cite at least one visual EvidenceUnit unless the cue describes independent "
        "OCR product or offer text. Prioritize PRODUCT_IDENTITY, PRODUCT_ATTRIBUTE, PRODUCT_VARIANT, "
        "PROCESS_DEMONSTRATION, OUTCOME_DISPLAY, BEFORE_AFTER, VICARIOUS_TRIAL, USAGE_SCENARIO, "
        "COMPARISON_ANCHOR, and CREDIBILITY_SIGNAL. PROCESS_DEMONSTRATION, OUTCOME_DISPLAY, "
        "BEFORE_AFTER, and VICARIOUS_TRIAL must cite visual EvidenceUnits showing the action or state; "
        "a spoken description of a test is not a visual demonstration. content_en and attributes must use "
        "English. source_text_native must be empty for visual cues and may copy only verbatim source text "
        "for OCR cues. directness must be DIRECT, INFERRED, or NEEDS_REVIEW and confidence must be a "
        "JSON number from 0 to 1. Do not infer effects, audience response, trust, purchase, conversion, "
        "or interaction outcomes. Do not output IDs created by local code, questions, or answers."
    )
    user = _json({"video_id": video_id, "visual_ocr_evidence_units": evidence_units})
    return system, user


def build_commercial_relation_prompt(
    video_id: str,
    evidence_units: list[dict[str, object]],
    commerce_cues: list[dict[str, object]],
    source_capabilities: dict[str, bool] | None = None,
) -> tuple[str, str]:
    endpoint_contract = "; ".join(
        (
            f"{relation_type.value}: source endpoint type in "
            f"[{', '.join(sorted(item.value for item in rule['source']))}], target endpoint type in "
            f"[{', '.join(sorted(item.value for item in rule['target']))}]"
        )
        for relation_type, rule in RELATION_RULES.items()
    )
    system = (
        "You are the SalesBench Commercial Relation Builder. Connect existing grounded CommerceCue "
        "nodes only when their cited EvidenceUnits support a controlled commercial argument relation. "
        f"Allowed relation_type values are: {_enum_values(RelationType)}. "
        "Return strict JSON with exactly two top-level arrays: commercial_relations and abstentions. "
        "Return no more than eight highest-value non-duplicate commercial_relations; do not enumerate one "
        "DESCRIPTION_REFERS_TO_PRODUCT relation for every descriptive cue. "
        "Each commercial_relations item must contain exactly relation_type, source_cue_ids, "
        "target_cue_ids, evidence_ids, status, rationale_en, directness, and confidence. Copy existing "
        "cue IDs and EvidenceUnit IDs exactly; never invent IDs. status must be SUPPORTED, "
        "PARTIALLY_SUPPORTED, CONTRADICTED, or TEMPORALLY_MISALIGNED. rationale_en must be a concise "
        "English explanation of the cited relationship. directness must be DIRECT, INFERRED, or "
        "NEEDS_REVIEW. Do not output relation_id, provenance, extractor, translations, questions, or "
        "answers; local code generates IDs and overrides provenance from the frozen ontology. "
        f"Use these directed endpoint contracts exactly: {endpoint_contract}. "
        "evidence_ids must include the union of EvidenceUnit IDs cited by all endpoint cues, not only "
        "the first or source endpoint. "
        "CLAIM_REPEATED_ACROSS_MODALITIES means two modalities repeat equivalent promotional wording; "
        "it is not independent evidence. CLAIM_SUPPORTED_BY_DEMONSTRATION requires the claimed property "
        "or outcome itself to be independently visible. Merely showing the product, feature, colored "
        "markings, applicator, or one method step does not demonstrate a claimed memory, health, durability, "
        "ease, fit, or performance effect. Abstain when endpoints are ambiguous or evidence is "
        "insufficient. Never claim that a viewer trusted, purchased, converted, or became less uncertain. "
        "Describe only relationships among the supplied content cues. "
        "When source_capabilities.asr_temporal_order is false, do not emit CONTENT_PRECEDES_CTA, "
        "CLAIM_TEMPORALLY_MISALIGNED, or any relation whose truth requires ASR order."
    )
    user = _json(
        {
            "video_id": video_id,
            "evidence_units": evidence_units,
            "commerce_cues": commerce_cues,
            "source_capabilities": source_capabilities or {},
        }
    )
    return system, user


def build_proposer_prompt(
    generator: str,
    video_id: str,
    evidence_units: list[dict[str, object]],
    commerce_cues: list[dict[str, object]] | None = None,
    commercial_relations: list[dict[str, object]] | None = None,
) -> tuple[str, str]:
    generator = generator.lower()
    contracts = {
        "cm_proposer": (
            "Generate CM candidates only. Allowed capabilities are SPEECH_VISUAL_COREFERENCE, "
            "OCR_SPEECH_OFFER_ALIGNMENT, CLAIM_DEMONSTRATION_STATUS, "
            "REPETITION_VS_INDEPENDENT_EVIDENCE, PARTIAL_SUPPORT, CONTRADICTION, "
            "TEMPORAL_MISALIGNMENT, and NOT_DEMONSTRATED. Each candidate must test a real relation "
            "or information difference between at least two distinct modalities. CLAIM_DEMONSTRATION_STATUS "
            "must distinguish independent demonstration from repeated wording. NOT_DEMONSTRATED is "
            "allowed only with a complete observation window; sampled frames are insufficient."
        ),
        "ss_proposer": (
            "Generate SS candidates only. Allowed capabilities are PROBLEM_SOLUTION, FEATURE_BENEFIT, "
            "PROCESS_DEMONSTRATION, OUTCOME_DISPLAY, BEFORE_AFTER_COMPARISON, VICARIOUS_TRIAL, "
            "PRICE_VALUE_FRAMING, REFERENCE_PRICE_ANCHORING, CREDIBILITY_SIGNAL, SOCIAL_PROOF, "
            "LIMITATION_DISCLOSURE, OBJECTION_HANDLING, SCARCITY_AND_URGENCY, and CTA_SEQUENCE. "
            "Every candidate must reconstruct a specific observable commercial relation or ordered "
            "relation path. Do not use generic labels such as strategy, mechanism, or trust without "
            "naming the cited claim, product feature, demonstration, offer, objection, or action cue."
        ),
        "ae_proposer": (
            "Generate AE candidates only. Allowed capabilities are CONTENT_IMPLIED_NEED, USAGE_CONTEXT, "
            "FIT_CONSTRAINT, QUALITY_UNCERTAINTY, USAGE_UNCERTAINTY, PRICE_UNCERTAINTY, "
            "SERVICE_OR_RISK_CONCERN, DECISION_BARRIER, and OFFER_NEED_ALIGNMENT. Interpret only "
            "needs, contexts, constraints, objections, or offer alignment explicitly represented by "
            "the content. The answer must not claim a real audience profile. Never infer a real "
            "demographic audience, viewer psychology, purchase "
            "intention, conversion, or popularity."
        ),
    }
    if generator not in contracts:
        raise ValueError(f"Unknown task generator: {generator}")
    task_type = {"cm_proposer": "CM", "ss_proposer": "SS", "ae_proposer": "AE"}[generator]
    task = GoldTaskType(task_type)
    operator_contract = "; ".join(
        f"{subtype} -> {default_reasoning_operator(task, subtype)}"
        for subtype in sorted(allowed_subtypes(task))
    )
    system = (
        f"You are the SalesBench-QA {generator}. {contracts[generator]} "
        "Return strict JSON with exactly two top-level arrays: proposals and abstentions, with no more "
        f"than three proposals. Every proposal must use task_type={task_type}, set capability equal to "
        f'task_subtype, and include the literal pair "task_type":"{task_type}". Each proposal must '
        'contain "task_subtype", "capability", "reasoning_operator", "target", "proposed_gold", '
        '"evidence_ids", "commerce_cue_ids", "commercial_relation_ids", "reasoning_edges", '
        '"question_intent", "forbidden_inferences", and "proposal_confidence". '
        f"Use this exact capability-to-operator mapping: {operator_contract}. "
        "target and proposed_gold must be non-empty objects, and proposed_gold must contain "
        "answer. Do not output proposal_id; local code creates it. Every proposal must cite at least "
        "two distinct evidence_ids copied exactly from the EvidenceUnits input and at least one existing "
        "CommerceCue. An evidence_ids value must never be a cue ID or relation ID. "
        "Capabilities that require a relation or "
        "path must cite existing commercial_relation_ids; never invent IDs. The first value of every "
        "reasoning_edges entry must be a cited EvidenceUnit ID. All natural-language values must be "
        "English. proposal_confidence must be a JSON number from 0 to 1. Never "
        "copy instruction wording, field descriptions, generic placeholders, or meta-text into any "
        "proposal field; every target, answer, reasoning claim, and question intent must name content "
        "specific to this video. "
        "Do not use titles, follower counts, interaction metrics, private metadata, or external knowledge. "
        "When cited Evidence or CommerceCues contain a seller-stated effect, benefit, fit, or performance "
        "claim, the proposed_gold answer must explicitly attribute it with wording such as 'the speaker "
        "claims', 'the video presents', or 'the product is promoted as'; never rewrite it as a verified fact. "
        "Never claim content caused trust, purchase, sales, conversion, interaction, or reduced viewer "
        "uncertainty. If a graph path is incomplete or the conclusion has a reasonable alternative, "
        "place task_type, task_subtype, and an English reason in abstentions."
    )
    user = _json(
        {
            "video_id": video_id,
            "evidence_units": evidence_units,
            "commerce_cues": commerce_cues or [],
            "commercial_relations": commercial_relations or [],
        }
    )
    return system, user


def build_challenger_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
    commerce_cues: list[dict[str, object]] | None = None,
    commercial_relations: list[dict[str, object]] | None = None,
) -> tuple[str, str]:
    system = (
        "You are the SalesBench-QA Gold Challenger. Return strict JSON whose only top-level key is "
        "reviews. Return exactly one review for every input proposal and copy proposal_id exactly. You "
        "must not default to PASS when evidence is missing, ambiguous, or schema-invalid. Check that "
        "Evidence, CommerceCue, and CommercialRelation IDs exist, graph endpoints support each claim, "
        "fact and inference are separated, "
        "reasonable alternatives are addressed, candidates do not duplicate or conflict, no title, "
        "interaction metric, private metadata, or external knowledge is used, and no sales or "
        "interaction causation is claimed. Task rubrics: BP must be a directly localizable fact and a "
        "spoken effect claim is not a verified product fact. CM must use two distinct modalities and a "
        "valid claim relation, including the complete observation-window requirement for NOT_DEMONSTRATED. "
        "SS needs a specific commercial relation or ordered path and must not inflate generic description "
        "into sales logic or performance prediction. AE must be a bounded interpretation of needs, contexts, or decision "
        "barriers represented by the content, never a real audience profile or conversion conclusion. "
        "Showing a feature or method step does not prove its claimed benefit: colored markings do not "
        "establish memory improvement, pointing at words does not establish learning ease, and tilting an "
        "unloaded product does not establish load capacity or long-term stability. Reject such overreach. "
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
    user = _json(
        {
            "video_id": video_id,
            "proposals": proposals,
            "evidence_units": evidence_units,
            "commerce_cues": commerce_cues or [],
            "commercial_relations": commercial_relations or [],
        }
    )
    return system, user


def build_adjudicator_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    reviews: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
    commerce_cues: list[dict[str, object]] | None = None,
    commercial_relations: list[dict[str, object]] | None = None,
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
    user = _json(
        {
            "video_id": video_id,
            "proposals": proposals,
            "reviews": reviews,
            "evidence_units": evidence_units,
            "commerce_cues": commerce_cues or [],
            "commercial_relations": commercial_relations or [],
        }
    )
    return system, user
