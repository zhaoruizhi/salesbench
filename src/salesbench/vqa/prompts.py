"""All-English prompts for evidence-specific question realization."""

from __future__ import annotations

import json

from .specs import QuestionSpec


QUESTION_REALIZER_PROMPT_VERSION = "question-realizer-prompt-v3"
QA_QUALITY_PROMPT_VERSION = "qa-quality-prompt-v2"


QUESTION_REALIZER_SYSTEM_PROMPT = """You are the SalesBench English Question Realizer. Convert one approved semantic QuestionSpec into one natural, concise, content-specific English question for a presenter-led e-commerce short-video benchmark. Preserve the requested task capability, reasoning operator, intended modality, and assertion scope. Name the concrete product, claim, functional operation, observable outcome, offer, condition, comparison, objection, need, or decision barrier when supported. The question must require the cited commercial evidence; do not turn generic holding, pointing, showing, rotating, page flipping, or an empty background object into a benchmark question. BP may ask a short localized fact, but should prioritize product identity, offer terms, OCR, measurable attributes, functional operation, or observable state change. CM must require an actual relationship between modalities. SS must reconstruct a specific commercial argument rather than ask for a generic strategy label. AE must ask one bounded need, use context, constraint, objection, or decision barrier without inventing a real audience profile. Preserve 'the speaker claims', conditions, instructions, hypotheticals, and promises instead of converting them into facts. Do not add facts or external knowledge. Do not include the answer in the question. Do not ask generic questions such as 'What mechanism is used?', 'What strategy is used?', 'What audience is targeted?', or 'What evidence supports the answer?'. Never append evidence instructions. Do not mention EvidenceUnit, CommerceCue, CommercialRelation, IDs, schemas, annotations, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


QUESTION_REPAIR_SYSTEM_PROMPT = """You are the SalesBench English Question Surface Repairer. A previous question was rejected by deterministic local validation. Rewrite only the question surface while preserving the supplied task capability, reasoning operator, and answer target. Resolve every local error code. Never copy the complete gold answer into the question. For BP PRODUCT_IDENTITY, ask for the type or identity of the featured item using a generic referent such as 'the featured product'. For BP ATTRIBUTE_AND_VARIANT, ask which visible or stated feature, variant, quantity, or design detail is highlighted without naming the answer phrase. Keep questions content-specific for CM, SS, and AE by referring to the relevant claim, demonstration, offer, objection, usage situation, or contrast, but do not reveal the answer. Do not add facts, external knowledge, IDs, schemas, evidence instructions, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


QA_QUALITY_SYSTEM_PROMPT = """You are the strict semantic quality gate for SalesBench, a presenter-led e-commerce short-video VQA benchmark. Decide whether one candidate Question and Gold Answer should survive production using only the supplied QuestionSpec, EvidenceUnits, CommerceCues, and CommercialRelations. Return strict JSON with exactly: spec_id, verdict, reason, answerable_from_evidence, gold_supported, unique_answer, task_aligned, content_specific, commerce_relevant, natural_question, non_trivial, commercially_diagnostic, claim_scope_preserved, intended_modality_required, and reference_closed. Every quality field is boolean. verdict must be PASS, REJECT, or HUMAN_REVIEW, and PASS requires every boolean to be true. REJECT formulaic, generic, awkward, answer-leaking, unsupported, overclaimed, wrong-target, wrong-task, non-unique, externally dependent, or low-value background-action items. Reject questions about merely holding, pointing to, showing, rotating, flipping through, or an empty background object unless that observation is necessary to identify the product, verify an offer, perform a functional operation, or establish an observable state change. Spoken seller claims remain claims; conditions, instructions, hypotheticals, and promises must retain their scope. A short demonstration cannot prove long-term efficacy, consumer response, causality, sales, conversion, trust, or popularity. intended_modality_required is false when the question can be answered from only one modality despite claiming a CM capability. For BP require one useful localized fact. For CM require a concrete cross-modal relationship. For SS require a specific observable commercial argument mechanism. For AE allow only one bounded need, scenario, constraint, objection, or decision barrier traceable to the content. Use HUMAN_REVIEW only when two materially plausible supported readings remain; plain insufficiency is REJECT. Write the reason in English, preserve spec_id, and never use external knowledge."""


def build_question_realizer_prompt(
    spec: QuestionSpec,
    evidence_context: list[dict[str, object]],
    cue_context: list[dict[str, object]],
    relation_context: list[dict[str, object]],
) -> tuple[str, str]:
    payload = {
        "question_spec": spec.to_dict(),
        "evidence_context": evidence_context,
        "commerce_cue_context": cue_context,
        "commercial_relation_context": relation_context,
    }
    return QUESTION_REALIZER_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_question_repair_prompt(
    spec: QuestionSpec,
    rejected_question: str,
    local_error_codes: list[str],
    evidence_context: list[dict[str, object]],
    cue_context: list[dict[str, object]],
    relation_context: list[dict[str, object]],
) -> tuple[str, str]:
    payload = {
        "spec_id": spec.spec_id,
        "question_spec": spec.to_dict(),
        "rejected_question": rejected_question,
        "local_error_codes": local_error_codes,
        "evidence_context": evidence_context,
        "commerce_cue_context": cue_context,
        "commercial_relation_context": relation_context,
    }
    return QUESTION_REPAIR_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_qa_quality_prompt(
    spec: QuestionSpec,
    question: str,
    evidence_context: list[dict[str, object]],
    cue_context: list[dict[str, object]],
    relation_context: list[dict[str, object]],
) -> tuple[str, str]:
    user = json.dumps(
        {
            "question_spec": spec.to_dict(),
            "question": question,
            "gold_answer": spec.gold_answer,
            "evidence_units": evidence_context,
            "commerce_cues": cue_context,
            "commercial_relations": relation_context,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return QA_QUALITY_SYSTEM_PROMPT, user
