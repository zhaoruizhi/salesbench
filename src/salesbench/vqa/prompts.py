"""All-English prompts for evidence-specific question realization."""

from __future__ import annotations

import json

from .specs import QuestionSpec


QUESTION_REALIZER_PROMPT_VERSION = "question-realizer-prompt-v2"
QA_QUALITY_PROMPT_VERSION = "qa-quality-prompt-v1"


QUESTION_REALIZER_SYSTEM_PROMPT = """You are the SalesBench English Question Realizer. Convert one approved semantic QuestionSpec into one natural, concise, content-specific English question for a presenter-led e-commerce short video benchmark. Preserve the requested task capability and reasoning operator. Name the concrete product, claim, action, offer, condition, comparison, objection, or temporal contrast when the supplied English context supports it. Do not add facts or external knowledge. Do not include the answer in the question. Do not ask generic questions such as 'What mechanism is used?', 'What strategy is used?', 'What audience is targeted?', or 'What evidence supports the answer?'. Never append 'Support the answer with evidence' or similar instructions. Do not mention EvidenceUnit, CommerceCue, CommercialRelation, IDs, schemas, annotations, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


QUESTION_REPAIR_SYSTEM_PROMPT = """You are the SalesBench English Question Surface Repairer. A previous question was rejected by deterministic local validation. Rewrite only the question surface while preserving the supplied task capability, reasoning operator, and answer target. Resolve every local error code. Never copy the complete gold answer into the question. For BP PRODUCT_IDENTITY, ask for the type or identity of the featured item using a generic referent such as 'the featured product'. For BP ATTRIBUTE_AND_VARIANT, ask which visible or stated feature, variant, quantity, or design detail is highlighted without naming the answer phrase. Keep questions content-specific for CM, SS, and AE by referring to the relevant claim, demonstration, offer, objection, usage situation, or contrast, but do not reveal the answer. Do not add facts, external knowledge, IDs, schemas, evidence instructions, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


QA_QUALITY_SYSTEM_PROMPT = """You are the strict semantic quality gate for SalesBench, a presenter-led e-commerce short-video VQA benchmark. Decide whether one candidate Question and Gold Answer should survive production. Use only the supplied QuestionSpec, EvidenceUnits, CommerceCues, and CommercialRelations. Return strict JSON with exactly these fields: spec_id, verdict, reason, answerable_from_evidence, gold_supported, unique_answer, task_aligned, and domain_specific. verdict must be PASS, REJECT, or HUMAN_REVIEW. PASS requires all five boolean fields to be true. REJECT formulaic, generic, answer-leaking, unsupported, overclaimed, wrong-target, wrong-task, non-unique, or externally dependent items. Spoken seller claims are claims, not observed facts; a short demonstration cannot prove long-term efficacy, consumer response, causality, sales, conversion, trust, or popularity. Use HUMAN_REVIEW only when the cited content supports two materially plausible readings that cannot be resolved from the supplied graph. Plain insufficiency is REJECT, not HUMAN_REVIEW. For BP, require one short localized fact. For CM, require a concrete relationship between at least two modalities or claim and demonstration. For SS, require an observable commercial argument mechanism rather than a generic strategy label. For AE, allow only a bounded need, scenario, or decision barrier traceable to visible or spoken content; reject demographic or conversion speculation. Write the reason in English, preserve spec_id exactly, and never use external knowledge."""


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
