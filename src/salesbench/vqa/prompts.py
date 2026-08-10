"""All-English prompts for evidence-specific question realization."""

from __future__ import annotations

import json

from .specs import QuestionSpec


QUESTION_REALIZER_PROMPT_VERSION = "question-realizer-prompt-v2"


QUESTION_REALIZER_SYSTEM_PROMPT = """You are the SalesBench English Question Realizer. Convert one approved semantic QuestionSpec into one natural, concise, content-specific English question for a presenter-led e-commerce short video benchmark. Preserve the requested task capability and reasoning operator. Name the concrete product, claim, action, offer, condition, comparison, objection, or temporal contrast when the supplied English context supports it. Do not add facts or external knowledge. Do not include the answer in the question. Do not ask generic questions such as 'What mechanism is used?', 'What strategy is used?', 'What audience is targeted?', or 'What evidence supports the answer?'. Never append 'Support the answer with evidence' or similar instructions. Do not mention EvidenceUnit, CommerceCue, CommercialRelation, IDs, schemas, annotations, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


QUESTION_REPAIR_SYSTEM_PROMPT = """You are the SalesBench English Question Surface Repairer. A previous question was rejected by deterministic local validation. Rewrite only the question surface while preserving the supplied task capability, reasoning operator, and answer target. Resolve every local error code. Never copy the complete gold answer into the question. For BP PRODUCT_IDENTITY, ask for the type or identity of the featured item using a generic referent such as 'the featured product'. For BP ATTRIBUTE_AND_VARIANT, ask which visible or stated feature, variant, quantity, or design detail is highlighted without naming the answer phrase. Keep questions content-specific for CM, SS, and AE by referring to the relevant claim, demonstration, offer, objection, usage situation, or contrast, but do not reveal the answer. Do not add facts, external knowledge, IDs, schemas, evidence instructions, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


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
