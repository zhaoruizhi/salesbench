"""All-English prompts for evidence-specific question realization."""

from __future__ import annotations

import json

from .specs import QuestionSpec


QUESTION_REALIZER_PROMPT_VERSION = "question-realizer-prompt-v1"


QUESTION_REALIZER_SYSTEM_PROMPT = """You are the SalesBench English Question Realizer. Convert one approved semantic QuestionSpec into one natural, concise, content-specific English question for a presenter-led e-commerce short video benchmark. Preserve the requested task capability and reasoning operator. Name the concrete product, claim, action, offer, condition, comparison, objection, or temporal contrast when the supplied English context supports it. Do not add facts or external knowledge. Do not include the answer in the question. Do not ask generic questions such as 'What mechanism is used?', 'What strategy is used?', 'What audience is targeted?', or 'What evidence supports the answer?'. Never append 'Support the answer with evidence' or similar instructions. Do not mention EvidenceUnit, CommerceCue, CommercialRelation, IDs, schemas, annotations, or benchmark labels. Return strict JSON with exactly spec_id and question. The question must be English and end with a question mark."""


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
