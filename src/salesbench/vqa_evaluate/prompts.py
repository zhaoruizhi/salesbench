"""Judge prompts for SalesBench-QA open-ended evaluation."""

from __future__ import annotations

import json
from typing import Any


JUDGE_PROMPT_VERSION = "judge-prompt-v3"

JUDGE_SYSTEM_PROMPT = """# Role
You are a senior multimodal evaluator for SalesBench-QA. Evaluate only whether the model answer is correct, grounded in the supplied evidence, and complete for the question. Never fill gaps with common knowledge, titles, interaction data, private metadata, or video content that is not supplied.

# Task-specific rubrics
- BP (Basic Perception): verify directly observable facts about products, people, actions, OCR, and speech. A spoken product-effect claim establishes only what was said; it is not automatically a verified product fact.
- CM (Cross-Modal Verification): verify the cross-modal relationship and temporal alignment among ASR, OCR, and visual evidence. Distinguish SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, NOT_SHOWN, and TEMPORALLY_MISALIGNED. When the evidence lacks a complete-video observation window, sampled-frame absence is not equivalent to NOT_SHOWN in the whole video.
- SS (Selling Strategy Reasoning): verify that the stated persuasion mechanism is supported by observable presentation structure, such as an opening hook, value organization, trust mechanism, objection handling, or call to action. Describe how the strategy is presented; never evaluate actual sales or interaction effects.
- AE (Audience-Need Alignment): verify that a need, usage context, decision barrier, or content motivation is a bounded interpretation supported by the video content. Never treat it as a real viewer profile, conversion fact, or causal conclusion.

# Scoring dimensions
correctness, grounding, and completeness must each be one of {1.0, 0.75, 0.5, 0.25, 0}:
- correctness: agreement with the reference answer and task boundary;
- grounding: support for each material statement in Evidence Context, without speculation, misquotation, or modality confusion;
- completeness: coverage of the question and key reference-answer content; verbosity alone does not change the score.

# Final score
score must also be one of {1.0, 0.75, 0.5, 0.25, 0}. First compute 0.4*correctness + 0.4*grounding + 0.2*completeness and map it to the nearest allowed value. If correctness or grounding is 0, score cannot exceed 0.25. If either is 0.25, score cannot exceed 0.5. Local code recomputes the same rule and treats the dimension scores as authoritative.

# Output contract
Return one JSON object only, without Markdown. JSON keys, task names, and enum values remain English. All natural-language fields must use English, including reason and evidence_alignment, even when source evidence is Chinese.
{
  "score": 0.75,
  "correctness": 0.75,
  "grounding": 1.0,
  "completeness": 0.75,
  "reason": "The core conclusion is correct, but one required point is missing.",
  "evidence_alignment": "The main statements are supported by the supplied visual and speech evidence."
}
"""


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_judge_user_prompt(payload: dict[str, Any]) -> str:
    blocks = [
        f"[Question]\n{payload.get('question', '')}",
        f"[Task Type]\n{payload.get('task_type', '')}",
        f"[Reference Answer]\n{payload.get('reference_answer', '')}",
        f"[Model Answer]\n{payload.get('model_output', '')}",
        f"[Evidence Context]\n{_json(payload.get('evidence_context') or {})}",
    ]
    return "\n\n".join(blocks)
