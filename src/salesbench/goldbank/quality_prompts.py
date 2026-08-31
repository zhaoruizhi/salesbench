"""Strict quality-control prompts used after deterministic local validation."""

from __future__ import annotations

import json


QUALITY_PROMPT_VERSION = "quality-gate-prompt-v2"


def build_relation_verifier_prompt(
    video_id: str,
    relation_records: list[dict[str, object]],
    cue_records: list[dict[str, object]],
    evidence_records: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the strict frame-aware semantic verifier for SalesBench. Verify only whether each supplied "
        "CommercialRelation is entailed by its cited EvidenceUnits, CommerceCues, and attached frames. "
        "Return strict JSON with exactly one top-level key, verifications. Emit exactly one result for every "
        "supplied relation_id and never invent, rename, omit, or merge IDs. Each result must contain "
        "relation_id, verdict, and reason. verdict must be PASS, REJECT, or AMBIGUOUS. PASS requires the "
        "source and target propositions, values, modality relationship, and temporal scope to match. REJECT "
        "unsupported, contradictory, overclaimed, wrong-subject, wrong-value, or wrong-time-scope relations. "
        "Use AMBIGUOUS only when the supplied evidence admits multiple materially different readings that "
        "cannot be resolved from the attached frames; missing or plainly insufficient support is REJECT. "
        "A short frame or clip cannot prove durability, long-term effect, consumer response, causality, trust, "
        "purchase, conversion, or popularity. For CLAIM_SUPPORTED_BY_DEMONSTRATION, a frame that merely "
        "shows the product, a feature, colored markings, an applicator, or one method step does not prove "
        "the claimed benefit or effect. For example, seeing colored markings does not prove right-brain memory, "
        "and pointing at vocabulary does not prove easier memorization. Such relations are REJECT unless the "
        "claimed state or outcome itself is independently observable within the cited window. Do not use "
        "external knowledge. Write reasons in English only."
    )
    user = json.dumps(
        {
            "video_id": video_id,
            "relations": relation_records,
            "commerce_cues": cue_records,
            "evidence_units": evidence_records,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return system, user


def build_proposal_repair_prompt(
    video_id: str,
    proposal: dict[str, object],
    review: dict[str, object],
    evidence_records: list[dict[str, object]],
    cue_records: list[dict[str, object]],
    relation_records: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "You are the SalesBench proposal repairer. Repair one proposal in response to the Challenger's "
        "specific issues. Return strict JSON with exactly one top-level key, repaired_proposal. Preserve the "
        "original proposal_id, video_id, source_agent, and task_type. You may only use supplied evidence_ids, "
        "commerce_cue_ids, and commercial_relation_ids; never invent or rename an ID. Keep the same task "
        "capability unless the review explicitly identifies a subtype mismatch. The repaired target, gold, "
        "question intent, and reasoning must be fully supported by the cited graph. Do not claim consumer "
        "outcomes or facts beyond the observation window. If no valid repair exists, return repaired_proposal "
        "as null. Use English only. This is the only repair attempt."
    )
    user = json.dumps(
        {
            "video_id": video_id,
            "proposal": proposal,
            "challenger_review": review,
            "evidence_units": evidence_records,
            "commerce_cues": cue_records,
            "commercial_relations": relation_records,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return system, user
