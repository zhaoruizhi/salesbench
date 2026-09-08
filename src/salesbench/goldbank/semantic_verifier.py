"""Batched, frame-aware verification for accepted commercial relations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from ..utils import clean_text
from ..vlm.api_client import APICallResult, VLMClient
from .commerce_schema import CommerceCue, CommercialRelation
from .quality_prompts import build_relation_verifier_prompt
from .schema import EvidenceUnit


class SemanticVerdict(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class SemanticVerification:
    relation_id: str
    verdict: SemanticVerdict
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relation_id": self.relation_id,
            "verdict": self.verdict.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SemanticVerificationBatch:
    verifications: tuple[SemanticVerification, ...]
    call: APICallResult


def parse_semantic_verifications(
    raw_response: str,
    expected_relation_ids: set[str],
) -> list[SemanticVerification]:
    try:
        payload = json.loads(raw_response or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid semantic verifier JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("verifications"), list):
        raise ValueError("semantic verifier response requires a verifications list")
    parsed: list[SemanticVerification] = []
    seen: set[str] = set()
    for raw in payload["verifications"]:
        if not isinstance(raw, dict):
            raise ValueError("semantic verification rows must be objects")
        relation_id = clean_text(raw.get("relation_id"))
        if relation_id not in expected_relation_ids:
            raise ValueError(f"unknown relation_id from semantic verifier: {relation_id}")
        if relation_id in seen:
            raise ValueError(f"duplicate relation_id from semantic verifier: {relation_id}")
        reason = clean_text(raw.get("reason"))
        if not reason:
            raise ValueError(f"semantic verifier reason is required for {relation_id}")
        try:
            verdict = SemanticVerdict(clean_text(raw.get("verdict")).upper())
        except ValueError as exc:
            raise ValueError(f"invalid semantic verifier verdict for {relation_id}") from exc
        seen.add(relation_id)
        parsed.append(SemanticVerification(relation_id, verdict, reason))
    missing = expected_relation_ids - seen
    if missing:
        raise ValueError(f"semantic verifier omitted relation_ids: {sorted(missing)}")
    return parsed


def cited_frame_indices(
    relation: CommercialRelation,
    evidence: dict[str, EvidenceUnit],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                frame_index
                for evidence_id in relation.evidence_ids
                if evidence_id in evidence
                for frame_index in evidence[evidence_id].frame_indices
            }
        )
    )


def verify_relations(
    client: VLMClient,
    video_id: str,
    relations: list[CommercialRelation],
    cues: dict[str, CommerceCue],
    evidence: dict[str, EvidenceUnit],
    frame_images: dict[int, str],
) -> SemanticVerificationBatch:
    relation_ids = {relation.relation_id for relation in relations}
    cited_cue_ids = {
        cue_id
        for relation in relations
        for cue_id in (*relation.source_cue_ids, *relation.target_cue_ids)
    }
    cited_evidence_ids = {
        evidence_id for relation in relations for evidence_id in relation.evidence_ids
    }
    system, user = build_relation_verifier_prompt(
        video_id,
        [relation.to_dict() for relation in relations],
        [cues[cue_id].to_dict() for cue_id in sorted(cited_cue_ids) if cue_id in cues],
        [evidence[evidence_id].to_dict() for evidence_id in sorted(cited_evidence_ids) if evidence_id in evidence],
    )
    frame_indices = sorted(
        {
            frame_index
            for relation in relations
            for frame_index in cited_frame_indices(relation, evidence)
            if frame_index in frame_images
        }
    )
    content: list[dict[str, object]] = [{"type": "text", "text": user}]
    for frame_index in frame_indices:
        content.extend(
            [
                {"type": "text", "text": f"[CITED_FRAME frame_index={frame_index}]"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_images[frame_index]}"},
                },
            ]
        )
    call = client.call(system, content, response_format="json_object")
    if not call.success:
        raise ValueError(call.error or "semantic verifier model call failed")
    return SemanticVerificationBatch(
        tuple(parse_semantic_verifications(call.raw_response, relation_ids)),
        call,
    )
