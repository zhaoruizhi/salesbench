"""Normalization helpers for evidence and grounded annotations."""

from __future__ import annotations

import re
from typing import Any

from ..utils import clean_text
from .ontology import allowed_subtypes
from .schema import (
    EvidenceModality,
    EvidenceUnit,
    GoldItem,
    GoldProposal,
    GoldTaskType,
    make_evidence_id,
    parse_gold_proposal,
    stable_digest,
)


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", clean_text(value)).strip()


def normalize_task_subtype(task_type: GoldTaskType, value: object) -> str:
    subtype = normalize_text(value).upper()
    if subtype not in allowed_subtypes(task_type):
        raise ValueError(f"Unknown subtype for {task_type.value}: {subtype}")
    return subtype


def normalize_evidence_units(video_id: str, raw_units: list[dict[str, object]]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for idx, raw in enumerate(raw_units):
        modality = EvidenceModality(normalize_text(raw.get("modality") or EvidenceModality.METADATA.value))
        subject = normalize_text(raw.get("subject"))
        predicate = normalize_text(raw.get("predicate"))
        if not subject or not predicate or raw.get("value") in (None, ""):
            raise ValueError("Evidence unit requires subject, predicate, and value")
        start_s = raw.get("start_s")
        end_s = raw.get("end_s")
        if start_s is not None and end_s is not None and float(start_s) > float(end_s):
            raise ValueError("Evidence start_s cannot exceed end_s")
        evidence_id = normalize_text(raw.get("evidence_id")) or make_evidence_id(video_id, modality, idx)
        units.append(
            EvidenceUnit(
                evidence_id=evidence_id,
                video_id=video_id,
                modality=modality,
                start_s=None if start_s is None else float(start_s),
                end_s=None if end_s is None else float(end_s),
                frame_indices=tuple(int(value) for value in raw.get("frame_indices", []) or []),
                text_span=normalize_text(raw.get("text_span")),
                subject=subject,
                predicate=predicate,
                value=raw.get("value"),
                attributes=dict(raw.get("attributes") or {}),
                source_domains=tuple(normalize_text(value) for value in raw.get("source_domains", []) or []),
                extractor=normalize_text(raw.get("extractor") or "objective_evidence_extractor"),
                confidence=float(raw.get("confidence", 0.0)),
                timestamp_status=normalize_text(raw.get("timestamp_status") or "unavailable"),
            )
        )
    return units


def normalize_proposals(video_id: str, proposer: str, raw: list[dict[str, object]]) -> list[GoldProposal]:
    proposals: list[GoldProposal] = []
    for idx, record in enumerate(raw):
        payload = dict(record)
        task_type = GoldTaskType(normalize_text(payload.get("task_type")).upper())
        subtype = normalize_task_subtype(task_type, payload.get("task_subtype"))
        payload["task_subtype"] = subtype
        payload["video_id"] = video_id
        payload["source_agent"] = normalize_text(payload.get("source_agent") or proposer)
        payload["proposal_id"] = normalize_text(payload.get("proposal_id")) or (
            f"{video_id}_{payload['source_agent']}_{task_type.value.lower()}_{idx:03d}_"
            f"{stable_digest(payload.get('target', {}))}"
        )
        proposals.append(parse_gold_proposal(payload))
    return proposals


def semantic_key(item: GoldProposal | GoldItem) -> tuple[str, str, str, str]:
    if isinstance(item, GoldProposal):
        value: Any = item.proposed_gold
    else:
        value = item.gold_value
    return (
        item.task_type.value,
        normalize_text(item.task_subtype).upper(),
        stable_digest(item.target),
        stable_digest(value),
    )


def semantic_target_key(item: GoldProposal | GoldItem) -> tuple[str, str, str]:
    return (
        item.task_type.value,
        normalize_text(item.task_subtype).upper(),
        stable_digest(item.target),
    )
