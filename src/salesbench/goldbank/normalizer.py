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
_FRAME_REFERENCE_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)
_CONFIDENCE_LABELS = {
    "very_high": 0.95,
    "very high": 0.95,
    "high": 0.90,
    "medium": 0.75,
    "moderate": 0.75,
    "low": 0.50,
    "very_low": 0.25,
    "very low": 0.25,
}
_PROPOSER_ALLOWED_SUBTYPES = {
    "consumer": {
        (GoldTaskType.AE, "AUDIENCE_NEED_FIT"),
        (GoldTaskType.AE, "USAGE_CONTEXT"),
        (GoldTaskType.AE, "DECISION_STATE"),
        (GoldTaskType.AE, "CONTENT_MOTIVATION"),
    },
    "operator": {
        (GoldTaskType.CM, "CLAIM_EVIDENCE_RELATION"),
        (GoldTaskType.CM, "CLAIM_PARTIAL_SUPPORT"),
        (GoldTaskType.CM, "TEXT_VISUAL_CONSISTENCY"),
    },
    "strategist": {
        (GoldTaskType.SS, "HOOK_MECHANISM"),
        (GoldTaskType.SS, "VALUE_PROPOSITION"),
        (GoldTaskType.SS, "TRUST_MECHANISM"),
        (GoldTaskType.SS, "OBJECTION_HANDLING"),
        (GoldTaskType.SS, "URGENCY_CTA"),
        (GoldTaskType.SS, "FUNNEL_ROLE"),
    },
}


def normalize_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", clean_text(value)).strip()


def normalize_task_subtype(task_type: GoldTaskType, value: object) -> str:
    subtype = normalize_text(value).upper()
    if subtype not in allowed_subtypes(task_type):
        raise ValueError(f"Unknown subtype for {task_type.value}: {subtype}")
    return subtype


def _normalize_confidence(value: object) -> float:
    text = normalize_text(value).lower()
    if text in _CONFIDENCE_LABELS:
        return _CONFIDENCE_LABELS[text]
    try:
        confidence = float(text)
    except ValueError:
        return 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    return confidence if 0.0 <= confidence <= 1.0 else 0.0


def _normalize_modality(raw: dict[str, object]) -> EvidenceModality:
    value = normalize_text(raw.get("modality") or EvidenceModality.METADATA.value).lower()
    try:
        return EvidenceModality(value)
    except ValueError:
        pass

    if value in {"image", "frame", "photo", "video"}:
        return EvidenceModality.VISUAL
    if value in {"speech", "audio", "subtitle", "subtitles", "voice"}:
        return EvidenceModality.ASR
    if value in {"on_screen_text", "on-screen text", "screen_text"}:
        return EvidenceModality.OCR
    if value != "text":
        return EvidenceModality.METADATA

    source_blob = " ".join(
        [
            normalize_text(raw.get("evidence_id")),
            normalize_text(raw.get("text_span")),
            " ".join(normalize_text(item) for item in raw.get("source_domains", []) or []),
        ]
    ).lower()
    if any(marker in source_blob for marker in ("asr", "subtitle", "speech", "audio", "口播", "字幕")):
        return EvidenceModality.ASR
    if any(marker in source_blob for marker in ("ocr", "on-screen", "screen_text", "画面文字")):
        return EvidenceModality.OCR
    return EvidenceModality.METADATA


def _frame_indices(raw: dict[str, object], source_locator: str) -> tuple[int, ...]:
    supplied = raw.get("frame_indices", []) or []
    if supplied:
        return tuple(int(value) for value in supplied)
    match = _FRAME_REFERENCE_RE.search(source_locator)
    return (int(match.group(1)),) if match else ()


def normalize_evidence_units(video_id: str, raw_units: list[dict[str, object]]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    used_ids: set[str] = set()
    for idx, raw in enumerate(raw_units):
        modality = _normalize_modality(raw)
        subject = normalize_text(raw.get("subject"))
        predicate = normalize_text(raw.get("predicate"))
        if not subject or not predicate or raw.get("value") in (None, ""):
            raise ValueError("Evidence unit requires subject, predicate, and value")
        start_s = raw.get("start_s")
        end_s = raw.get("end_s")
        if start_s is not None and end_s is not None and float(start_s) > float(end_s):
            raise ValueError("Evidence start_s cannot exceed end_s")
        source_locator = normalize_text(raw.get("evidence_id"))
        expected_prefix = f"{normalize_text(video_id)}_{modality.value}_"
        evidence_id = source_locator if source_locator.startswith(expected_prefix) else ""
        if not evidence_id or evidence_id in used_ids:
            evidence_id = make_evidence_id(video_id, modality, idx)
        used_ids.add(evidence_id)
        attributes = dict(raw.get("attributes") or {})
        if source_locator and source_locator != evidence_id:
            attributes.setdefault("source_locator", source_locator)
        text_span = normalize_text(raw.get("text_span"))
        if not text_span and modality in {EvidenceModality.ASR, EvidenceModality.OCR}:
            text_span = normalize_text(raw.get("value"))
        source_domains = tuple(normalize_text(value) for value in raw.get("source_domains", []) or [])
        if not source_domains:
            source_domains = {
                EvidenceModality.ASR: ("C2_audio_speech",),
                EvidenceModality.OCR: ("C6_raw_video",),
                EvidenceModality.VISUAL: ("C6_raw_video",),
            }.get(modality, ())
        units.append(
            EvidenceUnit(
                evidence_id=evidence_id,
                video_id=video_id,
                modality=modality,
                start_s=None if start_s is None else float(start_s),
                end_s=None if end_s is None else float(end_s),
                frame_indices=_frame_indices(raw, source_locator),
                text_span=text_span,
                subject=subject,
                predicate=predicate,
                value=raw.get("value"),
                attributes=attributes,
                source_domains=source_domains,
                extractor=normalize_text(raw.get("extractor") or "objective_evidence_extractor"),
                confidence=_normalize_confidence(raw.get("confidence", 0.0)),
                timestamp_status=normalize_text(raw.get("timestamp_status") or "unavailable"),
            )
        )
    return units


def normalize_proposals(video_id: str, proposer: str, raw: list[dict[str, object]]) -> list[GoldProposal]:
    proposals: list[GoldProposal] = []
    proposer = normalize_text(proposer).lower()
    for idx, record in enumerate(raw):
        payload = dict(record)
        task_type = GoldTaskType(normalize_text(payload.get("task_type")).upper())
        subtype = normalize_task_subtype(task_type, payload.get("task_subtype"))
        if (task_type, subtype) not in _PROPOSER_ALLOWED_SUBTYPES.get(proposer, set()):
            raise ValueError(f"{proposer} proposer cannot emit {task_type.value}/{subtype}")
        target = payload.get("target")
        proposed_gold = payload.get("proposed_gold")
        if not isinstance(target, dict) or not target:
            raise ValueError("Proposal target must be a non-empty object")
        if not isinstance(proposed_gold, dict) or not proposed_gold:
            raise ValueError("Proposal proposed_gold must be a non-empty object")
        payload["task_subtype"] = subtype
        payload["video_id"] = video_id
        payload["source_agent"] = proposer
        proposal_id = normalize_text(payload.get("proposal_id"))
        if proposal_id.lower() in {"optional", "optional string", "none", "null", "n/a"}:
            proposal_id = ""
        payload["proposal_id"] = proposal_id or (
            f"{video_id}_{payload['source_agent']}_{task_type.value.lower()}_{idx:03d}_"
            f"{stable_digest(payload.get('target', {}))}"
        )
        proposals.append(parse_gold_proposal(payload))
    return proposals


def semantic_key(item: GoldProposal | GoldItem) -> tuple[str, str, str, str, str]:
    if isinstance(item, GoldProposal):
        value: Any = item.proposed_gold
    else:
        value = item.gold_value
    return (
        normalize_text(item.video_id),
        item.task_type.value,
        normalize_text(item.task_subtype).upper(),
        stable_digest(item.target),
        stable_digest(value),
    )


def semantic_target_key(item: GoldProposal | GoldItem) -> tuple[str, str, str, str]:
    return (
        normalize_text(item.video_id),
        item.task_type.value,
        normalize_text(item.task_subtype).upper(),
        stable_digest(item.target),
    )
