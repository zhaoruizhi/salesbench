"""Normalization helpers for evidence and grounded annotations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..utils import clean_text, contains_cjk
from .commerce_ontology import relation_rule
from .commerce_schema import (
    CommerceCue,
    CommercialRelation,
    CueType,
    RelationType,
    make_cue_id,
    make_relation_id,
    parse_commerce_cue,
    parse_commercial_relation,
)
from .ontology import allowed_subtypes, default_reasoning_operator
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
_PROPOSER_TASKS = {
    "cm_proposer": GoldTaskType.CM,
    "ss_proposer": GoldTaskType.SS,
    "ae_proposer": GoldTaskType.AE,
}

_MODEL_PLACEHOLDER_TEXT = {
    "english content-specific focus",
    "specific supported claim",
    "ask a specific question about the cited product, offer, claim, or sequence.",
    "a concise english answer bounded by the cited graph.",
}

_DEFAULT_FORBIDDEN_INFERENCES = {
    GoldTaskType.BP: (
        "Do not infer interaction, sales, conversion, or unshown product properties.",
    ),
    GoldTaskType.CM: (
        "Do not treat repeated promotional wording as independent visual proof.",
        "Do not infer outcomes beyond the cited observation window.",
    ),
    GoldTaskType.SS: (
        "Do not claim that the content caused trust, purchase, conversion, or interaction.",
    ),
    GoldTaskType.AE: (
        "Do not infer a real viewer profile, purchase intention, conversion, or popularity.",
    ),
}


def normalize_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", clean_text(value)).strip()


def normalize_task_subtype(task_type: GoldTaskType, value: object) -> str:
    subtype = normalize_text(value).upper()
    if subtype not in allowed_subtypes(task_type):
        subtype = next(
            (
                candidate
                for candidate in allowed_subtypes(task_type)
                if default_reasoning_operator(task_type, candidate) == subtype
            ),
            subtype,
        )
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


def _contains_model_placeholder(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_model_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_model_placeholder(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = normalize_text(value).lower()
    return normalized in _MODEL_PLACEHOLDER_TEXT or (
        normalized.startswith("<") and normalized.endswith(">")
    )


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
            normalize_text(raw.get("source_text_native")),
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


def _normalize_attributes(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return {"items": list(value)} if value else {}
    normalized = normalize_text(value)
    return {"value": normalized} if normalized else {}


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
        attributes = _normalize_attributes(raw.get("attributes"))
        if source_locator and source_locator != evidence_id:
            attributes.setdefault("source_locator", source_locator)
        text_span = normalize_text(raw.get("source_text_native") or raw.get("text_span"))
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
                content_en=normalize_text(raw.get("content_en")),
                source_text_native=text_span,
            )
        )
    return units


def normalize_commerce_cues(
    video_id: str,
    raw_cues: list[dict[str, object]],
    evidence: dict[str, EvidenceUnit],
) -> list[CommerceCue]:
    cues: list[CommerceCue] = []
    seen_ids: set[str] = set()
    for raw in raw_cues:
        payload = dict(raw)
        cue_type = CueType(normalize_text(payload.get("cue_type")).upper())
        content_en = normalize_text(payload.get("content_en"))
        if not content_en:
            raise ValueError("Commerce cue requires content_en")
        if contains_cjk(content_en):
            raise ValueError("Commerce cue content_en must use English")
        evidence_ids = tuple(
            dict.fromkeys(normalize_text(item) for item in payload.get("evidence_ids", []) or [] if normalize_text(item))
        )
        if not evidence_ids:
            raise ValueError("Commerce cue requires evidence_ids")
        for evidence_id in evidence_ids:
            unit = evidence.get(evidence_id)
            if unit is None:
                raise ValueError(f"Unknown evidence id: {evidence_id}")
            if unit.video_id != video_id:
                raise ValueError(f"Evidence {evidence_id} belongs to another video")
        cue_id = make_cue_id(video_id, cue_type, evidence_ids, content_en)
        if cue_id in seen_ids:
            continue
        seen_ids.add(cue_id)
        payload.update(
            {
                "cue_id": cue_id,
                "video_id": video_id,
                "cue_type": cue_type.value,
                "content_en": content_en,
                "source_text_native": normalize_text(payload.get("source_text_native")),
                "evidence_ids": list(evidence_ids),
                "directness": normalize_text(payload.get("directness") or "DIRECT").upper(),
                "extractor": normalize_text(payload.get("extractor") or "commerce_cue_extractor"),
                "confidence": _normalize_confidence(payload.get("confidence", 0.0)),
            }
        )
        cues.append(parse_commerce_cue(payload))
    return cues


def normalize_commercial_relations(
    video_id: str,
    raw_relations: list[dict[str, object]],
    cues: dict[str, CommerceCue],
    evidence: dict[str, EvidenceUnit],
) -> list[CommercialRelation]:
    relations: list[CommercialRelation] = []
    seen_ids: set[str] = set()
    for raw in raw_relations:
        payload = dict(raw)
        relation_type = RelationType(normalize_text(payload.get("relation_type")).upper())
        source_cue_ids = tuple(
            dict.fromkeys(normalize_text(item) for item in payload.get("source_cue_ids", []) or [] if normalize_text(item))
        )
        target_cue_ids = tuple(
            dict.fromkeys(normalize_text(item) for item in payload.get("target_cue_ids", []) or [] if normalize_text(item))
        )
        if not source_cue_ids or not target_cue_ids:
            raise ValueError("Commercial relation requires source and target cue ids")
        endpoint_cues: list[CommerceCue] = []
        for cue_id in (*source_cue_ids, *target_cue_ids):
            cue = cues.get(cue_id)
            if cue is None:
                raise ValueError(f"Unknown cue id: {cue_id}")
            if cue.video_id != video_id:
                raise ValueError(f"Cue {cue_id} belongs to another video")
            endpoint_cues.append(cue)
        supplied_evidence = [
            normalize_text(item)
            for item in payload.get("evidence_ids", []) or []
            if normalize_text(item)
        ]
        endpoint_evidence = [item for cue in endpoint_cues for item in cue.evidence_ids]
        evidence_ids = tuple(dict.fromkeys([*supplied_evidence, *endpoint_evidence]))
        for evidence_id in evidence_ids:
            unit = evidence.get(evidence_id)
            if unit is None:
                raise ValueError(f"Unknown evidence id: {evidence_id}")
            if unit.video_id != video_id:
                raise ValueError(f"Evidence {evidence_id} belongs to another video")
        rationale_en = normalize_text(payload.get("rationale_en"))
        if not rationale_en or contains_cjk(rationale_en):
            raise ValueError("Commercial relation rationale_en must use English")
        relation_id = make_relation_id(video_id, relation_type, source_cue_ids, target_cue_ids)
        if relation_id in seen_ids:
            continue
        seen_ids.add(relation_id)
        payload.update(
            {
                "relation_id": relation_id,
                "video_id": video_id,
                "relation_type": relation_type.value,
                "source_cue_ids": list(source_cue_ids),
                "target_cue_ids": list(target_cue_ids),
                "evidence_ids": list(evidence_ids),
                "rationale_en": rationale_en,
                "provenance": relation_rule(relation_type)["provenance"].value,
                "directness": normalize_text(payload.get("directness") or "INFERRED").upper(),
                "extractor": normalize_text(payload.get("extractor") or "commercial_relation_builder"),
                "confidence": _normalize_confidence(payload.get("confidence", 0.0)),
            }
        )
        relations.append(parse_commercial_relation(payload))
    return relations


def normalize_proposals(
    video_id: str,
    generator: str,
    raw: list[dict[str, object]],
    commerce_cues: set[str] | Mapping[str, CommerceCue] | None = None,
    commercial_relations: set[str] | Mapping[str, CommercialRelation] | None = None,
    evidence_ids: set[str] | None = None,
) -> list[GoldProposal]:
    proposals: list[GoldProposal] = []
    generator = normalize_text(generator).lower()
    if generator not in _PROPOSER_TASKS:
        raise ValueError(f"Unknown task generator: {generator}")
    for idx, record in enumerate(raw):
        payload = dict(record)
        task_type = GoldTaskType(normalize_text(payload.get("task_type")).upper())
        subtype = normalize_task_subtype(task_type, payload.get("task_subtype"))
        if task_type != _PROPOSER_TASKS[generator]:
            raise ValueError(f"{generator} cannot emit {task_type.value}/{subtype}")
        target = payload.get("target")
        proposed_gold = payload.get("proposed_gold")
        if not isinstance(target, dict) or not target:
            raise ValueError("Proposal target must be a non-empty object")
        if not isinstance(proposed_gold, dict) or not proposed_gold:
            raise ValueError("Proposal proposed_gold must be a non-empty object")
        capability = subtype
        reasoning_operator = default_reasoning_operator(task_type, subtype)
        cue_ids = tuple(
            dict.fromkeys(
                normalize_text(value)
                for value in payload.get("commerce_cue_ids", []) or []
                if normalize_text(value)
            )
        )
        relation_ids = tuple(
            dict.fromkeys(
                normalize_text(value)
                for value in payload.get("commercial_relation_ids", []) or []
                if normalize_text(value)
            )
        )
        if commerce_cues is not None and any(cue_id not in commerce_cues for cue_id in cue_ids):
            raise ValueError("Proposal references an unknown CommerceCue")
        if commercial_relations is not None and any(
            relation_id not in commercial_relations for relation_id in relation_ids
        ):
            raise ValueError("Proposal references an unknown CommercialRelation")
        question_intent = normalize_text(payload.get("question_intent")) or (
            f"Ask a specific evidence-grounded question about {subtype.lower().replace('_', ' ')}."
        )
        forbidden_inferences = tuple(
            normalize_text(value)
            for value in payload.get("forbidden_inferences", []) or _DEFAULT_FORBIDDEN_INFERENCES[task_type]
            if normalize_text(value)
        )
        if _contains_model_placeholder((target, proposed_gold, payload.get("reasoning_edges", []), question_intent)):
            raise ValueError("Proposal contains copied schema placeholder text")
        if contains_cjk(
            (
                target,
                proposed_gold,
                payload.get("reasoning_edges", []),
                question_intent,
                forbidden_inferences,
            )
        ):
            raise ValueError("Proposal normalized natural-language fields must use English")
        supplied_evidence_ids = [
            normalize_text(value)
            for value in payload.get("evidence_ids", []) or []
            if normalize_text(value)
        ]
        if evidence_ids is not None:
            resolved_evidence_ids = [value for value in supplied_evidence_ids if value in evidence_ids]
            if isinstance(commerce_cues, Mapping):
                resolved_evidence_ids.extend(
                    evidence_id
                    for cue_id in cue_ids
                    for evidence_id in commerce_cues[cue_id].evidence_ids
                )
            if isinstance(commercial_relations, Mapping):
                resolved_evidence_ids.extend(
                    evidence_id
                    for relation_id in relation_ids
                    for evidence_id in commercial_relations[relation_id].evidence_ids
                )
            supplied_evidence_ids = [
                value for value in dict.fromkeys(resolved_evidence_ids) if value in evidence_ids
            ]
        reasoning_edges: list[list[str]] = []
        if isinstance(commerce_cues, Mapping):
            for cue_id in cue_ids:
                cue = commerce_cues[cue_id]
                for evidence_id in cue.evidence_ids:
                    if evidence_id in supplied_evidence_ids:
                        reasoning_edges.append([evidence_id, cue.content_en, "SUPPORTED"])
        if isinstance(commercial_relations, Mapping):
            for relation_id in relation_ids:
                relation = commercial_relations[relation_id]
                for evidence_id in relation.evidence_ids:
                    if evidence_id in supplied_evidence_ids:
                        reasoning_edges.append(
                            [evidence_id, relation.rationale_en, relation.status]
                        )
        if reasoning_edges:
            reasoning_edges = list(dict.fromkeys(tuple(edge) for edge in reasoning_edges))
            reasoning_edges = [list(edge) for edge in reasoning_edges]
        else:
            reasoning_edges = [
                [normalize_text(part) for part in edge[:3]]
                for edge in payload.get("reasoning_edges", []) or []
                if isinstance(edge, (list, tuple)) and len(edge) >= 3
            ]
        payload["task_subtype"] = subtype
        payload["video_id"] = video_id
        payload["source_agent"] = generator
        payload["capability"] = capability
        payload["reasoning_operator"] = reasoning_operator
        payload["evidence_ids"] = supplied_evidence_ids
        payload["reasoning_edges"] = reasoning_edges
        payload["commerce_cue_ids"] = list(cue_ids)
        payload["commercial_relation_ids"] = list(relation_ids)
        payload["question_intent"] = question_intent
        payload["forbidden_inferences"] = list(forbidden_inferences)
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
