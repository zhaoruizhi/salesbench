"""EvidenceDataset quality audit helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from ..utils import clean_text, contains_cjk
from .schema import GoldItem, GoldTier, parse_gold_item
from .validators import PRIVATE_KEYS, find_duplicate_and_conflicting_items


def _contains_private(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(clean_text(key) in PRIVATE_KEYS or _contains_private(value) for key, value in payload.items())
    if isinstance(payload, list):
        return any(_contains_private(value) for value in payload)
    return False


def _evidence_lookup(evidence: list[dict[str, object]]) -> dict[str, str]:
    return {
        clean_text(record.get("evidence_id")): clean_text(record.get("video_id"))
        for record in evidence
        if clean_text(record.get("evidence_id"))
    }


def _parse_items(records: list[dict[str, object]]) -> list[GoldItem]:
    items: list[GoldItem] = []
    for record in records:
        annotations = record.get("grounded_annotations")
        if annotations is None:
            annotations = record.get("gold_items", [])
        for raw_item in annotations or []:
            if isinstance(raw_item, dict) and raw_item.get("video_id"):
                try:
                    items.append(parse_gold_item(raw_item))
                except ValueError:
                    continue
    return items


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _cjk_string_count(payload: object) -> int:
    if isinstance(payload, dict):
        return sum(_cjk_string_count(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_cjk_string_count(value) for value in payload)
    return int(isinstance(payload, str) and contains_cjk(payload))


def _translation_metrics(
    evidence: list[dict[str, object]],
    cues: list[dict[str, object]],
    relations: list[dict[str, object]],
    translations: list[dict[str, object]],
) -> tuple[float, int]:
    sources: dict[tuple[str, str, str], str] = {}
    for rows, object_type, id_field, text_field in (
        (evidence, "evidence_unit", "evidence_id", "content_en"),
        (cues, "commerce_cue", "cue_id", "content_en"),
        (relations, "commercial_relation", "relation_id", "rationale_en"),
    ):
        for row in rows:
            text = clean_text(row.get(text_field))
            object_id = clean_text(row.get(id_field))
            if text and object_id:
                sources[(object_type, object_id, text_field)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    current = 0
    stale = 0
    seen: set[tuple[str, str, str]] = set()
    for row in translations:
        key = (
            clean_text(row.get("object_type")),
            clean_text(row.get("object_id")),
            clean_text(row.get("source_field")),
        )
        if key not in sources or key in seen:
            continue
        seen.add(key)
        if row.get("audit_only") is True and clean_text(row.get("source_sha256")) == sources[key]:
            current += 1
        else:
            stale += 1
    return _ratio(current, len(sources)), stale


def audit_gold_bank(
    records: list[dict[str, object]],
    evidence: list[dict[str, object]],
    commerce_cues: list[dict[str, object]] | None = None,
    commercial_relations: list[dict[str, object]] | None = None,
    audit_translations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    commerce_cues = commerce_cues or []
    commercial_relations = commercial_relations or []
    audit_translations = audit_translations or []
    items = _parse_items(records)
    evidence_by_id = _evidence_lookup(evidence)
    task_counts = Counter(item.task_type.value for item in items)
    tier_counts = Counter(item.gold_tier.value for item in items)
    valid_evidence_items = 0
    cross_modal_items = 0
    for item in items:
        refs = list(item.evidence_ids)
        if refs and all(ref in evidence_by_id and evidence_by_id[ref] == item.video_id for ref in refs):
            valid_evidence_items += 1
        if item.task_type.value == "CM" and len(set(refs)) >= 2:
            cross_modal_items += 1

    duplicate_conflict = find_duplicate_and_conflicting_items(items)
    duplicate_count = sum(1 for issue in duplicate_conflict if issue.code == "DUPLICATE_SEMANTICS")
    conflict_count = sum(1 for issue in duplicate_conflict if issue.code == "CONFLICTING_VALUE")
    video_ids = [clean_text(record.get("video_id")) for record in records if clean_text(record.get("video_id"))]
    by_video_task = {(item.video_id, item.task_type.value) for item in items}
    missing_required = [
        video_id
        for video_id in video_ids
        if (video_id, "BP") not in by_video_task or (video_id, "CM") not in by_video_task
    ]
    private_leakage = 0
    for record in records:
        if _contains_private(record):
            private_leakage += 1
    item_count = len(items)
    cue_by_id = {
        clean_text(row.get("cue_id")): row
        for row in commerce_cues
        if clean_text(row.get("cue_id"))
    }
    relation_by_id = {
        clean_text(row.get("relation_id")): row
        for row in commercial_relations
        if clean_text(row.get("relation_id"))
    }
    cue_evidence_valid = 0
    for cue in commerce_cues:
        video_id = clean_text(cue.get("video_id"))
        refs = [clean_text(value) for value in cue.get("evidence_ids") or []]
        if refs and all(ref in evidence_by_id and evidence_by_id[ref] == video_id for ref in refs):
            cue_evidence_valid += 1
    valid_relations = 0
    valid_provenance = 0
    allowed_provenance = {"THEORY_DIRECT", "THEORY_OPERATIONALIZED", "BENCHMARK_OPERATIONAL"}
    causal_outcome_types = {"INCREASES_TRUST", "CAUSES_PURCHASE", "IMPROVES_CONVERSION"}
    causal_outcome_relations = 0
    for relation in commercial_relations:
        video_id = clean_text(relation.get("video_id"))
        source_ids = [clean_text(value) for value in relation.get("source_cue_ids") or []]
        target_ids = [clean_text(value) for value in relation.get("target_cue_ids") or []]
        refs = [clean_text(value) for value in relation.get("evidence_ids") or []]
        endpoints = source_ids + target_ids
        endpoints_valid = bool(source_ids and target_ids) and all(
            cue_id in cue_by_id and clean_text(cue_by_id[cue_id].get("video_id")) == video_id
            for cue_id in endpoints
        )
        evidence_valid = bool(refs) and all(
            ref in evidence_by_id and evidence_by_id[ref] == video_id for ref in refs
        )
        if endpoints_valid and evidence_valid:
            valid_relations += 1
        provenance = clean_text(relation.get("provenance")).upper()
        if provenance in allowed_provenance:
            valid_provenance += 1
        relation_type = clean_text(relation.get("relation_type")).upper()
        rationale = clean_text(relation.get("rationale_en")).lower()
        if relation_type in causal_outcome_types or any(
            phrase in rationale
            for phrase in ("causes purchase", "improves conversion", "increases trust", "causes sales")
        ):
            causal_outcome_relations += 1
    cue_covered_items = sum(
        1
        for item in items
        if item.commerce_cue_ids
        and all(cue_id in cue_by_id for cue_id in item.commerce_cue_ids)
    )
    relation_covered_items = sum(
        1
        for item in items
        if item.commercial_relation_ids
        and all(relation_id in relation_by_id for relation_id in item.commercial_relation_ids)
    )
    capability_operator_items = sum(
        1 for item in items if clean_text(item.capability) and clean_text(item.reasoning_operator)
    )
    canonical_chinese_fields = sum(
        _cjk_string_count(
            {
                "target": item.target,
                "gold_value": item.gold_value,
                "question_intent": item.question_intent,
            }
        )
        for item in items
    )
    canonical_chinese_fields += sum(
        int(contains_cjk(clean_text(row.get("content_en")))) for row in evidence
    )
    canonical_chinese_fields += sum(
        int(contains_cjk(clean_text(row.get("content_en")))) for row in commerce_cues
    )
    canonical_chinese_fields += sum(
        int(contains_cjk(clean_text(row.get("rationale_en")))) for row in commercial_relations
    )
    translation_coverage, translation_stale = _translation_metrics(
        evidence, commerce_cues, commercial_relations, audit_translations
    )
    return {
        "video_count": len(video_ids),
        "item_count": item_count,
        "item_count_by_task": dict(task_counts),
        "item_count_by_quality": {
            "DIRECT": tier_counts.get(GoldTier.GOLD_A.value, 0),
            "INFERRED": tier_counts.get(GoldTier.GOLD_B.value, 0),
            "NEEDS_REVIEW": tier_counts.get(GoldTier.SILVER.value, 0),
            "REJECTED": tier_counts.get(GoldTier.REJECTED.value, 0),
        },
        "evidence_coverage": round(valid_evidence_items / item_count, 6) if item_count else 0.0,
        "cross_modal_coverage": round(cross_modal_items / max(task_counts.get("CM", 0), 1), 6) if task_counts.get("CM", 0) else 0.0,
        "duplicate_rate": round(duplicate_count / item_count, 6) if item_count else 0.0,
        "conflict_rate": round(conflict_count / item_count, 6) if item_count else 0.0,
        "needs_review_rate": round(tier_counts.get(GoldTier.SILVER.value, 0) / item_count, 6) if item_count else 0.0,
        "human_review_rate": round(sum(1 for item in items if item.review_status == "human_accepted") / item_count, 6) if item_count else 0.0,
        "abstention_count": sum(len(record.get("abstentions", []) or []) for record in records),
        "videos_missing_required_tasks": missing_required,
        "private_field_leakage": private_leakage,
        "commerce_cue_count": len(commerce_cues),
        "commercial_relation_count": len(commercial_relations),
        "commerce_cue_coverage": _ratio(cue_covered_items, item_count),
        "commerce_cue_evidence_validity": _ratio(cue_evidence_valid, len(commerce_cues)),
        "commercial_relation_coverage": _ratio(relation_covered_items, item_count),
        "commercial_relation_validity": _ratio(valid_relations, len(commercial_relations)),
        "relation_provenance_validity": _ratio(valid_provenance, len(commercial_relations)),
        "capability_operator_coverage": _ratio(capability_operator_items, item_count),
        "causal_outcome_relation_count": causal_outcome_relations,
        "canonical_chinese_field_count": canonical_chinese_fields,
        "audit_translation_coverage": translation_coverage,
        "audit_translation_stale_count": translation_stale,
    }
