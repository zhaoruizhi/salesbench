"""Evidence-context assembly for SalesBench-QA judge evaluation."""

from __future__ import annotations

from typing import Any

from ..utils import clean_text
from ..vqa.schema import answer_field
from ..goldbank.validators import PRIVATE_KEYS
from .schema import normalize_task_type


def _strip_private(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _strip_private(value)
            for key, value in payload.items()
            if clean_text(key) not in PRIVATE_KEYS
        }
    if isinstance(payload, list):
        return [_strip_private(value) for value in payload]
    return payload


def build_evidence_context(gold_record: dict[str, Any]) -> dict[str, Any]:
    evidence = gold_record.get("evidence_context")
    if evidence is None:
        evidence = gold_record.get("evidence_refs") or []
    if isinstance(evidence, list):
        evidence = [
            {
                key: item[key]
                for key in (
                    "evidence_id",
                    "modality",
                    "content_en",
                    "start_s",
                    "end_s",
                    "frame_indices",
                    "confidence",
                )
                if key in item
            }
            if isinstance(item, dict)
            else item
            for item in evidence
        ]
    return {
        "evidence_refs": _strip_private(evidence),
        "answer_type": clean_text(gold_record.get("answer_type")) or "open",
    }


def build_graph_context(gold_record: dict[str, Any]) -> dict[str, Any]:
    graph = gold_record.get("graph_context")
    if not isinstance(graph, dict):
        return {"commerce_cues": [], "commercial_relations": []}
    cues = []
    for item in graph.get("commerce_cues") or []:
        if isinstance(item, dict):
            cues.append(
                {
                    key: item[key]
                    for key in ("cue_id", "cue_type", "content_en", "evidence_ids", "directness")
                    if key in item
                }
            )
    relations = []
    for item in graph.get("commercial_relations") or []:
        if isinstance(item, dict):
            relations.append(
                {
                    key: item[key]
                    for key in (
                        "relation_id",
                        "relation_type",
                        "source_cue_ids",
                        "target_cue_ids",
                        "evidence_ids",
                        "status",
                        "rationale_en",
                        "provenance",
                    )
                    if key in item
                }
            )
    return {"commerce_cues": cues, "commercial_relations": relations}


def build_judge_payload(gold_record: dict[str, Any], answer_record: dict[str, Any]) -> dict[str, Any]:
    task_type = normalize_task_type(gold_record.get("task_type"))
    payload: dict[str, Any] = {
        "vqa_id": clean_text(gold_record.get("vqa_id")),
        "video_id": clean_text(gold_record.get("video_id")),
        "question": clean_text(gold_record.get("question")),
        "task_type": task_type,
        "task_subtype": clean_text(gold_record.get("task_subtype")),
        "capability": clean_text(gold_record.get("capability")),
        "reasoning_operator": clean_text(gold_record.get("reasoning_operator")),
        "reference_answer": gold_record.get("gold_answer"),
        "model_output": answer_field(answer_record),
        "evidence_context": build_evidence_context(gold_record),
        "graph_context": build_graph_context(gold_record),
    }
    return payload
