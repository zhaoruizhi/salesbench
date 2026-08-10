"""Resumable runner for natural English QA surface realization."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..goldbank.parsing import ModelOutputError, parse_json_object
from ..goldbank.schema import stable_digest
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..utils import clean_text, contains_cjk
from ..vlm.api_client import VLMClient
from .prompts import (
    QUESTION_REALIZER_PROMPT_VERSION,
    build_question_realizer_prompt,
)
from .specs import QuestionRealization, QuestionSpec, build_question_specs


FORMULAIC_MARKERS = (
    "what mechanism",
    "what strategy",
    "what audience",
    "support the answer",
    "support your answer",
    "according to the evidence",
    "based on the evidence",
)


def validate_realized_question(question: str, gold_answer: str) -> list[str]:
    issues: list[str] = []
    normalized = clean_text(question)
    lowered = normalized.lower()
    if not normalized or contains_cjk(normalized):
        issues.append("NON_ENGLISH_QUESTION")
    if normalized and not normalized.endswith("?"):
        issues.append("MISSING_QUESTION_MARK")
    if any(marker in lowered for marker in FORMULAIC_MARKERS):
        issues.append("FORMULAIC_QUESTION")
    answer = clean_text(gold_answer).lower()
    if answer and len(answer) > 3 and answer in lowered:
        issues.append("ANSWER_LEAKAGE")
    return issues


def _safe_context(record: dict[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: record[field] for field in fields if field in record}


def _context_by_ids(
    ids: tuple[str, ...],
    lookup: dict[str, dict[str, object]],
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    return [_safe_context(lookup[item_id], fields) for item_id in ids if item_id in lookup]


def _part_path(output_dir: Path, spec_id: str) -> Path:
    return output_dir / ".parts" / f"{spec_id}.json"


def _fingerprint(spec: QuestionSpec, model: str) -> str:
    return stable_digest(
        {
            "spec": spec.to_dict(),
            "model": model,
            "prompt_version": QUESTION_REALIZER_PROMPT_VERSION,
        },
        length=24,
    )


def _parse_realization(raw: str, spec: QuestionSpec, model: str) -> QuestionRealization:
    payload = parse_json_object(raw, "spec_id")
    if clean_text(payload.get("spec_id")) != spec.spec_id:
        raise ModelOutputError("realizer changed spec_id")
    question = clean_text(payload.get("question"))
    issues = validate_realized_question(question, spec.gold_answer)
    if issues:
        raise ModelOutputError(",".join(issues))
    return QuestionRealization(
        spec_id=spec.spec_id,
        video_id=spec.video_id,
        annotation_id=spec.annotation_id,
        question=question,
        model=model,
        prompt_version=QUESTION_REALIZER_PROMPT_VERSION,
    )


def run_qa_realizer(
    evidence_dir: Path,
    output_dir: Path,
    client: VLMClient,
    *,
    dataset_filename: str = "video_evidence_dataset.jsonl",
    max_workers: int = 1,
    resume: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(evidence_dir / dataset_filename)
    specs = build_question_specs(records)
    write_jsonl(output_dir / "qa_specs.jsonl", [spec.to_dict() for spec in specs])

    evidence_lookup = {
        clean_text(item.get("evidence_id")): item
        for item in read_jsonl(evidence_dir / "evidence_units.jsonl")
    }
    cue_lookup = {
        clean_text(item.get("cue_id")): item
        for item in read_jsonl(evidence_dir / "commerce_cues.jsonl")
    }
    relation_lookup = {
        clean_text(item.get("relation_id")): item
        for item in read_jsonl(evidence_dir / "commercial_relations.jsonl")
    }

    realizations: dict[str, QuestionRealization] = {}
    resumed_count = 0
    pending: list[QuestionSpec] = []
    for spec in specs:
        part = _part_path(output_dir, spec.spec_id)
        if resume and part.exists():
            try:
                payload = json.loads(part.read_text(encoding="utf-8"))
                if clean_text(payload.get("fingerprint")) == _fingerprint(spec, client.model):
                    realization = QuestionRealization(**payload["realization"])
                    realizations[spec.spec_id] = realization
                    resumed_count += 1
                    continue
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass
        pending.append(spec)

    failures: list[dict[str, object]] = []

    def realize(spec: QuestionSpec) -> tuple[QuestionSpec, QuestionRealization | None, str | None]:
        evidence_context = _context_by_ids(
            spec.evidence_refs,
            evidence_lookup,
            ("evidence_id", "modality", "content_en", "start_s", "end_s", "frame_indices"),
        )
        cue_context = _context_by_ids(
            spec.commerce_cue_ids,
            cue_lookup,
            ("cue_id", "cue_type", "content_en", "evidence_ids", "attributes"),
        )
        relation_context = _context_by_ids(
            spec.commercial_relation_ids,
            relation_lookup,
            (
                "relation_id",
                "relation_type",
                "source_cue_ids",
                "target_cue_ids",
                "status",
                "rationale_en",
            ),
        )
        system, user = build_question_realizer_prompt(
            spec,
            evidence_context,
            cue_context,
            relation_context,
        )
        call = client.call_text_only(system, user, response_format="json_object")
        if not call.success:
            return spec, None, call.error or "api_call_failed"
        try:
            return spec, _parse_realization(call.raw_response, spec, call.model), None
        except (ModelOutputError, ValueError) as exc:
            return spec, None, str(exc)

    workers = max(1, int(max_workers))
    if workers == 1:
        completed = [realize(spec) for spec in pending]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(realize, spec): spec.spec_id for spec in pending}
            for future in as_completed(futures):
                completed.append(future.result())

    for spec, realization, error in completed:
        if realization is None:
            failures.append(
                {
                    "spec_id": spec.spec_id,
                    "video_id": spec.video_id,
                    "annotation_id": spec.annotation_id,
                    "error": error or "unknown",
                }
            )
            continue
        realizations[spec.spec_id] = realization
        part = _part_path(output_dir, spec.spec_id)
        part.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            part,
            {
                "fingerprint": _fingerprint(spec, client.model),
                "realization": realization.to_dict(),
            },
        )

    ordered = [realizations[spec.spec_id].to_dict() for spec in specs if spec.spec_id in realizations]
    write_jsonl(output_dir / "qa_realizations.jsonl", ordered)
    write_jsonl(output_dir / "qa_realizer_failures.jsonl", sorted(failures, key=lambda item: item["spec_id"]))
    summary = {
        "prompt_version": QUESTION_REALIZER_PROMPT_VERSION,
        "model": client.model,
        "dataset_file": str(evidence_dir / dataset_filename),
        "counts": {
            "specs": len(specs),
            "realized": len(ordered),
            "failed": len(failures),
            "resumed": resumed_count,
        },
        "fingerprint": stable_digest(
            {"specs": [spec.to_dict() for spec in specs], "model": client.model},
            length=24,
        ),
    }
    write_json(output_dir / "qa_realizer_meta.json", summary)
    return summary
