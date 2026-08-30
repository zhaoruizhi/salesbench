"""Resumable runner for natural English QA surface realization."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..goldbank.parsing import ModelOutputError, parse_json_object
from ..goldbank.schema import stable_digest
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..utils import clean_text, contains_cjk
from ..vlm.api_client import VLMClient
from .prompts import (
    QA_QUALITY_PROMPT_VERSION,
    QUESTION_REALIZER_PROMPT_VERSION,
    build_question_repair_prompt,
    build_question_realizer_prompt,
)
from .item_validator import validate_qa_candidate
from .semantic_verifier import QASemanticVerdict, QASemanticVerification, verify_qa_candidate
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


def _fingerprint(
    spec: QuestionSpec,
    model: str,
    *,
    strict_semantic_verification: bool = False,
    verifier_model: str = "",
) -> str:
    return stable_digest(
        {
            "spec": spec.to_dict(),
            "model": model,
            "prompt_version": QUESTION_REALIZER_PROMPT_VERSION,
            "strict_semantic_verification": strict_semantic_verification,
            "quality_prompt_version": (
                QA_QUALITY_PROMPT_VERSION if strict_semantic_verification else "disabled"
            ),
            "verifier_model": verifier_model if strict_semantic_verification else "",
        },
        length=24,
    )


@dataclass(frozen=True)
class _RealizeOutcome:
    spec: QuestionSpec
    realization: QuestionRealization | None = None
    error: str | None = None
    repair_attempted: bool = False
    disposition: str = "PASS"
    verification: QASemanticVerification | None = None
    candidate_snapshot: dict[str, object] | None = None


def _parse_realization(raw: str, spec: QuestionSpec, model: str) -> QuestionRealization:
    payload = parse_json_object(raw, "spec_id")
    if clean_text(payload.get("spec_id")) != spec.spec_id:
        raise ModelOutputError("realizer changed spec_id")
    question = clean_text(payload.get("question"))
    issues = list(
        dict.fromkeys(
            [
                *validate_realized_question(question, spec.gold_answer),
                *validate_qa_candidate(spec, question, spec.gold_answer),
            ]
        )
    )
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
    allow_auto_candidates: bool = False,
    strict_semantic_verification: bool = False,
    semantic_verifier_client: VLMClient | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(evidence_dir / dataset_filename)
    specs = build_question_specs(
        records,
        allow_auto_candidates=allow_auto_candidates,
    )
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

    verifier_client = semantic_verifier_client or client
    realizations: dict[str, QuestionRealization] = {}
    semantic_verifications: dict[str, dict[str, object]] = {}
    resumed_count = 0
    pending: list[QuestionSpec] = []
    for spec in specs:
        part = _part_path(output_dir, spec.spec_id)
        if resume and part.exists():
            try:
                payload = json.loads(part.read_text(encoding="utf-8"))
                fingerprint = _fingerprint(
                    spec,
                    client.model,
                    strict_semantic_verification=strict_semantic_verification,
                    verifier_model=verifier_client.model,
                )
                cached_verification = payload.get("semantic_verification")
                cache_is_acceptable = not strict_semantic_verification or (
                    isinstance(cached_verification, dict)
                    and clean_text(cached_verification.get("verdict")).upper() == "PASS"
                )
                if clean_text(payload.get("fingerprint")) == fingerprint and cache_is_acceptable:
                    realization = QuestionRealization(**payload["realization"])
                    realizations[spec.spec_id] = realization
                    if isinstance(cached_verification, dict):
                        semantic_verifications[spec.spec_id] = cached_verification
                    resumed_count += 1
                    continue
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass
        pending.append(spec)

    failures: list[dict[str, object]] = []
    rejected_candidates: list[dict[str, object]] = []
    human_review_queue: list[dict[str, object]] = []
    pipeline_diagnostics: list[dict[str, object]] = []

    def realize(
        spec: QuestionSpec,
    ) -> _RealizeOutcome:
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
            return _RealizeOutcome(
                spec,
                error=call.error or "api_call_failed",
                disposition="PIPELINE_DIAGNOSTIC",
            )
        realization: QuestionRealization | None = None
        repair_attempted = False
        try:
            realization = _parse_realization(call.raw_response, spec, call.model)
        except (ModelOutputError, ValueError) as exc:
            initial_error = str(exc)
            try:
                rejected_payload = parse_json_object(call.raw_response, "spec_id")
                rejected_question = clean_text(rejected_payload.get("question"))
            except ModelOutputError:
                rejected_question = ""
            local_errors = list(
                dict.fromkeys(
                    [
                        *validate_realized_question(rejected_question, spec.gold_answer),
                        *validate_qa_candidate(spec, rejected_question, spec.gold_answer),
                    ]
                )
            )
            repair_system, repair_user = build_question_repair_prompt(
                spec,
                rejected_question,
                local_errors or [initial_error],
                evidence_context,
                cue_context,
                relation_context,
            )
            repair_call = client.call_text_only(
                repair_system,
                repair_user,
                response_format="json_object",
            )
            repair_attempted = True
            if not repair_call.success:
                return _RealizeOutcome(
                    spec,
                    error=repair_call.error or initial_error,
                    repair_attempted=True,
                    disposition="PIPELINE_DIAGNOSTIC",
                )
            try:
                realization = _parse_realization(
                    repair_call.raw_response,
                    spec,
                    repair_call.model,
                )
            except (ModelOutputError, ValueError) as repair_exc:
                return _RealizeOutcome(
                    spec,
                    error=str(repair_exc),
                    repair_attempted=True,
                    disposition="REJECT",
                    candidate_snapshot={
                        "question_spec": spec.to_dict(),
                        "rejected_question": rejected_question,
                    },
                )

        if realization is None:
            return _RealizeOutcome(
                spec,
                error="realization_missing_after_surface_stage",
                repair_attempted=repair_attempted,
                disposition="PIPELINE_DIAGNOSTIC",
            )
        candidate_snapshot = {
            "question_spec": spec.to_dict(),
            "question": realization.question,
            "evidence_units": evidence_context,
            "commerce_cues": cue_context,
            "commercial_relations": relation_context,
        }
        if not strict_semantic_verification:
            return _RealizeOutcome(
                spec,
                realization=realization,
                repair_attempted=repair_attempted,
                candidate_snapshot=candidate_snapshot,
            )
        try:
            verification = verify_qa_candidate(
                verifier_client,
                spec,
                realization.question,
                evidence_context,
                cue_context,
                relation_context,
            )
        except (ValueError, ModelOutputError) as exc:
            return _RealizeOutcome(
                spec,
                error=str(exc),
                repair_attempted=repair_attempted,
                disposition="PIPELINE_DIAGNOSTIC",
                candidate_snapshot=candidate_snapshot,
            )
        disposition = {
            QASemanticVerdict.PASS: "PASS",
            QASemanticVerdict.REJECT: "REJECT",
            QASemanticVerdict.HUMAN_REVIEW: "HUMAN_REVIEW",
        }[verification.verdict]
        return _RealizeOutcome(
            spec,
            realization=realization if disposition == "PASS" else None,
            repair_attempted=repair_attempted,
            disposition=disposition,
            verification=verification,
            candidate_snapshot=candidate_snapshot,
        )

    workers = max(1, int(max_workers))
    if workers == 1:
        completed = [realize(spec) for spec in pending]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(realize, spec): spec.spec_id for spec in pending}
            for future in as_completed(futures):
                completed.append(future.result())

    repaired_count = 0
    repair_attempted_count = 0
    for outcome in completed:
        spec = outcome.spec
        if outcome.repair_attempted:
            repair_attempted_count += 1
        verification_record = (
            outcome.verification.to_dict() if outcome.verification is not None else None
        )
        if verification_record is not None:
            semantic_verifications[spec.spec_id] = verification_record
        quality_record = {
            "spec_id": spec.spec_id,
            "video_id": spec.video_id,
            "annotation_id": spec.annotation_id,
            "task_type": spec.task_type.value,
            "reason": (
                outcome.verification.reason
                if outcome.verification is not None
                else outcome.error or "unknown"
            ),
            "candidate_snapshot": outcome.candidate_snapshot or {"question_spec": spec.to_dict()},
            "semantic_verification": verification_record,
        }
        if outcome.disposition == "REJECT":
            rejected_candidates.append(
                {
                    **quality_record,
                    "reason_code": (
                        "QA_SEMANTIC_REJECT"
                        if outcome.verification is not None
                        else "QA_LOCAL_VALIDATION_REJECT"
                    ),
                }
            )
        elif outcome.disposition == "HUMAN_REVIEW":
            human_review_queue.append(
                {**quality_record, "reason_code": "QA_SEMANTIC_AMBIGUITY"}
            )
        elif outcome.disposition == "PIPELINE_DIAGNOSTIC":
            diagnostic = {
                **quality_record,
                "reason_code": "QA_PIPELINE_FAILURE",
                "stage": (
                    "semantic_verification"
                    if strict_semantic_verification
                    and isinstance(outcome.candidate_snapshot, dict)
                    and "question" in outcome.candidate_snapshot
                    else "surface_realization"
                ),
            }
            pipeline_diagnostics.append(diagnostic)
            failures.append(
                {
                    "spec_id": spec.spec_id,
                    "video_id": spec.video_id,
                    "annotation_id": spec.annotation_id,
                    "error": outcome.error or "unknown",
                }
            )
        if outcome.realization is None:
            continue
        if outcome.repair_attempted:
            repaired_count += 1
        realizations[spec.spec_id] = outcome.realization
        part = _part_path(output_dir, spec.spec_id)
        part.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            part,
            {
                "fingerprint": _fingerprint(
                    spec,
                    client.model,
                    strict_semantic_verification=strict_semantic_verification,
                    verifier_model=verifier_client.model,
                ),
                "realization": outcome.realization.to_dict(),
                "semantic_verification": verification_record,
            },
        )

    ordered = [realizations[spec.spec_id].to_dict() for spec in specs if spec.spec_id in realizations]
    write_jsonl(output_dir / "qa_realizations.jsonl", ordered)
    write_jsonl(output_dir / "qa_realizer_failures.jsonl", sorted(failures, key=lambda item: item["spec_id"]))
    write_jsonl(
        output_dir / "qa_semantic_verifications.jsonl",
        [semantic_verifications[key] for key in sorted(semantic_verifications)],
    )
    write_jsonl(
        output_dir / "qa_rejected_candidates.jsonl",
        sorted(rejected_candidates, key=lambda item: item["spec_id"]),
    )
    write_jsonl(
        output_dir / "qa_human_review_queue.jsonl",
        sorted(human_review_queue, key=lambda item: item["spec_id"]),
    )
    write_jsonl(
        output_dir / "qa_pipeline_diagnostics.jsonl",
        sorted(pipeline_diagnostics, key=lambda item: item["spec_id"]),
    )
    summary = {
        "prompt_version": QUESTION_REALIZER_PROMPT_VERSION,
        "model": client.model,
        "strict_semantic_verification": strict_semantic_verification,
        "quality_prompt_version": (
            QA_QUALITY_PROMPT_VERSION if strict_semantic_verification else "disabled"
        ),
        "semantic_verifier_model": (
            verifier_client.model if strict_semantic_verification else "disabled"
        ),
        "dataset_file": str(evidence_dir / dataset_filename),
        "allow_auto_candidates": allow_auto_candidates,
        "counts": {
            "specs": len(specs),
            "realized": len(ordered),
            "failed": len(failures),
            "resumed": resumed_count,
            "repair_attempted": repair_attempted_count,
            "repaired": repaired_count,
            "semantic_rejected": sum(
                1
                for item in rejected_candidates
                if item.get("reason_code") == "QA_SEMANTIC_REJECT"
            ),
            "local_rejected": sum(
                1
                for item in rejected_candidates
                if item.get("reason_code") == "QA_LOCAL_VALIDATION_REJECT"
            ),
            "human_review": len(human_review_queue),
            "pipeline_diagnostics": len(pipeline_diagnostics),
        },
        "fingerprint": stable_digest(
            {
                "specs": [spec.to_dict() for spec in specs],
                "model": client.model,
                "strict_semantic_verification": strict_semantic_verification,
                "quality_prompt_version": (
                    QA_QUALITY_PROMPT_VERSION if strict_semantic_verification else "disabled"
                ),
                "semantic_verifier_model": (
                    verifier_client.model if strict_semantic_verification else "disabled"
                ),
            },
            length=24,
        ),
    }
    write_json(output_dir / "qa_realizer_meta.json", summary)
    return summary
