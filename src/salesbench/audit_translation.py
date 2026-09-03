"""Audit-only translation job collection, caching, and OpenAI-compatible runner."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .audit_translation_prompts import (
    AUDIT_TRANSLATION_PROMPT_VERSION,
    build_audit_translation_prompt,
)
from .goldbank.parsing import ModelOutputError, parse_json_object
from .goldbank.validators import PRIVATE_KEYS
from .goldbank.prompts import (
    BP_COMPILER_CONTRACT,
    build_adjudicator_prompt,
    build_challenger_prompt,
    build_commerce_cue_prompt,
    build_commercial_relation_prompt,
    build_evidence_extractor_prompt,
    build_language_evidence_prompt,
    build_proposer_prompt,
    build_visual_evidence_prompt,
    build_visual_evidence_repair_prompt,
    build_visual_commerce_cue_prompt,
)
from .io_utils import read_json, read_jsonl, write_jsonl
from .utils import clean_text, contains_cjk
from .vlm.api_client import VLMClient
from .vqa.prompts import (
    QA_QUALITY_SYSTEM_PROMPT,
    QUESTION_REALIZER_SYSTEM_PROMPT,
    QUESTION_REPAIR_SYSTEM_PROMPT,
)
from .vqa_baseline.prompts import CLOSED_SOURCE_SYSTEM_PROMPT
from .vqa_evaluate.prompts import JUDGE_SYSTEM_PROMPT


_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_CONTROLLED_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_MACHINE_CODE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
_MACHINE_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_CACHEABLE_PROMPT_VERSIONS = {
    "audit-translation-prompt-v1",
    AUDIT_TRANSLATION_PROMPT_VERSION,
}


@dataclass(frozen=True)
class TranslationJob:
    translation_id: str
    object_type: str
    object_id: str
    source_field: str
    source_text: str
    source_language: str
    target_language: str
    source_sha256: str
    audit_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "translation_id": self.translation_id,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "source_field": self.source_field,
            "source_text": self.source_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_sha256": self.source_sha256,
            "audit_only": self.audit_only,
        }


@dataclass(frozen=True)
class AuditTranslation:
    translation_id: str
    object_type: str
    object_id: str
    source_field: str
    source_language: str
    target_language: str
    source_sha256: str
    translated_text: str
    translation_method: str
    prompt_version: str
    audit_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "translation_id": self.translation_id,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "source_field": self.source_field,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_sha256": self.source_sha256,
            "translated_text": self.translated_text,
            "translation_method": self.translation_method,
            "prompt_version": self.prompt_version,
            "audit_only": self.audit_only,
        }


def build_translation_job(
    *,
    object_type: str,
    object_id: str,
    source_field: str,
    source_text: str,
) -> TranslationJob:
    normalized_text = clean_text(source_text)
    if not normalized_text:
        raise ValueError("Audit translation source text cannot be empty")
    normalized_type = clean_text(object_type).lower()
    normalized_id = clean_text(object_id)
    normalized_field = clean_text(source_field)
    if not normalized_type or not normalized_id or not normalized_field:
        raise ValueError("Audit translation requires object type, id, and source field")
    source_sha256 = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return TranslationJob(
        translation_id=(
            f"audit_translation::{normalized_type}::{normalized_id}::{normalized_field}::"
            f"{source_sha256[:12]}"
        ),
        object_type=normalized_type,
        object_id=normalized_id,
        source_field=normalized_field,
        source_text=normalized_text,
        source_language="en",
        target_language="zh-CN",
        source_sha256=source_sha256,
        audit_only=True,
    )


def validate_translation(source_text: str, translated_text: str) -> list[str]:
    issues: list[str] = []
    if not clean_text(translated_text) or not contains_cjk(translated_text):
        issues.append("MISSING_CHINESE_TRANSLATION")
    if Counter(_NUMBER_RE.findall(source_text)) != Counter(_NUMBER_RE.findall(translated_text)):
        issues.append("NUMBER_MISMATCH")
    source_tokens = set(_CONTROLLED_TOKEN_RE.findall(source_text))
    target_tokens = set(_CONTROLLED_TOKEN_RE.findall(translated_text))
    if not source_tokens <= target_tokens:
        issues.append("CONTROLLED_TOKEN_MISSING")
    return issues


def _append_missing_controlled_tokens(source_text: str, translated_text: str) -> str:
    translated = clean_text(translated_text)
    if not contains_cjk(translated):
        return translated
    missing = sorted(
        set(_CONTROLLED_TOKEN_RE.findall(source_text))
        - set(_CONTROLLED_TOKEN_RE.findall(translated))
    )
    if not missing:
        return translated
    return f"{translated}（保留标记：{'、'.join(missing)}）"


def _is_structured_machine_diagnostic(source_text: str) -> bool:
    try:
        payload = json.loads(source_text)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, (dict, list)):
        return False
    values: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            values.append(value)

    collect(payload)
    return bool(values) and all(_MACHINE_VALUE_RE.fullmatch(value) for value in values)


def _prompt_jobs() -> list[TranslationJob]:
    evidence_system, _ = build_evidence_extractor_prompt("video", {})
    language_evidence_system, _ = build_language_evidence_prompt("video", {})
    visual_evidence_system, _ = build_visual_evidence_prompt("video", {})
    visual_evidence_repair_system, _ = build_visual_evidence_repair_prompt("video", {}, [], [])
    visual_commerce_cue_system, _ = build_visual_commerce_cue_prompt("video", [])
    cue_system, _ = build_commerce_cue_prompt("video", [])
    relation_system, _ = build_commercial_relation_prompt("video", [], [])
    prompts = {
        "evidence_extractor": evidence_system,
        "language_evidence_extractor": language_evidence_system,
        "visual_evidence_extractor": visual_evidence_system,
        "visual_evidence_repairer": visual_evidence_repair_system,
        "visual_commerce_cue_extractor": visual_commerce_cue_system,
        "bp_compiler": BP_COMPILER_CONTRACT,
        "commerce_cue_extractor": cue_system,
        "commercial_relation_builder": relation_system,
        "cm_proposer": build_proposer_prompt("cm_proposer", "video", [])[0],
        "ss_proposer": build_proposer_prompt("ss_proposer", "video", [])[0],
        "ae_proposer": build_proposer_prompt("ae_proposer", "video", [])[0],
        "challenger": build_challenger_prompt("video", [], [])[0],
        "adjudicator": build_adjudicator_prompt("video", [], [], [])[0],
        "question_realizer": QUESTION_REALIZER_SYSTEM_PROMPT,
        "question_repairer": QUESTION_REPAIR_SYSTEM_PROMPT,
        "qa_semantic_quality_gate": QA_QUALITY_SYSTEM_PROMPT,
        "model_runner": CLOSED_SOURCE_SYSTEM_PROMPT,
        "judge": JUDGE_SYSTEM_PROMPT,
    }
    return [
        build_translation_job(
            object_type="prompt",
            object_id=prompt_id,
            source_field="system",
            source_text=text,
        )
        for prompt_id, text in prompts.items()
    ]


def _path(raw: object, repo_root: Path) -> Path:
    candidate = Path(clean_text(raw))
    return candidate if candidate.is_absolute() else repo_root / candidate


def _existing_rows(path: Path) -> list[dict[str, object]]:
    return read_jsonl(path) if path.is_file() else []


def _add_jobs(
    jobs: list[TranslationJob],
    rows: list[dict[str, object]],
    object_type: str,
    id_fields: tuple[str, ...],
    text_fields: tuple[str, ...],
) -> None:
    for row in rows:
        object_id = next((clean_text(row.get(field)) for field in id_fields if clean_text(row.get(field))), "")
        if not object_id:
            continue
        for field in text_fields:
            text = clean_text(row.get(field))
            if text:
                jobs.append(
                    build_translation_job(
                        object_type=object_type,
                        object_id=object_id,
                        source_field=field,
                        source_text=text,
                    )
                )


def _is_private_key(key: object) -> bool:
    normalized = clean_text(key).lower()
    return normalized in PRIVATE_KEYS or normalized.startswith("followers")


def _public_audit_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            clean_text(key): _public_audit_value(item)
            for key, item in value.items()
            if not _is_private_key(key)
        }
    if isinstance(value, list):
        return [_public_audit_value(item) for item in value]
    if isinstance(value, tuple):
        return [_public_audit_value(item) for item in value]
    return value


def _audit_text(value: object) -> str:
    public_value = _public_audit_value(value)
    if public_value in (None, "", [], {}):
        return ""
    if isinstance(public_value, (dict, list)):
        return json.dumps(public_value, ensure_ascii=False, sort_keys=True)
    return clean_text(public_value)


def _add_audit_fields(
    jobs: list[TranslationJob],
    rows: list[dict[str, object]],
    object_type: str,
    id_fields: tuple[str, ...],
    fields: tuple[str, ...],
) -> None:
    for ordinal, row in enumerate(rows):
        object_id = next(
            (clean_text(row.get(field)) for field in id_fields if clean_text(row.get(field))),
            f"row-{ordinal:06d}",
        )
        for field in fields:
            text = _audit_text(row.get(field))
            if not text:
                continue
            jobs.append(
                build_translation_job(
                    object_type=object_type,
                    object_id=object_id,
                    source_field=field,
                    source_text=text,
                )
            )


def collect_audit_translation_jobs(
    manifest_path: Path,
    *,
    repo_root: Path,
    include_prompts: bool = True,
) -> list[TranslationJob]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Audit delivery manifest must be an object")
    jobs = _prompt_jobs() if include_prompts else []
    for section in manifest.values():
        if not isinstance(section, dict) or not isinstance(section.get("artifacts"), dict):
            continue
        artifacts = section["artifacts"]
        evidence_source = artifacts.get("evidence", {}).get("source") if isinstance(artifacts.get("evidence"), dict) else None
        if evidence_source:
            root = _path(evidence_source, repo_root)
            _add_jobs(jobs, _existing_rows(root / "evidence_units.jsonl"), "evidence_unit", ("evidence_id",), ("content_en",))
            _add_jobs(jobs, _existing_rows(root / "commerce_cues.jsonl"), "commerce_cue", ("cue_id",), ("content_en",))
            _add_jobs(jobs, _existing_rows(root / "commercial_relations.jsonl"), "commercial_relation", ("relation_id",), ("rationale_en",))
            annotations = [
                annotation
                for record in _existing_rows(root / "video_evidence_dataset.jsonl")
                for annotation in record.get("grounded_annotations", []) or []
                if isinstance(annotation, dict)
            ]
            _add_audit_fields(
                jobs,
                annotations,
                "annotation",
                ("annotation_id", "gold_id"),
                ("question_intent", "target", "gold_value", "forbidden_inferences"),
            )
            _add_audit_fields(
                jobs,
                _existing_rows(root / "gold_proposals.jsonl"),
                "gold_proposal",
                ("proposal_id",),
                ("question_intent", "target", "proposed_gold", "reasoning_edges"),
            )
            _add_audit_fields(
                jobs,
                _existing_rows(root / "gold_reviews.jsonl"),
                "gold_review",
                ("review_id", "proposal_id"),
                ("reason", "rationale", "issues", "suggested_revision"),
            )
            for filename, object_type in (
                ("human_review_queue.jsonl", "evidence_review"),
                ("rejected_candidates.jsonl", "evidence_rejection"),
                ("pipeline_diagnostics.jsonl", "pipeline_diagnostic"),
            ):
                _add_audit_fields(
                    jobs,
                    _existing_rows(root / filename),
                    object_type,
                    ("review_item_id", "decision_id", "proposal_id", "video_id"),
                    (
                        "reason",
                        "issues",
                        "target",
                        "candidate_gold",
                        "candidate_snapshot",
                    ),
                )
            _add_audit_fields(
                jobs,
                _existing_rows(root / "quality_decisions.jsonl"),
                "quality_decision",
                ("decision_id",),
                ("reason", "candidate_snapshot", "verifier_result"),
            )
        qa_source = artifacts.get("qa", {}).get("source") if isinstance(artifacts.get("qa"), dict) else None
        if qa_source:
            root = _path(qa_source, repo_root)
            _add_jobs(jobs, _existing_rows(root / "qa_specs.jsonl"), "question_spec", ("spec_id",), ("question_intent", "gold_answer"))
            _add_audit_fields(
                jobs,
                _existing_rows(root / "qa_specs.jsonl"),
                "question_spec",
                ("spec_id",),
                ("target", "forbidden_inferences"),
            )
            _add_jobs(jobs, _existing_rows(root / "qa_realizations.jsonl"), "question_realization", ("spec_id",), ("question",))
            _add_jobs(jobs, _existing_rows(root / "vqa_gold_private.jsonl"), "qa", ("vqa_id",), ("question", "gold_answer"))
            for filename, object_type in (
                ("qa_rejected_candidates.jsonl", "qa_rejection"),
                ("qa_human_review_queue.jsonl", "qa_human_review"),
                ("qa_pipeline_diagnostics.jsonl", "qa_diagnostic"),
                ("qa_accepted_sample.jsonl", "qa_accepted_sample"),
            ):
                _add_audit_fields(
                    jobs,
                    _existing_rows(root / filename),
                    object_type,
                    ("spec_id", "vqa_id"),
                    ("reason", "issues", "candidate_snapshot", "semantic_verification"),
                )
        model_run_source = artifacts.get("model_run", {}).get("source") if isinstance(artifacts.get("model_run"), dict) else None
        if model_run_source:
            root = _path(model_run_source, repo_root)
            _add_jobs(
                jobs,
                _existing_rows(root / "predictions.jsonl"),
                "model_prediction",
                ("vqa_id",),
                ("answer", "prediction"),
            )
        evaluation_source = artifacts.get("evaluation", {}).get("source") if isinstance(artifacts.get("evaluation"), dict) else None
        if evaluation_source:
            root = _path(evaluation_source, repo_root)
            _add_jobs(
                jobs,
                _existing_rows(root / "predictions_judge_details.jsonl"),
                "judge_result",
                ("vqa_id",),
                ("reason", "evidence_alignment"),
            )
    unique: dict[tuple[str, str, str, str], TranslationJob] = {}
    for job in jobs:
        unique[(job.object_type, job.object_id, job.source_field, job.source_sha256)] = job
    return sorted(unique.values(), key=lambda item: (item.object_type, item.object_id, item.source_field))


def _parse_translation(record: dict[str, object]) -> AuditTranslation:
    return AuditTranslation(
        translation_id=clean_text(record.get("translation_id")),
        object_type=clean_text(record.get("object_type")),
        object_id=clean_text(record.get("object_id")),
        source_field=clean_text(record.get("source_field")),
        source_language=clean_text(record.get("source_language")),
        target_language=clean_text(record.get("target_language")),
        source_sha256=clean_text(record.get("source_sha256")),
        translated_text=clean_text(record.get("translated_text")),
        translation_method=clean_text(record.get("translation_method")),
        prompt_version=clean_text(record.get("prompt_version")),
        audit_only=bool(record.get("audit_only")),
    )


def load_audit_translations(path: Path) -> list[AuditTranslation]:
    if not path.exists():
        return []
    return [_parse_translation(record) for record in read_jsonl(path)]


class _AuditTranslationIndex:
    def __init__(self, rows: list[AuditTranslation]) -> None:
        self._rows: dict[tuple[str, str, str, str], AuditTranslation] = {}
        self._field_keys: set[tuple[str, str, str]] = set()
        for row in rows:
            if not row.audit_only:
                raise ValueError("localized VQA output requires audit_only translations")
            key = (row.object_type, row.object_id, row.source_field, row.source_sha256)
            self._rows[key] = row
            self._field_keys.add((row.object_type, row.object_id, row.source_field))

    def translate(
        self,
        object_type: str,
        object_id: object,
        source_field: str,
        source_text: object,
    ) -> dict[str, str]:
        text = clean_text(source_text)
        if not text:
            return {
                "translated_text": "",
                "status": "not_applicable",
                "source_sha256": "",
                "translation_method": "",
            }
        normalized_key = (
            clean_text(object_type).lower(),
            clean_text(object_id),
            clean_text(source_field),
        )
        source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self._rows.get((*normalized_key, source_sha256))
        if row is not None:
            return {
                "translated_text": row.translated_text,
                "status": "current",
                "source_sha256": source_sha256,
                "translation_method": row.translation_method,
            }
        return {
            "translated_text": "",
            "status": "stale" if normalized_key in self._field_keys else "missing",
            "source_sha256": source_sha256,
            "translation_method": "",
        }


def _localized_translation(
    index: _AuditTranslationIndex,
    status_counts: Counter[str],
    *,
    object_type: str,
    object_id: object,
    source_field: str,
    source_text: object,
) -> dict[str, str]:
    translation = index.translate(object_type, object_id, source_field, source_text)
    if translation["status"] != "not_applicable":
        status_counts[translation["status"]] += 1
    return translation


def _copy_present_fields(
    source: dict[str, object],
    fields: tuple[str, ...],
) -> dict[str, object]:
    return {field: source[field] for field in fields if field in source}


def _localized_vqa_record(
    record: dict[str, object],
    index: _AuditTranslationIndex,
    status_counts: Counter[str],
) -> dict[str, object]:
    vqa_id = record.get("vqa_id")
    question = _localized_translation(
        index,
        status_counts,
        object_type="qa",
        object_id=vqa_id,
        source_field="question",
        source_text=record.get("question"),
    )
    answer = _localized_translation(
        index,
        status_counts,
        object_type="qa",
        object_id=vqa_id,
        source_field="gold_answer",
        source_text=record.get("gold_answer"),
    )
    localized: dict[str, object] = _copy_present_fields(
        record,
        (
            "vqa_id",
            "video_id",
            "task_type",
            "task_subtype",
            "capability",
            "reasoning_operator",
            "spec_id",
        ),
    )
    localized["question_zh"] = question["translated_text"]
    localized["answer_zh"] = answer["translated_text"]

    translation_statuses: dict[str, object] = {
        "question": question["status"],
        "answer": answer["status"],
        "evidence": {},
        "commerce_cues": {},
        "commercial_relations": {},
    }
    references = {
        key: list(record.get(key) or [])
        for key in (
            "source_annotation_ids",
            "evidence_refs",
            "commerce_cue_ids",
            "commercial_relation_ids",
        )
        if record.get(key)
    }
    if references:
        localized["references"] = references

    evidence_zh = []
    for ordinal, item in enumerate(record.get("evidence_context") or []):
        if not isinstance(item, dict):
            continue
        translated = _localized_translation(
            index,
            status_counts,
            object_type="evidence_unit",
            object_id=item.get("evidence_id"),
            source_field="content_en",
            source_text=item.get("content_en"),
        )
        evidence_id = clean_text(item.get("evidence_id")) or f"row-{ordinal:06d}"
        localized_item = _copy_present_fields(
            item,
            ("evidence_id", "modality", "start_s", "end_s", "frame_indices"),
        )
        localized_item["content_zh"] = translated["translated_text"]
        localized_item["translation_status"] = translated["status"]
        evidence_zh.append(localized_item)
        translation_statuses["evidence"][evidence_id] = translated["status"]  # type: ignore[index]
    localized["evidence_zh"] = evidence_zh

    graph_context = record.get("graph_context")
    commerce_cues_zh = []
    commercial_relations_zh = []
    if isinstance(graph_context, dict):
        for ordinal, cue in enumerate(graph_context.get("commerce_cues") or []):
            if not isinstance(cue, dict):
                continue
            translated = _localized_translation(
                index,
                status_counts,
                object_type="commerce_cue",
                object_id=cue.get("cue_id"),
                source_field="content_en",
                source_text=cue.get("content_en"),
            )
            cue_id = clean_text(cue.get("cue_id")) or f"row-{ordinal:06d}"
            localized_cue = _copy_present_fields(cue, ("cue_id", "cue_type", "directness"))
            localized_cue["content_zh"] = translated["translated_text"]
            localized_cue["translation_status"] = translated["status"]
            commerce_cues_zh.append(localized_cue)
            translation_statuses["commerce_cues"][cue_id] = translated["status"]  # type: ignore[index]

        for ordinal, relation in enumerate(graph_context.get("commercial_relations") or []):
            if not isinstance(relation, dict):
                continue
            translated = _localized_translation(
                index,
                status_counts,
                object_type="commercial_relation",
                object_id=relation.get("relation_id"),
                source_field="rationale_en",
                source_text=relation.get("rationale_en"),
            )
            relation_id = clean_text(relation.get("relation_id")) or f"row-{ordinal:06d}"
            localized_relation = _copy_present_fields(
                relation,
                ("relation_id", "relation_type", "status"),
            )
            localized_relation["rationale_zh"] = translated["translated_text"]
            localized_relation["translation_status"] = translated["status"]
            commercial_relations_zh.append(localized_relation)
            translation_statuses["commercial_relations"][relation_id] = translated["status"]  # type: ignore[index]
    localized["commerce_cues_zh"] = commerce_cues_zh
    localized["commercial_relations_zh"] = commercial_relations_zh
    localized["translation_statuses"] = translation_statuses
    return localized


def write_localized_vqa_private(
    qa_dir: Path,
    translations_path: Path,
    *,
    output_filename: str = "vqa_gold_private_zh.jsonl",
) -> dict[str, object]:
    source_path = qa_dir / "vqa_gold_private.jsonl"
    if not source_path.is_file():
        return {
            "source": str(source_path),
            "output": str(qa_dir / output_filename),
            "written": 0,
            "translation_statuses": {},
            "skipped": "missing_vqa_gold_private",
        }
    index = _AuditTranslationIndex(load_audit_translations(translations_path))
    status_counts: Counter[str] = Counter()
    localized = [
        _localized_vqa_record(record, index, status_counts)
        for record in read_jsonl(source_path)
    ]
    output_path = qa_dir / output_filename
    write_jsonl(output_path, localized)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "written": len(localized),
        "translation_statuses": dict(sorted(status_counts.items())),
    }


def write_localized_vqa_outputs_from_manifest(
    manifest_path: Path,
    translations_path: Path,
    *,
    repo_root: Path,
) -> list[dict[str, object]]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Audit delivery manifest must be an object")
    outputs: list[dict[str, object]] = []
    seen: set[Path] = set()
    for section in manifest.values():
        if not isinstance(section, dict) or not isinstance(section.get("artifacts"), dict):
            continue
        artifacts = section["artifacts"]
        qa_source = (
            artifacts.get("qa", {}).get("source")
            if isinstance(artifacts.get("qa"), dict)
            else None
        )
        if not qa_source:
            continue
        qa_dir = _path(qa_source, repo_root)
        if qa_dir in seen:
            continue
        seen.add(qa_dir)
        if (qa_dir / "vqa_gold_private.jsonl").is_file():
            outputs.append(write_localized_vqa_private(qa_dir, translations_path))
    return outputs


def run_audit_translations(
    jobs: list[TranslationJob],
    output_path: Path,
    client: VLMClient,
    *,
    batch_size: int = 20,
) -> dict[str, object]:
    existing = load_audit_translations(output_path)
    cache = {
        (row.object_type, row.object_id, row.source_field, row.source_sha256): row
        for row in existing
        if row.audit_only and row.prompt_version in _CACHEABLE_PROMPT_VERSIONS
    }
    current: dict[tuple[str, str, str, str], AuditTranslation] = {}
    pending: list[TranslationJob] = []
    reused = 0
    translated = 0
    for job in jobs:
        key = (job.object_type, job.object_id, job.source_field, job.source_sha256)
        if key in cache:
            current[key] = cache[key]
            reused += 1
        elif _MACHINE_CODE_RE.fullmatch(job.source_text):
            current[key] = AuditTranslation(
                translation_id=job.translation_id,
                object_type=job.object_type,
                object_id=job.object_id,
                source_field=job.source_field,
                source_language=job.source_language,
                target_language=job.target_language,
                source_sha256=job.source_sha256,
                translated_text=f"审计代码：{job.source_text}",
                translation_method="deterministic-machine-code-v1",
                prompt_version=AUDIT_TRANSLATION_PROMPT_VERSION,
                audit_only=True,
            )
            translated += 1
        elif _is_structured_machine_diagnostic(job.source_text):
            current[key] = AuditTranslation(
                translation_id=job.translation_id,
                object_type=job.object_type,
                object_id=job.object_id,
                source_field=job.source_field,
                source_language=job.source_language,
                target_language=job.target_language,
                source_sha256=job.source_sha256,
                translated_text=f"审计诊断：{job.source_text}",
                translation_method="deterministic-machine-diagnostic-v1",
                prompt_version=AUDIT_TRANSLATION_PROMPT_VERSION,
                audit_only=True,
            )
            translated += 1
        else:
            pending.append(job)

    failures: list[dict[str, object]] = []
    size = max(1, int(batch_size))
    for offset in range(0, len(pending), size):
        batch = pending[offset : offset + size]
        system, user = build_audit_translation_prompt([job.to_dict() for job in batch])
        call = client.call_text_only(system, user, response_format="json_object")
        if not call.success:
            failures.extend({"translation_id": job.translation_id, "error": call.error or "api_call_failed"} for job in batch)
            continue
        try:
            payload = parse_json_object(call.raw_response, "translations")
            rows = payload["translations"]
            if not isinstance(rows, list):
                raise ModelOutputError("translations must be a list")
            returned = {
                clean_text(row.get("translation_id")): clean_text(row.get("translated_text"))
                for row in rows
                if isinstance(row, dict)
            }
        except (ModelOutputError, ValueError) as exc:
            failures.extend({"translation_id": job.translation_id, "error": str(exc)} for job in batch)
            continue
        for job in batch:
            translated_text = _append_missing_controlled_tokens(
                job.source_text,
                returned.get(job.translation_id, ""),
            )
            issues = validate_translation(job.source_text, translated_text)
            if issues:
                failures.append({"translation_id": job.translation_id, "error": ",".join(issues)})
                continue
            record = AuditTranslation(
                translation_id=job.translation_id,
                object_type=job.object_type,
                object_id=job.object_id,
                source_field=job.source_field,
                source_language=job.source_language,
                target_language=job.target_language,
                source_sha256=job.source_sha256,
                translated_text=translated_text,
                translation_method=call.model,
                prompt_version=AUDIT_TRANSLATION_PROMPT_VERSION,
                audit_only=True,
            )
            current[(job.object_type, job.object_id, job.source_field, job.source_sha256)] = record
            translated += 1

    ordered = sorted(current.values(), key=lambda item: (item.object_type, item.object_id, item.source_field))
    write_jsonl(output_path, [row.to_dict() for row in ordered])
    return {
        "output": str(output_path),
        "prompt_version": AUDIT_TRANSLATION_PROMPT_VERSION,
        "model": client.model,
        "counts": {
            "jobs": len(jobs),
            "translated": translated,
            "reused": reused,
            "failed": len(failures),
            "written": len(ordered),
            "stale_removed": len(existing) - reused,
        },
        "failures": failures,
    }
