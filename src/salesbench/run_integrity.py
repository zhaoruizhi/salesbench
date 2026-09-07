"""Content-addressed run metadata and reference-closure validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


VOLATILE_FINGERPRINT_KEYS = frozenset(
    {
        "generated_at",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
        "elapsed_seconds",
        "latency_seconds",
        "latency_ms",
        "cost",
        "cost_usd",
        "output_dir",
        "output_path",
        "dataset_file",
        "bank_file",
        "realizations_file",
    }
)


def _canonical(value: object, excluded: frozenset[str]) -> object:
    if isinstance(value, dict):
        return {
            str(key): _canonical(item, excluded)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in excluded
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item, excluded) for item in value]
    if isinstance(value, Path):
        return value.name
    if hasattr(value, "value"):
        return _canonical(getattr(value, "value"), excluded)
    return value


def canonical_sha256(value: object, *, exclude_keys: Iterable[str] = ()) -> str:
    """Hash semantic JSON content while excluding explicitly volatile run metadata."""

    excluded = VOLATILE_FINGERPRINT_KEYS | frozenset(str(key) for key in exclude_keys)
    encoded = json.dumps(
        _canonical(value, excluded),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_evidence_fingerprint(
    *,
    source: object,
    components: object,
    models: object,
    policy: object,
    evidence_units: object,
    commerce_cues: object,
    commercial_relations: object,
    dataset_rows: object,
) -> str:
    return canonical_sha256(
        {
            "stage": "evidence",
            "source": source,
            "components": components,
            "models": models,
            "policy": policy,
            "evidence_units": evidence_units,
            "commerce_cues": commerce_cues,
            "commercial_relations": commercial_relations,
            "dataset_rows": dataset_rows,
        }
    )


def compute_qa_realization_fingerprint(
    *,
    evidence_fingerprint: str,
    specs: object,
    components: object,
    models: object,
    policy: object,
) -> str:
    if not str(evidence_fingerprint).strip():
        raise ValueError("evidence_fingerprint is required")
    return canonical_sha256(
        {
            "stage": "qa_realization",
            "evidence_fingerprint": evidence_fingerprint,
            "specs": specs,
            "components": components,
            "models": models,
            "policy": policy,
        }
    )


def compute_compile_fingerprint(
    *,
    evidence_fingerprint: str,
    qa_realization_fingerprint: str,
    compiler_version: str,
    policy: object,
    selected_records: object,
) -> str:
    if not str(evidence_fingerprint).strip() or not str(qa_realization_fingerprint).strip():
        raise ValueError("compile fingerprints require Evidence and QA realization fingerprints")
    return canonical_sha256(
        {
            "stage": "compile",
            "evidence_fingerprint": evidence_fingerprint,
            "qa_realization_fingerprint": qa_realization_fingerprint,
            "compiler_version": compiler_version,
            "policy": policy,
            "selected_records": selected_records,
        }
    )


def read_required_fingerprint(meta_path: Path, field: str) -> str:
    path = Path(meta_path)
    if not path.exists():
        raise ValueError(f"required metadata file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read required metadata file: {path}") from exc
    value = str(payload.get(field) or "").strip() if isinstance(payload, dict) else ""
    if not value:
        raise ValueError(f"required fingerprint {field} is missing from {path}")
    return value


class ReferenceClosureError(ValueError):
    """Raised when a candidate artifact points outside its own verified graph."""

    def __init__(self, issues: list[dict[str, str]], report: dict[str, int | float]) -> None:
        self.issues = issues
        self.report = report
        codes = ", ".join(sorted({issue["code"] for issue in issues}))
        super().__init__(f"reference closure failed: {codes}")


def _lookup(rows: Iterable[dict[str, Any]], id_field: str) -> dict[str, dict[str, Any]]:
    return {
        str(row.get(id_field) or "").strip(): row
        for row in rows
        if str(row.get(id_field) or "").strip()
    }


def validate_reference_closure(
    reference_rows: Iterable[dict[str, Any]],
    *,
    annotations: Iterable[dict[str, Any]],
    evidence_units: Iterable[dict[str, Any]],
    commerce_cues: Iterable[dict[str, Any]],
    commercial_relations: Iterable[dict[str, Any]],
) -> dict[str, int | float]:
    """Require every graph reference to exist and belong to the referencing video."""

    rows = list(reference_rows)
    contracts = (
        ("source_annotation_ids", "annotation_refs", "annotation_id", "ANNOTATION", _lookup(annotations, "annotation_id")),
        ("evidence_refs", "evidence_refs", "evidence_id", "EVIDENCE", _lookup(evidence_units, "evidence_id")),
        ("commerce_cue_ids", "commerce_cue_refs", "cue_id", "COMMERCE_CUE", _lookup(commerce_cues, "cue_id")),
        (
            "commercial_relation_ids",
            "commercial_relation_refs",
            "relation_id",
            "COMMERCIAL_RELATION",
            _lookup(commercial_relations, "relation_id"),
        ),
    )
    report: dict[str, int | float] = {
        "rows": len(rows),
        "annotation_refs": 0,
        "evidence_refs": 0,
        "commerce_cue_refs": 0,
        "commercial_relation_refs": 0,
        "resolved_refs": 0,
        "unresolved_refs": 0,
        "cross_video_refs": 0,
        "closure_percent": 100.0,
    }
    issues: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("vqa_id") or row.get("spec_id") or row.get("gold_id") or "")
        video_id = str(row.get("video_id") or "")
        for source_field, count_field, _id_field, label, lookup in contracts:
            raw_refs = row.get(source_field) or []
            refs = raw_refs if isinstance(raw_refs, (list, tuple)) else [raw_refs]
            for raw_ref in refs:
                ref = str(raw_ref or "").strip()
                if not ref:
                    continue
                report[count_field] = int(report[count_field]) + 1
                target = lookup.get(ref)
                if target is None:
                    report["unresolved_refs"] = int(report["unresolved_refs"]) + 1
                    issues.append(
                        {
                            "code": f"MISSING_{label}_REF",
                            "row_id": row_id,
                            "reference_id": ref,
                        }
                    )
                    continue
                if str(target.get("video_id") or "") != video_id:
                    report["cross_video_refs"] = int(report["cross_video_refs"]) + 1
                    issues.append(
                        {
                            "code": f"CROSS_VIDEO_{label}_REF",
                            "row_id": row_id,
                            "reference_id": ref,
                        }
                    )
                    continue
                report["resolved_refs"] = int(report["resolved_refs"]) + 1
    total = sum(int(report[field]) for _, field, _, _, _ in contracts)
    report["closure_percent"] = round(100.0 * int(report["resolved_refs"]) / total, 6) if total else 100.0
    if issues:
        raise ReferenceClosureError(issues, report)
    return report
