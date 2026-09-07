import json
import sys

import pytest

sys.path.insert(0, "src")

from salesbench.run_integrity import (
    ReferenceClosureError,
    canonical_sha256,
    compute_compile_fingerprint,
    compute_evidence_fingerprint,
    compute_qa_realization_fingerprint,
    read_required_fingerprint,
    validate_reference_closure,
)


def test_canonical_sha256_ignores_order_and_declared_volatile_fields():
    first = {
        "model": "qwen3-vl-plus",
        "counts": {"evidence": 2, "cues": 1},
        "generated_at": "2026-09-07T10:00:00Z",
        "output_dir": "/private/first/run",
    }
    second = {
        "output_dir": "/different/machine/run",
        "generated_at": "2026-09-07T11:00:00Z",
        "counts": {"cues": 1, "evidence": 2},
        "model": "qwen3-vl-plus",
    }

    assert canonical_sha256(first) == canonical_sha256(second)
    assert canonical_sha256({**second, "model": "qwen-vl-max"}) != canonical_sha256(first)


def test_stage_fingerprints_change_when_semantic_inputs_change():
    evidence_args = {
        "source": {"videos": [{"video_id": "v1", "sha256": "a"}], "asr_sha256": "b"},
        "components": {"schema": "schema-v4", "prompt": "prompt-v10.3"},
        "models": {"vision": "qwen3-vl-plus", "text": "deepseek-v4-pro"},
        "policy": {"min_confidence": 0.8},
        "evidence_units": [{"evidence_id": "e1", "content_en": "A red box."}],
        "commerce_cues": [{"cue_id": "c1", "content_en": "Red product box."}],
        "commercial_relations": [],
        "dataset_rows": [{"video_id": "v1", "annotations": [{"annotation_id": "a1"}]}],
    }
    first_evidence = compute_evidence_fingerprint(**evidence_args)
    changed_evidence = compute_evidence_fingerprint(
        **{**evidence_args, "evidence_units": [{"evidence_id": "e1", "content_en": "A blue box."}]}
    )
    assert first_evidence != changed_evidence

    first_qa = compute_qa_realization_fingerprint(
        evidence_fingerprint=first_evidence,
        specs=[{"spec_id": "s1", "gold_answer": "red"}],
        components={"realizer_prompt": "v3", "quality_prompt": "v2"},
        models={"realizer": "deepseek-v4-pro", "verifier": "qwen3-vl-plus"},
        policy={"strict": True},
    )
    changed_qa = compute_qa_realization_fingerprint(
        evidence_fingerprint=changed_evidence,
        specs=[{"spec_id": "s1", "gold_answer": "red"}],
        components={"realizer_prompt": "v3", "quality_prompt": "v2"},
        models={"realizer": "deepseek-v4-pro", "verifier": "qwen3-vl-plus"},
        policy={"strict": True},
    )
    assert first_qa != changed_qa

    first_compile = compute_compile_fingerprint(
        evidence_fingerprint=first_evidence,
        qa_realization_fingerprint=first_qa,
        compiler_version="v9",
        policy={"max_per_task": 2},
        selected_records=[{"vqa_id": "q1", "question": "What color is the box?"}],
    )
    changed_compile = compute_compile_fingerprint(
        evidence_fingerprint=first_evidence,
        qa_realization_fingerprint=first_qa,
        compiler_version="v9",
        policy={"max_per_task": 1},
        selected_records=[{"vqa_id": "q1", "question": "What color is the box?"}],
    )
    assert first_compile != changed_compile


def test_reference_closure_checks_existence_and_video_ownership():
    references = [
        {
            "vqa_id": "q1",
            "video_id": "v1",
            "source_annotation_ids": ["a1"],
            "evidence_refs": ["e1"],
            "commerce_cue_ids": ["c1"],
            "commercial_relation_ids": ["r1"],
        }
    ]
    report = validate_reference_closure(
        references,
        annotations=[{"annotation_id": "a1", "video_id": "v1"}],
        evidence_units=[{"evidence_id": "e1", "video_id": "v1"}],
        commerce_cues=[{"cue_id": "c1", "video_id": "v1"}],
        commercial_relations=[{"relation_id": "r1", "video_id": "v1"}],
    )

    assert report == {
        "rows": 1,
        "annotation_refs": 1,
        "evidence_refs": 1,
        "commerce_cue_refs": 1,
        "commercial_relation_refs": 1,
        "resolved_refs": 4,
        "unresolved_refs": 0,
        "cross_video_refs": 0,
        "closure_percent": 100.0,
    }

    with pytest.raises(ReferenceClosureError) as exc_info:
        validate_reference_closure(
            references,
            annotations=[{"annotation_id": "a1", "video_id": "v2"}],
            evidence_units=[],
            commerce_cues=[{"cue_id": "c1", "video_id": "v1"}],
            commercial_relations=[{"relation_id": "r1", "video_id": "v1"}],
        )
    assert {issue["code"] for issue in exc_info.value.issues} == {
        "CROSS_VIDEO_ANNOTATION_REF",
        "MISSING_EVIDENCE_REF",
    }
    assert exc_info.value.report["unresolved_refs"] == 1
    assert exc_info.value.report["cross_video_refs"] == 1


def test_read_required_fingerprint_rejects_missing_or_empty_value(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"evidence_fingerprint": "a" * 64}), encoding="utf-8")
    assert read_required_fingerprint(meta_path, "evidence_fingerprint") == "a" * 64

    meta_path.write_text(json.dumps({"evidence_fingerprint": ""}), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_fingerprint"):
        read_required_fingerprint(meta_path, "evidence_fingerprint")

    with pytest.raises(ValueError, match="does not exist"):
        read_required_fingerprint(tmp_path / "missing.json", "evidence_fingerprint")
