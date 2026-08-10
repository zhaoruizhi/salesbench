from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.audit_translation import (  # noqa: E402
    AUDIT_TRANSLATION_PROMPT_VERSION,
    build_translation_job,
    collect_audit_translation_jobs,
    load_audit_translations,
    run_audit_translations,
    validate_translation,
)
from salesbench.io_utils import write_json, write_jsonl  # noqa: E402
from salesbench.vlm.api_client import APICallResult  # noqa: E402


class FakeTranslationClient:
    model = "gpt-4o"

    def __init__(self) -> None:
        self.calls = 0

    def call_text_only(self, system_prompt: str, user_text: str, response_format: str | None = None):
        self.calls += 1
        rows = json.loads(user_text)["translation_jobs"]
        return APICallResult(
            raw_response=json.dumps(
                {
                    "translations": [
                        {
                            "translation_id": row["translation_id"],
                            "translated_text": "审计翻译：" + row["source_text"],
                        }
                        for row in rows
                    ]
                },
                ensure_ascii=False,
            ),
            model=self.model,
            input_tokens=10,
            output_tokens=10,
            latency_s=0.01,
            cost_usd=0.0,
            success=True,
        )


def test_audit_translation_is_separate_versioned_and_hash_bound():
    job = build_translation_job(
        object_type="qa",
        object_id="q1",
        source_field="question",
        source_text="What condition is required for the 9.9-yuan discount?",
    )
    payload = job.to_dict()

    assert len(job.source_sha256) == 64
    assert job.target_language == "zh-CN"
    assert job.audit_only is True
    assert job.translation_id == "audit_translation::qa::q1::question"
    assert "question_zh" not in payload
    assert AUDIT_TRANSLATION_PROMPT_VERSION == "audit-translation-prompt-v1"


def test_translation_validation_preserves_numbers_negation_and_enum_tokens():
    source = "The 9.9-yuan offer is NOT_SHOWN in frame 12."

    assert validate_translation(source, "画面第12帧中未显示（NOT_SHOWN）9.9元优惠。") == []
    assert "NUMBER_MISMATCH" in validate_translation(source, "画面中未显示（NOT_SHOWN）19.9元优惠。")
    assert "CONTROLLED_TOKEN_MISSING" in validate_translation(source, "画面第12帧中未显示9.9元优惠。")


def test_collect_and_run_translations_from_delivery_manifest(tmp_path: Path):
    evidence = tmp_path / "evidence"
    qa = tmp_path / "qa"
    write_jsonl(
        evidence / "evidence_units.jsonl",
        [{"evidence_id": "e1", "video_id": "v1", "content_en": "The price is 9.9 yuan.", "source_text_native": "9.9元"}],
    )
    write_jsonl(
        evidence / "commerce_cues.jsonl",
        [{"cue_id": "c1", "video_id": "v1", "content_en": "The offer price is 9.9 yuan.", "evidence_ids": ["e1"]}],
    )
    write_jsonl(evidence / "commercial_relations.jsonl", [])
    write_jsonl(
        qa / "vqa_gold_private.jsonl",
        [{"vqa_id": "q1", "video_id": "v1", "question": "What price is presented?", "gold_answer": "The price is 9.9 yuan.", "likes": 100}],
    )
    manifest = {
        "formal": {
            "artifacts": {
                "evidence": {"source": str(evidence)},
                "qa": {"source": str(qa)},
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)

    jobs = collect_audit_translation_jobs(manifest_path, repo_root=tmp_path, include_prompts=False)
    client = FakeTranslationClient()
    output = tmp_path / "audit_translations.jsonl"
    first = run_audit_translations(jobs, output, client, batch_size=20)
    second = run_audit_translations(jobs, output, client, batch_size=20)
    translations = load_audit_translations(output)
    serialized = json.dumps([row.to_dict() for row in translations], ensure_ascii=False)

    assert first["counts"]["translated"] == len(jobs)
    assert second["counts"]["reused"] == len(jobs)
    assert client.calls == 1
    assert all(row.audit_only for row in translations)
    assert "likes" not in serialized
    assert "translated_text" in serialized


def test_prompt_translation_jobs_include_split_evidence_prompts(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {})

    prompt_jobs = collect_audit_translation_jobs(manifest_path, repo_root=tmp_path)
    prompt_ids = {job.object_id for job in prompt_jobs if job.object_type == "prompt"}

    assert "language_evidence_extractor" in prompt_ids
    assert "visual_evidence_extractor" in prompt_ids
    assert "visual_evidence_repairer" in prompt_ids
    assert "visual_commerce_cue_extractor" in prompt_ids
    assert "question_repairer" in prompt_ids
    assert "model_runner" in prompt_ids
