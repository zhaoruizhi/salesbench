from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.audit_translation import (  # noqa: E402
    AUDIT_TRANSLATION_PROMPT_VERSION,
    AuditTranslation,
    build_translation_job,
    collect_audit_translation_jobs,
    load_audit_translations,
    run_audit_translations,
    validate_translation,
)
import salesbench.audit_translation as audit_translation  # noqa: E402
from salesbench.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from salesbench.vlm.api_client import APICallResult  # noqa: E402
from salesbench.audit_translation_prompts import build_audit_translation_prompt  # noqa: E402


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


class DropsControlledTokenClient(FakeTranslationClient):
    def call_text_only(self, system_prompt: str, user_text: str, response_format: str | None = None):
        self.calls += 1
        row = json.loads(user_text)["translation_jobs"][0]
        return APICallResult(
            raw_response=json.dumps(
                {
                    "translations": [
                        {
                            "translation_id": row["translation_id"],
                            "translated_text": "画面第12帧中未显示该优惠。",
                        }
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
    assert job.translation_id.startswith("audit_translation::qa::q1::question::")
    assert "question_zh" not in payload
    assert AUDIT_TRANSLATION_PROMPT_VERSION == "audit-translation-prompt-v2"


def test_translation_ids_do_not_collide_when_one_queue_id_has_multiple_snapshots():
    first = build_translation_job(
        object_type="evidence_rejection",
        object_id="duplicate-review-id",
        source_field="candidate_snapshot",
        source_text='{"answer":"first candidate"}',
    )
    second = build_translation_job(
        object_type="evidence_rejection",
        object_id="duplicate-review-id",
        source_field="candidate_snapshot",
        source_text='{"answer":"second candidate"}',
    )

    assert first.translation_id != second.translation_id


def test_translation_validation_preserves_numbers_negation_and_enum_tokens():
    source = "The 9.9-yuan offer is NOT_SHOWN in frame 12."

    assert validate_translation(source, "画面第12帧中未显示（NOT_SHOWN）9.9元优惠。") == []
    assert "NUMBER_MISMATCH" in validate_translation(source, "画面中未显示（NOT_SHOWN）19.9元优惠。")
    assert "CONTROLLED_TOKEN_MISSING" in validate_translation(source, "画面第12帧中未显示9.9元优惠。")


def test_translation_prompt_lists_required_numbers_and_controlled_tokens():
    job = build_translation_job(
        object_type="evidence_unit",
        object_id="e1",
        source_field="content_en",
        source_text="The USB cable shows 15W and the CTA follows in frame 12.",
    )

    _, user = build_audit_translation_prompt([job.to_dict()])
    payload = json.loads(user)["translation_jobs"][0]

    assert payload["required_numbers"] == ["15", "12"]
    assert payload["required_controlled_tokens"] == ["CTA", "USB"]


def test_machine_code_reason_gets_deterministic_chinese_audit_label(tmp_path: Path):
    job = build_translation_job(
        object_type="evidence_rejection",
        object_id="r1",
        source_field="reason",
        source_text="commercial_relation_validation_failed",
    )
    client = FakeTranslationClient()
    output = tmp_path / "translations.jsonl"

    summary = run_audit_translations([job], output, client, batch_size=1)
    translations = load_audit_translations(output)

    assert summary["counts"]["translated"] == 1
    assert client.calls == 0
    assert translations[0].translated_text == "审计代码：commercial_relation_validation_failed"
    assert translations[0].translation_method == "deterministic-machine-code-v1"


def test_structured_machine_diagnostic_gets_deterministic_chinese_label(tmp_path: Path):
    source = json.dumps(
        [
            {
                "code": "MISSING_COMMERCIAL_RELATION",
                "item_id": "7359538830657064192_ss_proposer_ss_000_c551ee7351f0",
                "message": "RELATION_PATH_REQUIRED",
                "severity": "ERROR",
            }
        ],
        sort_keys=True,
    )
    job = build_translation_job(
        object_type="evidence_rejection",
        object_id="r1",
        source_field="issues",
        source_text=source,
    )
    client = FakeTranslationClient()
    output = tmp_path / "translations.jsonl"

    summary = run_audit_translations([job], output, client, batch_size=1)
    translation = load_audit_translations(output)[0]

    assert summary["counts"]["translated"] == 1
    assert client.calls == 0
    assert translation.translated_text == f"审计诊断：{source}"
    assert translation.translation_method == "deterministic-machine-diagnostic-v1"


def test_runner_safely_appends_missing_controlled_tokens_to_chinese_translation(tmp_path: Path):
    job = build_translation_job(
        object_type="evidence_unit",
        object_id="e1",
        source_field="content_en",
        source_text="The offer is NOT_SHOWN in frame 12.",
    )
    output = tmp_path / "translations.jsonl"

    summary = run_audit_translations(
        [job],
        output,
        DropsControlledTokenClient(),
        batch_size=1,
    )
    translation = load_audit_translations(output)[0]

    assert summary["counts"]["failed"] == 0
    assert "NOT_SHOWN" in translation.translated_text
    assert translation.translated_text.endswith("（保留标记：NOT_SHOWN）")


def test_collect_and_run_translations_from_delivery_manifest(tmp_path: Path):
    evidence = tmp_path / "evidence"
    qa = tmp_path / "qa"
    qa_realizations = tmp_path / "qa-realizations"
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
    write_jsonl(
        qa_realizations / "qa_realizations.jsonl",
        [{"spec_id": "s1", "video_id": "v1", "question": "Which price appears in the video?"}],
    )
    manifest = {
        "formal": {
            "artifacts": {
                "evidence": {"source": str(evidence)},
                "qa_realizations": {"source": str(qa_realizations)},
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
    assert any(row.object_type == "question_realization" for row in translations)


def test_write_localized_vqa_private_keeps_only_chinese_review_fields(tmp_path: Path):
    qa = tmp_path / "qa"
    qa_row = {
        "vqa_id": "q1",
        "video_id": "v1",
        "task_type": "BP",
        "question": "What price is presented?",
        "gold_answer": "The price is 9.9 yuan.",
        "evidence_context": [
            {
                "evidence_id": "e1",
                "content_en": "The price shown on screen is 9.9 yuan.",
            }
        ],
        "graph_context": {
            "commerce_cues": [
                {
                    "cue_id": "c1",
                    "content_en": "The offer price is 9.9 yuan.",
                }
            ],
            "commercial_relations": [
                {
                    "relation_id": "r1",
                    "rationale_en": "The screen price supports the spoken offer.",
                }
            ],
        },
    }
    write_jsonl(qa / "vqa_gold_private.jsonl", [qa_row])
    jobs = [
        build_translation_job(
            object_type="qa",
            object_id="q1",
            source_field="question",
            source_text=qa_row["question"],
        ),
        build_translation_job(
            object_type="qa",
            object_id="q1",
            source_field="gold_answer",
            source_text=qa_row["gold_answer"],
        ),
        build_translation_job(
            object_type="evidence_unit",
            object_id="e1",
            source_field="content_en",
            source_text=qa_row["evidence_context"][0]["content_en"],
        ),
        build_translation_job(
            object_type="commerce_cue",
            object_id="c1",
            source_field="content_en",
            source_text=qa_row["graph_context"]["commerce_cues"][0]["content_en"],
        ),
        build_translation_job(
            object_type="commercial_relation",
            object_id="r1",
            source_field="rationale_en",
            source_text=qa_row["graph_context"]["commercial_relations"][0]["rationale_en"],
        ),
    ]
    translated_texts = {
        ("qa", "q1", "question"): "展示的价格是多少？",
        ("qa", "q1", "gold_answer"): "价格是9.9元。",
        ("evidence_unit", "e1", "content_en"): "屏幕上显示的价格是9.9元。",
        ("commerce_cue", "c1", "content_en"): "优惠价是9.9元。",
        ("commercial_relation", "r1", "rationale_en"): "屏幕价格支持口播优惠。",
    }
    translations = [
        AuditTranslation(
            translation_id=job.translation_id,
            object_type=job.object_type,
            object_id=job.object_id,
            source_field=job.source_field,
            source_language=job.source_language,
            target_language=job.target_language,
            source_sha256=job.source_sha256,
            translated_text=translated_texts[(job.object_type, job.object_id, job.source_field)],
            translation_method="fixture",
            prompt_version=AUDIT_TRANSLATION_PROMPT_VERSION,
            audit_only=True,
        ).to_dict()
        for job in jobs
    ]
    translations_path = tmp_path / "audit_translations.jsonl"
    write_jsonl(translations_path, translations)

    assert hasattr(audit_translation, "write_localized_vqa_private")
    summary = audit_translation.write_localized_vqa_private(qa, translations_path)
    localized = list(read_jsonl(qa / "vqa_gold_private_zh.jsonl"))
    serialized = json.dumps(localized[0], ensure_ascii=False)

    assert summary["written"] == 1
    assert summary["translation_statuses"] == {"current": 5}
    assert localized[0]["question_zh"] == "展示的价格是多少？"
    assert localized[0]["answer_zh"] == "价格是9.9元。"
    assert localized[0]["evidence_zh"] == [
        {
            "evidence_id": "e1",
            "content_zh": "屏幕上显示的价格是9.9元。",
            "translation_status": "current",
        }
    ]
    assert localized[0]["commerce_cues_zh"] == [
        {
            "cue_id": "c1",
            "content_zh": "优惠价是9.9元。",
            "translation_status": "current",
        }
    ]
    assert localized[0]["commercial_relations_zh"][0]["rationale_zh"] == (
        "屏幕价格支持口播优惠。"
    )
    assert localized[0]["translation_statuses"] == {
        "answer": "current",
        "commerce_cues": {"c1": "current"},
        "commercial_relations": {"r1": "current"},
        "evidence": {"e1": "current"},
        "question": "current",
    }
    assert "question" not in localized[0]
    assert "gold_answer" not in localized[0]
    assert "evidence_context" not in localized[0]
    assert "graph_context" not in localized[0]
    assert "content_en" not in serialized
    assert "rationale_en" not in serialized
    assert qa_row["question"] not in serialized
    assert qa_row["gold_answer"] not in serialized


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
    assert "qa_semantic_quality_gate" in prompt_ids
    assert "model_runner" in prompt_ids


def test_collects_all_human_readable_quality_artifacts_without_private_fields(tmp_path: Path):
    evidence = tmp_path / "evidence"
    qa = tmp_path / "qa"
    model_run = tmp_path / "model"
    evaluation = tmp_path / "judge"
    write_jsonl(
        evidence / "video_evidence_dataset.jsonl",
        [
            {
                "video_id": "v1",
                "grounded_annotations": [
                    {
                        "annotation_id": "a1",
                        "question_intent": "Ask about the presented household need.",
                        "target": {"need": "organize small household items"},
                        "gold_value": {"answer": "The video presents the rack as a storage solution."},
                        "forbidden_inferences": ["Do not infer purchase conversion."],
                        "likes": 99,
                    }
                ],
            }
        ],
    )
    write_jsonl(
        evidence / "gold_proposals.jsonl",
        [
            {
                "proposal_id": "p1",
                "question_intent": "Ask how the offer is framed.",
                "target": {"offer": "a discounted rack"},
                "proposed_gold": {"answer": "The speaker contrasts two prices."},
            }
        ],
    )
    write_jsonl(
        evidence / "gold_reviews.jsonl",
        [{"proposal_id": "p1", "issues": ["The claim needs attribution."], "rationale": "Revise the claim boundary."}],
    )
    queue_row = {
        "review_item_id": "r1",
        "reason": "The temporal order is ambiguous.",
        "issues": [{"message": "Timestamps are unavailable."}],
        "target": {"sequence": "offer before CTA"},
        "candidate_gold": {"answer": "The offer precedes the CTA."},
        "candidate_snapshot": {"rationale_en": "The offer is presented before the CTA."},
        "followers": 123,
    }
    write_jsonl(evidence / "human_review_queue.jsonl", [queue_row])
    write_jsonl(evidence / "rejected_candidates.jsonl", [queue_row])
    write_jsonl(evidence / "pipeline_diagnostics.jsonl", [queue_row])
    write_jsonl(
        evidence / "quality_decisions.jsonl",
        [{"decision_id": "qd1", "reason": "The candidate violates a local contract."}],
    )
    qa_quality_row = {
        "spec_id": "s1",
        "reason": "The Gold answer is unsupported.",
        "candidate_snapshot": {"question": "Which outcome is proven?"},
    }
    for name in (
        "qa_rejected_candidates.jsonl",
        "qa_human_review_queue.jsonl",
        "qa_pipeline_diagnostics.jsonl",
        "qa_accepted_sample.jsonl",
    ):
        write_jsonl(qa / name, [qa_quality_row])
    write_jsonl(
        model_run / "predictions.jsonl",
        [{"vqa_id": "q1", "answer": "The model says the rack has five tiers."}],
    )
    write_jsonl(
        evaluation / "predictions_judge_details.jsonl",
        [{"vqa_id": "q1", "reason": "The answer matches the visual evidence.", "evidence_alignment": "The cited frame shows five tiers."}],
    )
    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "smoke": {
                "artifacts": {
                    "evidence": {"source": str(evidence)},
                    "qa": {"source": str(qa)},
                    "model_run": {"source": str(model_run)},
                    "evaluation": {"source": str(evaluation)},
                }
            }
        },
    )

    jobs = collect_audit_translation_jobs(
        manifest_path,
        repo_root=tmp_path,
        include_prompts=False,
    )
    job_keys = {(job.object_type, job.object_id, job.source_field) for job in jobs}
    all_source_text = "\n".join(job.source_text for job in jobs)

    assert ("annotation", "a1", "gold_value") in job_keys
    assert ("gold_proposal", "p1", "proposed_gold") in job_keys
    assert ("gold_review", "p1", "issues") in job_keys
    assert ("evidence_review", "r1", "candidate_snapshot") in job_keys
    assert ("evidence_rejection", "r1", "reason") in job_keys
    assert ("pipeline_diagnostic", "r1", "issues") in job_keys
    assert ("quality_decision", "qd1", "reason") in job_keys
    assert ("qa_rejection", "s1", "candidate_snapshot") in job_keys
    assert ("qa_human_review", "s1", "reason") in job_keys
    assert ("qa_diagnostic", "s1", "reason") in job_keys
    assert ("qa_accepted_sample", "s1", "reason") in job_keys
    assert ("model_prediction", "q1", "answer") in job_keys
    assert "followers" not in all_source_text
    assert "likes" not in all_source_text
