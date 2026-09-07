from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(1, ".")

from tools.audit_workbench.build import (
    build_workbench_data,
    classify_review_bucket,
    compact_workbench_to_byte_budget,
    collect_prompt_snapshot,
    compact_workbench_data,
    detect_annotation_risks,
    load_translation_index,
    organize_delivery,
    render_preview_workbench,
    render_workbench,
    select_accepted_annotation_sample,
    select_accepted_qa_sample,
)
from tools.audit_workbench.evidence_assets import (
    enrich_evidence_refs,
    load_frame_manifests,
    materialize_thumbnails,
)
from salesbench.vqa_evaluate.prompts import JUDGE_PROMPT_VERSION


def test_module_cli_resolves_src_package_despite_salesbench_script_shadowing() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.audit_workbench.build", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Build the local SalesBench" in result.stdout


def test_review_bucket_classification_separates_human_decisions_from_pipeline_noise() -> None:
    assert classify_review_bucket({"item_type": "abstention"}) == "abstention_resample"
    assert classify_review_bucket({"reason_code": "VALIDATION_FAILED"}) == "auto_rejected"
    assert classify_review_bucket({"reason_code": "CHALLENGER_REJECT"}) == "auto_rejected"
    assert (
        classify_review_bucket(
            {
                "stage": "commercial_relation_building",
                "resolution_status": "diagnostic",
                "reason_code": "COMMERCIAL_RELATION_VALIDATION_FAILED",
            }
        )
        == "pipeline_diagnostics"
    )
    assert (
        classify_review_bucket(
            {
                "stage": "proposal",
                "reason_code": "PROPOSAL_GRAPH_VALIDATION_FAILED",
                "issues": [{"code": "MISSING_COMMERCIAL_RELATION"}],
            }
        )
        == "pipeline_diagnostics"
    )
    assert (
        classify_review_bucket(
            {"stage": "challenge", "item_type": "candidate", "reason_code": "CHALLENGER_REVISE"}
        )
        == "content_review"
    )
    assert (
        classify_review_bucket({"risk_codes": ["MISSING_TEMPORAL_LOCALIZATION"]})
        == "pipeline_diagnostics"
    )
    assert classify_review_bucket({"risk_codes": ["SELLER_CLAIM_AS_FACT"]}) == "content_review"


def test_accepted_annotation_sample_is_stable_stratified_and_excludes_risks() -> None:
    annotations = [
        {
            "annotation_id": f"{task.lower()}-{index}",
            "video_id": f"v-{index}",
            "task_type": task,
            "evidence_refs": [f"e-{task}-{index}"],
        }
        for task in ("BP", "CM")
        for index in range(10)
    ]
    risk_ids = {"bp-0", "cm-0"}

    first = select_accepted_annotation_sample(annotations, risk_ids, fraction=0.1)
    second = select_accepted_annotation_sample(list(reversed(annotations)), risk_ids, fraction=0.1)

    assert len(first) == 2
    assert {row["task_type"] for row in first} == {"BP", "CM"}
    assert not risk_ids.intersection(str(row["annotation_id"]) for row in first)
    assert [row["annotation_id"] for row in first] == [row["annotation_id"] for row in second]


def test_accepted_qa_sample_is_stable_stratified_and_excludes_rejected_rows() -> None:
    rows = [
        {"vqa_id": f"{task.lower()}-{index}", "task_type": task}
        for task in ("BP", "CM", "SS", "AE")
        for index in range(10)
    ]
    excluded = {"bp-0", "cm-0", "ss-0", "ae-0"}

    first = select_accepted_qa_sample(rows, excluded, fraction=0.1)
    second = select_accepted_qa_sample(list(reversed(rows)), excluded, fraction=0.1)

    assert len(first) == 4
    assert {row["task_type"] for row in first} == {"BP", "CM", "SS", "AE"}
    assert not excluded.intersection(str(row["vqa_id"]) for row in first)
    assert [row["vqa_id"] for row in first] == [row["vqa_id"] for row in second]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_detect_annotation_risks_catches_task_schema_claim_and_localization() -> None:
    evidence = {
        "e_asr": {
            "evidence_id": "e_asr",
            "modality": "asr",
            "text_span": "祛斑淡印",
            "start_s": None,
            "end_s": None,
            "timestamp_status": "unavailable",
        },
        "e_ocr": {
            "evidence_id": "e_ocr",
            "modality": "ocr",
            "text_span": "[0, 50]",
            "frame_indices": [3],
        },
    }
    annotations = [
        {
            "annotation_id": "ae_1",
            "video_id": "v1",
            "task_type": "AE",
            "task_subtype": "CONTENT_MOTIVATION",
            "target": {"claim": "产品有效"},
            "gold_value": {"relation": "SUPPORTED", "answer": "产品有效。"},
            "evidence_refs": ["e_asr", "e_ocr"],
            "source_proposal_ids": ["1"],
        },
        {
            "annotation_id": "bp_1",
            "video_id": "v1",
            "task_type": "BP",
            "task_subtype": "ASR_FACT",
            "target": {"subject": "产品", "predicate": "效果"},
            "gold_value": {"value": "祛斑淡印"},
            "evidence_refs": ["e_asr"],
            "source_proposal_ids": ["bp-source"],
        },
        {
            "annotation_id": "cm_1",
            "video_id": "v1",
            "task_type": "CM",
            "task_subtype": "CLAIM_EVIDENCE_RELATION",
            "target": {"claim": "产品有效"},
            "gold_value": {"relation": "SUPPORTED"},
            "evidence_refs": ["e_asr"],
            "source_proposal_ids": ["cm-source"],
        },
    ]

    risks = detect_annotation_risks(annotations, evidence)
    codes = {code for item in risks for code in item["risk_codes"]}

    assert "AE_SCHEMA_MISMATCH" in codes
    assert "SELLER_CLAIM_AS_FACT" in codes
    assert "INVALID_TEXT_SPAN" in codes
    assert "MISSING_TEMPORAL_LOCALIZATION" in codes
    assert "NUMERIC_PROPOSAL_ID" in codes
    assert "CM_NOT_CROSS_MODAL" in codes


def test_unavailable_asr_timing_is_not_an_audit_risk() -> None:
    evidence = {
        "e_asr": {
            "evidence_id": "e_asr",
            "modality": "asr",
            "text_span": "领券后九块九",
            "start_s": None,
            "end_s": None,
            "timestamp_status": "unavailable",
        }
    }
    annotations = [
        {
            "annotation_id": "bp_price",
            "video_id": "v1",
            "task_type": "BP",
            "task_subtype": "PRICE_AND_DISCOUNT",
            "target": {"subject": "offer"},
            "gold_value": {"answer": "The speaker states a 9.9-yuan coupon price."},
            "evidence_refs": ["e_asr"],
            "source_proposal_ids": ["bp-price"],
        }
    ]

    risks = detect_annotation_risks(annotations, evidence)

    assert not any(
        "MISSING_TEMPORAL_LOCALIZATION" in row["risk_codes"] for row in risks
    )


def test_organize_delivery_physically_separates_formal_and_smoke(tmp_path: Path) -> None:
    _write(tmp_path / "source/formal-evidence/evidence.jsonl", "{}\n")
    _write(tmp_path / "source/formal-vqa/vqa_public.jsonl", "{}\n")
    _write(tmp_path / "source/formal-vqa/public/bp.jsonl", "{}\n")
    _write(tmp_path / "source/formal-vqa/run_gpt4o/predictions.jsonl", "{}\n")
    _write(tmp_path / "source/formal-vqa/run_smoke_gpt4o/predictions.jsonl", "{}\n")
    _write(tmp_path / "source/formal-vqa/evaluation_gpt4o/report.json", "{}")
    _write(tmp_path / "source/formal-vqa/evaluation_smoke_gpt4o/report.json", "{}")
    _write(tmp_path / "source/smoke-evidence/evidence.jsonl", "{}\n")
    _write(tmp_path / "source/smoke-vqa/vqa_public.jsonl", "{}\n")
    _write(tmp_path / "source/analysis/profile.json", "{}")
    manifest = {
        "formal": {
            "root": "outputs/formal/pilot",
            "artifacts": {
                "evidence": {"source": "source/formal-evidence"},
                "qa": {
                    "source": "source/formal-vqa",
                    "exclude": ["run_*", "evaluation_*"],
                },
                "model_run": {"source": "source/formal-vqa/run_gpt4o"},
                "evaluation": {"source": "source/formal-vqa/evaluation_gpt4o"},
                "private_analysis": {"source": "source/analysis"},
            },
        },
        "smoke": {
            "root": "outputs/smoke/gpt4o",
            "artifacts": {
                "evidence": {"source": "source/smoke-evidence"},
                "qa": {"source": "source/smoke-vqa"},
                "model_run": {"source": "source/formal-vqa/run_smoke_gpt4o"},
                "evaluation": {"source": "source/formal-vqa/evaluation_smoke_gpt4o"},
            },
        },
    }

    organized = organize_delivery(manifest, tmp_path)

    formal = tmp_path / "outputs/formal/pilot"
    smoke = tmp_path / "outputs/smoke/gpt4o"
    assert (formal / "qa/vqa_public.jsonl").exists()
    assert (formal / "qa/public/bp.jsonl").exists()
    assert not (formal / "qa/run_smoke_gpt4o").exists()
    assert not (formal / "qa/evaluation_smoke_gpt4o").exists()
    assert (formal / "model_run/predictions.jsonl").exists()
    assert (smoke / "model_run/predictions.jsonl").exists()
    assert (smoke / "evaluation/report.json").exists()
    assert organized["formal_root"] == str(formal)
    assert organized["smoke_root"] == str(smoke)


def test_render_workbench_has_interaction_contract_without_private_fields() -> None:
    prompts = collect_prompt_snapshot()
    data = {
        "release": {"status": "pilot_draft", "prompt_version": "evidence-prompt-v6"},
        "counts": {"videos": 1, "evidence_units": 2, "annotations": 1, "review_queue": 1, "qa": 1},
        "delivery": {"formal_root": "/tmp/formal", "smoke_root": "/tmp/smoke"},
        "evidence": {
            "queue": [{"id": "r1", "video_id": "v1", "reason": "needs review", "task_type": "AE"}],
            "risks": [
                {
                    "id": "a1",
                    "video_id": "v1",
                    "task_type": "AE",
                    "risk_codes": ["AE_SCHEMA_MISMATCH"],
                }
            ],
            "missing_task_videos": ["v1"],
        },
        "qa": [{"vqa_id": "q1", "task_type": "AE", "question": "问题", "gold_answer": "答案"}],
        "judge": {
            "summary": {"scored_count": 1, "judge_failed_count": 0},
            "metrics": {"macro_average": {"relaxed_accuracy": 0.5}},
            "rows": [{"vqa_id": "q1", "task_type": "AE", "score": 0.5, "reason": "partial"}],
        },
        "private_analysis_metadata": {"likes": 999, "followers": 123},
    }

    html = render_workbench(data, prompts, fragment=True)

    assert "<!doctype" not in html.lower()
    assert "<html" not in html.lower()
    assert "salesbench-audit-workbench" in html
    assert "data-tab=\"prompts\"" in html
    assert "data-tab=\"evidence\"" in html
    assert "data-tab=\"qa\"" in html
    assert "data-tab=\"judge\"" in html
    assert "localStorage" in html
    assert "exportDecisions" in html
    assert "evidence-prompt-v6" in html
    assert "evidence-prompt-v10" in html
    assert "运行版本" in html
    assert "当前代码" in html
    assert "Gold Challenger" in html
    assert "senior multimodal evaluator" in html
    assert "商业图资产" in html
    assert "\"likes\"" not in html
    assert "\"followers\"" not in html
    json.loads(html.split('<script type="application/json" id="sbaw-data">', 1)[1].split("</script>", 1)[0])


def test_review_views_render_one_case_pagers_and_four_evidence_queues() -> None:
    data = {
        "release": {"status": "candidate", "prompt_version": "evidence-prompt-v9"},
        "counts": {
            "videos": 1,
            "evidence_units": 1,
            "annotations": 1,
            "review_queue": 1,
            "accepted_sample": 1,
            "qa": 1,
            "judge_rows": 1,
        },
        "delivery": {},
        "evidence": {
            "queue": [
                {
                    "id": "r1",
                    "video_id": "v1",
                    "task_type": "CM",
                    "reason": "needs review",
                    "audit_bucket": "content_review",
                }
            ],
            "risks": [],
            "abstentions": [],
            "accepted_sample": [
                {
                    "id": "a1",
                    "video_id": "v1",
                    "task_type": "BP",
                    "audit_bucket": "accepted_sample",
                }
            ],
            "missing_task_videos": [],
        },
        "qa": [{"vqa_id": "q1", "task_type": "BP", "question": "Question?", "gold_answer": "Answer."}],
        "final_qa": [{"vqa_id": "q1", "task_type": "BP", "question": "Question?", "gold_answer": "Answer."}],
        "integrity": {"status": "VERIFIED", "blocking": False, "issues": []},
        "judge": {
            "summary": {},
            "metrics": {},
            "rows": [{"vqa_id": "q1", "task_type": "BP", "question": "Question?", "score": 1}],
        },
    }

    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)

    assert 'class="case-pager"' in html
    assert html.count("${pager(") == 4
    assert 'aria-label="上一条"' in html
    assert 'aria-label="下一条"' in html
    assert 'aria-label="跳转到案例序号"' in html
    assert "ArrowLeft" in html and "ArrowRight" in html
    assert "event.key==='Enter'" in html
    assert "prev.onclick" in html and "next.onclick" in html
    assert "prev.addEventListener" not in html and "next.addEventListener" not in html
    assert "内容审核" in html
    assert "Abstention / 补采样" in html
    assert "流水线诊断" in html
    assert "自动通过抽样" in html
    assert "原始诊断队列" in html
    assert "slice(0,200)" not in html


def test_final_qa_browser_is_chinese_first_paginated_and_read_only() -> None:
    data = {
        "release": {
            "status": "pilot_candidate_requires_human_review",
            "prompt_version": "evidence-prompt-v10.3",
            "run_id": "run-1",
        },
        "counts": {"videos": 1, "qa": 1, "final_qa": 1, "judge_rows": 0},
        "delivery": {},
        "translations": {},
        "integrity": {"status": "VERIFIED", "blocking": False, "issues": []},
        "evidence": {
            "queue": [],
            "risks": [],
            "abstentions": [],
            "accepted_sample": [],
            "missing_task_videos": [],
            "commerce_cues": [],
            "commercial_relations": [],
        },
        "qa": [],
        "final_qa": [
            {
                "vqa_id": "q-final",
                "video_id": "v1",
                "task_type": "SS",
                "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                "capability": "CLAIM_DEMONSTRATION_STATUS",
                "question": "How does the demonstration support the seller's claim?",
                "question_zh": "演示如何支持卖家的主张？",
                "gold_answer": "It shows the claimed state change.",
                "gold_answer_zh": "演示呈现了所声称的状态变化。",
                "evidence_items": [
                    {
                        "evidence_id": "e1",
                        "content_en": "The product changes state during use.",
                        "content_zh": "产品在使用过程中发生状态变化。",
                        "frames": [{"thumbnail_path": "assets/f.jpg", "frame_index": 1}],
                    }
                ],
                "commerce_cues": [],
                "commercial_relations": [],
                "selection_score": 88.0,
            }
        ],
        "judge": {"summary": {}, "metrics": {}, "rows": []},
    }

    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)

    assert 'data-tab="finalqa"' in html
    assert "最终 QA 浏览" in html
    assert "演示如何支持卖家的主张？" in html
    assert "English canonical question and Gold" in html
    assert "pager('fq')" in html
    assert "pageState={ev:0,qa:0,fq:0,jd:0}" in html
    assert "loading=\"lazy\"" in html
    final_renderer = html.split("function renderFinalQA()", 1)[1].split("function renderJudge()", 1)[0]
    assert "decisionButtons" not in final_renderer


def test_current_artifact_integrity_mismatch_blocks_final_qa_browser(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    qa_dir = tmp_path / "qa"
    evidence = {
        "evidence_id": "e1",
        "video_id": "v1",
        "modality": "visual",
        "frame_indices": [0],
        "content_en": "The host demonstrates the product.",
    }
    annotation = {
        "annotation_id": "a1",
        "video_id": "v1",
        "task_type": "SS",
        "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
        "target": {"claim": "easy to use"},
        "gold_value": {"answer": "The host demonstrates the use step."},
        "evidence_refs": ["e1"],
    }
    qa = {
        "vqa_id": "q1",
        "video_id": "v1",
        "task_type": "SS",
        "question": "How is the claim demonstrated?",
        "gold_answer": "The host demonstrates the use step.",
        "source_annotation_ids": ["a1"],
        "evidence_refs": ["e1"],
        "commerce_cue_ids": [],
        "commercial_relation_ids": [],
    }
    _write(evidence_dir / "evidence_units.jsonl", json.dumps(evidence) + "\n")
    _write(evidence_dir / "commerce_cues.jsonl", "")
    _write(evidence_dir / "commercial_relations.jsonl", "")
    _write(
        evidence_dir / "video_evidence_dataset.jsonl",
        json.dumps(
            {
                "schema_version": "evidence-dataset-schema-v4",
                "video_id": "v1",
                "grounded_annotations": [annotation],
            }
        )
        + "\n",
    )
    _write(
        evidence_dir / "generation_meta.json",
        json.dumps({"video_ids": ["v1"], "run_id": "run-1", "evidence_fingerprint": "e" * 64}),
    )
    _write(qa_dir / "vqa_gold_private.jsonl", json.dumps(qa) + "\n")
    _write(
        qa_dir / "generation_meta.json",
        json.dumps(
            {
                "compiler_version": "evidence-qa-compiler-v9",
                "evidence_fingerprint": "x" * 64,
                "qa_realization_fingerprint": "q" * 64,
                "compile_fingerprint": "c" * 64,
            }
        ),
    )
    manifest = {
        "frame_cache_root": "frames",
        "formal": {
            "artifacts": {
                "evidence": {"source": "evidence"},
                "qa": {"source": "qa"},
            }
        },
    }

    data = build_workbench_data(manifest, tmp_path, {})

    assert data["integrity"]["status"] == "BLOCKED"
    assert data["integrity"]["blocking"] is True
    assert "EVIDENCE_FINGERPRINT_MISMATCH" in data["integrity"]["issues"]
    assert data["final_qa"] == []
    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)
    assert "最终 QA 已阻断" in html


def test_v10_read_only_buckets_do_not_render_review_decision_controls() -> None:
    data = {
        "release": {"status": "candidate", "prompt_version": "evidence-prompt-v10"},
        "counts": {"videos": 1, "review_queue": 0, "qa": 0, "judge_rows": 0},
        "delivery": {},
        "evidence": {
            "queue": [
                {
                    "id": "rejected-1",
                    "video_id": "v1",
                    "audit_bucket": "auto_rejected",
                    "reason": "invalid relation",
                }
            ],
            "risks": [],
            "abstentions": [],
            "accepted_sample": [],
            "missing_task_videos": [],
        },
        "qa": [],
        "judge": {"summary": {}, "metrics": {}, "rows": []},
    }

    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)

    assert "自动拒绝" in html
    assert "只读记录" in html
    assert "x.audit_bucket==='human_review'" in html
    assert "rows.some(x=>x.audit_bucket==='human_review')" in html


def test_qa_audit_uses_quality_buckets_and_only_reviews_ambiguity_or_monitoring_sample() -> None:
    data = {
        "release": {"status": "candidate", "prompt_version": "evidence-prompt-v10"},
        "counts": {"videos": 1, "review_queue": 0, "qa": 3, "judge_rows": 0},
        "delivery": {},
        "evidence": {
            "queue": [],
            "risks": [],
            "abstentions": [],
            "accepted_sample": [],
            "missing_task_videos": [],
        },
        "qa": [
            {"vqa_id": "q-human", "task_type": "AE", "audit_bucket": "human_review"},
            {"vqa_id": "q-sample", "task_type": "BP", "audit_bucket": "accepted_sample"},
            {"vqa_id": "q-reject", "task_type": "SS", "audit_bucket": "auto_rejected"},
        ],
        "judge": {"summary": {}, "metrics": {}, "rows": []},
    }

    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)

    assert "QA 人工语义审核" in html
    assert "自动通过全集（只读）" in html
    assert "x.audit_bucket==='human_review'||x.audit_bucket==='accepted_sample'" in html
    assert "该 QA 是只读生产记录" in html
    assert "rows.some(x=>x.audit_bucket==='human_review')" in html


def test_preview_renderer_reduces_record_limit_to_fit_byte_budget() -> None:
    large = "x" * 10_000
    rows = [{"id": f"row-{index}", "reason": large} for index in range(12)]
    data = {
        "release": {"prompt_version": "v9"},
        "counts": {},
        "delivery": {},
        "translations": {},
        "evidence": {
            "queue": rows,
            "risks": rows,
            "abstentions": [],
            "accepted_sample": rows,
            "missing_task_videos": [],
        },
        "qa": [{"vqa_id": f"q-{index}", "question": large} for index in range(12)],
        "judge": {"summary": {}, "metrics": {}, "rows": [{"vqa_id": f"j-{index}", "reason": large} for index in range(12)]},
    }

    html = render_preview_workbench(data, [], limit=12, max_bytes=100_000)

    assert len(html.encode("utf-8")) < 100_000
    assert "预览根据 100000 字节上限自动缩减" in html


def test_preview_budget_includes_asset_preparation_mutations() -> None:
    rows = [{"id": f"row-{index}", "reason": "small"} for index in range(12)]
    data = {
        "evidence": {
            "queue": rows,
            "risks": rows,
            "abstentions": [],
            "accepted_sample": rows,
        },
        "qa": [{"vqa_id": f"q-{index}"} for index in range(12)],
        "judge": {"rows": [{"vqa_id": f"j-{index}"} for index in range(12)]},
    }
    prepared_limits: list[int] = []

    def prepare(compact: dict) -> None:
        prepared_limits.append(len(compact["qa"]))
        for collection in (
            compact["evidence"]["queue"],
            compact["evidence"]["risks"],
            compact["evidence"]["accepted_sample"],
            compact["qa"],
            compact["judge"]["rows"],
        ):
            for row in collection:
                row["prepared_asset"] = "y" * 5_000

    compact = compact_workbench_to_byte_budget(
        data,
        [],
        limit=12,
        max_bytes=100_000,
        prepare=prepare,
    )
    html = render_workbench(compact, [], fragment=True)

    assert len(prepared_limits) > 1
    assert prepared_limits[-1] < prepared_limits[0]
    assert len(html.encode("utf-8")) < 100_000


def test_prompt_snapshot_uses_current_judge_prompt_version() -> None:
    judge = next(item for item in collect_prompt_snapshot() if item["id"] == "judge")

    assert judge["version"] == JUDGE_PROMPT_VERSION


def test_prompt_snapshot_includes_question_surface_repairer() -> None:
    prompts = {item["id"]: item for item in collect_prompt_snapshot()}

    assert "question_repairer" in prompts
    assert "Question Surface Repairer" in prompts["question_repairer"]["system"]
    assert "model_runner" in prompts
    assert "final answer in English" in prompts["model_runner"]["system"]
    assert "qa_semantic_quality_gate" in prompts
    assert "plain insufficiency is REJECT" in prompts["qa_semantic_quality_gate"]["system"]


def test_prompt_snapshot_includes_split_language_and_visual_evidence_stages() -> None:
    prompts = {item["id"]: item for item in collect_prompt_snapshot()}

    assert "language_evidence_extractor" in prompts
    assert "visual_evidence_extractor" in prompts
    assert "visual_evidence_repairer" in prompts
    assert "visual_commerce_cue_extractor" in prompts
    assert "ASR Evidence Extractor" in prompts["language_evidence_extractor"]["system"]
    assert "Visual and OCR Evidence Extractor" in prompts["visual_evidence_extractor"]["system"]
    assert "Visual Evidence Repairer" in prompts["visual_evidence_repairer"]["system"]


def test_compact_preview_prioritizes_structural_risks_over_systemic_missing_time() -> None:
    data = {
        "evidence": {
            "queue": [],
            "risks": [
                {"id": "time", "risk_codes": ["MISSING_TEMPORAL_LOCALIZATION"]},
                {"id": "schema", "risk_codes": ["AE_SCHEMA_MISMATCH"]},
            ],
        },
        "qa": [],
        "judge": {"rows": []},
    }

    compact = compact_workbench_data(data, limit=1)

    assert compact["evidence"]["risks"][0]["id"] == "schema"


def test_compact_preview_keeps_accepted_sample_stratified_by_task() -> None:
    accepted = [
        {"id": f"{task}-{index}", "task_type": task}
        for task in ("AE", "BP", "CM", "SS")
        for index in range(4)
    ]
    data = {
        "evidence": {"queue": [], "risks": [], "abstentions": [], "accepted_sample": accepted},
        "qa": [],
        "judge": {"rows": []},
    }

    compact = compact_workbench_data(data, limit=4)

    assert {row["task_type"] for row in compact["evidence"]["accepted_sample"]} == {
        "AE",
        "BP",
        "CM",
        "SS",
    }


def test_evidence_content_links_visual_unit_to_exact_cached_frame(tmp_path: Path) -> None:
    _write(tmp_path / "frames/v1/frame_000.jpg", "zero")
    _write(tmp_path / "frames/v1/frame_001.jpg", "one")
    _write(
        tmp_path / "frames/v1/manifest.json",
        json.dumps(
            {
                "video_id": "v1",
                "frames": [
                    {"frame_index": 0, "timestamp_s": 0.5, "path": "frames/v1/frame_000.jpg"},
                    {"frame_index": 1, "timestamp_s": 1.5, "path": "frames/v1/frame_001.jpg"},
                ],
            }
        ),
    )
    evidence = {
        "e_visual": {
            "evidence_id": "e_visual",
            "video_id": "v1",
            "modality": "visual",
            "frame_indices": [1],
            "subject": "产品",
            "predicate": "颜色",
            "value": "红色",
            "text_span": "",
            "confidence": 0.9,
        }
    }

    manifests = load_frame_manifests(tmp_path / "frames", {"v1"}, repo_root=tmp_path)
    items = enrich_evidence_refs("v1", ["e_visual"], evidence, manifests)

    assert items[0]["semantic_text"] == "产品｜颜色｜红色"
    assert items[0]["frames"] == [
        {
            "frame_index": 1,
            "timestamp_s": 1.5,
            "source_path": str(tmp_path / "frames/v1/frame_001.jpg"),
            "relation": "direct",
        }
    ]


def test_v3_evidence_exposes_canonical_english_and_native_source(tmp_path: Path) -> None:
    evidence = {
        "e_visual": {
            "evidence_id": "e_visual",
            "video_id": "v1",
            "modality": "ocr",
            "frame_indices": [],
            "content_en": "The frame shows a 9.9-yuan offer.",
            "source_text_native": "到手9.9元",
            "confidence": 0.9,
        }
    }

    items = enrich_evidence_refs("v1", ["e_visual"], evidence, {})

    assert items[0]["content_en"] == "The frame shows a 9.9-yuan offer."
    assert items[0]["source_text_native"] == "到手9.9元"
    assert items[0]["semantic_text"] == "The frame shows a 9.9-yuan offer."


def test_representative_frames_make_unlocalized_asr_auditable(tmp_path: Path) -> None:
    frames = []
    for index, timestamp in enumerate((0.5, 1.5, 2.5, 3.5, 4.5)):
        _write(tmp_path / f"frames/v1/frame_{index:03d}.jpg", str(index))
        frames.append(
            {
                "frame_index": index,
                "timestamp_s": timestamp,
                "path": f"frames/v1/frame_{index:03d}.jpg",
            }
        )
    _write(tmp_path / "frames/v1/manifest.json", json.dumps({"video_id": "v1", "frames": frames}))
    evidence = {
        "e_asr": {
            "evidence_id": "e_asr",
            "video_id": "v1",
            "modality": "asr",
            "frame_indices": [],
            "text_span": "现在下单",
            "subject": "主播",
            "predicate": "行动提示",
            "value": "下单",
            "start_s": None,
            "end_s": None,
            "confidence": 0.9,
        }
    }

    manifests = load_frame_manifests(tmp_path / "frames", {"v1"}, repo_root=tmp_path)
    items = enrich_evidence_refs("v1", ["e_asr"], evidence, manifests)

    assert [frame["frame_index"] for frame in items[0]["frames"]] == [0, 2, 4]
    assert {frame["relation"] for frame in items[0]["frames"]} == {"representative"}
    assert items[0]["localization_note"] == "ASR 缺少时间戳，以下为视频代表帧，不能精确定位到该语音片段。"
    assert items[0]["text_span"] == "现在下单"


def test_missing_evidence_ref_is_visible_instead_of_silently_dropped() -> None:
    items = enrich_evidence_refs("v1", ["missing"], {}, {})

    assert items == [
        {
            "evidence_id": "missing",
            "missing": True,
            "semantic_text": "EvidenceUnit 不存在",
            "localization_note": "无法解析该 evidence_ref。",
            "frames": [],
        }
    ]


def test_queue_without_evidence_refs_still_shows_representative_context() -> None:
    manifests = {
        "v1": {
            0: {"frame_index": 0, "timestamp_s": 0.5, "source_path": "/frames/0.jpg"},
            4: {"frame_index": 4, "timestamp_s": 4.5, "source_path": "/frames/4.jpg"},
        }
    }

    items = enrich_evidence_refs("v1", [], {}, manifests)

    assert items[0]["modality"] == "context"
    assert items[0]["localization_note"] == "该审计项没有直接 Evidence 引用，以下为视频代表帧。"
    assert [frame["frame_index"] for frame in items[0]["frames"]] == [0, 4]


def test_workbench_data_makes_queue_qa_and_judge_evidence_readable(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    qa_dir = tmp_path / "qa"
    evaluation_dir = tmp_path / "evaluation"
    _write(tmp_path / "frames/v1/frame_001.jpg", "frame")
    _write(tmp_path / "frames/v2/frame_002.jpg", "frame")
    _write(
        tmp_path / "frames/v1/manifest.json",
        json.dumps(
            {
                "video_id": "v1",
                "frames": [{"frame_index": 1, "timestamp_s": 1.5, "path": "frames/v1/frame_001.jpg"}],
            }
        ),
    )
    _write(
        tmp_path / "frames/v2/manifest.json",
        json.dumps(
            {
                "video_id": "v2",
                "frames": [{"frame_index": 2, "timestamp_s": 2.5, "path": "frames/v2/frame_002.jpg"}],
            }
        ),
    )
    unit = {
        "evidence_id": "e1",
        "video_id": "v1",
        "modality": "ocr",
        "frame_indices": [1],
        "text_span": "国家专利",
        "subject": "产品",
        "predicate": "认证",
        "value": "国家专利",
        "start_s": 1.0,
        "end_s": 2.0,
        "confidence": 0.9,
    }
    annotation = {
        "annotation_id": "a1",
        "video_id": "v1",
        "task_type": "BP",
        "task_subtype": "OCR_FACT",
        "target": {"subject": "产品", "predicate": "认证"},
        "gold_value": {"value": "国家专利"},
        "evidence_refs": ["e1"],
        "source_proposal_ids": ["p1"],
    }
    proposal = {
        "proposal_id": "p1",
        "video_id": "v1",
        "task_type": "BP",
        "task_subtype": "OCR_FACT",
        "target": annotation["target"],
        "proposed_gold": annotation["gold_value"],
        "evidence_ids": ["e1"],
        "proposal_confidence": 0.9,
    }
    abstained_unit = {
        "evidence_id": "e2",
        "video_id": "v2",
        "modality": "visual",
        "frame_indices": [2],
        "content_en": "A quiz card is shown before the product pitch.",
    }
    _write(
        evidence_dir / "evidence_units.jsonl",
        json.dumps(unit, ensure_ascii=False) + "\n" + json.dumps(abstained_unit) + "\n",
    )
    _write(
        evidence_dir / "video_evidence_dataset.jsonl",
        json.dumps({"video_id": "v1", "grounded_annotations": [annotation]}, ensure_ascii=False) + "\n",
    )
    _write(evidence_dir / "gold_proposals.jsonl", json.dumps(proposal, ensure_ascii=False) + "\n")
    _write(
        evidence_dir / "human_review_queue.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "review_item_id": "r1",
                        "video_id": "v1",
                        "source_proposal_ids": ["p1"],
                        "reason": "复核",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "review_item_id": "r1",
                        "video_id": "v1",
                        "stage": "commercial_relation_building",
                        "item_type": "commercial_relation",
                        "reason_code": "COMMERCIAL_RELATION_VALIDATION_FAILED",
                        "reason": "commercial_relation_validation_failed",
                        "evidence_refs": ["e1"],
                        "issues": [{"code": "INVALID_RELATION_TARGET_TYPE"}],
                    }
                ),
            ]
        )
        + "\n",
    )
    _write(
        evidence_dir / "audit_before_review.json",
        json.dumps({"video_count": 1, "videos_missing_required_tasks": []}),
    )
    _write(
        evidence_dir / "generation_meta.json",
        json.dumps(
            {
                "prompt_version": "evidence-prompt-v6",
                "video_ids": ["v1", "v2"],
                "statuses": {"v1": "ok", "v2": "partial"},
            }
        ),
    )
    qa = {
        "vqa_id": "q1",
        "video_id": "v1",
        "task_type": "BP",
        "task_subtype": "OCR_FACT",
        "question": "画面展示了什么认证？",
        "gold_answer": "国家专利",
        "evidence_refs": ["e1"],
        "source_annotation_ids": ["a1"],
    }
    _write(qa_dir / "vqa_gold_private.jsonl", json.dumps(qa, ensure_ascii=False) + "\n")
    _write(qa_dir / "generation_meta.json", json.dumps({"compiler_version": "test"}))
    _write(
        evaluation_dir / "predictions_judge_details.jsonl",
        json.dumps(
            {
                "vqa_id": "q1",
                "video_id": "v1",
                "task_type": "BP",
                "question": qa["question"],
                "reference_answer": "国家专利",
                "model_output": "国家专利",
                "score": 1,
                "reason": "正确",
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    _write(
        evaluation_dir / "predictions_salesbench_qa_eval.json",
        json.dumps({"summary": {}, "metrics": {}, "judge_model": "judge"}),
    )
    manifest = {
        "tested_model": "model",
        "frame_cache_root": "frames",
        "formal": {
            "artifacts": {
                "evidence": {"source": "evidence"},
                "qa": {"source": "qa"},
                "evaluation": {"source": "evaluation"},
            }
        },
    }

    data = build_workbench_data(
        manifest,
        tmp_path,
        {"formal_root": "/formal", "smoke_root": "/smoke"},
    )

    assert data["evidence"]["queue"][0]["evidence_items"][0]["text_span"] == "国家专利"
    assert len({row["id"] for row in data["evidence"]["queue"]}) == 2
    assert all(row["id"].startswith("r1:") for row in data["evidence"]["queue"])
    assert data["evidence"]["queue"][1]["resolution_status"] == "diagnostic"
    assert data["evidence"]["queue"][0]["audit_bucket"] == "content_review"
    assert data["evidence"]["queue"][1]["audit_bucket"] == "pipeline_diagnostics"
    assert data["qa"][0]["evidence_items"][0]["semantic_text"] == "产品｜认证｜国家专利"
    assert data["qa"][0]["evidence_items"][0]["frames"][0]["frame_index"] == 1
    assert data["judge"]["rows"][0]["evidence_items"] == data["qa"][0]["evidence_items"]
    assert data["release"]["runtime_prompt_version"] == "evidence-prompt-v6"
    assert data["release"]["current_prompt_version"] == "evidence-prompt-v10.5"
    assert data["release"]["current_judge_prompt_version"] == "judge-prompt-v5"
    assert data["counts"]["videos"] == 2
    assert data["counts"]["commercial_records"] == 1
    assert data["counts"]["abstentions"] == 1
    assert data["counts"]["accepted_sample"] == 1
    assert data["counts"]["human_review_queue"] == 1
    assert data["counts"]["pipeline_diagnostics"] == 1
    assert data["evidence"]["accepted_sample"][0]["id"] == "a1"
    assert data["evidence"]["accepted_sample"][0]["audit_bucket"] == "accepted_sample"
    abstention = data["evidence"]["abstentions"][0]
    assert abstention["video_id"] == "v2"
    assert abstention["reason_code"] == "NO_COMMERCIAL_RECORD"
    assert abstention["audit_bucket"] == "abstention_resample"
    assert abstention["evidence_items"][0]["content_en"] == abstained_unit["content_en"]
    assert abstention["evidence_items"][0]["frames"][0]["frame_index"] == 2


def test_v10_workbench_consumes_separated_quality_artifacts_directly(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    qa_dir = tmp_path / "qa"
    qa_realizations_dir = tmp_path / "qa-realizations"
    _write(
        evidence_dir / "generation_meta.json",
        json.dumps({"video_ids": ["v1"], "prompt_version": "evidence-prompt-v10"}),
    )
    _write(
        evidence_dir / "evidence_units.jsonl",
        json.dumps(
            {
                "evidence_id": "e1",
                "video_id": "v1",
                "modality": "visual",
                "frame_indices": [0],
                "content_en": "The host holds the product.",
            }
        )
        + "\n",
    )
    _write(evidence_dir / "video_evidence_dataset.jsonl", "")
    _write(evidence_dir / "gold_proposals.jsonl", "")
    common = {
        "video_id": "v1",
        "item_type": "commercial_relation",
        "evidence_refs": ["e1"],
        "candidate_snapshot": {
            "relation_id": "r1",
            "evidence_ids": ["e1"],
            "rationale_en": "The relation candidate cites the visible product.",
        },
    }
    _write(
        evidence_dir / "human_review_queue.jsonl",
        json.dumps(
            {
                **common,
                "review_item_id": "human-1",
                "reason_code": "SEMANTIC_VERIFIER_AMBIGUOUS",
                "reason": "Two readings remain possible.",
            }
        )
        + "\n",
    )
    _write(
        evidence_dir / "rejected_candidates.jsonl",
        json.dumps(
            {
                **common,
                "review_item_id": "reject-1",
                "reason_code": "SEMANTIC_VERIFIER_REJECT",
                "reason": "The cited frame does not support the relation.",
            }
        )
        + "\n",
    )
    _write(
        evidence_dir / "pipeline_diagnostics.jsonl",
        json.dumps(
            {
                "review_item_id": "diag-1",
                "video_id": "v1",
                "item_type": "stage_failure",
                "reason_code": "RELATION_PARSE_ERROR",
                "reason": "Invalid JSON.",
            }
        )
        + "\n",
    )
    _write(evidence_dir / "quality_decisions.jsonl", "")
    qa_pass = {
        "vqa_id": "q-pass",
        "video_id": "v1",
        "task_type": "BP",
        "task_subtype": "PRODUCT_IDENTITY",
        "question": "What product is held by the host?",
        "gold_answer": "The featured product.",
        "spec_id": "qs-pass",
        "evidence_refs": ["e1"],
        "source_annotation_ids": [],
    }
    _write(qa_dir / "vqa_gold_private.jsonl", json.dumps(qa_pass) + "\n")
    qa_spec = {
        "spec_id": "qs-human",
        "video_id": "v1",
        "annotation_id": "a-human",
        "task_type": "AE",
        "capability": "CONTENT_IMPLIED_NEED",
        "reasoning_operator": "INFER_BOUNDED_NEED",
        "gold_answer": "A bounded need.",
        "evidence_refs": ["e1"],
        "commerce_cue_ids": [],
        "commercial_relation_ids": [],
    }
    qa_quality_common = {
        "video_id": "v1",
        "task_type": "AE",
        "reason": "Quality decision.",
        "candidate_snapshot": {
            "question_spec": qa_spec,
            "question": "What bounded need is indicated by the content?",
        },
    }
    _write(
        qa_realizations_dir / "qa_human_review_queue.jsonl",
        json.dumps(
            {
                **qa_quality_common,
                "spec_id": "qs-human",
                "reason_code": "QA_SEMANTIC_AMBIGUITY",
            }
        )
        + "\n",
    )
    rejected_spec = {**qa_spec, "spec_id": "qs-reject", "annotation_id": "a-reject"}
    _write(
        qa_realizations_dir / "qa_rejected_candidates.jsonl",
        json.dumps(
            {
                **qa_quality_common,
                "spec_id": "qs-reject",
                "reason_code": "QA_SEMANTIC_REJECT",
                "candidate_snapshot": {
                    "question_spec": rejected_spec,
                    "question": "What unsupported need will make everyone buy?",
                },
            }
        )
        + "\n",
    )
    diagnostic_spec = {**qa_spec, "spec_id": "qs-diagnostic", "annotation_id": "a-diagnostic"}
    _write(
        qa_realizations_dir / "qa_pipeline_diagnostics.jsonl",
        json.dumps(
            {
                **qa_quality_common,
                "spec_id": "qs-diagnostic",
                "reason_code": "QA_PIPELINE_FAILURE",
                "candidate_snapshot": {"question_spec": diagnostic_spec},
            }
        )
        + "\n",
    )
    _write(
        qa_realizations_dir / "qa_accepted_sample.jsonl",
        json.dumps(
            {
                "spec_id": "qs-pass",
                "video_id": "v1",
                "task_type": "BP",
                "reason_code": "QA_ACCEPTED_MONITORING_SAMPLE",
            }
        )
        + "\n",
    )
    manifest = {
        "frame_cache_root": "frames",
        "formal": {
            "artifacts": {
                "evidence": {"source": "evidence"},
                "qa_realizations": {"source": "qa-realizations"},
                "qa": {"source": "qa"},
            }
        },
    }

    data = build_workbench_data(manifest, tmp_path, {})

    buckets = {row["id"]: row["audit_bucket"] for row in data["evidence"]["queue"]}
    assert buckets == {
        "human-1": "human_review",
        "reject-1": "auto_rejected",
        "diag-1": "pipeline_diagnostics",
    }
    assert data["counts"]["human_review_queue"] == 1
    assert data["counts"]["auto_rejected"] == 1
    assert data["counts"]["pipeline_diagnostics"] == 1
    assert data["evidence"]["queue"][0]["candidate_snapshot"]
    qa_buckets = {row["spec_id"]: row["audit_bucket"] for row in data["qa"]}
    assert qa_buckets == {
        "qs-pass": "accepted_sample",
        "qs-human": "human_review",
        "qs-reject": "auto_rejected",
        "qs-diagnostic": "pipeline_diagnostics",
    }
    assert data["counts"]["qa_human_review"] == 1
    assert data["counts"]["qa_accepted_sample"] == 1
    assert data["counts"]["qa_auto_rejected"] == 1
    assert data["counts"]["qa_pipeline_diagnostics"] == 1


def test_qa_with_missing_evidence_reference_is_auto_rejected_in_audit(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    qa_dir = tmp_path / "qa"
    _write(
        evidence_dir / "generation_meta.json",
        json.dumps({"video_ids": ["v1"], "prompt_version": "evidence-prompt-v10.2"}),
    )
    _write(evidence_dir / "evidence_units.jsonl", "")
    _write(evidence_dir / "video_evidence_dataset.jsonl", "")
    _write(evidence_dir / "gold_proposals.jsonl", "")
    _write(
        qa_dir / "vqa_gold_private.jsonl",
        json.dumps(
            {
                "vqa_id": "q-missing",
                "video_id": "v1",
                "task_type": "BP",
                "task_subtype": "USAGE_STEP",
                "question": "What action is demonstrated?",
                "gold_answer": "The presenter turns a page.",
                "spec_id": "qs-missing",
                "evidence_refs": ["missing-evidence"],
            }
        )
        + "\n",
    )
    _write(
        qa_dir / "qa_accepted_sample.jsonl",
        json.dumps({"spec_id": "qs-missing", "video_id": "v1", "task_type": "BP"})
        + "\n",
    )
    manifest = {
        "frame_cache_root": "frames",
        "formal": {
            "artifacts": {
                "evidence": {"source": "evidence"},
                "qa": {"source": "qa"},
            }
        },
    }

    data = build_workbench_data(manifest, tmp_path, {})
    qa = data["qa"][0]

    assert qa["audit_bucket"] == "auto_rejected"
    assert qa["risk_codes"] == ["MISSING_EVIDENCE_REF"]
    assert data["counts"]["qa_auto_rejected"] == 1


def test_workbench_localizes_review_payloads_without_counting_empty_fields_as_missing(
    tmp_path: Path,
) -> None:
    from salesbench.audit_translation import build_translation_job

    evidence_dir = tmp_path / "evidence"
    qa_dir = tmp_path / "qa"
    snapshot = {
        "relation_id": "rel1",
        "rationale_en": "The offer is presented before the call to action.",
        "semantic_verifier": {
            "verdict": "AMBIGUOUS",
            "reason": "The timestamps do not establish temporal order.",
        },
    }
    reason = "Two temporal readings remain possible."
    snapshot_source = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    qa_snapshot = {
        "question": "Which need is supported by the content?",
        "question_spec": {
            "spec_id": "qs-human",
            "video_id": "v1",
            "task_type": "AE",
            "capability": "CONTENT_IMPLIED_NEED",
            "gold_answer": "The content supports a bounded household need.",
        },
    }
    qa_snapshot_source = json.dumps(qa_snapshot, ensure_ascii=False, sort_keys=True)
    _write(
        evidence_dir / "generation_meta.json",
        json.dumps({"video_ids": ["v1"], "prompt_version": "evidence-prompt-v10.2"}),
    )
    _write(evidence_dir / "evidence_units.jsonl", "")
    annotation = {
        "annotation_id": "a1",
        "video_id": "v1",
        "task_type": "BP",
        "task_subtype": "PRODUCT_IDENTITY",
        "target": {"subject": "featured product"},
        "gold_value": {"answer": "The host presents a storage product."},
        "evidence_refs": [],
    }
    _write(
        evidence_dir / "video_evidence_dataset.jsonl",
        json.dumps({"video_id": "v1", "grounded_annotations": [annotation]}) + "\n",
    )
    _write(evidence_dir / "gold_proposals.jsonl", "")
    _write(
        evidence_dir / "human_review_queue.jsonl",
        json.dumps(
            {
                "review_item_id": "human-1",
                "video_id": "v1",
                "item_type": "commercial_relation",
                "reason": reason,
                "candidate_snapshot": snapshot,
            }
        )
        + "\n",
    )
    _write(evidence_dir / "rejected_candidates.jsonl", "")
    _write(evidence_dir / "pipeline_diagnostics.jsonl", "")
    _write(evidence_dir / "quality_decisions.jsonl", "")
    _write(qa_dir / "vqa_gold_private.jsonl", "")
    _write(
        qa_dir / "qa_human_review_queue.jsonl",
        json.dumps(
            {
                "spec_id": "qs-human",
                "video_id": "v1",
                "task_type": "AE",
                "reason": "The inferred need remains ambiguous.",
                "candidate_snapshot": qa_snapshot,
            }
        )
        + "\n",
    )
    _write(qa_dir / "qa_rejected_candidates.jsonl", "")
    _write(qa_dir / "qa_pipeline_diagnostics.jsonl", "")
    _write(qa_dir / "qa_accepted_sample.jsonl", "")
    jobs = [
        (
            build_translation_job(
                object_type="evidence_review",
                object_id="human-1",
                source_field="reason",
                source_text=reason,
            ),
            "仍存在两种可能的时间顺序解读。",
        ),
        (
            build_translation_job(
                object_type="evidence_review",
                object_id="human-1",
                source_field="candidate_snapshot",
                source_text=snapshot_source,
            ),
            "候选快照：优惠先于行动号召出现；时间戳不足以确定顺序。（AMBIGUOUS）",
        ),
        (
            build_translation_job(
                object_type="qa_human_review",
                object_id="qs-human",
                source_field="reason",
                source_text="The inferred need remains ambiguous.",
            ),
            "推断出的需求仍存在歧义。",
        ),
        (
            build_translation_job(
                object_type="qa_human_review",
                object_id="qs-human",
                source_field="candidate_snapshot",
                source_text=qa_snapshot_source,
            ),
            "候选问题：内容支持哪一种需求？候选答案：内容支持一种有边界的家庭需求。",
        ),
        (
            build_translation_job(
                object_type="annotation",
                object_id="a1",
                source_field="target",
                source_text=json.dumps(annotation["target"], sort_keys=True),
            ),
            "目标：画面中的商品。",
        ),
        (
            build_translation_job(
                object_type="annotation",
                object_id="a1",
                source_field="gold_value",
                source_text=json.dumps(annotation["gold_value"], sort_keys=True),
            ),
            "答案：主播展示了一件收纳商品。",
        ),
    ]
    translation_path = tmp_path / "translations.jsonl"
    _write(
        translation_path,
        "\n".join(
            json.dumps(
                {
                    **job.to_dict(),
                    "translated_text": translated,
                    "translation_method": "qwen3-vl-plus",
                    "prompt_version": "audit-translation-prompt-v2",
                },
                ensure_ascii=False,
            )
            for job, translated in jobs
        )
        + "\n",
    )
    manifest = {
        "tested_model": "qwen3-vl-plus",
        "judge_model": "deepseek-v4-pro",
        "frame_cache_root": "frames",
        "formal": {
            "artifacts": {
                "evidence": {"source": "evidence"},
                "qa": {"source": "qa"},
            }
        },
    }
    translations = load_translation_index(translation_path)

    data = build_workbench_data(manifest, tmp_path, {}, translations=translations)
    review = data["evidence"]["queue"][0]
    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)

    assert review["reason_zh"] == "仍存在两种可能的时间顺序解读。"
    assert "优惠先于行动号召" in review["candidate_snapshot_zh"]
    assert data["qa"][0]["reason_zh"] == "推断出的需求仍存在歧义。"
    assert "候选问题" in data["qa"][0]["candidate_snapshot_zh"]
    assert data["evidence"]["risks"][0]["target_zh"] == "目标：画面中的商品。"
    assert "收纳商品" in data["evidence"]["risks"][0]["gold_value_zh"]
    assert data["translations"]["missing"] == 0
    assert "仍存在两种可能的时间顺序解读。" in html
    assert "推断出的需求仍存在歧义。" in html
    assert "The timestamps do not establish temporal order." in html


def test_workbench_renders_chinese_translation_and_english_source(tmp_path: Path) -> None:
    source = "The frame shows a 9.9-yuan offer."
    question = "What price is shown in the offer?"
    gold = "The offer price is 9.9 yuan."
    from salesbench.audit_translation import build_translation_job

    jobs = [
        (build_translation_job(object_type="evidence_unit", object_id="e1", source_field="content_en", source_text=source), "画面显示到手价为9.9元。"),
        (build_translation_job(object_type="qa", object_id="q1", source_field="question", source_text=question), "优惠中显示的价格是多少？"),
        (build_translation_job(object_type="qa", object_id="q1", source_field="gold_answer", source_text=gold), "优惠价为9.9元。"),
    ]
    translation_path = tmp_path / "audit_translations.jsonl"
    _write(
        translation_path,
        "\n".join(
            json.dumps(
                {
                    **job.to_dict(),
                    "translated_text": translated,
                    "translation_method": "gpt-4o",
                    "prompt_version": "audit-translation-prompt-v1",
                },
                ensure_ascii=False,
            )
            for job, translated in jobs
        )
        + "\n",
    )
    translations = load_translation_index(translation_path)
    evidence_item = enrich_evidence_refs(
        "v1",
        ["e1"],
        {
            "e1": {
                "evidence_id": "e1",
                "video_id": "v1",
                "modality": "ocr",
                "content_en": source,
                "source_text_native": "到手9.9元",
                "frame_indices": [],
            }
        },
        {},
    )[0]
    evidence_item["content_zh"] = translations.translate("evidence_unit", "e1", "content_en", source)["translated_text"]
    data = {
        "release": {
            "status": "candidate",
            "prompt_version": "evidence-prompt-v9",
            "tested_model": "qwen3-vl-plus",
            "judge_model": "deepseek-v4-pro",
        },
        "counts": {"videos": 1, "evidence_units": 1, "annotations": 1, "review_queue": 0, "qa": 1, "judge_rows": 0},
        "delivery": {},
        "translations": {"missing": 0, "stale": 0},
        "evidence": {"queue": [], "risks": [], "missing_task_videos": []},
        "qa": [{
            "vqa_id": "q1",
            "video_id": "v1",
            "task_type": "BP",
            "task_subtype": "OFFER_PRICE",
            "question": question,
            "question_zh": translations.translate("qa", "q1", "question", question)["translated_text"],
            "gold_answer": gold,
            "gold_answer_zh": translations.translate("qa", "q1", "gold_answer", gold)["translated_text"],
            "capability": "PRICE_AND_DISCOUNT",
            "reasoning_operator": "READ_GROUNDED_VALUE",
            "evidence_items": [evidence_item],
        }],
        "judge": {"summary": {}, "metrics": {}, "rows": []},
    }

    html = render_workbench(data, collect_prompt_snapshot(), fragment=True)

    assert "中文审计翻译" in html
    assert "优惠中显示的价格是多少？" in html
    assert "优惠价为9.9元。" in html
    assert "The frame shows a 9.9-yuan offer." in html
    assert "到手9.9元" in html
    assert "PRICE_AND_DISCOUNT" in html
    assert 'loading="lazy"' in html
    assert "SalesBench · GPT-4o pilot candidate" not in html
    assert 'id="model-summary"' in html


def test_translation_index_rejects_duplicate_or_non_audit_rows(tmp_path: Path) -> None:
    base = {
        "translation_id": "audit_translation::qa::q1::question",
        "object_type": "qa",
        "object_id": "q1",
        "source_field": "question",
        "source_language": "en",
        "target_language": "zh-CN",
        "source_sha256": "a" * 64,
        "translated_text": "问题",
        "translation_method": "gpt-4o",
        "prompt_version": "audit-translation-prompt-v1",
        "audit_only": True,
    }
    duplicate_path = tmp_path / "duplicate.jsonl"
    _write(duplicate_path, json.dumps(base, ensure_ascii=False) + "\n" + json.dumps(base, ensure_ascii=False) + "\n")

    import pytest

    with pytest.raises(ValueError, match="duplicate audit translation"):
        load_translation_index(duplicate_path)
    base["audit_only"] = False
    invalid_path = tmp_path / "invalid.jsonl"
    _write(invalid_path, json.dumps(base, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="audit_only"):
        load_translation_index(invalid_path)


def test_thumbnail_materialization_deduplicates_and_uses_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"jpeg")
    frame = {
        "frame_index": 1,
        "timestamp_s": 1.5,
        "source_path": str(source),
        "relation": "direct",
    }
    data = {
        "evidence": {"queue": [{"evidence_items": [{"frames": [dict(frame)]}]}], "risks": []},
        "qa": [{"evidence_items": [{"frames": [dict(frame)]}]}],
        "judge": {"rows": [{"evidence_items": [{"frames": [dict(frame)]}]}]},
    }

    result = materialize_thumbnails(
        data,
        asset_root=tmp_path / "audit/assets/frames",
        html_parent=tmp_path / "audit",
        converter=lambda src, dst: dst.write_bytes(src.read_bytes()),
    )

    assert result == {"unique_frames": 1, "created": 1, "reused": 0, "missing": 0}
    thumbnail = tmp_path / "audit/assets/frames/unknown/frame_001.jpg"
    assert thumbnail.read_bytes() == b"jpeg"
    qa_frame = data["qa"][0]["evidence_items"][0]["frames"][0]
    judge_frame = data["judge"]["rows"][0]["evidence_items"][0]["frames"][0]
    assert qa_frame["thumbnail_src"] == "assets/frames/unknown/frame_001.jpg"
    assert judge_frame["thumbnail_src"] == qa_frame["thumbnail_src"]
    assert "source_path" not in qa_frame


def test_rendered_evidence_uses_lazy_images_and_readable_content() -> None:
    prompts = collect_prompt_snapshot()
    data = {
        "release": {"status": "pilot", "prompt_version": "v6"},
        "counts": {"videos": 1, "evidence_units": 1, "annotations": 1, "review_queue": 1, "qa": 1},
        "delivery": {},
        "evidence": {
            "queue": [
                {
                    "id": "r1",
                    "video_id": "v1",
                    "task_type": "BP",
                    "reason": "复核",
                    "evidence_items": [
                        {
                            "evidence_id": "e1",
                            "modality": "ocr",
                            "semantic_text": "产品｜认证｜国家专利",
                            "text_span": "国家专利",
                            "localization_note": "",
                            "frames": [
                                {
                                    "frame_index": 1,
                                    "timestamp_s": 1.5,
                                    "relation": "direct",
                                    "thumbnail_src": "assets/frames/v1/frame_001.jpg",
                                }
                            ],
                        }
                    ],
                }
            ],
            "risks": [],
            "missing_task_videos": [],
        },
        "qa": [],
        "judge": {"summary": {}, "metrics": {}, "rows": []},
    }

    html = render_workbench(data, prompts, fragment=True)

    assert 'loading="lazy"' in html
    assert "产品｜认证｜国家专利" in html
    assert "国家专利" in html
    assert "frame_001.jpg" in html
    assert "openLightbox" in html


def test_review_cards_distinguish_candidates_abstentions_and_unresolved_sources() -> None:
    prompts = collect_prompt_snapshot()
    data = {
        "release": {"status": "pilot", "prompt_version": "v6"},
        "counts": {"videos": 1, "evidence_units": 1, "annotations": 0, "review_queue": 4, "qa": 0},
        "delivery": {},
        "evidence": {
            "queue": [
                {
                    "id": "candidate",
                    "video_id": "v1",
                    "stage": "validation",
                    "item_type": "candidate",
                    "task_type": "SS",
                    "task_subtype": "TRUST_MECHANISM",
                    "reason": "Only one evidence unit was cited.",
                    "target": {"mechanism": "demonstration"},
                    "candidate_gold": {"answer": "The product is demonstrated in use."},
                    "evidence_refs": ["e1"],
                    "issues": [{"code": "INSUFFICIENT_EVIDENCE"}],
                    "evidence_items": [],
                },
                {
                    "id": "abstention",
                    "video_id": "v1",
                    "stage": "proposal",
                    "item_type": "abstention",
                    "task_type": "AE",
                    "reason": "Fewer than two evidence units support a bounded answer.",
                    "display_summary": "No candidate generated",
                    "evidence_items": [],
                },
                {
                    "id": "unresolved",
                    "video_id": "v1",
                    "stage": "adjudication",
                    "item_type": "candidate",
                    "task_type": "",
                    "reason": "Legacy source ID is missing.",
                    "resolution_status": "unresolved",
                    "unresolved_proposal_ids": ["missing"],
                    "display_summary": "Candidate source could not be resolved",
                    "evidence_items": [],
                },
                {
                    "id": "diagnostic",
                    "video_id": "v1",
                    "stage": "commercial_relation_building",
                    "item_type": "commercial_relation",
                    "task_type": "",
                    "reason": "commercial_relation_validation_failed",
                    "reason_code": "COMMERCIAL_RELATION_VALIDATION_FAILED",
                    "resolution_status": "diagnostic",
                    "issues": [{"code": "INVALID_RELATION_TARGET_TYPE"}],
                    "evidence_refs": ["e1"],
                    "target": {},
                    "candidate_gold": {},
                    "evidence_items": [],
                },
            ],
            "risks": [],
            "missing_task_videos": [],
        },
        "qa": [],
        "judge": {"summary": {}, "metrics": {}, "rows": []},
    }

    html = render_workbench(data, prompts, fragment=True)

    assert "No candidate generated" in html
    assert "The product is demonstrated in use." in html
    assert "Candidate source could not be resolved" in html
    assert "Candidate content and evidence" in html
    assert "Abstention details" in html
    assert "Unresolved source details" in html
    assert "管线诊断记录，不包含候选 Target 或 Gold" in html
    assert "'PIPELINE':'UNRESOLVED'" in html
    assert "查看 target / gold / proposal / evidence_refs" not in html


def test_qa_and_judge_cards_render_semantic_evidence_not_only_ids() -> None:
    prompts = collect_prompt_snapshot()
    evidence_item = {
        "evidence_id": "e1",
        "modality": "asr",
        "semantic_text": "主播｜行动提示｜立即下单",
        "text_span": "现在下单",
        "localization_note": "ASR 缺少时间戳",
        "frames": [],
    }
    data = {
        "release": {"status": "pilot", "prompt_version": "v6"},
        "counts": {"videos": 1, "evidence_units": 1, "annotations": 1, "review_queue": 0, "qa": 1},
        "delivery": {},
        "evidence": {"queue": [], "risks": [], "missing_task_videos": []},
        "qa": [
            {
                "vqa_id": "q1",
                "video_id": "v1",
                "task_type": "BP",
                "task_subtype": "ASR_FACT",
                "question": "口播给出了什么提示？",
                "gold_answer": "立即下单",
                "source_annotations": [{"annotation_id": "a1", "target": {"subject": "主播"}}],
                "evidence_items": [evidence_item],
            }
        ],
        "judge": {
            "summary": {},
            "metrics": {},
            "rows": [
                {
                    "vqa_id": "q1",
                    "video_id": "v1",
                    "task_type": "BP",
                    "question": "口播给出了什么提示？",
                    "reference_answer": "立即下单",
                    "model_output": "立即下单",
                    "score": 1,
                    "reason": "回答正确",
                    "evidence_items": [evidence_item],
                }
            ],
        },
    }

    html = render_workbench(data, prompts, fragment=True)

    assert html.count("renderEvidenceItems(x.evidence_items)") >= 3
    assert "来源 Annotation" in html
