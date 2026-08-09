from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(1, ".")

from tools.audit_workbench.build import (
    build_workbench_data,
    collect_prompt_snapshot,
    compact_workbench_data,
    detect_annotation_risks,
    organize_delivery,
    render_workbench,
)
from tools.audit_workbench.evidence_assets import (
    enrich_evidence_refs,
    load_frame_manifests,
    materialize_thumbnails,
)


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
    assert "evidence-prompt-v7" in html
    assert "v6 运行快照" in html
    assert "v7 当前代码" in html
    assert "Gold Challenger" in html
    assert "资深多模态评测员" in html
    assert "自动风险概览" in html
    assert "\"likes\"" not in html
    assert "\"followers\"" not in html
    json.loads(html.split('<script type="application/json" id="sbaw-data">', 1)[1].split("</script>", 1)[0])


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
    _write(
        tmp_path / "frames/v1/manifest.json",
        json.dumps(
            {
                "video_id": "v1",
                "frames": [{"frame_index": 1, "timestamp_s": 1.5, "path": "frames/v1/frame_001.jpg"}],
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
    _write(evidence_dir / "evidence_units.jsonl", json.dumps(unit, ensure_ascii=False) + "\n")
    _write(
        evidence_dir / "video_evidence_dataset.jsonl",
        json.dumps({"video_id": "v1", "grounded_annotations": [annotation]}, ensure_ascii=False) + "\n",
    )
    _write(evidence_dir / "gold_proposals.jsonl", json.dumps(proposal, ensure_ascii=False) + "\n")
    _write(
        evidence_dir / "human_review_queue.jsonl",
        json.dumps(
            {"review_item_id": "r1", "video_id": "v1", "source_proposal_ids": ["p1"], "reason": "复核"},
            ensure_ascii=False,
        )
        + "\n",
    )
    _write(
        evidence_dir / "audit_before_review.json",
        json.dumps({"video_count": 1, "videos_missing_required_tasks": []}),
    )
    _write(evidence_dir / "generation_meta.json", json.dumps({"prompt_version": "evidence-prompt-v6"}))
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
    assert data["qa"][0]["evidence_items"][0]["semantic_text"] == "产品｜认证｜国家专利"
    assert data["qa"][0]["evidence_items"][0]["frames"][0]["frame_index"] == 1
    assert data["judge"]["rows"][0]["evidence_items"] == data["qa"][0]["evidence_items"]
    assert data["release"]["runtime_prompt_version"] == "evidence-prompt-v6"
    assert data["release"]["current_prompt_version"] == "evidence-prompt-v7"


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
