from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(1, ".")

from tools.audit_workbench.build import (
    collect_prompt_snapshot,
    compact_workbench_data,
    detect_annotation_risks,
    organize_delivery,
    render_workbench,
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
    assert "Gold Challenger" in html
    assert "E-commerce Content Analyst" in html
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
