from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from salesbench.goldbank.prompts import (
    PROMPT_VERSION,
    build_adjudicator_prompt,
    build_challenger_prompt,
    build_evidence_extractor_prompt,
    build_proposer_prompt,
)
from salesbench.goldbank.validators import PRIVATE_KEYS
from salesbench.vqa_evaluate.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

from .evidence_assets import enrich_evidence_refs, load_frame_manifests, materialize_thumbnails


RISK_LABELS = {
    "AE_SCHEMA_MISMATCH": "AE 字段结构越界",
    "SELLER_CLAIM_AS_FACT": "销售主张被当作客观事实",
    "INVALID_TEXT_SPAN": "OCR/ASR 原文定位无效",
    "MISSING_TEMPORAL_LOCALIZATION": "缺少时间定位",
    "NUMERIC_PROPOSAL_ID": "来源 proposal ID 不可追踪",
    "CM_NOT_CROSS_MODAL": "CM 未形成跨模态证据",
    "INSUFFICIENT_EVIDENCE": "证据引用不足",
    "TEMPLATE_REPETITION": "模板重复度高",
    "LOW_JUDGE_SCORE": "Judge 低分",
}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _strip_private(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _strip_private(value)
            for key, value in payload.items()
            if str(key) not in PRIVATE_KEYS and str(key) != "private_analysis_metadata"
        }
    if isinstance(payload, list):
        return [_strip_private(value) for value in payload]
    return payload


def _copy_artifact(source: Path, destination: Path, excludes: Iterable[str] = ()) -> None:
    patterns = tuple(excludes)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    if not source.is_dir():
        raise FileNotFoundError(f"delivery source not found: {source}")

    def ignored(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)}

    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignored)


def organize_delivery(manifest: dict[str, Any], repo_root: Path) -> dict[str, str]:
    roots: dict[str, str] = {}
    for group in ("formal", "smoke"):
        section = manifest[group]
        destination_root = (repo_root / section["root"]).resolve()
        destination_root.mkdir(parents=True, exist_ok=True)
        for name, artifact in section.get("artifacts", {}).items():
            source = (repo_root / artifact["source"]).resolve()
            _copy_artifact(source, destination_root / name, artifact.get("exclude", []))
        roots[f"{group}_root"] = str(destination_root)
    return roots


def collect_prompt_snapshot() -> list[dict[str, Any]]:
    placeholder_evidence = [
        {
            "evidence_id": "{{existing_evidence_id_1}}",
            "modality": "asr",
            "text_span": "{{exact ASR text}}",
            "subject": "{{subject}}",
            "predicate": "{{predicate}}",
            "value": "{{value}}",
            "confidence": 0.9,
        },
        {
            "evidence_id": "{{existing_evidence_id_2}}",
            "modality": "visual",
            "frame_indices": [0],
            "subject": "{{subject}}",
            "predicate": "{{predicate}}",
            "value": "{{value}}",
            "confidence": 0.9,
        },
    ]
    evidence_system, evidence_user = build_evidence_extractor_prompt(
        "{{video_id}}",
        {"video_text": "{{ASR/subtitles}}", "frames": "{{image parts with frame labels}}"},
    )
    prompts: list[dict[str, Any]] = [
        {
            "id": "extractor",
            "name": "Objective Evidence Extractor",
            "stage": "Evidence",
            "version": PROMPT_VERSION,
            "system": evidence_system,
            "user_template": evidence_user,
            "observed": [
                "部分 OCR text_span 返回坐标字符串而不是原文。",
                "ASR/OCR 的 start_s/end_s 大量为空，证据可复核性不足。",
            ],
            "recommendations": [
                "对 OCR 明确禁止 [start,end] 坐标式 text_span，并要求逐字引用。",
                "将 ASR 切片时间戳作为输入并要求输出 start_s/end_s；OCR 保留 frame_indices。",
                "区分‘视频中声称 X’与‘客观事实 X’，效果类语句默认按 claim 编码。",
            ],
        }
    ]
    proposer_notes = {
        "consumer": (
            ["AE CONTENT_MOTIVATION 实际出现 CM 风格的 claim/relation 字段。"],
            ["为每个 AE subtype 给出独立 JSON Schema，禁止 relation/claim。", "减少 Consumer 与 Strategist 在 VALUE_PROPOSITION 上的职责重叠。"],
        ),
        "operator": (
            ["CM 当前只强制两个 evidence_id，没有强制两个不同模态。"],
            ["CM proposal 增加 modality_pair，并由本地规则验证至少两种模态。", "NOT_SHOWN 需要明确负证据窗口，避免把未采到当作未出现。"],
        ),
        "strategist": (
            ["部分模型返回 1、2 作为 proposal_id，跨样本追踪性较弱。"],
            ["模型不再自报 proposal_id，由本地代码始终生成 canonical ID。", "对子类型分别约束 target 与 proposed_gold 的必填字段。"],
        ),
    }
    for role in ("consumer", "operator", "strategist"):
        system, user = build_proposer_prompt(role, "{{video_id}}", placeholder_evidence)
        prompts.append(
            {
                "id": role,
                "name": f"{role.title()} Proposer",
                "stage": "GroundedAnnotation proposal",
                "version": PROMPT_VERSION,
                "system": system,
                "user_template": user,
                "observed": proposer_notes[role][0],
                "recommendations": proposer_notes[role][1],
            }
        )
    proposal = {
        "proposal_id": "{{proposal_id}}",
        "task_type": "AE",
        "task_subtype": "USAGE_CONTEXT",
        "target": {"scenario": "{{scenario}}"},
        "proposed_gold": {"usage_context": "{{context}}", "answer": "{{answer}}"},
        "evidence_ids": ["{{existing_evidence_id_1}}", "{{existing_evidence_id_2}}"],
        "reasoning_edges": [],
        "proposal_confidence": 0.85,
    }
    challenger_system, challenger_user = build_challenger_prompt(
        "{{video_id}}", [proposal], placeholder_evidence
    )
    prompts.append(
        {
            "id": "challenger",
            "name": "Gold Challenger",
            "stage": "Quality challenge",
            "version": PROMPT_VERSION,
            "system": challenger_system,
            "user_template": challenger_user,
            "observed": ["296 条 Adjudicator/本地队列存在，但汇总 human_review_rate 仍为 0。"],
            "recommendations": [
                "为 BP/CM/SS/AE 分别提供 challenge rubric，不再只用一组通用布尔项。",
                "将所有非 PASS review 物化为可追踪 queue item，并修正审核率口径。",
            ],
        }
    )
    adjudicator_system, adjudicator_user = build_adjudicator_prompt(
        "{{video_id}}",
        [proposal],
        [{"proposal_id": "{{proposal_id}}", "verdict": "PASS"}],
        placeholder_evidence,
    )
    prompts.append(
        {
            "id": "adjudicator",
            "name": "Evidence Adjudicator",
            "stage": "Merge and adjudication",
            "version": PROMPT_VERSION,
            "system": adjudicator_system,
            "user_template": adjudicator_user,
            "observed": ["Adjudicator 被要求重写完整 annotation，扩大了结构漂移面。"],
            "recommendations": [
                "让 Adjudicator 只输出 proposal_id 分组、accept/review 决策和理由，annotation 由本地代码重建。",
                "对合并、遗漏和冲突建立确定性回归用例。",
            ],
        }
    )
    judge_payload = {
        "question": "{{question}}",
        "task_type": "{{BP|CM|SS|AE}}",
        "reference_answer": "{{gold answer}}",
        "model_output": "{{model answer}}",
        "evidence_context": {"evidence_refs": ["{{public evidence context}}"], "answer_type": "open"},
    }
    prompts.append(
        {
            "id": "judge",
            "name": "LLM-as-Judge",
            "stage": "Evaluation",
            "version": "judge-prompt-v2",
            "system": JUDGE_SYSTEM_PROMPT,
            "user_template": build_judge_user_prompt(judge_payload),
            "observed": [
                "被测模型和 Judge 当前同为 GPT-4o，存在 self-judge 偏差风险。",
                "v6 使用同一套通用描述覆盖四种任务，对 BP/CM 的边界不够贴切。",
                "当前 0.7063 MacroRA 尚无人工校准集支持。",
            ],
            "recommendations": [
                "v7 已采用 task-specific rubric：BP 重事实，CM 重模态关系，SS/AE 重证据约束推理。",
                "v7 已增加 correctness、grounding、completeness，并由本地规则确定性映射五档总分。",
                "建立至少 64 题双人标注校准集，报告一致率、weighted kappa 和 MAE。",
                "正式榜单优先使用与被测模型不同的 Judge，或采用双 Judge + 分歧复核。",
            ],
        }
    )
    return prompts


def _risk_record(annotation: dict[str, Any], codes: set[str]) -> dict[str, Any]:
    return {
        "id": str(annotation.get("annotation_id") or ""),
        "video_id": str(annotation.get("video_id") or ""),
        "task_type": str(annotation.get("task_type") or ""),
        "task_subtype": str(annotation.get("task_subtype") or ""),
        "risk_codes": sorted(codes),
        "risk_labels": [RISK_LABELS[code] for code in sorted(codes)],
        "target": annotation.get("target") or {},
        "gold_value": annotation.get("gold_value") or {},
        "evidence_refs": annotation.get("evidence_refs") or [],
        "source_proposal_ids": annotation.get("source_proposal_ids") or [],
        "confidence": annotation.get("confidence"),
    }


def detect_annotation_risks(
    annotations: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    coordinate_span = re.compile(r"^\[\s*\d+\s*,\s*\d+\s*\]$")
    for annotation in annotations:
        codes: set[str] = set()
        task = str(annotation.get("task_type") or "")
        target = annotation.get("target") or {}
        gold = annotation.get("gold_value") or {}
        refs = [str(item) for item in annotation.get("evidence_refs") or []]
        units = [evidence_by_id[item] for item in refs if item in evidence_by_id]
        if task == "AE" and ("claim" in target or "relation" in gold):
            codes.add("AE_SCHEMA_MISMATCH")
        predicate = str(target.get("predicate") or "").lower()
        if task == "BP" and any(term in predicate for term in ("效果", "功效", "effect", "疗效")):
            if any(str(unit.get("modality")) in {"asr", "ocr"} for unit in units):
                codes.add("SELLER_CLAIM_AS_FACT")
        if any(coordinate_span.match(str(unit.get("text_span") or "")) for unit in units):
            codes.add("INVALID_TEXT_SPAN")
        if any(
            str(unit.get("modality")) in {"asr", "ocr"}
            and unit.get("start_s") is None
            and unit.get("end_s") is None
            for unit in units
        ):
            codes.add("MISSING_TEMPORAL_LOCALIZATION")
        if any(str(item).isdigit() for item in annotation.get("source_proposal_ids") or []):
            codes.add("NUMERIC_PROPOSAL_ID")
        if task == "CM" and len({str(unit.get("modality")) for unit in units}) < 2:
            codes.add("CM_NOT_CROSS_MODAL")
        minimum = 1 if task == "BP" else 2
        if len(set(refs)) < minimum:
            codes.add("INSUFFICIENT_EVIDENCE")
        if codes:
            risks.append(_risk_record(annotation, codes))
    return risks


def _source_paths(manifest: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    formal = manifest["formal"]["artifacts"]
    return {name: (repo_root / spec["source"]).resolve() for name, spec in formal.items()}


def build_workbench_data(
    manifest: dict[str, Any], repo_root: Path, delivery: dict[str, str]
) -> dict[str, Any]:
    paths = _source_paths(manifest, repo_root)
    evidence_dir = paths["evidence"]
    qa_dir = paths["qa"]
    evaluation_dir = paths["evaluation"]
    units = _read_jsonl(evidence_dir / "evidence_units.jsonl")
    evidence_by_id = {str(row.get("evidence_id")): row for row in units}
    video_records = _read_jsonl(evidence_dir / "video_evidence_dataset.jsonl")
    annotations = [item for row in video_records for item in row.get("grounded_annotations") or []]
    annotation_by_id = {str(item.get("annotation_id")): item for item in annotations}
    video_ids = {str(row.get("video_id")) for row in video_records if row.get("video_id")}
    frame_cache_root = (repo_root / manifest.get("frame_cache_root", "outputs/cache/frames")).resolve()
    frame_manifests = load_frame_manifests(frame_cache_root, video_ids, repo_root=repo_root)
    proposals = _read_jsonl(evidence_dir / "gold_proposals.jsonl")
    proposal_index = {
        (str(row.get("video_id")), str(row.get("proposal_id"))): row for row in proposals
    }
    queue_rows: list[dict[str, Any]] = []
    for row in _read_jsonl(evidence_dir / "human_review_queue.jsonl"):
        proposal_ids = [str(item) for item in row.get("source_proposal_ids") or []]
        linked = [
            proposal_index.get((str(row.get("video_id")), proposal_id), {}) for proposal_id in proposal_ids
        ]
        linked = [item for item in linked if item]
        linked_evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for proposal in linked
                for evidence_id in proposal.get("evidence_ids") or []
            )
        )
        queue_rows.append(
            {
                "id": str(row.get("review_item_id") or ""),
                "video_id": str(row.get("video_id") or ""),
                "task_type": str((linked[0] if linked else {}).get("task_type") or "UNKNOWN"),
                "task_subtype": str((linked[0] if linked else {}).get("task_subtype") or ""),
                "reason": str(row.get("reason") or ""),
                "source_proposal_ids": proposal_ids,
                "proposal_summary": [
                    {
                        "target": item.get("target") or {},
                        "proposed_gold": item.get("proposed_gold") or {},
                        "evidence_ids": item.get("evidence_ids") or [],
                        "confidence": item.get("proposal_confidence"),
                    }
                    for item in linked
                ],
                "evidence_items": enrich_evidence_refs(
                    str(row.get("video_id") or ""),
                    linked_evidence_ids,
                    evidence_by_id,
                    frame_manifests,
                ),
            }
        )
    audit = _read_json(evidence_dir / "audit_before_review.json", {})
    evidence_meta = _read_json(evidence_dir / "generation_meta.json", {})
    qa_rows = _read_jsonl(qa_dir / "vqa_gold_private.jsonl")
    annotation_risks = detect_annotation_risks(annotations, evidence_by_id)
    for risk in annotation_risks:
        risk["evidence_items"] = enrich_evidence_refs(
            str(risk.get("video_id") or ""),
            [str(item) for item in risk.get("evidence_refs") or []],
            evidence_by_id,
            frame_manifests,
        )
    annotation_risk_index = {row["id"]: row["risk_codes"] for row in annotation_risks}
    question_counts = Counter(str(row.get("question") or "") for row in qa_rows)
    compact_qa: list[dict[str, Any]] = []
    for row in qa_rows:
        risk_codes: set[str] = set()
        for source_id in row.get("source_annotation_ids") or []:
            risk_codes.update(annotation_risk_index.get(str(source_id), []))
        if question_counts[str(row.get("question") or "")] >= 8:
            risk_codes.add("TEMPLATE_REPETITION")
        compact_qa.append(
            {
                "vqa_id": row.get("vqa_id"),
                "video_id": row.get("video_id"),
                "task_type": row.get("task_type"),
                "task_subtype": row.get("task_subtype"),
                "question": row.get("question"),
                "gold_answer": row.get("gold_answer"),
                "evidence_refs": row.get("evidence_refs") or [],
                "source_annotation_ids": row.get("source_annotation_ids") or [],
                "risk_codes": sorted(risk_codes),
                "risk_labels": [RISK_LABELS[code] for code in sorted(risk_codes)],
                "source_annotations": [
                    annotation_by_id[str(source_id)]
                    for source_id in row.get("source_annotation_ids") or []
                    if str(source_id) in annotation_by_id
                ],
                "evidence_items": enrich_evidence_refs(
                    str(row.get("video_id") or ""),
                    [str(item) for item in row.get("evidence_refs") or []],
                    evidence_by_id,
                    frame_manifests,
                ),
            }
        )
    judge_report_files = sorted(evaluation_dir.glob("*_salesbench_qa_eval.json"))
    judge_detail_files = sorted(evaluation_dir.glob("*_judge_details.jsonl"))
    judge_report = _read_json(judge_report_files[0], {}) if judge_report_files else {}
    qa_by_id = {str(row.get("vqa_id")): row for row in compact_qa}
    compact_judge = []
    for row in (_read_jsonl(judge_detail_files[0]) if judge_detail_files else []):
        qa_item = qa_by_id.get(str(row.get("vqa_id")), {})
        compact_judge.append(
            {
            "vqa_id": row.get("vqa_id"),
            "video_id": row.get("video_id"),
            "task_type": row.get("task_type"),
            "question": row.get("question"),
            "reference_answer": row.get("reference_answer"),
            "model_output": row.get("model_output"),
            "score": row.get("score"),
            "reported_score": row.get("reported_score"),
            "correctness": row.get("correctness"),
            "grounding": row.get("grounding"),
            "completeness": row.get("completeness"),
            "reason": row.get("reason"),
            "evidence_alignment": row.get("evidence_alignment"),
            "judge_success": row.get("judge_success"),
            "evidence_items": deepcopy(qa_item.get("evidence_items") or []),
            }
        )
    return _strip_private(
        {
            "release": {
                "status": "pilot_draft_requires_human_review",
                "prompt_version": evidence_meta.get("prompt_version") or PROMPT_VERSION,
                "runtime_prompt_version": evidence_meta.get("prompt_version") or "unknown",
                "current_prompt_version": PROMPT_VERSION,
                "pipeline_version": evidence_meta.get("pipeline_version"),
                "compiler_version": _read_json(qa_dir / "generation_meta.json", {}).get("compiler_version"),
                "judge_model": judge_report.get("judge_model"),
                "tested_model": manifest.get("tested_model", "gpt-4o"),
            },
            "delivery": delivery,
            "counts": {
                "videos": audit.get("video_count", len(video_records)),
                "evidence_units": len(units),
                "annotations": len(annotations),
                "review_queue": len(queue_rows),
                "qa": len(qa_rows),
                "judge_rows": len(compact_judge),
            },
            "task_counts": audit.get("item_count_by_task") or {},
            "quality_counts": audit.get("item_count_by_quality") or {},
            "evidence": {
                "queue": queue_rows,
                "risks": annotation_risks,
                "missing_task_videos": audit.get("videos_missing_required_tasks") or [],
                "audit": audit,
            },
            "qa": compact_qa,
            "judge": {
                "summary": judge_report.get("summary") or {},
                "metrics": judge_report.get("metrics") or {},
                "rows": compact_judge,
            },
        }
    )


def compact_workbench_data(data: dict[str, Any], limit: int = 60) -> dict[str, Any]:
    compact = deepcopy(data)
    compact["evidence"]["queue"] = compact["evidence"]["queue"][:limit]
    priority = {
        "AE_SCHEMA_MISMATCH": 6,
        "SELLER_CLAIM_AS_FACT": 6,
        "INVALID_TEXT_SPAN": 5,
        "CM_NOT_CROSS_MODAL": 5,
        "INSUFFICIENT_EVIDENCE": 5,
        "NUMERIC_PROPOSAL_ID": 4,
        "MISSING_TEMPORAL_LOCALIZATION": 1,
    }
    compact["evidence"]["risks"] = sorted(
        compact["evidence"]["risks"],
        key=lambda row: max((priority.get(code, 3) for code in row.get("risk_codes") or []), default=0),
        reverse=True,
    )[:limit]
    qa = compact["qa"]
    risky = [row for row in qa if row.get("risk_codes")]
    compact["qa"] = (risky + [row for row in qa if not row.get("risk_codes")])[:limit]
    judge = compact["judge"]["rows"]
    judge.sort(key=lambda row: (row.get("score") is None, row.get("score") or 0))
    compact["judge"]["rows"] = judge[:limit]
    compact["preview_notice"] = f"会话预览每层最多展示 {limit} 条；完整本地 HTML 包含全部记录。"
    return compact


def _json_for_script(payload: Any) -> str:
    return json.dumps(_strip_private(payload), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_workbench(
    data: dict[str, Any], prompts: list[dict[str, Any]], *, fragment: bool = False
) -> str:
    payload = _json_for_script({"data": data, "prompts": prompts, "risk_labels": RISK_LABELS})
    content = r'''<section id="salesbench-audit-workbench" aria-labelledby="sbaw-title">
<style>
#salesbench-audit-workbench{--ink:var(--color-text-primary,#182230);--muted:var(--color-text-secondary,#627084);--panel:var(--color-background-secondary,#f5f7fa);--card:var(--color-background-primary,#fff);--line:var(--color-border-secondary,#d9e0e8);--accent:#1769e0;--warn:#a85800;--bad:#b42318;--ok:#067647;color:var(--ink);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1440px;margin:auto}
#salesbench-audit-workbench *{box-sizing:border-box}#salesbench-audit-workbench h1{font-size:clamp(25px,4vw,42px);line-height:1.1;margin:.25rem 0}#salesbench-audit-workbench h2{font-size:21px;margin:0 0 12px}#salesbench-audit-workbench h3{font-size:16px;margin:0 0 8px}#salesbench-audit-workbench button,#salesbench-audit-workbench select,#salesbench-audit-workbench input{font:inherit}
#salesbench-audit-workbench .hero{border:1px solid var(--line);border-radius:20px;padding:24px;background:linear-gradient(135deg,var(--card),var(--panel))}#salesbench-audit-workbench .eyebrow{font-weight:700;color:var(--accent);letter-spacing:.04em}#salesbench-audit-workbench .muted{color:var(--muted)}#salesbench-audit-workbench .alert{margin-top:15px;padding:12px 14px;border-left:4px solid var(--warn);background:color-mix(in srgb,#f79009 10%,var(--card));border-radius:8px}
#salesbench-audit-workbench .stats{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:10px;margin-top:18px}#salesbench-audit-workbench .stat{padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--card)}#salesbench-audit-workbench .stat b{display:block;font-size:23px}#salesbench-audit-workbench .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}#salesbench-audit-workbench .tab{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 13px;border-radius:999px;cursor:pointer}#salesbench-audit-workbench .tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
#salesbench-audit-workbench .panel{display:none}#salesbench-audit-workbench .panel.active{display:block}#salesbench-audit-workbench .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}#salesbench-audit-workbench .card{border:1px solid var(--line);border-radius:14px;background:var(--card);padding:16px;min-width:0}#salesbench-audit-workbench .flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;align-items:stretch}#salesbench-audit-workbench .gate{padding:14px;border-radius:12px;border-top:4px solid var(--accent);background:var(--panel)}#salesbench-audit-workbench .gate.primary{border-top-color:var(--bad)}#salesbench-audit-workbench .tag{display:inline-flex;padding:2px 8px;border-radius:999px;background:var(--panel);border:1px solid var(--line);font-size:12px;margin:2px}#salesbench-audit-workbench .tag.bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 40%,var(--line))}
#salesbench-audit-workbench .toolbar{display:grid;grid-template-columns:minmax(170px,1fr) 150px 170px auto;gap:8px;margin:10px 0}#salesbench-audit-workbench .control{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--card);color:var(--ink);min-width:0}#salesbench-audit-workbench .list{display:grid;gap:10px}#salesbench-audit-workbench .item{border:1px solid var(--line);border-radius:12px;padding:13px;background:var(--card)}#salesbench-audit-workbench .item-head{display:flex;gap:8px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap}#salesbench-audit-workbench .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}#salesbench-audit-workbench pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:470px;overflow:auto;padding:14px;background:#111827;color:#e5edf7;border-radius:11px;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}#salesbench-audit-workbench details{margin-top:8px}#salesbench-audit-workbench summary{cursor:pointer;font-weight:650}
#salesbench-audit-workbench .decision{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}#salesbench-audit-workbench .decision button,#salesbench-audit-workbench .button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 9px;border-radius:8px;cursor:pointer}#salesbench-audit-workbench .decision button.active{background:var(--accent);color:#fff;border-color:var(--accent)}#salesbench-audit-workbench .bars{display:grid;gap:8px}#salesbench-audit-workbench .bar{display:grid;grid-template-columns:44px 1fr 55px;gap:8px;align-items:center}#salesbench-audit-workbench .track{height:10px;background:var(--panel);border-radius:999px;overflow:hidden}#salesbench-audit-workbench .fill{height:100%;background:var(--accent)}#salesbench-audit-workbench .footer{margin-top:18px;padding:14px;border-top:1px solid var(--line);display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}
#salesbench-audit-workbench .evidence-stack{display:grid;gap:9px;margin:12px 0}#salesbench-audit-workbench .evidence-unit{padding:11px;border:1px solid var(--line);border-radius:11px;background:var(--panel)}#salesbench-audit-workbench .evidence-unit .quote{margin:7px 0;padding:8px 10px;border-left:3px solid var(--accent);background:var(--card);border-radius:5px}#salesbench-audit-workbench .localization{color:var(--warn);font-size:12px}#salesbench-audit-workbench .frame-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:9px}#salesbench-audit-workbench .frame{border:1px solid var(--line);padding:0;border-radius:9px;overflow:hidden;background:var(--card);color:var(--ink);text-align:left;cursor:pointer}#salesbench-audit-workbench .frame img{display:block;width:100%;aspect-ratio:9/16;object-fit:cover;background:var(--panel)}#salesbench-audit-workbench .frame span{display:block;padding:6px 8px;font-size:11px}#salesbench-audit-workbench .frame-missing{min-height:90px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:9px;color:var(--muted)}#salesbench-audit-workbench .lightbox{position:fixed;inset:0;z-index:9999;background:rgba(7,14,25,.88);display:none;align-items:center;justify-content:center;padding:24px}#salesbench-audit-workbench .lightbox.open{display:flex}#salesbench-audit-workbench .lightbox button{position:absolute;top:18px;right:18px;border:0;border-radius:999px;background:#fff;color:#111;padding:8px 12px;cursor:pointer}#salesbench-audit-workbench .lightbox img{max-width:min(92vw,1000px);max-height:88vh;object-fit:contain;border-radius:10px}
@media(max-width:900px){#salesbench-audit-workbench .stats{grid-template-columns:repeat(3,1fr)}#salesbench-audit-workbench .flow{grid-template-columns:repeat(2,1fr)}#salesbench-audit-workbench .toolbar{grid-template-columns:1fr 1fr}}@media(max-width:560px){#salesbench-audit-workbench .hero{padding:16px}#salesbench-audit-workbench .stats{grid-template-columns:repeat(2,1fr)}#salesbench-audit-workbench .grid,#salesbench-audit-workbench .flow,#salesbench-audit-workbench .toolbar{grid-template-columns:1fr}#salesbench-audit-workbench .tabs{display:grid;grid-template-columns:1fr 1fr}#salesbench-audit-workbench .tab{border-radius:10px}}
</style>
<header class="hero"><div class="eyebrow">SalesBench · 64-video GPT-4o pilot</div><h1 id="sbaw-title">Prompt 与人工审计工作台</h1><p class="muted">先冻结 Evidence，再重新编译 QA，最后校准 Judge。当前结果是可诊断的 pilot draft，不是已经人工验收的正式 benchmark。</p><div class="alert"><strong>审核结论：</strong>Evidence 和 QA 都要审，但 Evidence/Annotation 是第一质量门；QA 不能修复错误证据。Judge 还需要独立人工校准。</div><p><span class="tag">v6 运行快照：<b id="runtime-prompt-version"></b></span> <span class="tag">v7 当前代码：<b id="current-prompt-version"></b></span></p><div id="sbaw-stats" class="stats"></div></header>
<nav class="tabs" role="tablist" aria-label="工作台视图"><button class="tab" role="tab" data-tab="delivery" aria-selected="true">交付地图</button><button class="tab" role="tab" data-tab="prompts" aria-selected="false">当前 Prompt</button><button class="tab" role="tab" data-tab="evidence" aria-selected="false">Evidence 审计</button><button class="tab" role="tab" data-tab="qa" aria-selected="false">QA 审计</button><button class="tab" role="tab" data-tab="judge" aria-selected="false">Judge 审计</button></nav>
<main><section class="panel active" data-panel="delivery"><div id="delivery-view"></div></section><section class="panel" data-panel="prompts"><div id="prompt-view"></div></section><section class="panel" data-panel="evidence"><div id="evidence-view"></div></section><section class="panel" data-panel="qa"><div id="qa-view"></div></section><section class="panel" data-panel="judge"><div id="judge-view"></div></section></main>
<footer class="footer"><span class="muted" id="preview-note"></span><button class="button" type="button" onclick="exportDecisions()">导出人工审核决定 JSON</button></footer>
<div class="lightbox" id="frame-lightbox" role="dialog" aria-modal="true" aria-label="证据帧大图"><button type="button" onclick="closeLightbox()">关闭</button><img id="frame-lightbox-image" alt="放大的证据帧"></div>
<script type="application/json" id="sbaw-data">__PAYLOAD__</script>
<script>
(()=>{const ROOT=document.getElementById('salesbench-audit-workbench');const PACK=JSON.parse(document.getElementById('sbaw-data').textContent);const D=PACK.data;const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pretty=v=>esc(JSON.stringify(v,null,2));const key=id=>`salesbench-audit:${D.release.prompt_version}:${id}`;
function stat(label,value){return `<div class="stat"><span class="muted">${esc(label)}</span><b>${esc(value)}</b></div>`}document.getElementById('runtime-prompt-version').textContent=D.release.runtime_prompt_version||D.release.prompt_version||'unknown';document.getElementById('current-prompt-version').textContent=D.release.current_prompt_version||'evidence-prompt-v7';document.getElementById('sbaw-stats').innerHTML=[stat('视频',D.counts.videos),stat('EvidenceUnit',D.counts.evidence_units),stat('自动接受 Annotation',D.counts.annotations),stat('人工队列',D.counts.review_queue),stat('QA',D.counts.qa),stat('Judge',D.counts.judge_rows)].join('');document.getElementById('preview-note').textContent=D.preview_notice||'完整本地审计视图；审核决定仅保存在当前浏览器。';
function showTab(id){ROOT.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===id)));ROOT.querySelectorAll('[data-panel]').forEach(p=>p.classList.toggle('active',p.dataset.panel===id));}ROOT.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));
function renderDelivery(){const f=D.delivery.formal_root||'',s=D.delivery.smoke_root||'';document.getElementById('delivery-view').innerHTML=`<div class="grid"><article class="card"><h2>物理目录</h2><p><span class="tag">FORMAL</span> <span class="mono">${esc(f)}</span></p><p><span class="tag">SMOKE</span> <span class="mono">${esc(s)}</span></p><p class="muted">smoke 与正式结果物理分离；v7 新运行必须写入新目录，不能覆盖当前 v6 快照。</p><p><b>当前数据：</b>${esc(D.release.runtime_prompt_version)} · <b>下一次运行：</b>${esc(D.release.current_prompt_version)}</p></article><article class="card"><h2>四层正式交付</h2><ol><li><b>EvidenceDataset</b>：审核并冻结后才是 Gold 来源</li><li><b>Prompt manifest</b>：版本与运行契约</li><li><b>QA</b>：公开问题 + 私有 Gold</li><li><b>Evaluation</b>：predictions、Judge 明细、校准报告</li></ol><p class="muted">互动分析是独立私有实验，不属于这四层。</p></article></div><h2 style="margin-top:18px">三道人工质量门</h2><div class="flow"><article class="gate primary"><h3>Gate E · Evidence</h3><p>全审 296 条队列 + 221 条 INFERRED + 风险 DIRECT；优先处理 21 个缺 BP/CM 视频。</p></article><article class="gate"><h3>冻结 EvidenceDataset</h3><p>接受、修订、拒绝都写入审核记录；冻结后重新编译 QA。</p></article><article class="gate"><h3>Gate Q · QA</h3><p>64 视频 pilot 的 384 题建议全量通读，问题必须回溯到 annotation。</p></article><article class="gate"><h3>Gate J · Judge</h3><p>至少 64 题双人评分 + 裁决，校准五档评分后才能作为榜单指标。</p></article></div>`}
function renderPrompts(){const opts=PACK.prompts.map((p,i)=>`<option value="${i}">${esc(p.name)} · ${esc(p.stage)}</option>`).join('');document.getElementById('prompt-view').innerHTML=`<div class="alert"><strong>版本边界：</strong>这里展示的是 v7 当前代码 Prompt，用于下一次 smoke；Evidence/QA/Judge 审计数据仍来自 v6 运行快照。</div><div class="card"><div class="toolbar"><select class="control" id="prompt-select" aria-label="选择 Prompt">${opts}</select><span class="tag" id="prompt-version"></span></div><div id="prompt-detail"></div></div>`;const select=document.getElementById('prompt-select');const draw=()=>{const p=PACK.prompts[Number(select.value)];document.getElementById('prompt-version').textContent=p.version;document.getElementById('prompt-detail').innerHTML=`<div class="grid"><div><h2>${esc(p.name)}</h2><h3>v7 当前 System Prompt（代码原文）</h3><pre>${esc(p.system)}</pre><details><summary>下一次运行的 User Payload 模板</summary><pre>${typeof p.user_template==='string'?esc(p.user_template):pretty(p.user_template)}</pre></details></div><div><article class="card"><h3>v6 运行结果中观察到</h3><ul>${p.observed.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></article><article class="card" style="margin-top:10px"><h3>v7 已落实 / 后续验证</h3><ol>${p.recommendations.map(x=>`<li>${esc(x)}</li>`).join('')}</ol></article></div></div>`};select.addEventListener('change',draw);draw()}
function decisionButtons(id){const current=localStorage.getItem(key(id))||'';return `<div class="decision" data-decision="${esc(id)}">${[['accept','接受'],['revise','修订'],['reject','拒绝'],['defer','待定']].map(([v,l])=>`<button type="button" data-value="${v}" class="${current===v?'active':''}">${l}</button>`).join('')}</div>`}function bindDecisions(){ROOT.querySelectorAll('[data-decision] button').forEach(b=>b.addEventListener('click',()=>{const box=b.closest('[data-decision]');localStorage.setItem(key(box.dataset.decision),b.dataset.value);box.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));}))}
function renderFrames(frames){if(!frames||!frames.length)return '<div class="frame-missing">没有可显示的关联帧</div>';return `<div class="frame-grid">${frames.map(frame=>{const caption=`帧 ${frame.frame_index} · ${frame.timestamp_s==null?'时间未知':Number(frame.timestamp_s).toFixed(2)+'s'} · ${frame.relation||''}`;return frame.thumbnail_src?`<button class="frame" type="button" onclick="openLightbox('${esc(frame.thumbnail_src)}','${esc(caption)}')"><img loading="lazy" decoding="async" src="${esc(frame.thumbnail_src)}" alt="${esc(caption)}"><span>${esc(caption)}</span></button>`:`<div class="frame-missing">${esc(caption)}<br>缩略图缺失</div>`}).join('')}</div>`}
function renderEvidenceItems(items){return `<div class="evidence-stack">${(items||[]).map(item=>`<section class="evidence-unit"><div><span class="tag">${esc(item.modality||'missing')}</span><span class="mono">${esc(item.evidence_id||'无直接 Evidence ID')}</span></div><h3>${esc(item.semantic_text||'未提供结构化内容')}</h3>${item.text_span?`<div class="quote"><b>原文：</b>${esc(item.text_span)}</div>`:''}${item.localization_note?`<div class="localization">${esc(item.localization_note)}</div>`:''}${renderFrames(item.frames)}</section>`).join('')}</div>`}
window.openLightbox=function openLightbox(src,caption){const box=document.getElementById('frame-lightbox');const image=document.getElementById('frame-lightbox-image');image.src=src;image.alt=caption||'放大的证据帧';box.classList.add('open')};window.closeLightbox=function closeLightbox(){const box=document.getElementById('frame-lightbox');box.classList.remove('open');document.getElementById('frame-lightbox-image').removeAttribute('src')};document.getElementById('frame-lightbox').addEventListener('click',event=>{if(event.target.id==='frame-lightbox')closeLightbox()});
function toolbar(prefix){return `<div class="toolbar"><input class="control" id="${prefix}-search" placeholder="搜索 ID、问题、理由…"><select class="control" id="${prefix}-task"><option value="">全部任务</option><option>BP</option><option>CM</option><option>SS</option><option>AE</option><option>UNKNOWN</option></select><select class="control" id="${prefix}-risk"><option value="">全部风险</option>${Object.entries(PACK.risk_labels).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('')}</select><span class="muted" id="${prefix}-count"></span></div>`}
function tags(codes){return (codes||[]).map(code=>`<span class="tag bad">${esc(PACK.risk_labels[code]||code)}</span>`).join('')}
function renderEvidence(){const queue=D.evidence.queue.map(x=>({...x,kind:'QUEUE',risk_codes:['HUMAN_QUEUE']}));const risks=D.evidence.risks.map(x=>({...x,kind:'AUTO_RISK'}));const rows=[...queue,...risks];const riskCounts=risks.reduce((acc,row)=>{(row.risk_codes||[]).forEach(code=>acc[code]=(acc[code]||0)+1);return acc},{});const riskSummary=Object.entries(riskCounts).sort((a,b)=>b[1]-a[1]).map(([code,count])=>`<span class="tag bad">${esc(PACK.risk_labels[code]||code)} · ${count}</span>`).join('');document.getElementById('evidence-view').innerHTML=`<div class="grid"><article class="card"><h2>当前优先级</h2><p><b>P0：</b>${D.counts.review_queue} 条人工队列；${D.evidence.missing_task_videos.length} 个缺 BP/CM 视频。</p><p><b>P1：</b>全部 INFERRED 与自动规则风险；对其余 DIRECT 分层抽样。</p></article><article class="card"><h2>自动风险概览</h2><p>${riskSummary||'未命中自动风险规则'}</p><p class="muted">时间定位缺失是系统性问题；结构越界、主张事实化与非跨模态 CM 优先逐条处理。</p></article><article class="card"><h2>不要只审 QA</h2><p>如果 Evidence 把“口播声称有效”写成“产品客观有效”，QA 即使语句流畅也仍是错误 Gold。</p><p class="muted">审核决定不会自动写回数据集。</p></article></div>${toolbar('ev')}<div id="ev-list" class="list"></div>`;const draw=()=>{const q=document.getElementById('ev-search').value.toLowerCase(),task=document.getElementById('ev-task').value,risk=document.getElementById('ev-risk').value;const found=rows.filter(x=>(!task||x.task_type===task)&&(!risk||(x.risk_codes||[]).includes(risk))&&(!q||JSON.stringify(x).toLowerCase().includes(q)));document.getElementById('ev-count').textContent=`显示 ${found.length} / ${rows.length}`;document.getElementById('ev-list').innerHTML=found.slice(0,200).map(x=>`<article class="item"><div class="item-head"><div><span class="tag">${esc(x.kind)}</span><span class="tag">${esc(x.task_type||'UNKNOWN')}</span>${tags(x.risk_codes)}</div><span class="mono">${esc(x.video_id)}</span></div><h3>${esc(x.id)}</h3><p>${esc(x.reason||'自动接受记录命中本地风险规则')}</p>${renderEvidenceItems(x.evidence_items)}<details><summary>查看 target / gold / proposal / evidence_refs</summary><pre>${pretty({target:x.target,gold_value:x.gold_value,proposal_summary:x.proposal_summary,evidence_refs:x.evidence_refs,source_proposal_ids:x.source_proposal_ids})}</pre></details>${decisionButtons(x.id)}</article>`).join('')||'<p class="muted">没有匹配记录。</p>';bindDecisions()};['ev-search','ev-task','ev-risk'].forEach(id=>document.getElementById(id).addEventListener('input',draw));draw()}
function renderQA(){document.getElementById('qa-view').innerHTML=`<div class="card"><h2>QA 是第二道审核门</h2><p>当前 ${D.counts.qa} 题由未完成人工冻结的 annotation 编译，因此只能称为 provisional QA。对 64 视频 pilot 建议全量通读；发现问题时回溯 annotation 或模板。</p></div>${toolbar('qa')}<div id="qa-list" class="list"></div>`;const draw=()=>{const q=document.getElementById('qa-search').value.toLowerCase(),task=document.getElementById('qa-task').value,risk=document.getElementById('qa-risk').value;const found=D.qa.filter(x=>(!task||x.task_type===task)&&(!risk||(x.risk_codes||[]).includes(risk))&&(!q||JSON.stringify(x).toLowerCase().includes(q)));document.getElementById('qa-count').textContent=`显示 ${found.length} / ${D.qa.length}`;document.getElementById('qa-list').innerHTML=found.slice(0,200).map(x=>`<article class="item"><div class="item-head"><div><span class="tag">${esc(x.task_type)}</span><span class="tag">${esc(x.task_subtype)}</span>${tags(x.risk_codes)}</div><span class="mono">${esc(x.vqa_id)}</span></div><h3>${esc(x.question)}</h3><p><b>Gold：</b>${esc(typeof x.gold_answer==='string'?x.gold_answer:JSON.stringify(x.gold_answer))}</p><h3>可读 Evidence</h3>${renderEvidenceItems(x.evidence_items)}<details><summary>来源 Annotation 与完整引用</summary><pre>${pretty({source_annotations:x.source_annotations,source_annotation_ids:x.source_annotation_ids,evidence_refs:x.evidence_refs})}</pre></details>${decisionButtons(x.vqa_id)}</article>`).join('')||'<p class="muted">没有匹配记录。</p>';bindDecisions()};['qa-search','qa-task','qa-risk'].forEach(id=>document.getElementById(id).addEventListener('input',draw));draw()}
function renderJudge(){const per=D.judge.metrics.per_task||{};const bars=Object.entries(per).map(([task,m])=>`<div class="bar"><b>${esc(task)}</b><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,Number(m.relaxed_accuracy||0)*100))}%"></div></div><span>${(Number(m.relaxed_accuracy||0)*100).toFixed(1)}%</span></div>`).join('');document.getElementById('judge-view').innerHTML=`<div class="grid"><article class="card"><h2>当前结果：v6 诊断分数</h2><div class="bars">${bars}</div><p>Macro relaxed：<b>${(Number(D.judge.metrics.macro_average?.relaxed_accuracy||0)*100).toFixed(2)}%</b> · Judge failures：<b>${esc(D.judge.summary.judge_failed_count||0)}</b></p></article><article class="card"><h2>v7 Judge 与后续校准</h2><ol><li>已加入四任务独立 rubric；</li><li>已加入 correctness / grounding / completeness 分项与本地确定性映射；</li><li>仍需至少 64 题双人校准，报告 kappa 与 MAE；</li><li>正式榜单避免 GPT-4o 评 GPT-4o，或使用双 Judge + 分歧复核。</li></ol></article></div>${toolbar('jd')}<div id="jd-list" class="list"></div>`;document.querySelector('#jd-risk').innerHTML='<option value="">全部分数</option><option value="LOW">≤ 0.5</option><option value="HIGH">≥ 0.75</option>';const draw=()=>{const q=document.getElementById('jd-search').value.toLowerCase(),task=document.getElementById('jd-task').value,risk=document.getElementById('jd-risk').value;const found=D.judge.rows.filter(x=>(!task||x.task_type===task)&&(!risk||(risk==='LOW'?Number(x.score)<=.5:Number(x.score)>=.75))&&(!q||JSON.stringify(x).toLowerCase().includes(q)));document.getElementById('jd-count').textContent=`显示 ${found.length} / ${D.judge.rows.length}`;document.getElementById('jd-list').innerHTML=found.slice(0,200).map(x=>`<article class="item"><div class="item-head"><div><span class="tag">${esc(x.task_type)}</span><span class="tag ${Number(x.score)<=.5?'bad':''}">score ${esc(x.score)}</span></div><span class="mono">${esc(x.vqa_id)}</span></div><h3>${esc(x.question)}</h3><p><b>Reference：</b>${esc(x.reference_answer)}</p><p><b>Model：</b>${esc(x.model_output)}</p>${x.correctness==null?'<p class="muted">v6 结果没有分项分数；v7 重新评估后将显示 correctness / grounding / completeness。</p>':`<p><span class="tag">correctness ${esc(x.correctness)}</span> <span class="tag">grounding ${esc(x.grounding)}</span> <span class="tag">completeness ${esc(x.completeness)}</span></p>`}<h3>Judge 实际依据的 Evidence</h3>${renderEvidenceItems(x.evidence_items)}<details><summary>Judge 理由与证据对齐</summary><p>${esc(x.reason)}</p><p>${esc(x.evidence_alignment)}</p></details>${decisionButtons(`judge:${x.vqa_id}`)}</article>`).join('')||'<p class="muted">没有匹配记录。</p>';bindDecisions()};['jd-search','jd-task','jd-risk'].forEach(id=>document.getElementById(id).addEventListener('input',draw));draw()}
window.exportDecisions=function exportDecisions(){const decisions={};for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);if(k&&k.startsWith(`salesbench-audit:${D.release.prompt_version}:`))decisions[k.split(':').slice(2).join(':')]=localStorage.getItem(k)}const blob=new Blob([JSON.stringify({prompt_version:D.release.prompt_version,exported_at:new Date().toISOString(),decisions},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`salesbench-audit-decisions-${D.release.prompt_version}.json`;a.click();URL.revokeObjectURL(url)};renderDelivery();renderPrompts();renderEvidence();renderQA();renderJudge();})();
</script></section>'''.replace("__PAYLOAD__", payload)
    if fragment:
        return content
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>SalesBench Prompt 与人工审计工作台</title></head>"
        f"<body style=\"margin:0;padding:20px;background:#eef2f6\">{content}</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local SalesBench prompt and audit workbench")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path)
    parser.add_argument("--preview-limit", type=int, default=60)
    parser.add_argument("--skip-organize", action="store_true")
    args = parser.parse_args()
    repo_root = Path.cwd().resolve()
    manifest = _read_json(args.manifest.resolve())
    delivery = (
        {
            "formal_root": str((repo_root / manifest["formal"]["root"]).resolve()),
            "smoke_root": str((repo_root / manifest["smoke"]["root"]).resolve()),
        }
        if args.skip_organize
        else organize_delivery(manifest, repo_root)
    )
    data = build_workbench_data(manifest, repo_root, delivery)
    prompts = collect_prompt_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full_data = deepcopy(data)
    full_thumbnail_stats = materialize_thumbnails(
        full_data,
        asset_root=args.output.parent / "assets" / "frames",
        html_parent=args.output.parent,
    )
    full_data["thumbnail_stats"] = full_thumbnail_stats
    args.output.write_text(render_workbench(full_data, prompts), encoding="utf-8")
    preview_thumbnail_stats: dict[str, int] | None = None
    if args.fragment:
        args.fragment.parent.mkdir(parents=True, exist_ok=True)
        preview_data = compact_workbench_data(deepcopy(data), args.preview_limit)
        preview_thumbnail_stats = materialize_thumbnails(
            preview_data,
            asset_root=args.fragment.parent / "assets" / "frames",
            html_parent=args.fragment.parent,
        )
        preview_data["thumbnail_stats"] = preview_thumbnail_stats
        args.fragment.write_text(
            render_workbench(preview_data, prompts, fragment=True),
            encoding="utf-8",
        )
        if args.fragment.stat().st_size >= 1_000_000:
            raise ValueError(f"fragment exceeds 1 MB: {args.fragment.stat().st_size}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fragment": str(args.fragment or ""),
                "counts": data["counts"],
                "full_thumbnails": full_thumbnail_stats,
                "preview_thumbnails": preview_thumbnail_stats,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
