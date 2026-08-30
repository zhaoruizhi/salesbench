from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from salesbench.goldbank.prompts import (
    BP_COMPILER_CONTRACT,
    PROMPT_VERSION,
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
from salesbench.goldbank.validators import PRIVATE_KEYS
from salesbench.vqa_evaluate.prompts import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from salesbench.vqa.prompts import (
    QA_QUALITY_PROMPT_VERSION,
    QA_QUALITY_SYSTEM_PROMPT,
    QUESTION_REALIZER_PROMPT_VERSION,
    QUESTION_REALIZER_SYSTEM_PROMPT,
    QUESTION_REPAIR_SYSTEM_PROMPT,
)
from salesbench.vqa_baseline.prompts import BASELINE_PROMPT_VERSION, CLOSED_SOURCE_SYSTEM_PROMPT

from .evidence_assets import enrich_evidence_refs, load_frame_manifests, materialize_thumbnails
from .review_queue import normalize_review_queue_row


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
    "NO_COMMERCIAL_RECORD": "未生成可编译商业记录",
}

PIPELINE_RISK_CODES = {
    "INVALID_TEXT_SPAN",
    "MISSING_TEMPORAL_LOCALIZATION",
    "NUMERIC_PROPOSAL_ID",
}


def classify_review_bucket(row: dict[str, Any]) -> str:
    """Route an audit record to one actionable queue without duplicating root causes."""
    explicit = str(row.get("audit_bucket") or "")
    if explicit:
        return explicit
    item_type = str(row.get("item_type") or "").lower()
    reason_code = str(row.get("reason_code") or "").upper()
    stage = str(row.get("stage") or "").lower()
    resolution = str(row.get("resolution_status") or "").lower()
    issue_codes = {
        str(issue.get("code") or "").upper()
        for issue in row.get("issues") or []
        if isinstance(issue, dict)
    }
    risk_codes = {str(code).upper() for code in row.get("risk_codes") or []}
    if item_type == "abstention" or reason_code in {"ABSTENTION", "NO_COMMERCIAL_RECORD"}:
        return "abstention_resample"
    if reason_code in {
        "BELOW_MIN_CONFIDENCE",
        "VALIDATION_FAILED",
        "SEMANTIC_DUPLICATE",
        "CONFLICTING_VALUE",
        "CHALLENGER_REJECT",
        "CHALLENGER_REJECTED_OR_HUMAN_REVIEW",
        "SEMANTIC_VERIFIER_REJECT",
    }:
        return "auto_rejected"
    if (
        resolution in {"diagnostic", "unresolved"}
        or stage in {"commerce_cue_extraction", "commercial_relation_building"}
        or reason_code.endswith("_PARSE_ERROR")
        or reason_code in {
            "PROPOSAL_GRAPH_VALIDATION_FAILED",
            "COMMERCE_CUE_BELOW_MIN_CONFIDENCE",
            "COMMERCE_CUE_VALIDATION_FAILED",
            "COMMERCIAL_RELATION_VALIDATION_FAILED",
        }
        or "MISSING_COMMERCIAL_RELATION" in issue_codes
        or (risk_codes and risk_codes <= PIPELINE_RISK_CODES)
    ):
        return "pipeline_diagnostics"
    return "content_review"


def select_accepted_annotation_sample(
    annotations: list[dict[str, Any]],
    excluded_annotation_ids: set[str],
    *,
    fraction: float = 0.1,
) -> list[dict[str, Any]]:
    """Select a deterministic per-task sample from annotations that passed local risk rules."""
    if not 0 < fraction <= 1:
        raise ValueError("accepted annotation sample fraction must be in (0, 1]")
    groups: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        annotation_id = str(annotation.get("annotation_id") or "")
        if not annotation_id or annotation_id in excluded_annotation_ids:
            continue
        groups.setdefault(str(annotation.get("task_type") or "UNKNOWN"), []).append(annotation)
    selected: list[dict[str, Any]] = []
    for task_type in sorted(groups):
        ranked = sorted(
            groups[task_type],
            key=lambda row: (
                hashlib.sha256(str(row.get("annotation_id") or "").encode("utf-8")).hexdigest(),
                str(row.get("annotation_id") or ""),
            ),
        )
        selected.extend(ranked[: max(1, math.ceil(len(ranked) * fraction))])
    return selected


def select_accepted_qa_sample(
    qa_rows: list[dict[str, Any]],
    excluded_vqa_ids: set[str],
    *,
    fraction: float = 0.1,
) -> list[dict[str, Any]]:
    """Select a deterministic per-task QA monitoring sample without turning all PASS rows into review work."""
    if not 0 < fraction <= 1:
        raise ValueError("accepted QA sample fraction must be in (0, 1]")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in qa_rows:
        vqa_id = str(row.get("vqa_id") or "")
        if not vqa_id or vqa_id in excluded_vqa_ids:
            continue
        groups.setdefault(str(row.get("task_type") or "UNKNOWN"), []).append(row)
    selected: list[dict[str, Any]] = []
    for task_type in sorted(groups):
        ranked = sorted(
            groups[task_type],
            key=lambda row: (
                hashlib.sha256(str(row.get("vqa_id") or "").encode("utf-8")).hexdigest(),
                str(row.get("vqa_id") or ""),
            ),
        )
        selected.extend(ranked[: max(1, math.ceil(len(ranked) * fraction))])
    return selected


class TranslationIndex:
    """Hash-bound, audit-only translation lookup used by the HTML renderer."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._source_fields: set[tuple[str, str, str]] = set()
        self.missing = 0
        self.stale = 0
        for row in rows:
            if row.get("audit_only") is not True:
                raise ValueError("every audit translation must set audit_only=true")
            if any(str(key) in PRIVATE_KEYS or str(key) == "private_analysis_metadata" for key in row):
                raise ValueError("audit translation sidecar contains a private field")
            object_type = str(row.get("object_type") or "")
            object_id = str(row.get("object_id") or "")
            source_field = str(row.get("source_field") or "")
            source_hash = str(row.get("source_sha256") or "")
            if not all((object_type, object_id, source_field, source_hash, row.get("translated_text"))):
                raise ValueError("audit translation row is incomplete")
            key = (object_type, object_id, source_field, source_hash)
            if key in self._rows:
                raise ValueError(f"duplicate audit translation: {key}")
            self._rows[key] = dict(row)
            self._source_fields.add((object_type, object_id, source_field))

    def translate(
        self,
        object_type: str,
        object_id: str,
        source_field: str,
        source_text: object,
    ) -> dict[str, Any]:
        text = str(source_text or "").strip()
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (str(object_type), str(object_id), str(source_field), source_hash)
        row = self._rows.get(key)
        if row is not None:
            return {
                "translated_text": str(row["translated_text"]),
                "status": "current",
                "source_sha256": source_hash,
                "translation_method": str(row.get("translation_method") or ""),
            }
        if (str(object_type), str(object_id), str(source_field)) in self._source_fields:
            self.stale += 1
            status = "stale"
        else:
            self.missing += 1
            status = "missing"
        return {
            "translated_text": "",
            "status": status,
            "source_sha256": source_hash,
            "translation_method": "",
        }

    def summary(self) -> dict[str, int]:
        return {
            "loaded": len(self._rows),
            "missing": self.missing,
            "stale": self.stale,
        }


def load_translation_index(path: Path) -> TranslationIndex:
    if not path.is_file():
        raise FileNotFoundError(f"audit translation sidecar not found: {path}")
    return TranslationIndex(_read_jsonl(path))


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


def _translation(
    translations: TranslationIndex | None,
    object_type: str,
    object_id: object,
    source_field: str,
    source_text: object,
) -> dict[str, Any]:
    if translations is None:
        return {
            "translated_text": "",
            "status": "unavailable",
            "source_sha256": "",
            "translation_method": "",
        }
    return translations.translate(object_type, str(object_id or ""), source_field, source_text)


def _localize_evidence_items(
    items: list[dict[str, Any]], translations: TranslationIndex | None
) -> list[dict[str, Any]]:
    for item in items:
        content_en = str(item.get("content_en") or item.get("semantic_text") or "")
        if not item.get("evidence_id") or not content_en:
            item["translation_status"] = "not_applicable"
            continue
        translated = _translation(
            translations,
            "evidence_unit",
            item.get("evidence_id"),
            "content_en",
            content_en,
        )
        item["content_zh"] = translated["translated_text"]
        item["translation_status"] = translated["status"]
        item["translation_source_sha256"] = translated["source_sha256"]
        item["translation_method"] = translated["translation_method"]
    return items


def _localized_graph_item(
    row: dict[str, Any], object_type: str, id_field: str, text_field: str, translations: TranslationIndex | None
) -> dict[str, Any]:
    item = dict(row)
    translated = _translation(
        translations,
        object_type,
        row.get(id_field),
        text_field,
        row.get(text_field),
    )
    item["content_zh"] = translated["translated_text"]
    item["translation_status"] = translated["status"]
    item["translation_source_sha256"] = translated["source_sha256"]
    item["translation_method"] = translated["translation_method"]
    return item


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
            "content_en": "{{canonical English evidence sentence}}",
            "source_text_native": "{{exact ASR text}}",
            "subject": "{{subject}}",
            "predicate": "{{predicate}}",
            "value": "{{value}}",
            "confidence": 0.9,
        },
        {
            "evidence_id": "{{existing_evidence_id_2}}",
            "modality": "visual",
            "frame_indices": [0],
            "content_en": "{{canonical English visual evidence sentence}}",
            "source_text_native": "",
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
            "id": "language_evidence_extractor",
            "name": "ASR Evidence Extractor",
            "stage": "Evidence / ASR",
            "version": PROMPT_VERSION,
            "system": build_language_evidence_prompt(
                "{{video_id}}", {"asr_subtitles": {"video_text": "{{ASR/subtitles}}"}}
            )[0],
            "user_template": build_language_evidence_prompt(
                "{{video_id}}", {"asr_subtitles": {"video_text": "{{ASR/subtitles}}"}}
            )[1],
            "observed": ["独立处理 ASR，避免长帧输入挤占语言证据输出。"],
            "recommendations": ["核对 source_text_native 是否逐字来自 ASR，英文语义是否保持 claim 边界。"],
        },
        {
            "id": "visual_evidence_extractor",
            "name": "Visual and OCR Evidence Extractor",
            "stage": "Evidence / Frames",
            "version": PROMPT_VERSION,
            "system": build_visual_evidence_prompt(
                "{{video_id}}", {"sampled_frames": "{{image parts with frame labels}}"}
            )[0],
            "user_template": build_visual_evidence_prompt(
                "{{video_id}}", {"sampled_frames": "{{image parts with frame labels}}"}
            )[1],
            "observed": ["独立处理帧与 OCR，并要求至少一条可见事实。"],
            "recommendations": ["核对 OCR 是否排除了重复字幕、账号水印和互动计数。"],
        },
        {
            "id": "visual_evidence_repairer",
            "name": "Visual Evidence Repairer",
            "stage": "Evidence / Local-validation repair",
            "version": PROMPT_VERSION,
            "system": build_visual_evidence_repair_prompt(
                "{{video_id}}",
                {"sampled_frames": "{{image parts with frame labels}}"},
                [{"content_en": "{{rejected candidate}}"}],
                [{"code": "{{local validation issue}}"}],
            )[0],
            "user_template": build_visual_evidence_repair_prompt(
                "{{video_id}}",
                {"sampled_frames": "{{image parts with frame labels}}"},
                [{"content_en": "{{rejected candidate}}"}],
                [{"code": "{{local validation issue}}"}],
            )[1],
            "observed": ["仅在有采样帧但首轮没有任何 visual EvidenceUnit 通过本地验证时触发。"],
            "recommendations": ["修复输出仍须重新看帧，并再次通过英文、模态与帧定位规则。"],
        },
        {
            "id": "visual_commerce_cue_extractor",
            "name": "Visual Commerce Cue Repair",
            "stage": "CommerceCue / Visual coverage",
            "version": PROMPT_VERSION,
            "system": build_visual_commerce_cue_prompt("{{video_id}}", placeholder_evidence)[0],
            "user_template": build_visual_commerce_cue_prompt("{{video_id}}", placeholder_evidence)[1],
            "observed": ["仅在综合 Cue 阶段未覆盖视觉证据时触发。"],
            "recommendations": ["演示、结果和前后对比 Cue 必须引用可定位视觉帧。"],
        },
        {
            "id": "evidence_extractor",
            "name": "Fallback Combined Evidence Extractor",
            "stage": "Evidence / Fallback",
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
    cue_system, cue_user = build_commerce_cue_prompt("{{video_id}}", placeholder_evidence)
    prompts.append(
        {
            "id": "commerce_cue_extractor",
            "name": "Commerce Cue Extractor",
            "stage": "CommerceCue",
            "version": PROMPT_VERSION,
            "system": cue_system,
            "user_template": cue_user,
            "observed": ["将原子事实归纳为可审计的带货表达线索。"],
            "recommendations": ["检查 claim 与可观察事实是否保持分离。"],
        }
    )
    cue_placeholder = [
        {
            "cue_id": "{{existing_cue_id_1}}",
            "cue_type": "FUNCTION_CLAIM",
            "content_en": "{{specific commercial cue}}",
            "evidence_ids": ["{{existing_evidence_id_1}}"],
        },
        {
            "cue_id": "{{existing_cue_id_2}}",
            "cue_type": "PROCESS_DEMONSTRATION",
            "content_en": "{{specific demonstration cue}}",
            "evidence_ids": ["{{existing_evidence_id_2}}"],
        },
    ]
    relation_system, relation_user = build_commercial_relation_prompt(
        "{{video_id}}", placeholder_evidence, cue_placeholder
    )
    prompts.append(
        {
            "id": "commercial_relation_builder",
            "name": "Commercial Relation Builder",
            "stage": "CommercialRelation",
            "version": PROMPT_VERSION,
            "system": relation_system,
            "user_template": relation_user,
            "observed": ["连接 claim、demonstration、offer、objection 与 CTA。"],
            "recommendations": ["检查端点类型、证据覆盖和消费者结果越界。"],
        }
    )
    prompts.append(
        {
            "id": "bp_compiler",
            "name": "BP Deterministic Compiler",
            "stage": "BP candidate generation",
            "version": PROMPT_VERSION,
            "system": BP_COMPILER_CONTRACT,
            "user_template": "Validated EvidenceUnits",
            "observed": ["BP 不调用 LLM，由本地规则从直接证据生成。"],
            "recommendations": ["人工审核应重点检查上游 EvidenceUnit，而不是改写 BP 模板。"],
        }
    )
    prompts.append(
        {
            "id": "question_realizer",
            "name": "Question Realizer",
            "stage": "English QA realization",
            "version": QUESTION_REALIZER_PROMPT_VERSION,
            "system": QUESTION_REALIZER_SYSTEM_PROMPT,
            "user_template": "QuestionSpec + cited English Evidence/Commerce graph context",
            "observed": ["负责减少问题模板化，不改变 QuestionSpec 的语义边界。"],
            "recommendations": ["检查自然度、答案泄漏、跨任务重复和英文一致性。"],
        }
    )
    prompts.append(
        {
            "id": "question_repairer",
            "name": "Question Surface Repairer",
            "stage": "QA / Local-validation repair",
            "version": QUESTION_REALIZER_PROMPT_VERSION,
            "system": QUESTION_REPAIR_SYSTEM_PROMPT,
            "user_template": {
                "spec_id": "{{spec_id}}",
                "question_spec": "{{same semantic specification}}",
                "rejected_question": "{{first-pass question}}",
                "local_error_codes": ["{{deterministic validation code}}"],
                "evidence_context": "{{English evidence context}}",
            },
            "observed": ["仅在首轮问题违反语言、公式化、问号或答案泄漏规则时触发一次。"],
            "recommendations": ["修复只能改写问句表面，不得改变 Gold、能力或证据引用。"],
        }
    )
    prompts.append(
        {
            "id": "qa_semantic_quality_gate",
            "name": "QA Semantic Quality Gate",
            "stage": "QA / Semantic production gate",
            "version": QA_QUALITY_PROMPT_VERSION,
            "system": QA_QUALITY_SYSTEM_PROMPT,
            "user_template": "QuestionSpec + realized question + Gold + cited English Evidence/Commerce graph context",
            "observed": ["明确缺证据、Gold 过度推断、答案不唯一、任务错位和领域性不足会在生产阶段自动拒绝。"],
            "recommendations": ["HUMAN_REVIEW 只用于证据支持两种实质性解释且图上下文无法消歧的少量案例。"],
        }
    )
    prompts.append(
        {
            "id": "model_runner",
            "name": "OpenAI-compatible VQA Model Runner",
            "stage": "Evaluation / Model answer",
            "version": BASELINE_PROMPT_VERSION,
            "system": CLOSED_SOURCE_SYSTEM_PROMPT,
            "user_template": "Sampled video frames + source-language ASR/subtitles + public English question",
            "observed": ["公开问题与最终答案统一为英文，视频中的原语言内容由被测模型翻译。"],
            "recommendations": ["模型答案不得读取 Gold、Evidence graph、任务类型或互动元数据。"],
        }
    )
    proposer_notes = {
        "cm_proposer": (
            ["CM 需要严格的跨模态与观察窗口约束。"],
            ["验证至少两种模态；NOT_SHOWN 仅允许完整视频观察窗口。"],
        ),
        "ss_proposer": (
            ["SS 容易把一般产品描述过度解释为说服策略。"],
            ["要求两条证据共同支持明确的机制，不预测效果。"],
        ),
        "ae_proposer": (
            ["AE 容易越界为真实用户画像或转化结论。"],
            ["限定为内容所对应的需求、场景或决策障碍。"],
        ),
    }
    for generator in ("cm_proposer", "ss_proposer", "ae_proposer"):
        system, user = build_proposer_prompt(generator, "{{video_id}}", placeholder_evidence)
        prompts.append(
            {
                "id": generator,
                "name": generator.replace("_", " ").title(),
                "stage": "GroundedAnnotation proposal",
                "version": PROMPT_VERSION,
                "system": system,
                "user_template": user,
                "observed": proposer_notes[generator][0],
                "recommendations": proposer_notes[generator][1],
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
            "version": JUDGE_PROMPT_VERSION,
            "system": JUDGE_SYSTEM_PROMPT,
            "user_template": build_judge_user_prompt(judge_payload),
            "observed": [
                "被测模型和 Judge 当前同为 GPT-4o，存在 self-judge 偏差风险。",
                "v6 使用同一套通用描述覆盖四种任务，对 BP/CM 的边界不够贴切。",
                "当前 0.7063 MacroRA 尚无人工校准集支持。",
            ],
            "recommendations": [
                "v8 已采用 task-specific rubric：BP 重事实，CM 重模态关系，SS/AE 重证据约束推理。",
                "v8 已增加 correctness、grounding、completeness，并由本地规则确定性映射五档总分。",
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


def _qa_quality_source_row(record: dict[str, Any], bucket: str) -> dict[str, Any]:
    snapshot = record.get("candidate_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    spec = snapshot.get("question_spec")
    spec = spec if isinstance(spec, dict) else {}
    spec_id = str(record.get("spec_id") or spec.get("spec_id") or "")
    annotation_id = str(record.get("annotation_id") or spec.get("annotation_id") or "")
    reason_code = str(record.get("reason_code") or "QA_QUALITY_EVENT")
    return {
        "vqa_id": f"qa-quality:{spec_id or hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest()[:16]}",
        "video_id": record.get("video_id") or spec.get("video_id"),
        "task_type": record.get("task_type") or spec.get("task_type") or "UNKNOWN",
        "task_subtype": spec.get("capability") or "",
        "question": snapshot.get("question") or snapshot.get("rejected_question") or "",
        "gold_answer": spec.get("gold_answer") or "",
        "capability": spec.get("capability") or "",
        "reasoning_operator": spec.get("reasoning_operator") or "",
        "spec_id": spec_id,
        "commerce_cue_ids": spec.get("commerce_cue_ids") or [],
        "commercial_relation_ids": spec.get("commercial_relation_ids") or [],
        "evidence_refs": spec.get("evidence_refs") or [],
        "source_annotation_ids": [annotation_id] if annotation_id else [],
        "audit_bucket": bucket,
        "reason": record.get("reason") or reason_code,
        "risk_codes": [reason_code],
        "semantic_verification": record.get("semantic_verification"),
        "candidate_snapshot": snapshot,
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


def localize_prompt_snapshot(
    prompts: list[dict[str, Any]], translations: TranslationIndex | None
) -> list[dict[str, Any]]:
    localized = deepcopy(prompts)
    for prompt in localized:
        translated = _translation(
            translations,
            "prompt",
            prompt.get("id"),
            "system",
            prompt.get("system"),
        )
        prompt["system_zh"] = translated["translated_text"]
        prompt["translation_status"] = translated["status"]
        prompt["translation_source_sha256"] = translated["source_sha256"]
        prompt["translation_method"] = translated["translation_method"]
    return localized


def _source_paths(manifest: dict[str, Any], repo_root: Path, group: str) -> dict[str, Path]:
    artifacts = manifest[group]["artifacts"]
    return {name: (repo_root / spec["source"]).resolve() for name, spec in artifacts.items()}


def build_workbench_data(
    manifest: dict[str, Any],
    repo_root: Path,
    delivery: dict[str, str],
    *,
    translations: TranslationIndex | None = None,
    group: str = "formal",
) -> dict[str, Any]:
    paths = _source_paths(manifest, repo_root, group)
    evidence_dir = paths["evidence"]
    qa_dir = paths["qa"]
    evaluation_dir = paths.get("evaluation", repo_root / "__missing_evaluation__")
    units = _read_jsonl(evidence_dir / "evidence_units.jsonl")
    evidence_by_id = {str(row.get("evidence_id")): row for row in units}
    cues = _read_jsonl(evidence_dir / "commerce_cues.jsonl")
    relations = _read_jsonl(evidence_dir / "commercial_relations.jsonl")
    cue_by_id = {
        str(row.get("cue_id")): _localized_graph_item(
            row, "commerce_cue", "cue_id", "content_en", translations
        )
        for row in cues
    }
    relation_by_id = {
        str(row.get("relation_id")): _localized_graph_item(
            row, "commercial_relation", "relation_id", "rationale_en", translations
        )
        for row in relations
    }
    evidence_meta = _read_json(evidence_dir / "generation_meta.json", {})
    video_records = _read_jsonl(evidence_dir / "video_evidence_dataset.jsonl")
    annotations = [item for row in video_records for item in row.get("grounded_annotations") or []]
    annotation_by_id = {str(item.get("annotation_id")): item for item in annotations}
    commercial_video_ids = {
        str(row.get("video_id")) for row in video_records if row.get("video_id")
    }
    sampled_video_ids = [
        str(row.get("video_id"))
        for row in _read_jsonl(evidence_dir / "video_samples.jsonl")
        if row.get("video_id")
    ]
    processed_video_ids = list(
        dict.fromkeys(
            [str(video_id) for video_id in evidence_meta.get("video_ids") or [] if video_id]
            + sampled_video_ids
            + sorted(commercial_video_ids)
        )
    )
    video_ids = set(processed_video_ids)
    frame_cache_root = (repo_root / manifest.get("frame_cache_root", "outputs/cache/frames")).resolve()
    frame_manifests = load_frame_manifests(frame_cache_root, video_ids, repo_root=repo_root)
    proposals = _read_jsonl(evidence_dir / "gold_proposals.jsonl")
    proposal_index = {
        (str(row.get("video_id")), str(row.get("proposal_id"))): row for row in proposals
    }
    queue_rows: list[dict[str, Any]] = []
    human_rows = _read_jsonl(evidence_dir / "human_review_queue.jsonl")
    rejected_rows = _read_jsonl(evidence_dir / "rejected_candidates.jsonl")
    diagnostic_rows = _read_jsonl(evidence_dir / "pipeline_diagnostics.jsonl")
    has_separated_quality_outputs = any(
        (evidence_dir / filename).exists()
        for filename in (
            "quality_decisions.jsonl",
            "rejected_candidates.jsonl",
            "pipeline_diagnostics.jsonl",
        )
    )
    raw_queue_rows = [
        *[
            (row, "human_review" if has_separated_quality_outputs else "legacy")
            for row in human_rows
        ],
        *[(row, "auto_rejected") for row in rejected_rows],
        *[
            (
                row,
                "abstention_resample"
                if str(row.get("item_type") or "").lower() == "abstention"
                else "pipeline_diagnostics",
            )
            for row in diagnostic_rows
        ],
    ]
    review_id_counts = Counter(
        str(row.get("review_item_id") or row.get("id") or "")
        for row, _bucket in raw_queue_rows
    )
    for row, source_bucket in raw_queue_rows:
        normalized = normalize_review_queue_row(row, proposal_index)
        normalized["audit_bucket"] = (
            classify_review_bucket(normalized)
            if source_bucket == "legacy"
            else source_bucket
        )
        review_item_id = normalized["review_item_id"]
        if review_id_counts[review_item_id] > 1:
            fingerprint = hashlib.sha256(
                json.dumps(_strip_private(row), ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:10]
            normalized["id"] = f"{review_item_id}:{fingerprint}"
        else:
            normalized["id"] = review_item_id
        normalized["evidence_items"] = enrich_evidence_refs(
            normalized["video_id"],
            normalized["evidence_refs"],
            evidence_by_id,
            frame_manifests,
        )
        _localize_evidence_items(normalized["evidence_items"], translations)
        normalized["commerce_cues"] = [
            cue_by_id[cue_id]
            for cue_id in normalized.get("commerce_cue_ids") or []
            if cue_id in cue_by_id
        ]
        normalized["commercial_relations"] = [
            relation_by_id[relation_id]
            for relation_id in normalized.get("commercial_relation_ids") or []
            if relation_id in relation_by_id
        ]
        queue_rows.append(normalized)
    audit = _read_json(evidence_dir / "audit_before_review.json", {})
    qa_rows = _read_jsonl(qa_dir / "vqa_gold_private.jsonl")
    qa_human_rows = _read_jsonl(qa_dir / "qa_human_review_queue.jsonl")
    qa_rejected_rows = _read_jsonl(qa_dir / "qa_rejected_candidates.jsonl")
    qa_diagnostic_rows = _read_jsonl(qa_dir / "qa_pipeline_diagnostics.jsonl")
    qa_accepted_sample_rows = _read_jsonl(qa_dir / "qa_accepted_sample.jsonl")
    qa_sample_spec_ids = {
        str(row.get("spec_id") or "")
        for row in qa_accepted_sample_rows
        if row.get("spec_id")
    }
    compiled_spec_ids = {str(row.get("spec_id") or "") for row in qa_rows if row.get("spec_id")}
    qa_quality_source_rows = [
        *[_qa_quality_source_row(row, "human_review") for row in qa_human_rows],
        *[_qa_quality_source_row(row, "auto_rejected") for row in qa_rejected_rows],
        *[_qa_quality_source_row(row, "pipeline_diagnostics") for row in qa_diagnostic_rows],
        *[
            _qa_quality_source_row(row, "accepted_sample")
            for row in qa_accepted_sample_rows
            if str(row.get("spec_id") or "") not in compiled_spec_ids
        ],
    ]
    annotation_risks = detect_annotation_risks(annotations, evidence_by_id)
    for risk in annotation_risks:
        risk["audit_bucket"] = "auto_rejected"
        risk["evidence_items"] = enrich_evidence_refs(
            str(risk.get("video_id") or ""),
            [str(item) for item in risk.get("evidence_refs") or []],
            evidence_by_id,
            frame_manifests,
        )
        _localize_evidence_items(risk["evidence_items"], translations)
        annotation = annotation_by_id.get(str(risk.get("id") or ""), {})
        risk["commerce_cues"] = [
            cue_by_id[cue_id]
            for cue_id in annotation.get("commerce_cue_ids") or []
            if cue_id in cue_by_id
        ]
        risk["commercial_relations"] = [
            relation_by_id[relation_id]
            for relation_id in annotation.get("commercial_relation_ids") or []
            if relation_id in relation_by_id
        ]
    risk_annotation_ids = {str(row.get("id") or "") for row in annotation_risks}
    accepted_sample: list[dict[str, Any]] = []
    for annotation in select_accepted_annotation_sample(annotations, risk_annotation_ids):
        sample = _risk_record(annotation, set())
        sample.update(
            {
                "audit_bucket": "accepted_sample",
                "reason": "该 Annotation 已通过当前本地规则，进入按任务分层的约 10% 质量抽样。",
                "evidence_items": enrich_evidence_refs(
                    str(annotation.get("video_id") or ""),
                    [str(item) for item in annotation.get("evidence_refs") or []],
                    evidence_by_id,
                    frame_manifests,
                ),
                "commerce_cues": [
                    cue_by_id[cue_id]
                    for cue_id in annotation.get("commerce_cue_ids") or []
                    if cue_id in cue_by_id
                ],
                "commercial_relations": [
                    relation_by_id[relation_id]
                    for relation_id in annotation.get("commercial_relation_ids") or []
                    if relation_id in relation_by_id
                ],
            }
        )
        _localize_evidence_items(sample["evidence_items"], translations)
        accepted_sample.append(sample)
    evidence_ids_by_video: dict[str, list[str]] = {}
    for unit in units:
        video_id = str(unit.get("video_id") or "")
        evidence_id = str(unit.get("evidence_id") or "")
        if video_id and evidence_id:
            evidence_ids_by_video.setdefault(video_id, []).append(evidence_id)
    cue_ids_by_video: dict[str, list[str]] = {}
    for cue in cues:
        video_id = str(cue.get("video_id") or "")
        cue_id = str(cue.get("cue_id") or "")
        if video_id and cue_id:
            cue_ids_by_video.setdefault(video_id, []).append(cue_id)
    relation_ids_by_video: dict[str, list[str]] = {}
    for relation in relations:
        video_id = str(relation.get("video_id") or "")
        relation_id = str(relation.get("relation_id") or "")
        if video_id and relation_id:
            relation_ids_by_video.setdefault(video_id, []).append(relation_id)
    abstentions: list[dict[str, Any]] = []
    for video_id in processed_video_ids:
        if video_id in commercial_video_ids:
            continue
        evidence_refs = evidence_ids_by_video.get(video_id, [])
        evidence_items = enrich_evidence_refs(
            video_id,
            evidence_refs,
            evidence_by_id,
            frame_manifests,
        )
        _localize_evidence_items(evidence_items, translations)
        abstentions.append(
            {
                "id": f"abstention:{video_id}",
                "video_id": video_id,
                "stage": "commercial_compilation",
                "item_type": "abstention",
                "task_type": "UNKNOWN",
                "task_subtype": "NO_COMMERCIAL_RECORD",
                "reason_code": "NO_COMMERCIAL_RECORD",
                "risk_codes": ["NO_COMMERCIAL_RECORD"],
                "audit_bucket": "abstention_resample",
                "display_summary": "该视频已完成 Evidence 提取，但未生成可编译的商业记录",
                "reason": (
                    "请核对抽帧是否覆盖实际带货段、商业线索是否确实缺失，并决定保留 abstention、"
                    "增加尾段抽帧后重跑，或从正式 cohort 重采样。"
                ),
                "pipeline_status": (evidence_meta.get("statuses") or {}).get(video_id),
                "evidence_refs": evidence_refs,
                "evidence_items": evidence_items,
                "commerce_cues": [
                    cue_by_id[cue_id]
                    for cue_id in cue_ids_by_video.get(video_id, [])
                    if cue_id in cue_by_id
                ],
                "commercial_relations": [
                    relation_by_id[relation_id]
                    for relation_id in relation_ids_by_video.get(video_id, [])
                    if relation_id in relation_by_id
                ],
            }
        )
    annotation_risk_index = {row["id"]: row["risk_codes"] for row in annotation_risks}
    question_counts = Counter(str(row.get("question") or "") for row in qa_rows)
    compact_qa: list[dict[str, Any]] = []
    for row in [*qa_rows, *qa_quality_source_rows]:
        risk_codes: set[str] = {
            str(code) for code in row.get("risk_codes") or [] if str(code)
        }
        for source_id in row.get("source_annotation_ids") or []:
            risk_codes.update(annotation_risk_index.get(str(source_id), []))
        if question_counts[str(row.get("question") or "")] >= 8:
            risk_codes.add("TEMPLATE_REPETITION")
        explicit_bucket = str(row.get("audit_bucket") or "")
        audit_bucket = explicit_bucket or (
            "auto_rejected"
            if risk_codes
            else (
                "accepted_sample"
                if str(row.get("spec_id") or "") in qa_sample_spec_ids
                else "accepted_all"
            )
        )
        question_translation = _translation(
            translations, "qa", row.get("vqa_id"), "question", row.get("question")
        )
        gold_translation = _translation(
            translations, "qa", row.get("vqa_id"), "gold_answer", row.get("gold_answer")
        )
        evidence_items = enrich_evidence_refs(
            str(row.get("video_id") or ""),
            [str(item) for item in row.get("evidence_refs") or []],
            evidence_by_id,
            frame_manifests,
        )
        _localize_evidence_items(evidence_items, translations)
        compact_qa.append(
            {
                "vqa_id": row.get("vqa_id"),
                "video_id": row.get("video_id"),
                "task_type": row.get("task_type"),
                "task_subtype": row.get("task_subtype"),
                "question": row.get("question"),
                "question_zh": question_translation["translated_text"],
                "question_translation_status": question_translation["status"],
                "gold_answer": row.get("gold_answer"),
                "gold_answer_zh": gold_translation["translated_text"],
                "gold_translation_status": gold_translation["status"],
                "capability": row.get("capability"),
                "reasoning_operator": row.get("reasoning_operator"),
                "spec_id": row.get("spec_id"),
                "commerce_cue_ids": row.get("commerce_cue_ids") or [],
                "commercial_relation_ids": row.get("commercial_relation_ids") or [],
                "evidence_refs": row.get("evidence_refs") or [],
                "source_annotation_ids": row.get("source_annotation_ids") or [],
                "audit_bucket": audit_bucket,
                "reason": row.get("reason") or "",
                "semantic_verification": row.get("semantic_verification"),
                "risk_codes": sorted(risk_codes),
                "risk_labels": [RISK_LABELS.get(code, code) for code in sorted(risk_codes)],
                "source_annotations": [
                    annotation_by_id[str(source_id)]
                    for source_id in row.get("source_annotation_ids") or []
                    if str(source_id) in annotation_by_id
                ],
                "commerce_cues": [
                    cue_by_id[cue_id]
                    for cue_id in row.get("commerce_cue_ids") or []
                    if cue_id in cue_by_id
                ],
                "commercial_relations": [
                    relation_by_id[relation_id]
                    for relation_id in row.get("commercial_relation_ids") or []
                    if relation_id in relation_by_id
                ],
                "evidence_items": evidence_items,
            }
        )
    if not qa_accepted_sample_rows:
        excluded_ids = {
            str(row.get("vqa_id") or "")
            for row in compact_qa
            if row.get("audit_bucket") != "accepted_all"
        }
        legacy_sample_ids = {
            str(row.get("vqa_id") or "")
            for row in select_accepted_qa_sample(compact_qa, excluded_ids, fraction=0.1)
        }
        for row in compact_qa:
            if row.get("audit_bucket") == "accepted_all" and str(row.get("vqa_id") or "") in legacy_sample_ids:
                row["audit_bucket"] = "accepted_sample"
                row["reason"] = "该 QA 通过现有自动规则，进入按任务分层的约 10% 监控抽样；这不是全量人工审核。"
    judge_report_files = sorted(evaluation_dir.glob("*_salesbench_qa_eval.json"))
    judge_detail_files = sorted(evaluation_dir.glob("*_judge_details.jsonl"))
    judge_report = _read_json(judge_report_files[0], {}) if judge_report_files else {}
    qa_by_id = {str(row.get("vqa_id")): row for row in compact_qa}
    compact_judge = []
    for row in (_read_jsonl(judge_detail_files[0]) if judge_detail_files else []):
        qa_item = qa_by_id.get(str(row.get("vqa_id")), {})
        reason_translation = _translation(
            translations, "judge_result", row.get("vqa_id"), "reason", row.get("reason")
        )
        alignment_translation = _translation(
            translations,
            "judge_result",
            row.get("vqa_id"),
            "evidence_alignment",
            row.get("evidence_alignment"),
        )
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
            "reason_zh": reason_translation["translated_text"],
            "reason_translation_status": reason_translation["status"],
            "evidence_alignment": row.get("evidence_alignment"),
            "evidence_alignment_zh": alignment_translation["translated_text"],
            "evidence_alignment_translation_status": alignment_translation["status"],
            "judge_success": row.get("judge_success"),
            "evidence_items": deepcopy(qa_item.get("evidence_items") or []),
            }
        )
    human_review_count = sum(
        1
        for row in queue_rows
        if row.get("audit_bucket") in {"human_review", "content_review"}
    )
    pipeline_diagnostic_count = sum(
        1 for row in queue_rows if row.get("audit_bucket") == "pipeline_diagnostics"
    )
    artifact_abstention_count = sum(
        1 for row in queue_rows if row.get("audit_bucket") == "abstention_resample"
    )
    auto_rejected_count = sum(
        1 for row in queue_rows if row.get("audit_bucket") == "auto_rejected"
    ) + len(annotation_risks)
    return _strip_private(
        {
            "release": {
                "status": "pilot_candidate_requires_human_review",
                "group": group,
                "prompt_version": evidence_meta.get("prompt_version") or PROMPT_VERSION,
                "runtime_prompt_version": evidence_meta.get("prompt_version") or "unknown",
                "current_prompt_version": PROMPT_VERSION,
                "pipeline_version": evidence_meta.get("pipeline_version"),
                "compiler_version": _read_json(qa_dir / "generation_meta.json", {}).get("compiler_version"),
                "judge_model": judge_report.get("judge_model"),
                "current_judge_prompt_version": JUDGE_PROMPT_VERSION,
                "tested_model": manifest.get("tested_model", "gpt-4o"),
            },
            "delivery": delivery,
            "counts": {
                "videos": len(processed_video_ids),
                "commercial_records": len(video_records),
                "abstentions": len(abstentions) + artifact_abstention_count,
                "accepted_sample": len(accepted_sample),
                "evidence_units": len(units),
                "annotations": len(annotations),
                "review_queue": len(queue_rows),
                "human_review_queue": human_review_count,
                "auto_rejected": auto_rejected_count,
                "pipeline_diagnostics": pipeline_diagnostic_count,
                "qa": len(qa_rows),
                "qa_audit_rows": len(compact_qa),
                "qa_human_review": sum(
                    1 for row in compact_qa if row.get("audit_bucket") == "human_review"
                ),
                "qa_accepted_sample": sum(
                    1 for row in compact_qa if row.get("audit_bucket") == "accepted_sample"
                ),
                "qa_auto_rejected": sum(
                    1 for row in compact_qa if row.get("audit_bucket") == "auto_rejected"
                ),
                "qa_pipeline_diagnostics": sum(
                    1
                    for row in compact_qa
                    if row.get("audit_bucket") == "pipeline_diagnostics"
                ),
                "judge_rows": len(compact_judge),
            },
            "translations": translations.summary() if translations else {"loaded": 0, "missing": 0, "stale": 0},
            "task_counts": audit.get("item_count_by_task") or {},
            "quality_counts": audit.get("item_count_by_quality") or {},
            "evidence": {
                "queue": queue_rows,
                "risks": annotation_risks,
                "abstentions": abstentions,
                "accepted_sample": accepted_sample,
                "commerce_cues": list(cue_by_id.values()),
                "commercial_relations": list(relation_by_id.values()),
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


def _stratified_prefix(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("task_type") or "UNKNOWN"), []).append(row)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit:
        added = False
        for task_type in sorted(groups):
            if index < len(groups[task_type]):
                selected.append(groups[task_type][index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    return selected


def compact_workbench_data(data: dict[str, Any], limit: int = 60) -> dict[str, Any]:
    compact = deepcopy(data)
    compact["evidence"]["queue"] = compact["evidence"]["queue"][:limit]
    compact["evidence"]["abstentions"] = compact["evidence"].get("abstentions", [])[:limit]
    compact["evidence"]["accepted_sample"] = _stratified_prefix(
        compact["evidence"].get("accepted_sample", []), limit
    )
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
#salesbench-audit-workbench .stats{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:10px;margin-top:18px}#salesbench-audit-workbench .stat{padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--card)}#salesbench-audit-workbench .stat b{display:block;font-size:23px}#salesbench-audit-workbench .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}#salesbench-audit-workbench .tab{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 13px;border-radius:999px;cursor:pointer}#salesbench-audit-workbench .tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
#salesbench-audit-workbench .panel{display:none}#salesbench-audit-workbench .panel.active{display:block}#salesbench-audit-workbench .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}#salesbench-audit-workbench .card{border:1px solid var(--line);border-radius:14px;background:var(--card);padding:16px;min-width:0}#salesbench-audit-workbench .flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;align-items:stretch}#salesbench-audit-workbench .gate{padding:14px;border-radius:12px;border-top:4px solid var(--accent);background:var(--panel)}#salesbench-audit-workbench .gate.primary{border-top-color:var(--bad)}#salesbench-audit-workbench .tag{display:inline-flex;padding:2px 8px;border-radius:999px;background:var(--panel);border:1px solid var(--line);font-size:12px;margin:2px}#salesbench-audit-workbench .tag.bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 40%,var(--line))}
#salesbench-audit-workbench .toolbar{display:grid;grid-template-columns:minmax(170px,1fr) 150px 170px auto;gap:8px;margin:10px 0}#salesbench-audit-workbench .toolbar.with-bucket{grid-template-columns:minmax(170px,1fr) 150px 180px 170px auto}#salesbench-audit-workbench .control{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--card);color:var(--ink);min-width:0}#salesbench-audit-workbench .case-pager{position:sticky;top:8px;z-index:20;display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;margin:10px 0;padding:9px;border:1px solid var(--line);border-radius:12px;background:color-mix(in srgb,var(--card) 94%,transparent);box-shadow:0 4px 14px rgba(16,24,40,.08);backdrop-filter:blur(8px)}#salesbench-audit-workbench .case-pager button{border:1px solid var(--line);border-radius:8px;padding:7px 11px;background:var(--card);color:var(--ink);cursor:pointer}#salesbench-audit-workbench .case-pager button:disabled{cursor:not-allowed;opacity:.45}#salesbench-audit-workbench .case-pager input{width:76px;border:1px solid var(--line);border-radius:8px;padding:6px 8px;background:var(--card);color:var(--ink)}#salesbench-audit-workbench .list{display:grid;gap:10px}#salesbench-audit-workbench .item{border:1px solid var(--line);border-radius:12px;padding:13px;background:var(--card)}#salesbench-audit-workbench .item-head{display:flex;gap:8px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap}#salesbench-audit-workbench .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}#salesbench-audit-workbench pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:470px;overflow:auto;padding:14px;background:#111827;color:#e5edf7;border-radius:11px;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}#salesbench-audit-workbench details{margin-top:8px}#salesbench-audit-workbench summary{cursor:pointer;font-weight:650}
#salesbench-audit-workbench .decision{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}#salesbench-audit-workbench .decision button,#salesbench-audit-workbench .button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 9px;border-radius:8px;cursor:pointer}#salesbench-audit-workbench .decision button.active{background:var(--accent);color:#fff;border-color:var(--accent)}#salesbench-audit-workbench .bars{display:grid;gap:8px}#salesbench-audit-workbench .bar{display:grid;grid-template-columns:44px 1fr 55px;gap:8px;align-items:center}#salesbench-audit-workbench .track{height:10px;background:var(--panel);border-radius:999px;overflow:hidden}#salesbench-audit-workbench .fill{height:100%;background:var(--accent)}#salesbench-audit-workbench .footer{margin-top:18px;padding:14px;border-top:1px solid var(--line);display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}
#salesbench-audit-workbench .evidence-stack{display:grid;gap:9px;margin:12px 0}#salesbench-audit-workbench .evidence-unit{padding:11px;border:1px solid var(--line);border-radius:11px;background:var(--panel)}#salesbench-audit-workbench .evidence-unit .quote{margin:7px 0;padding:8px 10px;border-left:3px solid var(--accent);background:var(--card);border-radius:5px}#salesbench-audit-workbench .localization{color:var(--warn);font-size:12px}#salesbench-audit-workbench .frame-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:9px}#salesbench-audit-workbench .frame{border:1px solid var(--line);padding:0;border-radius:9px;overflow:hidden;background:var(--card);color:var(--ink);text-align:left;cursor:pointer}#salesbench-audit-workbench .frame img{display:block;width:100%;aspect-ratio:9/16;object-fit:cover;background:var(--panel)}#salesbench-audit-workbench .frame span{display:block;padding:6px 8px;font-size:11px}#salesbench-audit-workbench .frame-missing{min-height:90px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:9px;color:var(--muted)}#salesbench-audit-workbench .lightbox{position:fixed;inset:0;z-index:9999;background:rgba(7,14,25,.88);display:none;align-items:center;justify-content:center;padding:24px}#salesbench-audit-workbench .lightbox.open{display:flex}#salesbench-audit-workbench .lightbox button{position:absolute;top:18px;right:18px;border:0;border-radius:999px;background:#fff;color:#111;padding:8px 12px;cursor:pointer}#salesbench-audit-workbench .lightbox img{max-width:min(92vw,1000px);max-height:88vh;object-fit:contain;border-radius:10px}
#salesbench-audit-workbench .review-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin:10px 0}#salesbench-audit-workbench .review-field{padding:10px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}#salesbench-audit-workbench .review-field pre{max-height:220px;margin:7px 0 0}#salesbench-audit-workbench .empty-state{padding:12px;border:1px dashed var(--warn);border-radius:10px;background:color-mix(in srgb,#f79009 8%,var(--card))}
#salesbench-audit-workbench .translation-warning{padding:6px 9px;border-left:3px solid var(--warn);background:color-mix(in srgb,#f79009 8%,var(--card));font-size:12px;margin:7px 0}#salesbench-audit-workbench .native-source{padding:8px 10px;border-left:3px solid var(--ok);background:var(--card);border-radius:5px;margin:7px 0}#salesbench-audit-workbench .graph-stack{display:grid;gap:8px;margin:10px 0}#salesbench-audit-workbench .graph-item{padding:10px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
@media(max-width:900px){#salesbench-audit-workbench .stats{grid-template-columns:repeat(3,1fr)}#salesbench-audit-workbench .flow{grid-template-columns:repeat(2,1fr)}#salesbench-audit-workbench .toolbar{grid-template-columns:1fr 1fr}}@media(max-width:560px){#salesbench-audit-workbench .hero{padding:16px}#salesbench-audit-workbench .stats{grid-template-columns:repeat(2,1fr)}#salesbench-audit-workbench .grid,#salesbench-audit-workbench .flow,#salesbench-audit-workbench .toolbar{grid-template-columns:1fr}#salesbench-audit-workbench .tabs{display:grid;grid-template-columns:1fr 1fr}#salesbench-audit-workbench .tab{border-radius:10px}}
</style>
<header class="hero"><div class="eyebrow">SalesBench · GPT-4o pilot candidate</div><h1 id="sbaw-title">双语 Prompt、Evidence 与 QA 人工审计工作台</h1><p class="muted">Canonical Evidence、QA、Prompt 与 Judge 全部为英文；中文审计翻译是独立 sidecar，只帮助人工阅读，不进入 Gold、模型输入或评分。</p><div class="alert"><strong>审核结论：</strong>EvidenceUnit、CommerceCue、CommercialRelation 和候选 Annotation 是第一质量门；QA 是第二质量门；Judge 需要人工校准。当前产物在人审前一律是 candidate。</div><p><span class="tag">运行版本：<b id="runtime-prompt-version"></b></span> <span class="tag">当前代码：<b id="current-prompt-version"></b></span> <span class="tag">翻译缺失/过期：<b id="translation-warning-count"></b></span></p><div id="sbaw-stats" class="stats"></div></header>
<nav class="tabs" role="tablist" aria-label="工作台视图"><button class="tab" role="tab" data-tab="delivery" aria-selected="true">交付地图</button><button class="tab" role="tab" data-tab="prompts" aria-selected="false">当前 Prompt</button><button class="tab" role="tab" data-tab="evidence" aria-selected="false">Evidence 审计</button><button class="tab" role="tab" data-tab="qa" aria-selected="false">QA 审计</button><button class="tab" role="tab" data-tab="judge" aria-selected="false">Judge 审计</button></nav>
<main><section class="panel active" data-panel="delivery"><div id="delivery-view"></div></section><section class="panel" data-panel="prompts"><div id="prompt-view"></div></section><section class="panel" data-panel="evidence"><div id="evidence-view"></div></section><section class="panel" data-panel="qa"><div id="qa-view"></div></section><section class="panel" data-panel="judge"><div id="judge-view"></div></section></main>
<footer class="footer"><span class="muted" id="preview-note"></span><button class="button" type="button" onclick="exportDecisions()">导出人工审核决定 JSON</button></footer>
<div class="lightbox" id="frame-lightbox" role="dialog" aria-modal="true" aria-label="证据帧大图"><button type="button" onclick="closeLightbox()">关闭</button><img id="frame-lightbox-image" alt="放大的证据帧"></div>
<script type="application/json" id="sbaw-data">__PAYLOAD__</script>
<script>
(()=>{const ROOT=document.getElementById('salesbench-audit-workbench');const PACK=JSON.parse(document.getElementById('sbaw-data').textContent);const D=PACK.data;const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pretty=v=>esc(JSON.stringify(v,null,2));const key=id=>`salesbench-audit:${D.release.prompt_version}:${id}`;
function stat(label,value){return `<div class="stat"><span class="muted">${esc(label)}</span><b>${esc(value)}</b></div>`}document.getElementById('runtime-prompt-version').textContent=D.release.runtime_prompt_version||D.release.prompt_version||'unknown';document.getElementById('current-prompt-version').textContent=D.release.current_prompt_version||'evidence-prompt-v9';document.getElementById('translation-warning-count').textContent=Number(D.translations?.missing||0)+Number(D.translations?.stale||0);document.getElementById('sbaw-stats').innerHTML=[stat('已处理视频',D.counts.videos),stat('商业记录',D.counts.commercial_records??D.counts.videos),stat('Abstention',D.counts.abstentions??0),stat('EvidenceUnit',D.counts.evidence_units),stat('Annotation',D.counts.annotations),stat('Evidence 人工待审',D.counts.human_review_queue??D.counts.review_queue),stat('Evidence 自动拒绝',D.counts.auto_rejected??0),stat('原始诊断队列',D.counts.pipeline_diagnostics??0),stat('QA',D.counts.qa),stat('QA 人工待审',D.counts.qa_human_review??0),stat('QA 抽样',D.counts.qa_accepted_sample??0),stat('Judge',D.counts.judge_rows)].join('');document.getElementById('preview-note').textContent=D.preview_notice||'完整本地审计视图；审核决定仅保存在当前浏览器。';
function showTab(id){ROOT.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===id)));ROOT.querySelectorAll('[data-panel]').forEach(p=>p.classList.toggle('active',p.dataset.panel===id));}ROOT.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));
function renderDelivery(){const f=D.delivery.formal_root||'',s=D.delivery.smoke_root||'';document.getElementById('delivery-view').innerHTML=`<div class="grid"><article class="card"><h2>物理目录</h2><p><span class="tag">FORMAL</span> <span class="mono">${esc(f)}</span></p><p><span class="tag">SMOKE</span> <span class="mono">${esc(s)}</span></p><p class="muted">smoke 与 64 条正式 candidate 结果物理分离，不能互相覆盖。</p><p><b>当前数据：</b>${esc(D.release.runtime_prompt_version)} · <b>审计组：</b>${esc(D.release.group||'formal')}</p></article><article class="card"><h2>四层正式交付</h2><ol><li><b>EvidenceDataset</b>：人工审核并冻结后才是 Gold 来源</li><li><b>Prompt manifest</b>：全英文版本与运行契约</li><li><b>QA</b>：全英文公开问题 + 私有 Gold</li><li><b>Evaluation</b>：predictions、Judge 明细、校准报告</li></ol><p class="muted">中文审计翻译和互动分析都不进入公开 benchmark。</p></article></div><h2 style="margin-top:18px">三道人工质量门</h2><div class="flow"><article class="gate primary"><h3>Gate E · Evidence Graph</h3><p>核查 EvidenceUnit、CommerceCue、CommercialRelation、Annotation 及关联帧。</p></article><article class="gate"><h3>冻结 EvidenceDataset</h3><p>接受、修订、拒绝都写入审核记录；冻结后重新实现并编译 QA。</p></article><article class="gate"><h3>Gate Q · QA</h3><p>逐题检查自然度、领域特异性、答案和图引用。</p></article><article class="gate"><h3>Gate J · Judge</h3><p>抽取人工评分集校准五档主评分和错误标签。</p></article></div>`}
function translationWarning(status){return status&&status!=='current'&&status!=='not_applicable'?`<div class="translation-warning">中文审计翻译状态：${esc(status)}。请以英文 canonical 原文为准并重新生成 sidecar。</div>`:''}
function renderPrompts(){const opts=PACK.prompts.map((p,i)=>`<option value="${i}">${esc(p.name)} · ${esc(p.stage)}</option>`).join('');document.getElementById('prompt-view').innerHTML=`<div class="alert"><strong>语言边界：</strong>v9 Prompt 的唯一规范版本是英文；中文审计翻译仅用于阅读，并通过 source SHA-256 与英文原文绑定。</div><div class="card"><div class="toolbar"><select class="control" id="prompt-select" aria-label="选择 Prompt">${opts}</select><span class="tag" id="prompt-version"></span></div><div id="prompt-detail"></div></div>`;const select=document.getElementById('prompt-select');const draw=()=>{const p=PACK.prompts[Number(select.value)];document.getElementById('prompt-version').textContent=p.version;document.getElementById('prompt-detail').innerHTML=`<div class="grid"><div><h2>${esc(p.name)}</h2><h3>中文审计翻译</h3>${translationWarning(p.translation_status)}<pre>${esc(p.system_zh||'尚无有效中文审计翻译')}</pre><details open><summary>English canonical System Prompt</summary><pre>${esc(p.system)}</pre></details><details><summary>User Payload 模板</summary><pre>${typeof p.user_template==='string'?esc(p.user_template):pretty(p.user_template)}</pre></details><p class="mono">source sha256: ${esc(p.translation_source_sha256||'not loaded')} · ${esc(p.translation_method||'')}</p></div><div><article class="card"><h3>审计观察</h3><ul>${p.observed.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></article><article class="card" style="margin-top:10px"><h3>已落实 / 后续验证</h3><ol>${p.recommendations.map(x=>`<li>${esc(x)}</li>`).join('')}</ol></article></div></div>`};select.addEventListener('change',draw);draw()}
function decisionButtons(id){const current=localStorage.getItem(key(id))||'';return `<div class="decision" data-decision="${esc(id)}">${[['accept','接受'],['revise','修订'],['reject','拒绝'],['defer','待定']].map(([v,l])=>`<button type="button" data-value="${v}" class="${current===v?'active':''}">${l}</button>`).join('')}</div>`}function bindDecisions(){ROOT.querySelectorAll('[data-decision] button').forEach(b=>b.addEventListener('click',()=>{const box=b.closest('[data-decision]');localStorage.setItem(key(box.dataset.decision),b.dataset.value);box.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));}))}
function renderFrames(frames){if(!frames||!frames.length)return '<div class="frame-missing">没有可显示的关联帧</div>';return `<div class="frame-grid">${frames.map(frame=>{const caption=`帧 ${frame.frame_index} · ${frame.timestamp_s==null?'时间未知':Number(frame.timestamp_s).toFixed(2)+'s'} · ${frame.relation||''}`;return frame.thumbnail_src?`<button class="frame" type="button" onclick="openLightbox('${esc(frame.thumbnail_src)}','${esc(caption)}')"><img loading="lazy" decoding="async" src="${esc(frame.thumbnail_src)}" alt="${esc(caption)}"><span>${esc(caption)}</span></button>`:`<div class="frame-missing">${esc(caption)}<br>缩略图缺失</div>`}).join('')}</div>`}
function renderEvidenceItems(items){return `<div class="evidence-stack">${(items||[]).map(item=>`<section class="evidence-unit"><div><span class="tag">${esc(item.modality||'missing')}</span><span class="mono">${esc(item.evidence_id||'无直接 Evidence ID')}</span></div>${translationWarning(item.translation_status)}<h3>${esc(item.content_zh||item.semantic_text||'未提供可读内容')}</h3><details open><summary>English canonical evidence</summary><p>${esc(item.content_en||item.semantic_text||'')}</p></details>${item.source_text_native?`<div class="native-source"><b>视频原语言 ASR/OCR：</b>${esc(item.source_text_native)}</div>`:''}${item.localization_note?`<div class="localization">${esc(item.localization_note)}</div>`:''}${renderFrames(item.frames)}</section>`).join('')}</div>`}
function renderGraph(cues,relations){const cueHtml=(cues||[]).map(c=>`<section class="graph-item"><div><span class="tag">Cue</span><span class="tag">${esc(c.cue_type)}</span><span class="mono">${esc(c.cue_id)}</span></div>${translationWarning(c.translation_status)}<b>${esc(c.content_zh||c.content_en||'')}</b><details><summary>English canonical cue 与完整引用</summary><p>${esc(c.content_en)}</p><pre>${pretty({evidence_ids:c.evidence_ids,directness:c.directness,theory_tags:c.theory_tags,attributes:c.attributes})}</pre></details></section>`).join('');const relationHtml=(relations||[]).map(r=>`<section class="graph-item"><div><span class="tag">Relation</span><span class="tag">${esc(r.relation_type)}</span><span class="mono">${esc(r.relation_id)}</span></div>${translationWarning(r.translation_status)}<b>${esc(r.content_zh||r.rationale_en||'')}</b><details><summary>English rationale 与图端点</summary><p>${esc(r.rationale_en)}</p><pre>${pretty({source_cue_ids:r.source_cue_ids,target_cue_ids:r.target_cue_ids,evidence_ids:r.evidence_ids,status:r.status,provenance:r.provenance})}</pre></details></section>`).join('');return `<div class="graph-stack">${cueHtml}${relationHtml}</div>`}
function renderReviewDetails(x){if(x.kind==='AUTO_RISK'||x.kind==='ACCEPTED_SAMPLE')return `<details open><summary>${x.kind==='ACCEPTED_SAMPLE'?'自动通过抽样详情':'Annotation risk details'}</summary><pre>${pretty({target:x.target,gold_value:x.gold_value,evidence_refs:x.evidence_refs,risk_codes:x.risk_codes,capability:x.capability,reasoning_operator:x.reasoning_operator})}</pre></details>`;if(x.item_type==='abstention')return `<div class="empty-state"><b>${esc(x.display_summary||'No candidate generated')}</b><p>${esc(x.reason||'The generator abstained.')}</p></div><details><summary>Abstention details</summary><pre>${pretty({stage:x.stage,task_type:x.task_type,task_subtype:x.task_subtype,reason_code:x.reason_code})}</pre></details>`;if(x.resolution_status==='unresolved')return `<div class="empty-state"><b>${esc(x.display_summary||'Candidate source could not be resolved')}</b><p>${esc(x.reason)}</p></div><details><summary>Unresolved source details</summary><pre>${pretty({source_proposal_ids:x.source_proposal_ids,unresolved_proposal_ids:x.unresolved_proposal_ids,stage:x.stage})}</pre></details>`;if(x.resolution_status==='diagnostic')return `<div class="empty-state"><b>管线诊断记录，不包含候选 Target 或 Gold</b><p>${esc(x.reason)}</p></div><details open><summary>诊断错误与来源</summary><pre>${pretty({stage:x.stage,item_type:x.item_type,reason_code:x.reason_code,issues:x.issues,evidence_refs:x.evidence_refs,commerce_cue_ids:x.commerce_cue_ids,commercial_relation_ids:x.commercial_relation_ids,candidate_snapshot:x.candidate_snapshot})}</pre></details>`;return `<p><span class="tag">${esc(x.capability||x.task_subtype||'')}</span><span class="tag">${esc(x.reasoning_operator||'')}</span></p><div class="review-fields"><section class="review-field"><b>Target</b><pre>${pretty(x.target||{})}</pre></section><section class="review-field"><b>Candidate Gold</b><pre>${pretty(x.candidate_gold||{})}</pre></section></div><details open><summary>Candidate content and evidence / 完整候选内容与验证结果</summary><pre>${pretty({candidate_snapshot:x.candidate_snapshot,semantic_verifier:x.semantic_verifier,stage:x.stage,item_type:x.item_type,task_subtype:x.task_subtype,question_intent:x.question_intent,issues:x.issues,evidence_refs:x.evidence_refs,commerce_cue_ids:x.commerce_cue_ids,commercial_relation_ids:x.commercial_relation_ids,source_proposal_ids:x.source_proposal_ids,source_candidates:x.source_candidates})}</pre></details>`}
window.openLightbox=function openLightbox(src,caption){const box=document.getElementById('frame-lightbox');const image=document.getElementById('frame-lightbox-image');image.src=src;image.alt=caption||'放大的证据帧';box.classList.add('open')};window.closeLightbox=function closeLightbox(){const box=document.getElementById('frame-lightbox');box.classList.remove('open');document.getElementById('frame-lightbox-image').removeAttribute('src')};document.getElementById('frame-lightbox').addEventListener('click',event=>{if(event.target.id==='frame-lightbox')closeLightbox()});
function toolbar(prefix,withBucket=false){const bucket=withBucket?`<select class="control" id="${prefix}-bucket" aria-label="审计队列"><option value="human_review">人工语义审核</option><option value="content_review">历史内容审核</option><option value="accepted_sample">自动通过抽样</option><option value="auto_rejected">自动拒绝</option><option value="pipeline_diagnostics">流水线诊断</option><option value="abstention_resample">Abstention / 补采样</option><option value="accepted_all">自动通过全集（只读）</option><option value="">全部队列</option></select>`:'';return `<div class="toolbar ${withBucket?'with-bucket':''}"><input class="control" id="${prefix}-search" placeholder="搜索 ID、问题、理由…"><select class="control" id="${prefix}-task"><option value="">全部任务</option><option>BP</option><option>CM</option><option>SS</option><option>AE</option><option>UNKNOWN</option></select>${bucket}<select class="control" id="${prefix}-risk"><option value="">全部风险</option>${Object.entries(PACK.risk_labels).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('')}</select><span class="muted" id="${prefix}-count"></span></div>`}
const pageState={ev:0,qa:0,jd:0};const pageDraws={};
function pager(prefix){return `<nav class="case-pager" aria-label="案例翻页"><button type="button" id="${prefix}-prev" aria-label="上一条">← 上一条</button><strong id="${prefix}-page">第 0 / 0 条</strong><label>跳转到第 <input id="${prefix}-jump" type="number" min="1" value="1" aria-label="跳转到案例序号"> 条</label><button type="button" id="${prefix}-next" aria-label="下一条">下一条 →</button></nav>`}
function movePage(prefix,delta){const draw=pageDraws[prefix];if(!draw)return;pageState[prefix]+=delta;draw()}
function bindPager(prefix,total,draw){pageDraws[prefix]=draw;const last=Math.max(0,total-1);pageState[prefix]=Math.max(0,Math.min(pageState[prefix],last));const prev=document.getElementById(`${prefix}-prev`),next=document.getElementById(`${prefix}-next`),jump=document.getElementById(`${prefix}-jump`),label=document.getElementById(`${prefix}-page`);label.textContent=total?`第 ${pageState[prefix]+1} / ${total} 条`:'第 0 / 0 条';prev.disabled=!total||pageState[prefix]===0;next.disabled=!total||pageState[prefix]===last;jump.disabled=!total;jump.max=String(Math.max(1,total));jump.value=String(total?pageState[prefix]+1:1);prev.onclick=()=>movePage(prefix,-1);next.onclick=()=>movePage(prefix,1);const applyJump=()=>{const requested=Math.max(1,Math.min(total,Number(jump.value)||1));pageState[prefix]=requested-1;draw()};jump.onchange=applyJump;jump.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();applyJump()}}}
function tags(codes){return (codes||[]).map(code=>`<span class="tag bad">${esc(PACK.risk_labels[code]||code)}</span>`).join('')}
function renderEvidence(){const queue=(D.evidence.queue||[]).map(x=>({...x,kind:'QUEUE',risk_codes:[...(x.risk_codes||[]),...(x.audit_bucket==='human_review'?['HUMAN_QUEUE']:[])]}));const abstentions=(D.evidence.abstentions||[]).map(x=>({...x,kind:'ABSTENTION',audit_bucket:'abstention_resample'}));const risks=(D.evidence.risks||[]).map(x=>({...x,kind:'AUTO_RISK',audit_bucket:x.audit_bucket||'auto_rejected'}));const accepted=(D.evidence.accepted_sample||[]).map(x=>({...x,kind:'ACCEPTED_SAMPLE',audit_bucket:'accepted_sample'}));const rows=[...queue,...abstentions,...risks,...accepted];const bucketLabels={human_review:'人工语义审核',content_review:'历史内容审核',accepted_sample:'自动通过抽样',auto_rejected:'自动拒绝',pipeline_diagnostics:'流水线诊断',abstention_resample:'Abstention / 补采样'};const bucketCounts=rows.reduce((acc,row)=>{const bucket=row.audit_bucket||'human_review';acc[bucket]=(acc[bucket]||0)+1;return acc},{});const bucketSummary=Object.entries(bucketLabels).map(([bucket,label])=>`<span class="tag">${esc(label)} · ${bucketCounts[bucket]||0}</span>`).join('');document.getElementById('evidence-view').innerHTML=`<div class="grid"><article class="card"><h2>六类质量视图</h2><p>${bucketSummary}</p><p class="muted">只有人工语义审核提供决定按钮；自动拒绝与流水线诊断是只读生产记录，Abstention 用于补帧或重采样，自动通过抽样用于监控漏检。</p></article><article class="card"><h2>商业图资产</h2><p>CommerceCue：<b>${D.evidence.commerce_cues?.length||0}</b> · CommercialRelation：<b>${D.evidence.commercial_relations?.length||0}</b></p><p>缺任务视频：<b>${D.evidence.missing_task_videos.length}</b> · 自动通过抽样：<b>${D.counts.accepted_sample||accepted.length}</b></p></article></div>${toolbar('ev',true)}${pager('ev')}<div id="ev-list" class="list"></div>`;document.getElementById('ev-bucket').value=rows.some(x=>x.audit_bucket==='human_review')?'human_review':(rows.some(x=>x.audit_bucket==='content_review')?'content_review':'');const draw=()=>{const q=document.getElementById('ev-search').value.toLowerCase(),task=document.getElementById('ev-task').value,bucket=document.getElementById('ev-bucket').value,risk=document.getElementById('ev-risk').value;const found=rows.filter(x=>(!task||x.task_type===task)&&(!bucket||x.audit_bucket===bucket)&&(!risk||(x.risk_codes||[]).includes(risk))&&(!q||JSON.stringify(x).toLowerCase().includes(q)));pageState.ev=Math.max(0,Math.min(pageState.ev,Math.max(0,found.length-1)));const x=found[pageState.ev];document.getElementById('ev-count').textContent=`匹配 ${found.length} / ${rows.length}`;document.getElementById('ev-list').innerHTML=x?`<article class="item"><div class="item-head"><div><span class="tag">${esc(bucketLabels[x.audit_bucket]||x.audit_bucket||x.kind)}</span><span class="tag">${esc(x.task_type||(x.resolution_status==='diagnostic'?'PIPELINE':'UNRESOLVED'))}</span>${x.stage?`<span class="tag">${esc(x.stage)}</span>`:''}${x.item_type?`<span class="tag">${esc(x.item_type)}</span>`:''}${tags(x.risk_codes)}</div><span class="mono">${esc(x.video_id)}</span></div><h3>${esc(x.id)}</h3><p>${esc(x.reason||'自动接受记录命中本地风险规则')}</p>${renderReviewDetails(x)}<h3>CommerceCue / CommercialRelation</h3>${renderGraph(x.commerce_cues,x.commercial_relations)}<h3>可核对 Evidence 与关联帧</h3>${renderEvidenceItems(x.evidence_items)}${x.audit_bucket==='human_review'||x.audit_bucket==='content_review'?decisionButtons(x.id):'<p class="muted">只读记录：该案例不应产生人工发布决定。</p>'}</article>`:'<p class="muted">没有匹配记录。</p>';bindDecisions();bindPager('ev',found.length,draw)};['ev-search','ev-task','ev-bucket','ev-risk'].forEach(id=>document.getElementById(id).addEventListener('input',()=>{pageState.ev=0;draw()}));draw()}
function renderQA(){const rows=D.qa||[];const bucketLabels={human_review:'QA 人工语义审核',content_review:'历史 QA 内容审核',accepted_sample:'QA 自动通过抽样',auto_rejected:'QA 自动拒绝',pipeline_diagnostics:'QA 流水线诊断',accepted_all:'自动通过全集（只读）'};document.getElementById('qa-view').innerHTML=`<div class="card"><h2>QA 生产质量视图</h2><p>人工只处理语义歧义和约 10% 自动通过监控样本；明确错误、公式化问题、缺证据和工程故障在生产阶段分流。中文只用于通读，结论必须同时核对英文 canonical QA、商业图、Evidence 与帧。</p></div>${toolbar('qa',true)}${pager('qa')}<div id="qa-list" class="list"></div>`;document.getElementById('qa-bucket').value=rows.some(x=>x.audit_bucket==='human_review')?'human_review':(rows.some(x=>x.audit_bucket==='accepted_sample')?'accepted_sample':(rows.some(x=>x.audit_bucket==='content_review')?'content_review':'accepted_all'));const draw=()=>{const q=document.getElementById('qa-search').value.toLowerCase(),task=document.getElementById('qa-task').value,bucket=document.getElementById('qa-bucket').value,risk=document.getElementById('qa-risk').value;const found=rows.filter(x=>(!task||x.task_type===task)&&(!bucket||x.audit_bucket===bucket)&&(!risk||(x.risk_codes||[]).includes(risk))&&(!q||JSON.stringify(x).toLowerCase().includes(q)));pageState.qa=Math.max(0,Math.min(pageState.qa,Math.max(0,found.length-1)));const x=found[pageState.qa];document.getElementById('qa-count').textContent=`匹配 ${found.length} / ${rows.length}`;document.getElementById('qa-list').innerHTML=x?`<article class="item"><div class="item-head"><div><span class="tag">${esc(bucketLabels[x.audit_bucket]||x.audit_bucket||'QA')}</span><span class="tag">${esc(x.task_type)}</span><span class="tag">${esc(x.task_subtype)}</span><span class="tag">${esc(x.capability||'')}</span><span class="tag">${esc(x.reasoning_operator||'')}</span>${tags(x.risk_codes)}</div><span class="mono">${esc(x.vqa_id)}</span></div>${x.reason?`<p><b>质量路由：</b>${esc(x.reason)}</p>`:''}${translationWarning(x.question_translation_status)}${translationWarning(x.gold_translation_status)}<h3>${esc(x.question_zh||x.question||'无可展示问题')}</h3><p><b>中文 Gold：</b>${esc(x.gold_answer_zh||x.gold_answer||'无')}</p><details open><summary>English canonical question and Gold</summary><p><b>Question:</b> ${esc(x.question)}</p><p><b>Gold:</b> ${esc(typeof x.gold_answer==='string'?x.gold_answer:JSON.stringify(x.gold_answer))}</p></details>${x.semantic_verification?`<details open><summary>QA semantic quality verification</summary><pre>${pretty(x.semantic_verification)}</pre></details>`:''}<h3>CommerceCue / CommercialRelation</h3>${renderGraph(x.commerce_cues,x.commercial_relations)}<h3>可读 Evidence 与关联帧</h3>${renderEvidenceItems(x.evidence_items)}<details><summary>来源 Annotation 与完整引用</summary><pre>${pretty({source_annotations:x.source_annotations,source_annotation_ids:x.source_annotation_ids,evidence_refs:x.evidence_refs,commerce_cue_ids:x.commerce_cue_ids,commercial_relation_ids:x.commercial_relation_ids,spec_id:x.spec_id})}</pre></details>${x.audit_bucket==='human_review'||x.audit_bucket==='accepted_sample'?decisionButtons(x.vqa_id):'<p class="muted">该 QA 是只读生产记录，不产生人工发布决定。</p>'}</article>`:'<p class="muted">没有匹配记录。</p>';bindDecisions();bindPager('qa',found.length,draw)};['qa-search','qa-task','qa-bucket','qa-risk'].forEach(id=>document.getElementById(id).addEventListener('input',()=>{pageState.qa=0;draw()}));draw()}
function renderJudge(){const per=D.judge.metrics.per_task||{};const bars=Object.entries(per).map(([task,m])=>`<div class="bar"><b>${esc(task)}</b><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,Number(m.relaxed_accuracy||0)*100))}%"></div></div><span>${(Number(m.relaxed_accuracy||0)*100).toFixed(1)}%</span></div>`).join('');document.getElementById('judge-view').innerHTML=`<div class="grid"><article class="card"><h2>Candidate 诊断分数</h2><div class="bars">${bars}</div><p>Macro relaxed：<b>${(Number(D.judge.metrics.macro_average?.relaxed_accuracy||0)*100).toFixed(2)}%</b> · Judge failures：<b>${esc(D.judge.summary.judge_failed_count||0)}</b></p></article><article class="card"><h2>Judge v5 与后续校准</h2><ol><li>四任务独立 rubric 与五档主分数；</li><li>英文 canonical 理由和诊断标签；</li><li>中文理由只由 audit sidecar 生成；</li><li>仍需人工样本校准一致性和偏差。</li></ol></article></div>${toolbar('jd')}${pager('jd')}<div id="jd-list" class="list"></div>`;document.querySelector('#jd-risk').innerHTML='<option value="">全部分数</option><option value="LOW">≤ 0.5</option><option value="HIGH">≥ 0.75</option>';const draw=()=>{const q=document.getElementById('jd-search').value.toLowerCase(),task=document.getElementById('jd-task').value,risk=document.getElementById('jd-risk').value;const found=D.judge.rows.filter(x=>(!task||x.task_type===task)&&(!risk||(risk==='LOW'?Number(x.score)<=.5:Number(x.score)>=.75))&&(!q||JSON.stringify(x).toLowerCase().includes(q)));pageState.jd=Math.max(0,Math.min(pageState.jd,Math.max(0,found.length-1)));const x=found[pageState.jd];document.getElementById('jd-count').textContent=`匹配 ${found.length} / ${D.judge.rows.length}`;document.getElementById('jd-list').innerHTML=x?`<article class="item"><div class="item-head"><div><span class="tag">${esc(x.task_type)}</span><span class="tag ${Number(x.score)<=.5?'bad':''}">score ${esc(x.score)}</span></div><span class="mono">${esc(x.vqa_id)}</span></div><h3>${esc(x.question)}</h3><p><b>Reference：</b>${esc(x.reference_answer)}</p><p><b>Model：</b>${esc(x.model_output)}</p>${x.correctness==null?'<p class="muted">当前结果没有完整分项分数。</p>':`<p><span class="tag">correctness ${esc(x.correctness)}</span> <span class="tag">grounding ${esc(x.grounding)}</span> <span class="tag">completeness ${esc(x.completeness)}</span></p>`}<h3>Judge 实际依据的 Evidence</h3>${renderEvidenceItems(x.evidence_items)}${translationWarning(x.reason_translation_status)}${translationWarning(x.evidence_alignment_translation_status)}<details open><summary>中文 Judge 理由与证据对齐</summary><p>${esc(x.reason_zh||'尚无中文审计翻译')}</p><p>${esc(x.evidence_alignment_zh||'')}</p></details><details><summary>English canonical Judge output</summary><p>${esc(x.reason)}</p><p>${esc(x.evidence_alignment)}</p></details>${decisionButtons(`judge:${x.vqa_id}`)}</article>`:'<p class="muted">没有匹配记录。</p>';bindDecisions();bindPager('jd',found.length,draw)};['jd-search','jd-task','jd-risk'].forEach(id=>document.getElementById(id).addEventListener('input',()=>{pageState.jd=0;draw()}));draw()}
document.addEventListener('keydown',event=>{if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;if(['INPUT','SELECT','TEXTAREA'].includes(String(event.target?.tagName||''))||document.getElementById('frame-lightbox').classList.contains('open'))return;const panel=ROOT.querySelector('[data-panel].active')?.dataset.panel;const prefix={evidence:'ev',qa:'qa',judge:'jd'}[panel];if(!prefix)return;event.preventDefault();movePage(prefix,event.key==='ArrowLeft'?-1:1)});
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


def compact_workbench_to_byte_budget(
    data: dict[str, Any],
    prompts: list[dict[str, Any]],
    *,
    limit: int,
    max_bytes: int,
    prepare: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Reduce the per-layer preview limit until the rendered fragment fits its host budget."""
    if limit < 1 or max_bytes < 1:
        raise ValueError("preview limit and byte budget must be positive")
    effective_limit = limit
    while True:
        compact = compact_workbench_data(data, effective_limit)
        if effective_limit < limit:
            compact["preview_notice"] = (
                f"预览根据 {max_bytes} 字节上限自动缩减：每层最多 {effective_limit} 条；"
                "完整本地 HTML 保留全部记录。"
            )
        if prepare is not None:
            prepare(compact)
        rendered = render_workbench(compact, prompts, fragment=True)
        if len(rendered.encode("utf-8")) < max_bytes:
            return compact
        if effective_limit == 1:
            raise ValueError(f"fragment exceeds byte budget even at one record per layer: {max_bytes}")
        effective_limit = max(1, effective_limit // 2)


def render_preview_workbench(
    data: dict[str, Any],
    prompts: list[dict[str, Any]],
    *,
    limit: int,
    max_bytes: int = 1_000_000,
) -> str:
    compact = compact_workbench_to_byte_budget(
        data,
        prompts,
        limit=limit,
        max_bytes=max_bytes,
    )
    return render_workbench(compact, prompts, fragment=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local SalesBench prompt and audit workbench")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path)
    parser.add_argument("--group", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--preview-limit", type=int, default=60)
    parser.add_argument("--skip-organize", action="store_true")
    args = parser.parse_args()
    repo_root = Path.cwd().resolve()
    manifest = _read_json(args.manifest.resolve())
    translations = load_translation_index(args.translations.resolve())
    delivery = (
        {
            "formal_root": str((repo_root / manifest["formal"]["root"]).resolve()),
            "smoke_root": str((repo_root / manifest["smoke"]["root"]).resolve()),
        }
        if args.skip_organize
        else organize_delivery(manifest, repo_root)
    )
    data = build_workbench_data(
        manifest,
        repo_root,
        delivery,
        translations=translations,
        group=args.group,
    )
    prompts = localize_prompt_snapshot(collect_prompt_snapshot(), translations)
    data["translations"] = translations.summary()
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

        def prepare_preview(preview_data: dict[str, Any]) -> None:
            nonlocal preview_thumbnail_stats
            preview_thumbnail_stats = materialize_thumbnails(
                preview_data,
                asset_root=args.fragment.parent / "assets" / "frames",
                html_parent=args.fragment.parent,
            )
            preview_data["thumbnail_stats"] = preview_thumbnail_stats

        preview_data = compact_workbench_to_byte_budget(
            deepcopy(data),
            prompts,
            limit=args.preview_limit,
            max_bytes=1_000_000,
            prepare=prepare_preview,
        )
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
