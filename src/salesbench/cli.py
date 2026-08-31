from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .assets import build_asset_index
from .config import config_summary, ensure_output_dirs, load_config
from .excel_reader import load_sheet_records
from .input_builder import build_input_artifacts
from .interaction_analysis import run_interaction_analysis
from .io_utils import read_records, write_json, write_jsonl
from .preprocess import build_benchmark_dataset


def _env_or_arg(args: argparse.Namespace, attr: str, env_var: str) -> str | None:
    return getattr(args, attr, None) or os.environ.get(env_var)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _arg_or_env(args: argparse.Namespace, attr: str, *env_vars: str) -> str | None:
    return getattr(args, attr, None) or _first_env(*env_vars)


def _path(value: str, repo_root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def prepare_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config)
    raw_records = load_sheet_records(config.research_workbook)
    asset_index, asset_manifest = build_asset_index(config.raw_video_dir, config.raw_sales_dir)
    processed_records, _, _, profile = build_benchmark_dataset(
        raw_records=raw_records,
        asset_index=asset_index,
        asset_manifest=asset_manifest,
        config=config,
    )
    write_json(config.asset_manifest, asset_manifest)
    write_jsonl(config.processed_main, processed_records)
    write_json(config.data_profile, profile)
    print(json.dumps({"config": config_summary(config), "record_count": len(processed_records)}, ensure_ascii=False, indent=2))
    return 0


def build_inputs_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config)
    print(json.dumps(build_input_artifacts(config), ensure_ascii=False, indent=2))
    return 0


def select_evidence_cohort_command(args: argparse.Namespace) -> int:
    from .goldbank.cohort import select_goldbank_cohort
    from .goldbank.prompts import PROMPT_VERSION
    from .goldbank.schema import SCHEMA_VERSION

    config = load_config(args.config)
    cohort = select_goldbank_cohort(
        read_records(config.processed_main),
        total=args.total,
        seed=args.seed,
        require_video_asset=not args.allow_missing_video_asset,
    )
    cohort["version"] = "evidence-cohort-v3"
    cohort["prompt_version"] = PROMPT_VERSION
    cohort["schema_version"] = SCHEMA_VERSION
    output = _path(args.output, config.repo_root)
    write_json(output, cohort)
    print(json.dumps({"output": str(output), "video_count": len(cohort["video_ids"])}, ensure_ascii=False, indent=2))
    return 0


def build_evidence_dataset_command(args: argparse.Namespace) -> int:
    from .goldbank.runner import build_gold_bank_dataset

    config = load_config(args.config)
    ensure_output_dirs(config)
    default_key = _env_or_arg(args, "api_key", "OPENAI_API_KEY")
    vision_key = _arg_or_env(
        args,
        "vision_api_key",
        "VISION_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
    ) or default_key
    text_key = _arg_or_env(
        args,
        "text_api_key",
        "TEXT_API_KEY",
        "DEEPSEEK_API_KEY",
    ) or default_key
    if not vision_key or not text_key:
        raise ValueError("Evidence generation requires vision and text provider API keys")
    summary = build_gold_bank_dataset(
        config=config,
        pilot_config_path=_path(args.cohort_config, config.repo_root),
        output_dir=_path(args.output_dir, config.repo_root),
        api_key=default_key or vision_key,
        model=args.model,
        base_url=_env_or_arg(args, "base_url", "OPENAI_BASE_URL"),
        max_workers=args.max_workers,
        resume=args.resume,
        vision_api_key=vision_key,
        vision_model=_arg_or_env(
            args,
            "vision_model",
            "VISION_MODEL",
            "QWEN_VISION_MODEL",
        ) or args.model,
        vision_base_url=_arg_or_env(
            args,
            "vision_base_url",
            "VISION_BASE_URL",
            "QWEN_BASE_URL",
            "DASHSCOPE_BASE_URL",
        ),
        text_api_key=text_key,
        text_model=_arg_or_env(
            args,
            "text_model",
            "TEXT_MODEL",
            "DEEPSEEK_MODEL",
        ) or args.model,
        text_base_url=_arg_or_env(
            args,
            "text_base_url",
            "TEXT_BASE_URL",
            "DEEPSEEK_BASE_URL",
        ) or _env_or_arg(args, "base_url", "OPENAI_BASE_URL"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def apply_evidence_reviews_command(args: argparse.Namespace) -> int:
    from .goldbank.review_io import apply_human_reviews

    evidence_dir = Path(args.evidence_dir)
    reviewed = apply_human_reviews(
        read_records(evidence_dir / "video_evidence_dataset.jsonl"),
        read_records(evidence_dir / "human_review_queue.jsonl"),
        read_records(Path(args.decisions)),
    )
    write_jsonl(Path(args.output), reviewed)
    print(json.dumps({"output": args.output, "record_count": len(reviewed)}, ensure_ascii=False, indent=2))
    return 0


def realize_qa_command(args: argparse.Namespace) -> int:
    from .vlm.api_client import VLMClient
    from .vqa.realizer import run_qa_realizer

    api_key = _arg_or_env(
        args,
        "text_api_key",
        "TEXT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    if not api_key:
        raise ValueError(
            "QA realization requires --text-api-key, TEXT_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY"
        )
    client = VLMClient(
        api_key=api_key,
        model=_arg_or_env(
            args,
            "text_model",
            "TEXT_MODEL",
            "DEEPSEEK_MODEL",
        ) or "gpt-4o",
        base_url=_arg_or_env(
            args,
            "text_base_url",
            "TEXT_BASE_URL",
            "DEEPSEEK_BASE_URL",
            "OPENAI_BASE_URL",
        ),
        max_tokens=1024,
    )
    summary = run_qa_realizer(
        Path(args.evidence_dir),
        Path(args.output_dir),
        client,
        dataset_filename=args.dataset_file,
        max_workers=args.max_workers,
        resume=args.resume,
        allow_auto_candidates=args.allow_auto_candidates,
        strict_semantic_verification=args.strict_semantic_verification,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_audit_translations_command(args: argparse.Namespace) -> int:
    from .audit_translation import collect_audit_translation_jobs, run_audit_translations
    from .vlm.api_client import VLMClient

    api_key = _arg_or_env(
        args,
        "api_key",
        "TEXT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    if not api_key:
        raise ValueError(
            "Audit translation requires --api-key, TEXT_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY"
        )
    manifest = Path(args.manifest)
    client = VLMClient(
        api_key=api_key,
        model=_arg_or_env(
            args,
            "model",
            "TEXT_MODEL",
            "DEEPSEEK_MODEL",
        ) or "gpt-4o",
        base_url=_arg_or_env(
            args,
            "base_url",
            "TEXT_BASE_URL",
            "DEEPSEEK_BASE_URL",
            "OPENAI_BASE_URL",
        ),
        max_tokens=4096,
    )
    jobs = collect_audit_translation_jobs(
        manifest,
        repo_root=manifest.resolve().parent.parent,
        include_prompts=not args.skip_prompts,
    )
    summary = run_audit_translations(
        jobs,
        Path(args.output),
        client,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def compile_vqa_command(args: argparse.Namespace) -> int:
    from .vqa.compiler import CompilePolicy, compile_vqa_from_gold

    summary = compile_vqa_from_gold(
        gold_bank_dir=Path(args.evidence_dir),
        output_dir=Path(args.output_dir),
        policy=CompilePolicy(
            max_questions_per_video=args.max_questions_per_video,
            max_per_task=args.max_per_task,
            require_all_tasks=not args.allow_missing_tasks,
            allow_auto_candidates=args.allow_auto_candidates,
        ),
        bank_filename=args.dataset_file,
        realizations_path=Path(args.realizations),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_vqa_benchmark_command(args: argparse.Namespace) -> int:
    from .vqa_baseline.runner import run_salesbench_qa_baseline

    config = load_config(args.config)
    api_key = _arg_or_env(
        args,
        "api_key",
        "VISION_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
    )
    if not api_key:
        raise ValueError(
            "OpenAI-compatible runner requires --api-key, VISION_API_KEY, QWEN_API_KEY, "
            "DASHSCOPE_API_KEY, or OPENAI_API_KEY"
        )
    model = _arg_or_env(
        args,
        "model",
        "VISION_MODEL",
        "QWEN_VISION_MODEL",
    ) or "gpt-4o"
    summary = run_salesbench_qa_baseline(
        config=config,
        vqa_path=_path(args.vqa, config.repo_root),
        output_dir=_path(args.output_dir, config.repo_root),
        api_key=api_key,
        model=model,
        base_url=_arg_or_env(
            args,
            "base_url",
            "VISION_BASE_URL",
            "QWEN_BASE_URL",
            "DASHSCOPE_BASE_URL",
            "OPENAI_BASE_URL",
        ),
        max_samples=args.max_samples,
        max_workers=args.max_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def evaluate_vqa_benchmark_command(args: argparse.Namespace) -> int:
    from .vqa_evaluate.runner import evaluate_salesbench_qa_files

    config = load_config(args.config)
    api_key = _arg_or_env(
        args,
        "api_key",
        "TEXT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    if not api_key:
        raise ValueError(
            "LLM-as-Judge requires --api-key, TEXT_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY"
        )
    report = evaluate_salesbench_qa_files(
        gold_path=_path(args.gold, config.repo_root),
        answers_path=_path(args.predictions, config.repo_root),
        output_dir=_path(args.output_dir, config.repo_root),
        api_key=api_key,
        judge_model=_arg_or_env(
            args,
            "judge_model",
            "TEXT_MODEL",
            "DEEPSEEK_MODEL",
        ) or "gpt-4o",
        base_url=_arg_or_env(
            args,
            "base_url",
            "TEXT_BASE_URL",
            "DEEPSEEK_BASE_URL",
            "OPENAI_BASE_URL",
        ),
        max_workers=args.max_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def audit_evidence_dataset_command(args: argparse.Namespace) -> int:
    from .goldbank.audit import audit_gold_bank

    evidence_path = Path(args.evidence)
    evidence_dir = evidence_path.parent
    cue_path = Path(args.cues) if args.cues else evidence_dir / "commerce_cues.jsonl"
    relation_path = (
        Path(args.relations) if args.relations else evidence_dir / "commercial_relations.jsonl"
    )
    report = audit_gold_bank(
        read_records(Path(args.dataset)),
        read_records(evidence_path),
        read_records(cue_path) if cue_path.exists() else [],
        read_records(relation_path) if relation_path.exists() else [],
        read_records(Path(args.translations)) if args.translations else [],
    )
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def analyze_interactions_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_interaction_analysis(
        records_path=_path(args.records, config.repo_root) if args.records else config.processed_main,
        output_dir=_path(args.output_dir, config.repo_root),
        judge_details_path=_path(args.judge_details, config.repo_root) if args.judge_details else None,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SalesBench Evidence-First multimodal VQA benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="构建内部数据资产")
    prepare.add_argument("--config", default=None)
    prepare.set_defaults(func=prepare_command)

    inputs = sub.add_parser("build-inputs", help="构建 C1-C6 内部上下文资产")
    inputs.add_argument("--config", default=None)
    inputs.set_defaults(func=build_inputs_command)

    cohort = sub.add_parser("select-evidence-cohort", help="选择 EvidenceDataset cohort")
    cohort.add_argument("--config", default=None)
    cohort.add_argument("--total", type=int, default=128)
    cohort.add_argument("--seed", type=int, default=42)
    cohort.add_argument("--output", default="configs/evidence_alpha_128videos.json")
    cohort.add_argument("--allow-missing-video-asset", action="store_true")
    cohort.set_defaults(func=select_evidence_cohort_command)

    evidence = sub.add_parser("build-evidence-dataset", help="运行 Evidence-First 生成流程")
    evidence.add_argument("--config", default=None)
    evidence.add_argument("--cohort-config", default="configs/evidence_alpha_128videos.json")
    evidence.add_argument("--output-dir", default="outputs/evidence/v2")
    evidence.add_argument("--api-key", default=None)
    evidence.add_argument("--base-url", default=None)
    evidence.add_argument("--model", default="gpt-4o")
    evidence.add_argument("--vision-api-key", default=None)
    evidence.add_argument("--vision-base-url", default=None)
    evidence.add_argument("--vision-model", default=None)
    evidence.add_argument("--text-api-key", default=None)
    evidence.add_argument("--text-base-url", default=None)
    evidence.add_argument("--text-model", default=None)
    evidence.add_argument("--max-workers", type=int, default=1)
    evidence.add_argument("--no-resume", action="store_false", dest="resume")
    evidence.set_defaults(func=build_evidence_dataset_command, resume=True)

    reviews = sub.add_parser("apply-evidence-reviews", help="应用人工 Evidence 复核")
    reviews.add_argument("--evidence-dir", required=True)
    reviews.add_argument("--decisions", required=True)
    reviews.add_argument("--output", required=True)
    reviews.set_defaults(func=apply_evidence_reviews_command)

    realize = sub.add_parser("realize-qa", help="将审核后的 QuestionSpec 实现为自然英文问题")
    realize.add_argument("--evidence-dir", required=True)
    realize.add_argument("--dataset-file", default="video_evidence_dataset.jsonl")
    realize.add_argument("--output-dir", required=True)
    realize.add_argument("--text-api-key", default=None)
    realize.add_argument("--text-base-url", default=None)
    realize.add_argument("--text-model", default=None)
    realize.add_argument("--max-workers", type=int, default=2)
    realize.add_argument("--no-resume", action="store_false", dest="resume")
    realize.add_argument(
        "--allow-auto-candidates",
        action="store_true",
        help="候选集调试模式；正式 QA 仅接受人工确认的 annotation",
    )
    realize.add_argument(
        "--strict-semantic-verification",
        action="store_true",
        help="生成后使用同一文本模型进行严格 QA 语义门禁",
    )
    realize.set_defaults(func=realize_qa_command, resume=True)

    translations = sub.add_parser("build-audit-translations", help="生成仅供人工审计的中文翻译 sidecar")
    translations.add_argument("--manifest", required=True)
    translations.add_argument("--output", required=True)
    translations.add_argument("--api-key", default=None)
    translations.add_argument("--base-url", default=None)
    translations.add_argument("--model", default=None)
    translations.add_argument("--batch-size", type=int, default=20)
    translations.add_argument("--skip-prompts", action="store_true")
    translations.set_defaults(func=build_audit_translations_command)

    compile_parser = sub.add_parser("compile-vqa", help="从 EvidenceDataset 编译 BP/CM/SS/AE")
    compile_parser.add_argument("--evidence-dir", required=True)
    compile_parser.add_argument("--dataset-file", default="video_evidence_dataset_reviewed.jsonl")
    compile_parser.add_argument("--realizations", required=True)
    compile_parser.add_argument("--output-dir", required=True)
    compile_parser.add_argument("--max-questions-per-video", type=int, default=8)
    compile_parser.add_argument("--max-per-task", type=int, default=2)
    compile_parser.add_argument("--allow-missing-tasks", action="store_true", help="仅限 pilot 调试")
    compile_parser.add_argument(
        "--allow-auto-candidates",
        action="store_true",
        help="候选集调试模式；正式编译默认关闭",
    )
    compile_parser.set_defaults(func=compile_vqa_command)

    run_parser = sub.add_parser("run-vqa-benchmark", help="OpenAI-compatible 便利 runner，输出 predictions.jsonl")
    run_parser.add_argument("--config", default=None)
    run_parser.add_argument("--vqa", required=True)
    run_parser.add_argument("--output-dir", default="outputs/vqa/v2/run")
    run_parser.add_argument("--api-key", default=None)
    run_parser.add_argument("--base-url", default=None)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--max-samples", type=int, default=None)
    run_parser.add_argument("--max-workers", type=int, default=2)
    run_parser.set_defaults(func=run_vqa_benchmark_command)

    evaluate = sub.add_parser("evaluate-vqa-benchmark", help="四任务 LLM-as-Judge 评测")
    evaluate.add_argument("--config", default=None)
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output-dir", default="outputs/vqa/v2/evaluation")
    evaluate.add_argument("--api-key", default=None)
    evaluate.add_argument("--base-url", default=None)
    evaluate.add_argument("--judge-model", default=None)
    evaluate.add_argument("--max-workers", type=int, default=4)
    evaluate.set_defaults(func=evaluate_vqa_benchmark_command)

    audit = sub.add_parser("audit-evidence-dataset", help="审计 EvidenceDataset")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--evidence", required=True)
    audit.add_argument("--cues", default=None)
    audit.add_argument("--relations", default=None)
    audit.add_argument("--translations", default=None)
    audit.add_argument("--output", default=None)
    audit.set_defaults(func=audit_evidence_dataset_command)

    interaction = sub.add_parser("analyze-interactions", help="独立私有互动关联与诊断分析")
    interaction.add_argument("--config", default=None)
    interaction.add_argument("--records", default=None)
    interaction.add_argument("--judge-details", default=None)
    interaction.add_argument("--output-dir", default="outputs/analysis/interactions")
    interaction.add_argument("--bootstrap-samples", type=int, default=1000)
    interaction.set_defaults(func=analyze_interactions_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))
