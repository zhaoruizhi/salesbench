"""Run closed-source VLM baselines on SalesBench-QA."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
from typing import Any

from ..config import BenchmarkConfig
from ..io_utils import read_records, write_json, write_jsonl
from ..utils import clean_text
from ..vlm.api_client import VLMClient
from ..vlm.frame_sampler import sample_frames
from ..vqa.schema import PUBLIC_TASKS, sanitize_model_name
from ..vqa_evaluate.runner import evaluate_salesbench_qa_files
from .analysis import build_analysis_markdown
from .parser import parse_closed_source_response
from .prompts import BASELINE_PROMPT_VERSION, CLOSED_SOURCE_SYSTEM_PROMPT, build_closed_source_user_prompt


def _answer_file_stem(model: str) -> str:
    return f"{sanitize_model_name(model)}_closed_source"


def _validate_public_items(items: list[dict[str, Any]]) -> None:
    invalid_tasks = sorted(
        {
            clean_text(record.get("task_type")).upper()
            for record in items
            if clean_text(record.get("task_type")).upper() not in PUBLIC_TASKS
        }
    )
    if invalid_tasks:
        raise ValueError(f"Unsupported public VQA task types: {invalid_tasks}")


def _video_lookup(config: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    return {
        clean_text(record.get("video_id")): record
        for record in read_records(config.input_raw_video)
        if clean_text(record.get("video_id"))
    }


def _frame_blocks(video_record: dict[str, Any], max_frames: int = 16) -> list[dict[str, Any]]:
    video_id = clean_text(video_record.get("video_id"))
    video_path = clean_text(video_record.get("primary_video_path"))
    if not video_id or not video_path or not video_record.get("has_video_asset"):
        return [{"type": "text", "text": "Video: no usable video frames."}]
    frames = sample_frames(
        video_path=video_path,
        video_id=video_id,
        strategy="hook_plus_uniform",
        total_frames=max_frames,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": "Video: sampled key frames."}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_base64}", "detail": "low"},
            }
        )
    return content


def build_user_content(item: dict[str, Any], video_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    video_id = clean_text(item.get("video_id"))
    video_record = video_lookup.get(video_id, {"video_id": video_id})
    return [
        *_frame_blocks(video_record),
        {"type": "text", "text": build_closed_source_user_prompt({"question": item.get("question")}, video_record)},
    ]


def _process_item(item: dict[str, Any], video_lookup: dict[str, dict[str, Any]], client: VLMClient) -> dict[str, Any]:
    result = client.call(
        system_prompt=CLOSED_SOURCE_SYSTEM_PROMPT,
        user_content=build_user_content(item, video_lookup),
        response_format=None,
    )
    parsed = parse_closed_source_response(result.raw_response) if result.success else {
        "thought": "",
        "answer": "",
        "parse_warning": "api_call_failed",
    }
    answer = parsed["answer"]
    return {
        "vqa_id": clean_text(item.get("vqa_id")),
        "video_id": clean_text(item.get("video_id")),
        "task_layer": clean_text(item.get("task_layer")),
        "task_type": clean_text(item.get("task_type")),
        "answer": answer,
        "thought": parsed["thought"],
        "parse_warning": parsed["parse_warning"],
        "model": result.model,
        "success": result.success and bool(answer),
        "raw_response": result.raw_response,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": result.latency_s,
        "cost_usd": result.cost_usd,
    }


def run_salesbench_qa_baseline_records(
    items: list[dict[str, Any]],
    video_lookup: dict[str, dict[str, Any]],
    output_dir: Path,
    model: str,
    client: VLMClient,
    max_workers: int = 2,
) -> dict[str, Any]:
    _validate_public_items(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model_answers"
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = _answer_file_stem(model)
    answer_path = output_dir / "predictions.jsonl"
    meta_path = model_dir / f"{stem}_run_meta.json"

    answers: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    lock = threading.Lock()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(_process_item, item, video_lookup, client): clean_text(item.get("vqa_id"))
            for item in items
        }
        for future in as_completed(futures):
            vqa_id = futures[future]
            try:
                answer = future.result()
            except Exception as exc:
                answer = {"vqa_id": vqa_id, "success": False, "answer": "", "error": str(exc), "cost_usd": 0.0}
            answers.append(answer)
            with lock:
                completed += 1
                if not answer.get("success"):
                    failed += 1
                if completed % 10 == 0 or completed == len(items):
                    elapsed = time.time() - start_time
                    print(f"  [{completed}/{len(items)}] answered={completed - failed} failed={failed} elapsed={elapsed/60:.1f}min")

    answers.sort(key=lambda record: clean_text(record.get("vqa_id")))
    write_jsonl(answer_path, answers)
    summary = {
        "model": model,
        "prompt_version": BASELINE_PROMPT_VERSION,
        "vqa_count": len(items),
        "answer_count": len(answers),
        "failed_count": failed,
        "total_cost_usd": round(sum(float(item.get("cost_usd") or 0.0) for item in answers), 6),
        "outputs": {
            "answers": str(answer_path),
            "run_meta": str(meta_path),
        },
    }
    write_json(meta_path, summary)
    return summary


def run_salesbench_qa_baseline(
    config: BenchmarkConfig,
    vqa_path: Path,
    output_dir: Path,
    api_key: str,
    model: str = "gpt-4o",
    base_url: str | None = None,
    max_samples: int | None = None,
    max_workers: int = 2,
    evaluate: bool = False,
    judge_model: str = "gpt-4o",
) -> dict[str, Any]:
    items = [
        record for record in read_records(vqa_path)
        if clean_text(record.get("task_layer")) == "salesbench_qa"
    ]
    _validate_public_items(items)
    if max_samples:
        items = items[:max_samples]

    client = VLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=300,
        rate_limit_rpm=0,
        disable_thinking=True,
    )
    summary = run_salesbench_qa_baseline_records(
        items=items,
        video_lookup=_video_lookup(config),
        output_dir=output_dir,
        model=model,
        client=client,
        max_workers=max_workers,
    )

    if evaluate:
        answers_path = Path(summary["outputs"]["answers"])
        evaluation_dir = output_dir / "evaluation"
        eval_report = evaluate_salesbench_qa_files(
            gold_path=vqa_path,
            answers_path=answers_path,
            output_dir=evaluation_dir,
            api_key=api_key,
            judge_model=judge_model,
            base_url=base_url,
            max_workers=max_workers,
        )
        details_path = Path(eval_report["outputs"]["judge_details"])
        details = read_records(details_path)
        analysis_text = build_analysis_markdown(eval_report, details, model=model)
        analysis_path = evaluation_dir / f"{answers_path.stem}_analysis.md"
        analysis_path.write_text(analysis_text, encoding="utf-8")
        summary["evaluation"] = eval_report
        summary["outputs"]["analysis"] = str(analysis_path)

    return summary
