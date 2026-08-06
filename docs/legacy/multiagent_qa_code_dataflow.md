# SalesBench Multi-Agent QA Code Data Flow

本文档从代码执行路径说明 `src/salesbench/multiagent/` 的 QA 生成数据流。它关注函数、对象、prompt payload、JSONL 产物之间的数据如何传递，而不是重复设计文档中的研究动机。

## 1. Top-Level Entry

命令入口在 `src/salesbench/cli.py`：

```bash
python salesbench.py build-multiagent-qa \
  --model gpt-4o \
  --max-videos 30 \
  --questions-per-video 5 \
  --proposals-per-perspective 3 \
  --max-workers 2 \
  --output-dir outputs/vqa/v1_0_multiagent
```

调用链：

```text
salesbench.py
  -> salesbench.cli.main()
  -> build_parser()
  -> build_multiagent_qa_command(args)
  -> multiagent.runner.build_multiagent_qa_dataset(...)
```

`build_multiagent_qa_command()` 做三件事：

1. `load_config()` 读取 `configs/benchmark_v1.json`。
2. 从 CLI 或环境变量读取 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。
3. 把参数传给 `build_multiagent_qa_dataset()`。

其中 `--questions-per-video` 控制最终每条视频保留多少个 QA，`--proposals-per-perspective` 控制 Stage 1 每个 Proposer 视角生成多少候选。两者解耦后，默认 30 条视频的 Stage 1 候选量约为 `30 × 3 perspectives × 3 = 270`，而最终仍目标保留 `30 × 5 = 150` 个 QA。

## 2. Runner-Level Data Flow

主入口是 `src/salesbench/multiagent/runner.py::build_multiagent_qa_dataset()`。

整体流程：

```text
BenchmarkConfig
  -> SalesBenchContextStore.from_config(config)
  -> selected_raw_records(max_videos=30)
  -> video_samples.jsonl
  -> ThreadPoolExecutor
  -> _process_video(video_record, store, pipeline)
  -> PipelineResult
  -> stage*.jsonl + salesbench_qa_gold.jsonl + generation_meta.json
```

### 2.1 Input Files Loaded

`SalesBenchContextStore.from_config()` 加载并按 `video_id` 建索引：

| Context domain | Source path from config | Stored as |
| --- | --- | --- |
| C6 raw video/text | `config.input_raw_video` | `raw_lookup` |
| C1 visual features | `config.input_visual_features` | `visual_lookup` |
| C2 audio/speech | `config.input_audio_speech_features` | `audio_lookup` |
| C3 text/language | `config.input_text_language_features` | `text_lookup` |
| C4 publish context | `config.input_publish_context_features` | `publish_lookup` |
| C5 cross-modal | `config.input_cross_modal_consistency_features` | `cross_lookup` |
| PD performance data | `config.processed_main` | `processed_lookup` |

### 2.2 Sampling

`SalesBenchContextStore.selected_raw_records()` 复用 `salesbench.vqa.sampler.select_video_samples()`：

```text
processed_records
  -> stratified deterministic sample by product_bucket/fan_segment
  -> selected video_id list
  -> raw_lookup[video_id]
```

选中的 raw video records 立即写入：

```text
outputs/vqa/v1_0_multiagent/video_samples.jsonl
```

### 2.3 Per-Video Processing

每条视频走 `_process_video()`：

```text
video_record
  -> _sample_video_frames()
      -> sample_frames(...)
      -> frame_meta: [{frame_index, timestamp_s, path}]
      -> frames_b64: [base64 jpeg, ...]
  -> store.bundle_for_video(video_id, frames=frame_meta)
  -> pipeline.run_video(bundle, frames_b64)
```

`frame_meta` 写入 C6 上下文供 prompt 文本引用；`frames_b64` 只在需要 VLM 看图时作为 image blocks 传给 API。

## 3. Context Bundle And Firewall

上下文组装在 `src/salesbench/multiagent/context.py`。

核心对象是 `VideoContextBundle`：

```python
VideoContextBundle(
    video_id="...",
    content_context={
        "C1_visual": {...},
        "C2_audio_speech": {...},
        "C3_text_language": {...},
        "C4_publish_context": {...},
        "C5_cross_modal": {...},
        "C6_raw_video": {..., "frames": [...]},
    },
    performance_data={
        "likes": ...,
        "collects": ...,
        "shares": ...,
        "comments": ...,
    },
)
```

### 3.1 Performance Data Isolation

`build_context_bundle()` 使用两个 helper 隔离数据：

| Helper | Role |
| --- | --- |
| `_without_performance_keys(record)` | 从内容域记录中删除 `likes/collects/shares/comments/performance_data` |
| `_performance_data(record)` | 只从 `processed_record` 抽取 `likes/collects/shares/comments` |

因此：

- `content_context` 永远不应包含真实互动指标。
- `performance_data` 只在 Stage 4 calibration 使用。

### 3.2 Task Routing

`CONTEXT_ROUTES` 是硬编码路由表：

| Task | `route_context()` visible domains |
| --- | --- |
| BP | C1, C2, C6 |
| CM | C2, C3, C5, C6 |
| SS | C2, C3, C4, C6 |
| AE | C3, C4, C6 |
| PA | C1, C2, C3, C4, C5 |

两类上下文读取函数：

```text
all_content_context(bundle, include_video=True)
  -> Stage 1/2 agents get content domains only, no PD

route_context(bundle, task_type)
  -> Stage 3 Answerer/Verifier get task-specific visible domains only
```

PA 路由特别重要：它不包含 C6 raw video，也不包含 PD。

## 4. Pipeline Objects

Schema 在 `src/salesbench/multiagent/schema.py`。

主要 dataclass：

| Class | Created in | Purpose |
| --- | --- | --- |
| `CandidateQuestion` | Stage 1 | 三个 Proposer 输出的候选问题 |
| `ReviewResult` | Stage 2 Challenger | 对候选问题的 PASS/REVISE/REJECT |
| `SelectedQuestion` | Stage 2 Synthesizer | 最终进入 Answerer 的问题 |
| `AgentCallTrace` | 每次 API call | 记录 raw response、parsed output、tokens、cost |
| `PipelineResult` | `run_video()` return | 汇总单视频所有阶段产物 |

Task and role enums：

```python
TaskType = BP | CM | SS | AE | PA
Perspective = consumer | operator | strategist
ReviewStatus = PASS | REVISE | REJECT
```

## 5. API Call Wrapper

所有 agent API 调用经过 `pipeline.py::_call_agent()`：

```text
_call_agent(client, stage, agent_name, system_prompt, user_text, frames_b64?)
  -> if frames_b64:
       client.call(system_prompt, user_content=[images..., text], response_format="json_object")
     else:
       client.call_text_only(system_prompt, user_text, response_format="json_object")
  -> _try_parse_json(raw_response)
  -> AgentCallTrace
```

VLM image payload 由 `_image_content()` 构造：

```json
[
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}},
  {"type": "text", "text": "...prompt..."}
]
```

## 6. Stage 1: Multi-Perspective Proposal

Function:

```text
MultiAgentQAPipeline._stage1_propose(bundle, frames_b64, traces)
```

Input:

```text
bundle
  -> all_content_context(bundle)
frames_b64
  -> passed to VLM proposer calls
```

Agent loop:

```text
for Perspective in [consumer, operator, strategist]:
  build_proposer_system_prompt(perspective)
  build_proposer_user_prompt(video_id, all_content_context, proposals_per_perspective)
  _call_agent(vlm_client, "stage1_proposal", f"proposer_{perspective}", ...)
  parsed["candidates"]
  -> _coerce_candidate(...)
  -> CandidateQuestion.to_dict()
```

Expected LLM JSON:

```json
{
  "candidates": [
    {
      "task_type": "BP",
      "question": "...",
      "answer_type": "open",
      "evidence_sources": ["C1_visual", "C6_raw_video"],
      "rationale": "..."
    }
  ]
}
```

Output appended to:

```text
PipelineResult.stage1_candidates
outputs/vqa/v1_0_multiagent/stage1_candidates.jsonl
```

Candidate ids are generated locally:

```text
{video_id}_{perspective}_{task_type}_{index:03d}
```

## 7. Stage 2A: Challenger Review

Function:

```text
MultiAgentQAPipeline._stage2_challenge(bundle, candidates, traces)
```

Input:

```text
candidates from Stage 1
all_content_context(bundle)
```

Call:

```text
_call_agent(
  llm_client,
  stage="stage2_review",
  agent_name="challenger",
  system_prompt=CHALLENGER_SYSTEM_PROMPT,
  user_text=build_challenger_user_prompt(...)
)
```

Expected LLM JSON:

```json
{
  "reviews": [
    {
      "question_id": "...",
      "status": "PASS",
      "reason": "...",
      "suggested_revision": "...",
      "corrected_task_type": "BP"
    }
  ]
}
```

Parsing:

```text
parsed["reviews"]
  -> _coerce_review()
  -> ReviewResult.to_dict()
```

Fallback:

If no parseable reviews exist, `_fallback_reviews()` marks all candidates `PASS` with a local fallback reason. This keeps the pipeline runnable even when a review call fails or returns malformed JSON.

Output appended to:

```text
PipelineResult.stage2_reviews
outputs/vqa/v1_0_multiagent/stage2_reviews.jsonl
```

## 8. Stage 2B: Synthesizer Selection

Function:

```text
MultiAgentQAPipeline._stage2_synthesize(bundle, candidates, reviews, traces)
```

Input:

```text
Stage 1 candidates
Stage 2 Challenger reviews
questions_per_video
```

Call:

```text
_call_agent(
  llm_client,
  stage="stage2_synthesis",
  agent_name="synthesizer",
  system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
  user_text=build_synthesizer_user_prompt(...)
)
```

Expected LLM JSON:

```json
{
  "selected_questions": [
    {
      "source_question_id": "...",
      "task_type": "PA",
      "question": "...",
      "answer_type": "open",
      "evidence_sources": ["C1_visual", "C2_audio_speech"],
      "selection_reason": "按任务类型比较后的选择理由"
    }
  ]
}
```

Parsing:

```text
parsed["selected_questions"]
  -> _coerce_selected(raw, video_id, index, candidate_lookup)
  -> SelectedQuestion
```

Fallback:

`_fallback_selected()`:

1. Drops candidates whose review status is `REJECT`.
2. Tries to pick one candidate per `TaskType` in BP/CM/SS/AE/PA order.
3. Fills remaining slots with other accepted candidates.
4. Truncates to `questions_per_video`.

Output appended to:

```text
PipelineResult.stage2_selected_questions
outputs/vqa/v1_0_multiagent/stage2_selected_questions.jsonl
```

Selected ids are generated locally:

```text
{video_id}_{task_type}_{index:03d}
```

## 9. Stage 3: Answer And Verify

Function:

```text
MultiAgentQAPipeline._stage3_answer_and_verify(bundle, selected, frames_b64, traces)
```

Loop:

```text
for selected question:
  routed = route_context(bundle, question.task_type)
  for round in 1..max_verify_rounds:
    Answerer generates QA item
    Verifier checks QA item
    PASS -> append to verified_items and stop retry
    REJECT -> stop retry and drop item
    REVISE/other -> pass verifier feedback into next Answerer round
```

### 9.1 Answerer Model Selection

```python
answer_context_has_video = question.task_type != TaskType.PA and "C6_raw_video" in routed
```

If `answer_context_has_video`:

```text
client = vlm_client
frames_b64 passed to _call_agent()
```

Otherwise:

```text
client = llm_client
text-only call
```

This means PA is always text-only because PA never sees C6.

### 9.2 Answerer Input

```text
build_answerer_user_prompt(question.to_dict(), routed_context, feedback)
```

`routed_context` is task-specific:

```text
BP -> C1/C2/C6
CM -> C2/C3/C5/C6
SS -> C2/C3/C4/C6
AE -> C3/C4/C6
PA -> C1/C2/C3/C4/C5
```

Expected Answerer JSON:

```json
{
  "answer": "30-60字简要结论",
  "gold_answer": "可作为参考答案的结论",
  "evidence": {
    "visual": "...",
    "speech": "...",
    "text": "...",
    "context": "...",
    "cross_modal": "..."
  },
  "reasoning": "证据到结论的推理链"
}
```

### 9.3 QA Item Construction

`_build_qa_item(question, parsed, index)` creates:

```json
{
  "vqa_id": "{video_id}_{task_type_lower}_{index:03d}",
  "video_id": "...",
  "task_layer": "salesbench_qa",
  "task_type": "PA",
  "question": "...",
  "answer_type": "open",
  "gold_answer": "...",
  "answer": "...",
  "evidence": {...},
  "reasoning": "...",
  "source_question_id": "...",
  "evidence_sources": ["C1_visual"],
  "provenance_tier": "Gold-B",
  "human_review_status": "auto_generated_pending_review"
}
```

The duplicate `answer` field is kept for generation trace convenience, but it is hidden by `salesbench.vqa.schema.public_vqa_item()` before model-facing evaluation use.

Raw answer attempts go to:

```text
PipelineResult.stage3_qa_raw
outputs/vqa/v1_0_multiagent/stage3_qa_raw.jsonl
```

### 9.4 Verifier Input And Output

Verifier receives:

```text
question.to_dict()
qa_item
routed_context
```

Expected Verifier JSON:

```json
{
  "verdict": "PASS",
  "reason": "...",
  "issues": ["..."],
  "revision_advice": "..."
}
```

PASS behavior:

```text
qa_item["verification"] = parsed_verdict
verified_items.append(qa_item)
```

Verified QA goes to:

```text
PipelineResult.stage3_qa_verified
outputs/vqa/v1_0_multiagent/stage3_qa_verified.jsonl
```

## 10. Stage 4: PA Calibration

Function:

```text
MultiAgentQAPipeline._stage4_calibrate(bundle, verified_items, traces)
```

Branching:

```text
for verified item:
  if task_type != "PA":
    final_items.append(item)
  else:
    run Calibrator
    calibrate_pa_item(item, bundle.performance_data, parsed)
    accepted -> attach private metadata and append
    rejected -> record calibration only, drop from final_items
```

### 10.1 Calibrator Input

```text
build_calibrator_user_prompt(
  qa_item,
  all_content_context(bundle, include_video=False),
  bundle.performance_data
)
```

This is the only model prompt that receives:

```json
{
  "likes": ...,
  "collects": ...,
  "shares": ...,
  "comments": ...
}
```

The Calibrator sees C1-C5 content context only, not C6 video frames.

Expected Calibrator JSON:

```json
{
  "status": "ACCEPT",
  "reason": "...",
  "performance_labels": {
    "like": "低/中/高",
    "collect": "低/中/高",
    "share": "低/中/高",
    "comment": "低/中/高"
  }
}
```

### 10.2 Local Calibration Guard

`calibrate_pa_item()` applies deterministic checks after model output:

- If answer text says `收藏高于转发` but `collects <= shares`, reject with `collect_share_direction_conflict`.
- If answer text says `转发高于收藏` but `shares <= collects`, reject with `share_collect_direction_conflict`.
- If Calibrator JSON says `REJECT`, reject.
- Otherwise accept and add performance labels.

Important invariant:

```text
calibrate_pa_item() never rewrites question, answer, gold_answer, evidence, or reasoning.
```

Accepted PA items receive private fields:

```json
{
  "performance_metadata": {"likes": ..., "collects": ..., "shares": ..., "comments": ...},
  "calibration_metadata": {...}
}
```

Calibration records go to:

```text
PipelineResult.stage4_calibration
outputs/vqa/v1_0_multiagent/stage4_calibration.jsonl
```

Final accepted QA goes to:

```text
PipelineResult.final_items
outputs/vqa/v1_0_multiagent/salesbench_qa_gold.jsonl
outputs/vqa/v1_0_multiagent/vqa_gold.jsonl
```

## 11. Trace Data Flow

Every `_call_agent()` returns an `AgentCallTrace`.

Trace fields:

```json
{
  "stage": "stage3_answer",
  "agent_name": "answerer",
  "success": true,
  "raw_response": "...",
  "parsed_output": {...},
  "model": "gpt-4o",
  "input_tokens": 0,
  "output_tokens": 0,
  "latency_s": 0.0,
  "cost_usd": 0.0,
  "error": null
}
```

Per-video traces are packed by:

```text
PipelineResult.to_trace_record()
```

Then runner writes:

```text
outputs/vqa/v1_0_multiagent/agent_traces.jsonl
```

`generation_meta.json` aggregates:

- sample count
- failed video count
- stage counts
- final QA count
- total cost
- elapsed time
- output paths

## 12. Model-Facing Public Item Flow

Generated QA items are gold records. Before using them as questions for a model, existing VQA runner code should call:

```python
salesbench.vqa.schema.public_vqa_item(item)
```

`GOLD_KEYS` hides:

```python
{
    "gold_answer",
    "answer",
    "evidence",
    "provenance_tier",
    "human_review_status",
    "performance_metadata",
    "private_metadata",
    "calibration_metadata",
}
```

Therefore model-facing items include the question and options, but not:

- reference answers
- evidence chains
- human review labels
- PA private performance data
- calibration metadata

## 13. End-To-End Data Flow Diagram

```text
CLI args/env
  |
  v
build_multiagent_qa_command()
  |
  v
build_multiagent_qa_dataset()
  |
  +--> SalesBenchContextStore.from_config()
  |      |
  |      +--> input/raw_video/video_index.jsonl
  |      +--> input/visual_features/visual_features.jsonl
  |      +--> input/audio_speech/audio_speech_features.jsonl
  |      +--> input/text_language/text_language_features.jsonl
  |      +--> input/publish_context/publish_context_features.jsonl
  |      +--> input/cross_modal_consistency/cross_modal_consistency_features.jsonl
  |      +--> outputs/processed/videos_v1.jsonl
  |
  +--> selected_raw_records()
  |      |
  |      v
  |    video_samples.jsonl
  |
  +--> _process_video()
         |
         +--> sample_frames() -> frames_b64 + frame_meta
         |
         +--> bundle_for_video()
         |      |
         |      +--> content_context C1-C6
         |      +--> isolated performance_data PD
         |
         +--> MultiAgentQAPipeline.run_video()
                |
                +--> Stage 1 Proposers
                |      -> stage1_candidates.jsonl
                |
                +--> Stage 2 Challenger
                |      -> stage2_reviews.jsonl
                |
                +--> Stage 2 Synthesizer
                |      -> stage2_selected_questions.jsonl
                |
                +--> Stage 3 Answerer/Verifier
                |      -> stage3_qa_raw.jsonl
                |      -> stage3_qa_verified.jsonl
                |
                +--> Stage 4 Calibrator for PA only
                |      -> stage4_calibration.jsonl
                |
                +--> final_items
                       -> salesbench_qa_gold.jsonl
                       -> vqa_gold.jsonl
                       -> agent_traces.jsonl
                       -> generation_meta.json
```

## 14. Main Invariants

1. Stage 1-3 never receive `likes`, `collects`, `shares`, or `comments`.
2. PA never receives C6 raw video frames during answer generation.
3. Stage 4 sees PD only for PA filtering.
4. Stage 4 never rewrites QA text.
5. `public_vqa_item()` removes all reference-answer and private metadata fields before model-facing use.
6. If Challenger or Synthesizer returns malformed JSON, local fallback keeps the pipeline runnable while preserving trace errors.
