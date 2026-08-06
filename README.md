# SalesBench

SalesBench 是面向营销短视频的 Evidence-First 多模态 VQA benchmark。当前正式范围只有四个公开任务集；互动指标只用于独立分析实验，不参与 VQA 标注、模型输入、Judge 或排行榜。

## Benchmark 范围

| 任务 | 评测内容 | 允许回答 | 禁止回答 |
| --- | --- | --- | --- |
| BP | Basic Perception | 产品、人物、动作、画面文字、口播事实 | 互动效果、销量、传播效果 |
| CM | Cross-Modal Verification | ASR 与画面/OCR 是否一致、互补或冲突 | 根据标题或互动量推断内容真假 |
| SS | Selling Strategy Reasoning | 如何展示卖点、制造行动提示、组织说服结构 | 为什么一定会高赞、预计收藏量 |
| AE | Audience-Need Alignment | 面向什么需求、场景和决策障碍 | 依据粉丝画像或互动量推断真实转化 |

主榜分数是 BP、CM、SS、AE 四任务分数的宏平均；微平均仅作辅助统计。缺失答案在本地直接计 0 分。

## 数据边界

公开模型的统一输入只有：

- 16 帧 `hook_plus_uniform` 视频采样；
- ASR/字幕文本；
- 当前问题。

内部 C1-C6 是证据生产阶段的资产盘点层：视觉、语音、文本、发布上下文、跨模态信息和原始素材。它们不等于标准答案，也不直接进入公开 benchmark。只有能回指到视频帧、画面 OCR 或 ASR 的内容才能成为 `EvidenceUnit`；标题、粉丝量和互动量不能作为直接证据。

```text
C1-C6 内部资产
  -> EvidenceUnit（visual / OCR / ASR）
  -> GroundedAnnotation（BP / CM / SS / AE）
  -> 确定性问题编译
  -> vqa_public.jsonl + vqa_gold_private.jsonl
  -> 模型 predictions.jsonl
  -> 四任务 Judge 与宏平均

互动快照 -> 独立私有分析（不进入上述链路）
```

核心质量状态是 `DIRECT`、`INFERRED`、`NEEDS_REVIEW`、`REJECTED`。BP/CM 的合格项由直接证据支撑；SS/AE 是受控推理，必须保留证据引用。低置信度、证据冲突和语义歧义进入人工复核，不能自动编译成 VQA。

## 快速开始

项目要求 Python 3.13+。核心流程只使用标准库；调用 OpenAI-compatible 模型时安装 provider 依赖：

```bash
python -m pip install -e '.[providers]'
```

准备内部数据与 C1-C6 资产：

```bash
python salesbench.py prepare --config configs/benchmark_v1.json
python salesbench.py build-inputs --config configs/benchmark_v1.json
```

选择或使用固定 Evidence cohort：

```bash
python salesbench.py select-evidence-cohort \
  --config configs/benchmark_v1.json \
  --total 128 \
  --seed 42 \
  --output configs/evidence_alpha_128videos.json
```

构建 EvidenceDataset。API key 通过环境变量或命令行注入，不写入配置文件：

```bash
export OPENAI_API_KEY="..."

python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_pilot_5videos.json \
  --output-dir outputs/evidence/v2_pilot \
  --model gpt-4o
```

主要产物包括 `evidence_units.jsonl`、`video_evidence_dataset.jsonl`、`human_review_queue.jsonl` 和 `generation_meta.json`。运行支持断点续跑；缓存指纹包含视频哈希、采帧策略、帧数、模型和契约版本。

应用人工复核并编译四任务 VQA：

```bash
python salesbench.py apply-evidence-reviews \
  --evidence-dir outputs/evidence/v2_pilot \
  --decisions outputs/evidence/v2_pilot/human_review_decisions.jsonl \
  --output outputs/evidence/v2_pilot/video_evidence_dataset_reviewed.jsonl

python salesbench.py compile-vqa \
  --evidence-dir outputs/evidence/v2_pilot \
  --output-dir outputs/vqa/v2_pilot
```

正式编译要求 BP、CM、SS、AE 均非空。仅调试 pilot 时可显式传入 `--allow-missing-tasks`。公开文件位于 `public/{bp,cm,ss,ae}.jsonl`；标准答案与证据上下文只保存在 `vqa_gold_private.jsonl`。

运行模型和评测：

```bash
python salesbench.py run-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --vqa outputs/vqa/v2_pilot/vqa_public.jsonl \
  --output-dir outputs/vqa/v2_pilot/run \
  --model gpt-4o

python salesbench.py evaluate-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --gold outputs/vqa/v2_pilot/vqa_gold_private.jsonl \
  --predictions outputs/vqa/v2_pilot/run/predictions.jsonl \
  --output-dir outputs/vqa/v2_pilot/evaluation
```

模型提交格式为 JSONL，每条至少包含：

```json
{"vqa_id": "v1_bp_00001", "answer": "画面中展示了黄色包装。"}
```

## 独立互动分析

互动分析处理点赞、评论、分享、收藏等观测期不统一的快照指标，只报告描述性关联、分层诊断和 bootstrap 置信区间。它不产生标准答案，不推断因果，不进入公开模型输入，也不影响排行榜。

```bash
python salesbench.py analyze-interactions \
  --config configs/benchmark_v1.json \
  --judge-details outputs/vqa/v2_pilot/evaluation/predictions_judge_details.jsonl \
  --output-dir outputs/analysis/interactions
```

## 验证

```bash
pytest -q
python -m compileall -q src
```

旧的数值预测与 question-first 实验结果仅作为历史产物保存在 `outputs/legacy/`，不代表当前 benchmark 协议。
