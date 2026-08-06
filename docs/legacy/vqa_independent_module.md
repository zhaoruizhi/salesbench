# Marketing Effect VQA 独立模块说明

## 1. 模块定位

`src/salesbench/vqa/` 是一套独立的 VQA 数据构造、模型作答和评估模块。它不复用旧的 `baseline_multi_agent.py` final-score 预测主流程，也不要求模型输出 `final_pred_score`。

这个模块的目标是把 SalesBench 从“预测一个综合 final score”扩展为“回答营销内容理解与弱效果判断问题”：

- **Understanding VQA**：模型是否理解视频中的商品、卖点、信任证据、内容类型和文本-画面一致性。
- **Weak Effect VQA**：模型是否能判断点赞、评论、转发、收藏四个互动代理指标的相对表现。

四个互动指标只代表弱效果代理标签，不代表真实销售额、GMV、曝光或播放量。

## 2. 文件结构

```text
src/salesbench/vqa/
├── __init__.py
├── schema.py         # VQA schema、A-E 等级映射、答案规范化
├── effect_labels.py  # 单视频 effect VQA 生成、gold 合并
├── sampler.py        # pilot 视频抽样、pairwise 构造
├── composer.py       # understanding VQA 的多 agent 生成链
├── runner.py         # 模型回答 VQA
└── evaluator.py      # VQA 评估指标
```

CLI 入口仍然是仓库根目录的 `salesbench.py`：

```bash
python salesbench.py --help
```

新增命令：

```text
build-vqa-effect-labels
build-vqa-understanding
run-vqa
evaluate-vqa
```

## 3. VQA Item Schema

所有 VQA item 使用统一 JSONL schema：

```json
{
  "vqa_id": "735xxx_like_level_001",
  "video_id": "735xxx",
  "task_layer": "weak_effect",
  "task_type": "like_level",
  "question": "这条视频的点赞表现相对同类样本属于哪个水平？ 请从 E、D、C、B、A 五档中选择。",
  "answer_type": "ordinal_5",
  "options": ["E", "D", "C", "B", "A"],
  "gold_answer": "B",
  "evidence": {
    "source": "labels_v1_dimensions",
    "metric": "engagement_score",
    "raw_value": 0.68,
    "binning_rule": "global_quintile"
  },
  "provenance_tier": "WeakEffect-C",
  "human_review_status": "auto_derived"
}
```

模型作答时，`runner.py` 会隐藏 `gold_answer`、`evidence`、`provenance_tier`、`human_review_status`，只把题目、选项和视频输入给模型。

## 4. Effect VQA 生成逻辑

Effect VQA 是纯规则生成，不调用 LLM。

输入文件：

```text
outputs/processed/labels_v1_dimensions.jsonl
outputs/processed/videos_v1.jsonl
```

四个弱效果维度映射为：

| VQA 维度 | task_type | 标签字段 |
|----------|-----------|----------|
| 点赞 | `like_level` / `like_pairwise` | `engagement_score` |
| 评论 | `comment_level` / `comment_pairwise` | `discussion_score` |
| 转发 | `share_level` / `share_pairwise` | `virality_score` |
| 收藏 | `collect_level` / `collect_pairwise` | `conversion_score` |

### 4.1 Single Effect VQA

每条入选视频固定生成 4 个 single effect 问题：

- 点赞表现属于哪一档？
- 评论表现属于哪一档？
- 转发表现属于哪一档？
- 收藏表现属于哪一档？

每个维度按全量样本的全局五分位转成 A-E：

```text
A: >= q80
B: >= q60 and < q80
C: >= q40 and < q60
D: >= q20 and < q40
E: < q20
```

gold answer 直接来自 `labels_v1_dimensions.jsonl`，并在 `evidence` 中记录原始维度分数和分档阈值。

### 4.2 Pairwise Effect VQA

Pairwise effect VQA 在两个视频之间提问，例如：

```text
两条同类视频中，哪条点赞表现更高？ A：视频 xxx；B：视频 yyy。
```

构造条件：

- 同一个 `product_bucket`
- `fan_segment` 相同或相邻
- 对应维度分数差距 `>= pair_min_gap`，默认 `0.15`
- 每个维度最多采样 `pairwise_per_dim` 个 pair，默认 `100`
- 使用固定 seed，保证可复现

## 5. Understanding VQA 生成逻辑

Understanding VQA 由 `build-vqa-understanding` 调用 LLM/VLM 生成，采用独立的多 agent 链，不使用旧 `agents/orchestrator.py` 的 scoring pipeline。

生成链路：

```text
视频帧 + 标题 + 口播 + 商品标题
        │
        ▼
Evidence Agent
提取可见事实、商品展示、首屏内容、文本证据、画面证据
        │
        ▼
Marketing Semantics Agent
生成卖点、痛点、信任证据、说服路径、内容类型、一致性标签
        │
        ▼
VQA Composer Agent
按固定题型生成 2-4 个 understanding VQA item
        │
        ▼
QA Verifier Agent
过滤无证据或涉及销售额/GMV/曝光/播放量等不存在标签的问题
```

当前固定题型包括：

| task_type | answer_type | options |
|-----------|-------------|---------|
| `value_proposition` | `multi_label` | 功能效果、省钱性价比、使用便利、情绪共鸣、知识教程、不清楚 |
| `trust_evidence` | `multi_label` | 品牌背书、专家经验、用户口碑、效果演示、价格承诺、无明显信任证据 |
| `content_type` | `single_choice` | 教程、测评、种草、剧情、直播切片、图文、其他 |
| `text_visual_consistency` | `single_choice` | 高度一致、基本一致、部分不一致、严重不一致 |

Understanding VQA 的 `human_review_status` 默认是 `auto_generated_pending_review`，建议人工审核后再作为正式 gold label。

## 6. 输出目录

默认输出目录：

```text
outputs/vqa/v0_9/
```

主要文件：

```text
outputs/vqa/v0_9/
├── video_samples.jsonl
├── effect_vqa_single.jsonl
├── effect_vqa_pairwise.jsonl
├── understanding_vqa.jsonl
├── vqa_gold.jsonl
├── model_answers/
│   ├── {model_name}.jsonl
│   └── {model_name}_run_meta.json
└── evaluation/
    └── {model_name}.json
```

`vqa_gold.jsonl` 是合并文件。运行 effect 或 understanding 生成命令后，模块会把当前目录里已有的 VQA 文件重新合并成最新 gold。

## 7. 运行步骤

以下命令都在仓库根目录执行。

### Step 0: 准备四维标签

如果 `outputs/processed/labels_v1_dimensions.jsonl` 已存在，可以跳过这步。

```bash
python salesbench.py build-dimension-labels
```

### Step 1: 生成可复现 Effect VQA

默认构造 30 条 pilot 视频：

```bash
python salesbench.py build-vqa-effect-labels --max-videos 30
```

常用参数：

```bash
python salesbench.py build-vqa-effect-labels \
  --max-videos 30 \
  --seed 42 \
  --pair-min-gap 0.15 \
  --pairwise-per-dim 100
```

全量视频可以使用：

```bash
python salesbench.py build-vqa-effect-labels --max-videos 0
```

### Step 2: 生成 Pilot Understanding VQA

需要 API key：

```bash
python salesbench.py build-vqa-understanding \
  --api-key YOUR_API_KEY \
  --model gpt-4o \
  --max-videos 30 \
  --questions-per-video 4 \
  --max-workers 3
```

如果使用兼容 OpenAI API 的服务：

```bash
python salesbench.py build-vqa-understanding \
  --api-key YOUR_API_KEY \
  --base-url https://your-api-base/v1 \
  --model gpt-4o \
  --max-videos 30
```

也可以使用环境变量：

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://your-api-base/v1"
python salesbench.py build-vqa-understanding --model gpt-4o --max-videos 30
```

### Step 3: 让模型回答 VQA

默认读取：

```text
outputs/vqa/v0_9/vqa_gold.jsonl
```

Dry run：

```bash
python salesbench.py run-vqa \
  --api-key YOUR_API_KEY \
  --model gpt-4o \
  --dry-run
```

指定题数：

```bash
python salesbench.py run-vqa \
  --api-key YOUR_API_KEY \
  --model gpt-4o \
  --max-samples 50 \
  --max-workers 5
```

指定 gold 文件：

```bash
python salesbench.py run-vqa \
  --api-key YOUR_API_KEY \
  --model gpt-4o \
  --vqa-path outputs/vqa/v0_9/vqa_gold.jsonl
```

模型答案默认写到：

```text
outputs/vqa/v0_9/model_answers/{model_name}.jsonl
```

### Step 4: 评估 VQA 答案

```bash
python salesbench.py evaluate-vqa \
  --answers outputs/vqa/v0_9/model_answers/gpt-4o.jsonl
```

指定 gold 和输出路径：

```bash
python salesbench.py evaluate-vqa \
  --gold outputs/vqa/v0_9/vqa_gold.jsonl \
  --answers outputs/vqa/v0_9/model_answers/gpt-4o.jsonl \
  --output outputs/vqa/v0_9/evaluation/gpt-4o.json
```

## 8. 评估指标

`evaluate-vqa` 输出两层指标：

### Understanding Metrics

- `content_qa_accuracy`：单选理解题准确率。
- `marketing_macro_f1`：多标签营销理解题 Macro-F1。

### Weak Effect Metrics

- `effect_ordinal_accuracy_4d`：四个互动维度 A-E 等级题平均准确率。
- `effect_ordinal_qwk_4d`：四个互动维度等级题 Quadratic Weighted Kappa。
- `effect_pairwise_accuracy_4d`：四个互动维度 pairwise 题准确率。
- `metric_wise_spearman`：把 A-E 映射为 5-1 后，分维度计算排序相关。

如果当前 `vqa_gold.jsonl` 只有 effect 题，没有 understanding 题，understanding 指标会是 `null`。

## 9. 本地验证

运行标准库 unittest：

```bash
python -m unittest discover -s tests -v
```

验证 CLI 是否注册：

```bash
python salesbench.py --help
```

快速生成 effect pilot：

```bash
python salesbench.py build-vqa-effect-labels --max-videos 30
```

用 gold answer 构造一份 sanity-check 答案并评估：

```bash
python - <<'PY'
import json
from pathlib import Path

src = Path("outputs/vqa/v0_9/vqa_gold.jsonl")
out = Path("/private/tmp/vqa_answers_gold.jsonl")
with src.open(encoding="utf-8") as reader, out.open("w", encoding="utf-8") as writer:
    for line in reader:
        item = json.loads(line)
        writer.write(json.dumps({
            "vqa_id": item["vqa_id"],
            "answer": item["gold_answer"],
            "confidence": 1.0,
        }, ensure_ascii=False) + "\n")
print(out)
PY

python salesbench.py evaluate-vqa --answers /private/tmp/vqa_answers_gold.jsonl
```

在全对 sanity check 下，effect ordinal accuracy、QWK、pairwise accuracy 应该都是 `1.0`。

## 10. 注意事项

- `build-vqa-effect-labels` 不调用 LLM，结果可复现。
- `build-vqa-understanding` 和 `run-vqa` 会调用模型 API，可能产生费用。
- Understanding VQA 是 pilot 生成结果，正式论文实验前建议人工审核 `understanding_vqa.jsonl`。
- 模块不会构造销售额、GMV、曝光、播放量等强效果题，因为当前数据没有这些标签。
- 如果使用自定义输出目录，四个命令都可以传 `--output-dir` 保持同一套文件路径。
