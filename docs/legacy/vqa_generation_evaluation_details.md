# VQA 生成逻辑与评估指标实现说明

本文完全按当前项目代码说明 `src/salesbench/vqa/` 的 VQA 生成、作答和评估逻辑。对应实现主要在：

- `src/salesbench/vqa/schema.py`
- `src/salesbench/vqa/sampler.py`
- `src/salesbench/vqa/effect_labels.py`
- `src/salesbench/vqa/composer.py`
- `src/salesbench/vqa/runner.py`
- `src/salesbench/vqa/evaluator.py`

旧的 `src/salesbench/baseline_multi_agent.py` 和 `src/salesbench/agents/orchestrator.py` 是 final-score baseline 预测链路，当前 VQA 模块没有复用它们的评分主逻辑。

## 1. VQA 总体数据流

当前 VQA 模块分成两条生成路径：

```text
路径 A：Weak Effect VQA（纯规则，可复现）

labels_v1_dimensions.jsonl
processed videos_v1.jsonl
        │
        ▼
select_video_samples()
        │
        ├── build_effect_single_items()
        │       └── effect_vqa_single.jsonl
        │
        └── build_effect_pairwise_items()
                └── effect_vqa_pairwise.jsonl


路径 B：Understanding VQA（LLM/VLM 多智能体生成）

video_samples.jsonl 或 processed videos_v1.jsonl
        │
        ▼
Evidence Agent
        ▼
Marketing Semantics Agent
        ▼
VQA Composer Agent
        ▼
QA Verifier Agent + 本地过滤
        ▼
understanding_vqa.jsonl


合并：

understanding_vqa.jsonl
effect_vqa_single.jsonl
effect_vqa_pairwise.jsonl
        │
        ▼
merge_vqa_gold()
        │
        ▼
vqa_gold.jsonl
```

`merge_vqa_gold()` 会按以下顺序读取文件：

1. `understanding_vqa.jsonl`
2. `effect_vqa_single.jsonl`
3. `effect_vqa_pairwise.jsonl`

然后按 `vqa_id` 排序，写出 `vqa_gold.jsonl`。

## 2. 统一 VQA Schema

所有 VQA item 都写成 JSONL，每行是一个 dict。核心字段如下：

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

模型作答时，`runner.py` 会调用 `public_vqa_item()` 移除这些 gold-only 字段：

```text
gold_answer
evidence
provenance_tier
human_review_status
```

因此模型只能看到题目、选项和视频内容，不会看到标准答案或证据字段。

### 2.1 A-E 等级映射

`schema.py` 中定义：

```text
options = ["E", "D", "C", "B", "A"]

E -> 1
D -> 2
C -> 3
B -> 4
A -> 5
```

这个映射用于 QWK 和 Spearman 等排序/等级指标。

## 3. Weak Effect VQA 生成逻辑

Weak Effect VQA 由 `build-vqa-effect-labels` 调用，核心实现是 `build_effect_vqa_dataset()`。

输入：

```text
outputs/processed/labels_v1_dimensions.jsonl
outputs/processed/videos_v1.jsonl
```

四个互动维度在代码中定义为：

| 维度 | single task_type | pairwise task_type | 标签字段 |
|------|------------------|--------------------|----------|
| 点赞 | `like_level` | `like_pairwise` | `engagement_score` |
| 评论 | `comment_level` | `comment_pairwise` | `discussion_score` |
| 转发 | `share_level` | `share_pairwise` | `virality_score` |
| 收藏 | `collect_level` | `collect_pairwise` | `conversion_score` |

### 3.1 视频抽样

抽样函数是 `select_video_samples()`。

流程：

1. 从 `videos_v1.jsonl` 中筛选有 `video_id` 的记录。
2. 如果传入 `allowed_video_ids`，只保留同时出现在 `labels_v1_dimensions.jsonl` 的视频。
3. 如果 `max_videos is None`、`max_videos <= 0` 或候选样本数不超过 `max_videos`，直接按 `video_id` 排序返回全部候选样本。
4. 否则按 `(product_bucket, fan_segment)` 分组。
5. 每组先按 `video_id` 排序，再用固定 `seed` shuffle。
6. 按排序后的 group key 做 round-robin 采样，直到达到 `max_videos`。
7. 最终结果再按 `video_id` 排序。

因此，默认 30 条 pilot 是“按品类和粉丝段轻量分层”的可复现样本。

### 3.2 Single Effect VQA

生成函数是 `build_effect_single_items()`。

它对每个入选视频生成 4 道题：

```text
like_level
comment_level
share_level
collect_level
```

每道题的 gold answer 来自对应互动维度的全局五分位分档。

#### 3.2.1 五分位阈值

阈值由 `compute_global_quintile_thresholds()` 计算。对每个 metric，取所有可解析数值：

```text
values_m = [score_i,m]
```

然后计算：

```text
q20 = percentile(values_m, 0.20)
q40 = percentile(values_m, 0.40)
q60 = percentile(values_m, 0.60)
q80 = percentile(values_m, 0.80)
```

`percentile()` 的插值方式来自 `utils.py`：

```text
ordered = sorted(values)
position = (n - 1) * p
lower = floor(position)
upper = ceil(position)
weight = position - lower

percentile = ordered[lower] + (ordered[upper] - ordered[lower]) * weight
```

如果 `values` 为空，返回 `0.0`；如果只有一个值，返回该值。

#### 3.2.2 A-E 分档规则

`grade_for_score()` 的规则是：

```text
if score >= q80: A
elif score >= q60: B
elif score >= q40: C
elif score >= q20: D
else: E
```

每条 single effect VQA 的 `evidence` 会记录：

```text
source = labels_v1_dimensions
metric = 当前互动维度字段
raw_value = 原始维度分数
binning_rule = global_quintile
thresholds = q20/q40/q60/q80
```

### 3.3 Pairwise Effect VQA

生成函数是 `build_pairwise_items()`，由 `build_effect_pairwise_items()` 包装调用。

对每个互动维度，代码枚举入选视频的所有两两组合 `(left, right)`，并筛选满足以下条件的 pair：

1. 两个视频都存在于 `videos_v1.jsonl` 和 `labels_v1_dimensions.jsonl`。
2. `product_bucket` 完全相同。
3. `fan_segment` 相同或相邻。
4. 对应维度分数都能解析成数值。
5. 分数差距满足：

```text
abs(score_left - score_right) >= pair_min_gap
```

默认 `pair_min_gap = 0.15`。

#### 3.3.1 fan_segment 相邻规则

代码使用 `FAN_SEGMENTS_LOW_TO_HIGH` 的顺序：

```text
尾部 -> 腰部 -> 头腰 -> 头部
```

如果两个 segment 都在这个顺序表中：

```text
compatible = abs(index_left - index_right) <= 1
```

如果不在顺序表中，则要求两个非空字符串完全相同。

#### 3.3.2 可复现采样

每个互动维度内部：

1. 候选 pair 先按 `(left_id, right_id)` 排序。
2. 使用 `random.Random(seed + dim_index)` shuffle。
3. 如果 `pairwise_per_dim > 0`，取前 `pairwise_per_dim` 个。
4. 被选中的 pair 再按 `(left_id, right_id)` 排序。
5. 按排序后的位置生成 `..._001`、`..._002` 这样的 `vqa_id`。

`dim_index` 来自 `EFFECT_DIMENSIONS` 的顺序：点赞、评论、转发、收藏。

#### 3.3.3 Pairwise gold answer

Pairwise 题固定两个选项：

```text
options = ["A", "B"]
```

其中 A 表示左视频，B 表示右视频。gold answer 规则：

```text
gold = "A" if score_left > score_right else "B"
```

由于候选 pair 已经要求分差 `>= pair_min_gap`，正常情况下不会出现平分。

## 4. Understanding VQA 多智能体生成架构

Understanding VQA 由 `build-vqa-understanding` 调用，核心实现是 `build_understanding_vqa_dataset()` 和 `_process_video()`。

这条链路每条视频最多生成 2-4 道理解题。代码会把 `questions_per_video` 限制到：

```text
question_count = max(2, min(question_count, 4))
```

### 4.1 样本来源

`build_understanding_vqa_dataset()` 会优先读取：

```text
outputs/vqa/v0_9/video_samples.jsonl
```

如果该文件存在，就沿用 effect VQA 阶段选出的样本，并按 `max_videos` 截断。

如果不存在，就从 `config.processed_main` 读取 `videos_v1.jsonl`，调用 `select_video_samples()` 重新抽样，并写出 `video_samples.jsonl`。

### 4.2 Agent 1: Evidence Agent

实现位置：`composer.py` 的 `EVIDENCE_SYSTEM_PROMPT` / `EVIDENCE_USER_PROMPT`。

输入：

- 8 帧视频关键帧，来自 `sample_frames(..., strategy="hook_plus_uniform", total_frames=8)`
- 视频标题 `title`
- 口播/字幕 `video_text`
- 商品标题 `product_title`

调用方式：

```text
VLM call，传入图片 + 文本，response_format=json_object
```

职责：

- 只提取能从画面、标题、口播和商品标题中看到或读到的证据。
- 不做营销效果预测。
- 不推断销售额、曝光、播放量或 GMV。

期望 JSON 字段：

```json
{
  "visible_facts": ["可见事实1", "可见事实2"],
  "product_display": "商品如何出现或展示",
  "first_screen_content": "首屏/前几帧内容",
  "text_evidence": ["来自标题或口播的证据短句"],
  "frame_evidence": ["来自画面的证据描述"],
  "unsupported_or_missing": ["无法从现有数据确认的信息"]
}
```

### 4.3 Agent 2: Marketing Semantics Agent

实现位置：`SEMANTICS_SYSTEM_PROMPT` / `SEMANTICS_USER_PROMPT`。

输入：

- Evidence Agent 的 JSON 输出
- 视频标题
- 口播/字幕
- 商品标题

调用方式：

```text
LLM text-only call，response_format=json_object
```

职责：

- 根据证据生成营销内容理解标签。
- 不预测互动效果。

期望 JSON 字段：

```json
{
  "value_proposition": ["功能效果"],
  "pain_points": ["用户痛点"],
  "trust_evidence": ["效果演示"],
  "persuasion_path": "视频如何说服用户",
  "content_type": "教程",
  "text_visual_consistency": "高度一致",
  "evidence_summary": "可以支持上述标签的证据摘要"
}
```

### 4.4 Agent 3: VQA Composer Agent

实现位置：`COMPOSER_SYSTEM_PROMPT` / `COMPOSER_USER_PROMPT`。

输入：

- Evidence Agent 输出
- Marketing Semantics Agent 输出
- 固定题型和固定 options

调用方式：

```text
LLM text-only call，response_format=json_object
```

职责：

- 把证据和营销语义组装为 VQA item。
- 只生成能被现有帧、标题、口播或商品信息回答的问题。
- 不生成销售额、GMV、曝光、播放量、真实 CTA 等无标签问题。

当前固定题型来自 `UNDERSTANDING_OPTIONS`：

| task_type | answer_type 默认值 | options |
|-----------|-------------------|---------|
| `value_proposition` | `multi_label` | 功能效果、省钱性价比、使用便利、情绪共鸣、知识教程、不清楚 |
| `trust_evidence` | `multi_label` | 品牌背书、专家经验、用户口碑、效果演示、价格承诺、无明显信任证据 |
| `content_type` | `single_choice` | 教程、测评、种草、剧情、直播切片、图文、其他 |
| `text_visual_consistency` | `single_choice` | 高度一致、基本一致、部分不一致、严重不一致 |

Composer 期望输出：

```json
{
  "items": [
    {
      "task_type": "value_proposition",
      "question": "这条视频主要通过什么卖点说服用户？",
      "answer_type": "multi_label",
      "options": ["功能效果", "省钱性价比", "使用便利", "情绪共鸣", "知识教程", "不清楚"],
      "gold_answer": ["功能效果"],
      "evidence": {
        "text_span": "支持答案的标题或口播短句",
        "frames": ["frame_000"]
      }
    }
  ]
}
```

### 4.5 Agent 4: QA Verifier Agent

实现位置：`VERIFIER_SYSTEM_PROMPT` / `VERIFIER_USER_PROMPT`。

输入：

- Evidence Agent 输出
- Composer 生成的候选 VQA items

调用方式：

```text
LLM text-only call，response_format=json_object
```

职责：

- 检查候选 VQA 是否有明确答案来源。
- 涉及销售额、GMV、曝光、播放量或真实 CTA 的问题一律拒绝。

期望输出：

```json
{
  "accepted_task_types": ["value_proposition"],
  "rejections": [
    {"task_type": "xxx", "reason": "拒绝原因"}
  ]
}
```

注意：当前代码使用的是 `accepted_task_types` 级别过滤。也就是说，如果 verifier 接受某个 `task_type`，该类型的候选题会进入下一步本地校验；代码没有把单条 rejection reason 写回 VQA item，但完整 agent 输出会保存在 `understanding_traces.jsonl`。

### 4.6 本地规范化与过滤

Verifier 后，代码调用 `_coerce_understanding_items()` 做本地规范化。

保留条件：

1. `task_type` 必须在 `UNDERSTANDING_OPTIONS` 中。
2. 如果 verifier 返回了 `accepted_task_types`，则 `task_type` 必须在其中。
3. `question` 不能为空。
4. `gold_answer` 必须存在。
5. `question` 不能包含以下禁用词：

```text
销售额
GMV
曝光
播放量
真实CTA
真实 CTA
```

6. `evidence` 必须存在且非空。

保留下来的 item 会被补齐标准字段：

```text
vqa_id = {video_id}_{task_type}_{序号}
task_layer = understanding
provenance_tier = Gold-B
human_review_status = auto_generated_pending_review
```

## 5. 模型作答逻辑

模型作答由 `run-vqa` 调用，核心实现是 `run_vqa_model()`。

输入：

```text
vqa_gold.jsonl
input/raw_video/video_index.jsonl
```

流程：

1. 读取 VQA items。
2. 如果 `--dry-run`，默认只取 5 条；如果设置 `--max-samples`，按文件顺序取前 N 条。
3. 对 single-video item，给模型提供该视频 8 帧、标题、口播/字幕、商品标题和 VQA 题目。
4. 对 pairwise item，给模型提供视频 A 和视频 B 的帧与文本信息。
5. 调用 `public_vqa_item()` 删除 gold-only 字段后再放入 prompt。
6. 要求模型输出 JSON：

```json
{
  "answer": "你的答案；多选题使用字符串数组",
  "confidence": 0.0,
  "reasoning": "一句话说明依据"
}
```

7. 结果写入：

```text
outputs/vqa/v0_9/model_answers/{model_name}.jsonl
```

答案字段解析时，代码会依次尝试：

```text
answer
pred_answer
model_answer
prediction
```

## 6. 评估数据匹配与答案归一化

评估由 `evaluate-vqa` 调用，核心实现是 `evaluate_vqa_records()`。

### 6.1 匹配规则

预测答案按 `vqa_id` 建索引：

```text
answer_lookup[vqa_id] = answer_record
```

只有 gold 中 `vqa_id` 能在答案文件中找到的样本才参与评估。

输出 summary 包含：

```text
gold_count
answer_count
matched_answer_count
skipped_gold_count
effect_ordinal_count
effect_pairwise_count
understanding_single_count
understanding_multi_label_count
```

### 6.2 答案归一化

`normalize_answer()` 的规则：

多标签题：

- 如果答案是 list 或 tuple，逐项 `strip` 并去掉空项。
- 如果答案是字符串，按以下分隔符之一切开：

```text
，
,
、
;
；
|
```

- 如果没有分隔符，则把整个字符串作为单标签列表。

非多标签题：

- 如果答案是 list，取第一个元素。
- 否则转为字符串并 `strip`。

## 7. 当前评估指标

当前 `evaluator.py` 输出以下指标：

```text
content_qa_accuracy
marketing_macro_f1
effect_ordinal_accuracy_4d
effect_ordinal_qwk_4d
effect_pairwise_accuracy_4d
metric_wise_spearman
```

如果按论文主表的“四个指标”理解，当前最核心的四个是：

```text
content_qa_accuracy
marketing_macro_f1
effect_ordinal_qwk_4d
effect_pairwise_accuracy_4d
```

代码还额外输出 `effect_ordinal_accuracy_4d` 和 `metric_wise_spearman`，用于辅助观察。

下面按代码实现逐一说明公式。

## 8. 指标 1：Content QA Accuracy

字段名：

```text
content_qa_accuracy
```

参与样本：

```text
task_layer == "understanding"
answer_type != "multi_label"
```

也就是 understanding 层的单选或普通答案题。当前固定题型中主要包括：

```text
content_type
text_visual_consistency
```

令参与样本集合为：

```text
U_single = {i | task_layer_i = understanding, answer_type_i != multi_label}
```

对每个样本，先做 `normalize_answer()`，再比较字符串：

```text
correct_i = 1[ clean_text(normalize(gold_i)) = clean_text(normalize(pred_i)) ]
```

公式：

```text
Content QA Accuracy = (Σ_i correct_i) / |U_single|
```

如果没有参与样本，代码返回 `null`。

含义：

这个指标衡量模型对明确单选型内容理解问题的准确率，例如内容类型判断、文字画面一致性判断。它不评价多标签营销语义题。

## 9. 指标 2：Marketing Macro-F1

字段名：

```text
marketing_macro_f1
```

参与样本：

```text
task_layer == "understanding"
answer_type == "multi_label"
```

当前固定题型中主要包括：

```text
value_proposition
trust_evidence
```

对每个多标签样本，代码先构造标签全集：

```text
L_i = options_i ∪ gold_i ∪ pred_i
```

空字符串会被去掉。

对每个标签 `l ∈ L_i`：

```text
TP_l = 1[l ∈ gold_i and l ∈ pred_i]
FP_l = 1[l ∉ gold_i and l ∈ pred_i]
FN_l = 1[l ∈ gold_i and l ∉ pred_i]
```

单标签 F1：

```text
F1_l = 0, if 2TP_l + FP_l + FN_l = 0
F1_l = 2TP_l / (2TP_l + FP_l + FN_l), otherwise
```

单个 VQA item 的 macro-F1：

```text
F1_i = mean_{l ∈ L_i}(F1_l)
```

整体指标：

```text
Marketing Macro-F1 = mean_i(F1_i)
```

如果没有多标签 understanding 样本，代码返回 `null`。

含义：

这个指标衡量模型是否能识别营销语义标签，例如卖点类型、信任证据类型。它不是简单的 exact match；漏选和多选都会通过 per-label F1 扣分。

## 10. 指标 3：Effect Ordinal Accuracy@4D

字段名：

```text
effect_ordinal_accuracy_4d
```

这是代码当前额外输出的辅助指标，虽然不一定是论文主表四指标之一，但它直接反映 A-E 等级题的命中率。

参与样本：

```text
task_layer == "weak_effect"
answer_type == "ordinal_5"
```

对应四类 single effect 题：

```text
like_level
comment_level
share_level
collect_level
```

对每个样本，做 exact match：

```text
correct_i = 1[ normalize(gold_i) = normalize(pred_i) ]
```

公式：

```text
Effect Ordinal Accuracy@4D = (Σ_i correct_i) / N_ordinal
```

如果没有 ordinal effect 样本，代码返回 `null`。

含义：

这个指标衡量模型是否精确猜中互动表现的 A-E 档位。它很严格：预测 B、真实 A 也算错，和预测 E 一样都只记 0。

## 11. 指标 4：Effect Ordinal QWK@4D

字段名：

```text
effect_ordinal_qwk_4d
```

参与样本：

```text
task_layer == "weak_effect"
answer_type == "ordinal_5"
```

代码先把 A-E 映射为 1-5：

```text
E = 1
D = 2
C = 3
B = 4
A = 5
```

无法映射的预测或 gold 不进入 QWK 的 `true_ordinal_scores` / `pred_ordinal_scores`。

### 11.1 观测矩阵

设等级集合为：

```text
R = {1, 2, 3, 4, 5}
```

代码构造观测矩阵 `O`：

```text
O_ab = count(true = a, pred = b)
```

同时构造真实等级直方图和预测等级直方图：

```text
T_a = count(true = a)
P_b = count(pred = b)
```

总数：

```text
N = Σ_a T_a
```

### 11.2 期望矩阵

代码没有显式保存完整期望矩阵，但计算等价于：

```text
E_ab = T_a * P_b / N
```

### 11.3 二次权重

代码内部用矩阵 index `i, j = 0..4`，权重：

```text
w_ij = (i - j)^2 / (K - 1)^2
```

其中：

```text
K = 5
```

等价地，如果用等级值 `a,b ∈ {1,2,3,4,5}` 表示：

```text
w_ab = (a - b)^2 / 16
```

### 11.4 QWK 公式

代码计算：

```text
observed_weighted = Σ_a Σ_b w_ab * O_ab
expected_weighted = Σ_a Σ_b w_ab * E_ab
```

最终：

```text
QWK = 1 - observed_weighted / expected_weighted
```

特殊情况：

- 如果没有有效 ordinal 样本，返回 `null`。
- 如果 `expected_weighted == 0` 且 `observed_weighted == 0`，返回 `1.0`。
- 如果 `expected_weighted == 0` 且 `observed_weighted != 0`，返回 `0.0`。

含义：

这个指标衡量模型预测的 A-E 等级与真实等级的接近程度。它比 exact accuracy 更适合等级任务，因为预测 B、真实 A 的惩罚小于预测 E、真实 A。值越高越好，`1.0` 表示完全一致；低于 0 表示比按边际分布随机匹配还差。

## 12. 指标 5：Effect Pairwise Accuracy@4D

字段名：

```text
effect_pairwise_accuracy_4d
```

参与样本：

```text
task_layer == "weak_effect"
task_type.endswith("_pairwise")
```

对应四类 pairwise effect 题：

```text
like_pairwise
comment_pairwise
share_pairwise
collect_pairwise
```

每题 gold answer 是 `"A"` 或 `"B"`，模型答案也按 exact match 判断：

```text
correct_i = 1[ normalize(gold_i) = normalize(pred_i) ]
```

公式：

```text
Effect Pairwise Accuracy@4D = (Σ_i correct_i) / N_pairwise
```

如果没有 pairwise effect 样本，代码返回 `null`。

含义：

这个指标衡量模型在两个同品类、相近粉丝段、且真实互动分差足够大的视频之间，能否判断哪一个互动表现更高。它避免要求模型预测绝对 A-E 档位，重点考察相对排序能力。

## 13. 附加指标：Metric-wise Spearman

字段名：

```text
metric_wise_spearman
```

这是代码当前输出的附加字典，不是单个主指标。

参与样本：

```text
task_layer == "weak_effect"
answer_type == "ordinal_5"
gold 和 pred 都能映射到 A-E 分数
```

代码按 `task_type` 分组，例如：

```text
like_level
comment_level
share_level
collect_level
```

每组至少有 2 条样本才计算。

计算方式来自 `utils.spearman_correlation()`：

1. 对 true scores 和 pred scores 分别做 average-rank。
2. 对两个 rank 序列计算 Pearson correlation。

公式可以写作：

```text
Spearman(task_type) = Pearson(rank_avg(true_scores), rank_avg(pred_scores))
```

其中 Pearson 的代码公式是：

```text
Pearson(x, y) =
Σ_i (x_i - mean(x))(y_i - mean(y))
/ sqrt(Σ_i (x_i - mean(x))^2) sqrt(Σ_i (y_i - mean(y))^2)
```

如果长度不一致或样本数小于 2，`spearman_correlation()` 返回 `0.0`；但 `metric_wise_spearman` 在进入计算前已经要求每组至少 2 条。

含义：

这个附加指标观察模型在某个互动维度内的等级排序是否和 gold 排序一致。它比 exact accuracy 更关注相对顺序。

## 14. 当前代码中的“主四指标”与输出指标关系

如果论文主表只放四个指标，按当前代码最自然对应为：

| 论文指标 | 当前 JSON 字段 | 说明 |
|----------|----------------|------|
| Content QA Accuracy | `content_qa_accuracy` | understanding 单选题准确率 |
| Marketing Macro-F1 | `marketing_macro_f1` | understanding 多标签题 per-item macro-F1 均值 |
| Effect Ordinal QWK@4D | `effect_ordinal_qwk_4d` | 四个互动等级题的 QWK |
| Effect Pairwise Accuracy@4D | `effect_pairwise_accuracy_4d` | 四个互动 pairwise 题准确率 |

当前代码另外输出：

| 附加字段 | 用途 |
|----------|------|
| `effect_ordinal_accuracy_4d` | 看 A-E 档位 exact hit rate |
| `metric_wise_spearman` | 看每个 single effect task_type 的等级排序相关 |

## 15. 解释指标时的注意事项

- `content_qa_accuracy` 只统计 understanding 层非多标签题；如果 `vqa_gold.jsonl` 里没有这类题，结果是 `null`。
- `marketing_macro_f1` 只统计 understanding 层多标签题；如果没有多标签题，结果是 `null`。
- `effect_ordinal_qwk_4d` 只使用可映射到 A-E 的 ordinal 样本；模型输出不在 A-E 中的题不会进入 QWK 的分数列表。
- `effect_pairwise_accuracy_4d` 只看 `_pairwise` 题，且是 A/B exact match。
- 当前 evaluator 不按四个互动维度分别求均值后再平均；它是把所有符合条件的题汇总在一起计算整体 accuracy/QWK/pairwise accuracy。
- `@4D` 在当前代码中表示题目集合来自四个互动维度，而不是评估时显式先按四维分别算分再取平均。
