# SalesBench-QA Evidence-First 设计

## 1. 项目定位

SalesBench-QA 是面向主播带货短视频的多模态大模型 benchmark。正式 benchmark 只测模型对视频内容的感知、跨模态核验、销售策略和受众需求的理解，不测销量、转化或互动量预测。

研究系统由两个评分独立的部分组成：

1. 公开 VQA benchmark：BP、CM、SS、AE 四个任务集。
2. 私有互动分析：描述点赞、评论、分享、收藏与 VQA 表现之间的关联，仅作为额外实验。

互动分析不产生第五类 VQA，不计入主榜，不进入模型和 Judge 输入，也不支持因果结论。

## 2. 四任务边界

| 任务 | 研究问题 | 允许范围 | 排除范围 |
| --- | --- | --- | --- |
| BP | 模型看见或听见了什么 | 产品、人物、动作、画面文字、口播事实、时序 | 互动效果、销量、传播效果 |
| CM | 不同模态如何对应 | ASR 与画面/OCR 的一致、互补、冲突、未展示 | 根据标题、粉丝或互动量判断真假 |
| SS | 视频如何组织说服 | 卖点展示、Hook、信任、顾虑处理、CTA、说服结构 | 高赞原因、收藏量或传播量预测 |
| AE | 内容回应什么需求 | 使用场景、受众需求、决策状态和障碍 | 粉丝画像推断、真实受众转化 |

这四类任务按推理深度递进，但彼此不是因果链。SS/AE 可以使用受控推理，结论必须由视频帧、OCR 或 ASR 支撑。

## 3. 六维上下文与 EvidenceDataset

C1-C6 是内部资产组织方式，不是六份答案，也不是公开模型的额外输入：

| 层 | 内部作用 | 能否直接成为证据 |
| --- | --- | --- |
| C1 视觉特征 | 帮助定位可观察画面 | 只有回指到具体帧时可以 |
| C2 语音/ASR | 提供口播内容与时间段 | 可以 |
| C3 文本语言 | 内部检索、审计和候选发现 | 标题不可以；画面 OCR 可以 |
| C4 发布情境 | cohort 分层和私有分析 | 不可以 |
| C5 跨模态特征 | 候选关系发现和质量审计 | 派生分数不可以，原始两侧证据可以 |
| C6 原始视频 | 采样 16 帧 | 可以 |

因此两者不重合：C1-C6 回答“有哪些内部资产可供查找”，EvidenceDataset 回答“哪些可定位证据和受控结论已经通过质量规则”。QA 再从 GroundedAnnotation 确定性编译，不直接从六维上下文自由生成答案。

## 4. 数据契约

核心对象：

- `EvidenceUnit`：来自 `visual`、`ocr` 或 `asr` 的最小可定位事实。
- `GroundedAnnotation`：任务标签、结构化答案、`evidence_refs` 和质量状态。
- `VideoEvidenceRecord`：单视频证据引用、标注集合、覆盖状态和观察范围。
- `EvidenceDataset`：按视频组织的完整标注集。

质量状态：

- `DIRECT`：可由直接证据判断，主要用于 BP/CM。
- `INFERRED`：受控推理且证据充分，主要用于 SS/AE。
- `NEEDS_REVIEW`：低置信度、歧义、冲突或证据不足。
- `REJECTED`：不能进入 QA 编译。

模型生成的质量标签不可信；最终状态由本地规则决定。Challenger 只可审核达到置信度阈值且真实存在的 proposal，未送审或被拒绝的 proposal 不能由 Adjudicator 放行。

## 5. Evidence-First 流水线

```text
内部 C1-C6 + 16 帧
  -> 客观 Evidence Extractor
  -> 本地 BP proposal + CM/SS/AE proposers
  -> Challenger
  -> Adjudicator
  -> 本地验证、去重、冲突检查、质量赋值
  -> VideoEvidenceRecord
  -> 人工复核
  -> 确定性 QA Compiler
```

关键防线：

- 直接证据只允许视频帧、画面 OCR、ASR。
- 互动量、粉丝量、标题和商品元数据不能作为公开答案依据。
- 低置信度和冲突项进入人工复核队列。
- 语义重复只保留一个规范项。
- 正式编译要求四任务均非空。

## 6. 公开评测协议

每题公开输入为 16 个采样帧、ASR/字幕与问题。公开 JSONL 不包含标准答案、证据引用、内部结构化特征或互动信息。

统一字段：

```json
{
  "vqa_id": "v1_bp_00001",
  "video_id": "v1",
  "task_layer": "salesbench_qa",
  "task_type": "BP",
  "task_subtype": "ACTION",
  "question": "视频中人物或画面对产品做了什么？",
  "answer_type": "open"
}
```

模型提交 `predictions.jsonl`，每条包含 `vqa_id` 和 `answer`。未知任务类型直接拒绝；缺失答案本地计 0。

Judge 仅接收问题、任务、参考答案、模型答案和对应证据上下文。评分为 `0`、`0.25`、`0.5`、`0.75`、`1.0`。主指标为四任务 Relaxed Accuracy 宏平均，微平均仅作辅助。

## 7. 数据划分与报告

数据按 creator 分组划分约 80/10/10，保证同一创作者不跨 train/validation/test。正式报告至少包含：

- 四任务样本数、质量状态和证据模态分布；
- 四任务单项结果、宏平均和微平均；
- 缺失答案数与 Judge 失败数；
- 按品类、时长、证据模态等内容属性的诊断切片；
- 人工抽检准确率、重复率、冲突率和审稿一致性。

## 8. 独立互动分析

当前互动字段没有曝光量、观看时长、销量、统一观测窗口或快照时间，因此只能作为描述性快照。允许的额外实验包括 log1p profile、Spearman 关联、匹配/分层比较和 bootstrap 区间；所有输出必须标记 `diagnostic_only`、`non_causal`、`included_in_leaderboard=false`。

旧的数值预测、question-first 和五任务方案只作为历史实验，不再用于当前结论。
