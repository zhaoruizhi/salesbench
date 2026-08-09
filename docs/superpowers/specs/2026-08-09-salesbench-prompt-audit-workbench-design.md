# SalesBench Prompt 与人工审计工作台设计

## 目标

为 64 视频 GPT-4o pilot 生成一个本地、单文件、可交互 HTML 工作台，回答三个问题：

1. 哪些产物属于正式 pilot，哪些只是 smoke；
2. 当前 Evidence Extractor、Proposer、Challenger、Adjudicator 与 Judge 实际使用了什么 Prompt；
3. Evidence、QA 和 LLM-as-Judge 分别需要人工审核什么，当前数据中最值得优先修改的问题是什么。

工作台只读取本地运行产物，不调用模型，不修改 Gold，不向公网发送任何数据。包含实际视频内容的生成 HTML 放在被 Git 忽略的 `outputs/audit/` 和当前任务可视化目录；仓库只提交生成器、配置、测试和审核协议，避免把私有样本内容推到 GitHub。

## 正式与 Smoke 的物理边界

生成器将现有结果非破坏性地复制到两个独立根目录：

- `outputs/formal/pilot64_gpt4o_v6/`
  - `evidence/`：64 视频 EvidenceDataset 及审核队列；
  - `qa/`：确定性编译出的公开 QA、私有 Gold 和编译记录；
  - `model_run/`：完整 384 题 GPT-4o predictions；
  - `evaluation/`：完整 384 题 Judge 结果；
  - `private_analysis/`：互动诊断，明确不属于 benchmark。
- `outputs/smoke/gpt4o_v6/`
  - `evidence/`：5 视频 smoke；
  - `qa/`：5 视频 smoke 编译结果；
  - `model_run/`：8 题调用 smoke；
  - `evaluation/`：对应评估 smoke。

原目录保留，作为运行来源和兼容路径；正式交付索引只指向新的分层目录。

## 最终交付状态

最终 benchmark 交付由四层构成，但当前 64 视频结果在人工审核前均属于 pilot draft：

1. `EvidenceDataset`：冻结后的 EvidenceUnit 与 GroundedAnnotation，是唯一 Gold 来源；
2. `Prompt manifest`：Prompt 原文、版本、模型与流水线指纹，用于复现，不作为模型输入；
3. `QA`：由已冻结 EvidenceDataset 确定性编译出的公开 QA 与私有 Gold；
4. `Evaluation`：predictions、Judge 明细、聚合指标和人工校准报告。

互动分析始终作为第五个独立、私有、非排行榜研究产物。

## 人工审核职责

人工审核不是在 Evidence 和 QA 之间二选一，而是依次执行三层质量门：

### Gate E：Evidence 与 GroundedAnnotation（主要审核）

- 审核全部 `human_review_queue`；
- 审核全部 `INFERRED` 标注；
- 对 `DIRECT` 标注分层抽样，并全量审核自动规则标出的风险项；
- 核查帧/OCR/ASR 定位、证据是否真的支持结论、CM 是否真正跨模态、销售主张是否被误写为客观效果、任务子类型与字段结构是否匹配；
- 优先补查缺少 BP 或 CM 的 21 个视频。

Evidence 未冻结前，不得把下游 QA 称为正式 Gold。

### Gate Q：QA（下游审核）

- 在已审核 EvidenceDataset 上重新编译；
- 64 视频 pilot 对 384 题执行全量人工通读；
- 核查问题清晰度、唯一可答性、Gold 完整性、答案泄漏、模板重复和任务边界；
- QA 发现的问题必须回溯到 annotation 或模板，不得直接手工制造与 Evidence 脱节的答案。

### Gate J：Evaluation / Judge 校准

- 从四任务和五档分数中分层抽取至少 64 题；
- 两名人工评分者独立评分，分歧交由第三人裁决；
- 报告 exact agreement、相邻档 agreement、weighted kappa 和 MAE；
- Judge 未达到预设一致性门槛前，主分数只能称为 diagnostic；
- 被测模型与 Judge 同为 GPT-4o 的当前结果必须标记为 self-judge 风险。

## 工作台信息架构

单文件 HTML 包含五个视图：

1. **交付地图**：正式/Smoke 目录、四层交付和当前冻结状态；
2. **Prompt 浏览器**：逐阶段显示当前代码真实 Prompt、输入字段、已观察问题和下一版建议；
3. **Evidence 审计**：296 条队列、732 条自动接受标注中的规则风险、21 个缺任务视频；
4. **QA 审计**：384 题的任务分布、风险标记、问题/答案/证据摘要；
5. **Judge 审计**：总体与分任务指标、五档分数、低分样本、校准方案和发布门槛。

所有列表支持任务、风险、视频 ID 和关键词过滤。审核决定保存在浏览器 `localStorage`，可以导出 JSON；工作台本身不会写回 EvidenceDataset。

## 已观察的首批风险

工作台必须显式暴露而不是隐藏这些真实问题：

- AE `CONTENT_MOTIVATION` 中出现 `target.claim` 与 `gold_value.relation`，说明 proposer 的 subtype schema 约束不足；
- ASR 中的“淡化斑点痘印”等卖方声明被 BP 映射成产品客观“效果”，需要改成“口播声称”；
- 部分 OCR `text_span` 是 `[0, 50]` 一类坐标字符串而不是原文；
- ASR/OCR 时间定位大量为 `unavailable`，影响可复核性；
- 有些来源 proposal ID 只是 `1`、`2`，跨视频审计追踪性弱；
- `audit_before_review.json` 同时报告 296 条队列和 `human_review_rate = 0`，指标语义需要修正；
- Judge 使用与被测模型相同的 GPT-4o，且一套通用评分描述同时覆盖 BP/CM/SS/AE，尚未完成人工校准。

## 隐私与验收

- HTML 与复制后的运行结果不得包含 API key；
- 工作台不得读取 `.env`；
- 仓库提交不得包含实际私有 Evidence/QA/Judge 明细；
- 生成器必须从 Prompt builder 获取当前 Prompt，避免手工副本漂移；
- 生成器测试使用合成 fixture，验证正式/Smoke 物理分离、隐私字段清洗、审计计数和 HTML 交互契约；
- 最终运行 `pytest -q`、`compileall`、`git diff --check`，并在桌面宽度和移动宽度检查页面。
