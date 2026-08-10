# EvidenceDataset Data Card v3

## 状态与版本

当前 canonical 契约为 `evidence-dataset-schema-v3`。对应运行组件固定为：

- Evidence/Commerce Prompt：`evidence-prompt-v9`；
- Pipeline：`evidence-first-pipeline-v8`；
- QA Compiler：`evidence-qa-compiler-v5`；
- Judge：`judge-prompt-v4`；
- 中文审计翻译：`audit-translation-prompt-v1`。

v9 的 5 条 smoke 与 64 条 pilot 在人工完成 Evidence 和 QA 复核前均属于 `candidate_requires_human_review`，不能称为 frozen Gold。v2 Data Card 仅描述历史兼容契约。

## 研究目标

SalesBench 测量多模态模型对主播带货短视频中“可观察事实—商业表达线索—商业论证关系—任务答案”的理解能力。它不测真实消费者是否被说服，也不预测销量、转化或互动。

公开 benchmark 包含四个任务：

| 任务 | v3 能力边界 |
| --- | --- |
| BP | 商品身份、属性、规格、数量、套餐、价格、优惠条件、使用步骤、状态变化和场景 |
| CM | 口播/画面/OCR 共指与报价一致性、主张和展示的支持状态、重复证据、矛盾和时间错位 |
| SS | problem-solution、feature-benefit、演示、结果展示、价格锚定、异议处理、稀缺/紧迫和 CTA 顺序 |
| AE | 内容所表达的需求、场景、适配限制、质量/使用/价格/服务风险顾虑、决策障碍和 offer-need alignment |

## 四层数据模型

1. `EvidenceUnit`：只保存能够回指帧、OCR 或 ASR 时间段的原子事实。
2. `CommerceCue`：将事实归纳为商品、报价、展示、主张、需求、异议、可信线索或 CTA 等节点。
3. `CommercialRelation`：只使用冻结本体连接 Cue，例如 claim-demonstration、feature-benefit、problem-solution、offer-condition 和 content-before-CTA。
4. `GroundedAnnotation`：以 Capability、Reasoning Operator、Evidence/Cue/Relation 引用定义 BP/CM/SS/AE 的答案语义。

关系本体禁止 `INCREASES_TRUST`、`CAUSES_PURCHASE`、`IMPROVES_CONVERSION` 等消费者结果边。内容可以“回应一种顾虑”，但不能据此声称真实观众的顾虑被降低。

## 语言边界

- `content_en`、`rationale_en`、QuestionSpec、公开 QA、Gold Answer、模型答案和 Judge 输出均为英文 canonical 数据。
- 中文或其他语言的 ASR/OCR 逐字来源只保存在 `source_text_native`。
- 中文审计翻译只保存在独立 `audit_translations.jsonl`，记录源字段 SHA-256、翻译模型、Prompt 版本和 `audit_only=true`。
- 翻译不得写回 EvidenceDataset、QA、模型 payload 或 Judge payload；源 hash 变化后旧翻译必须标记 stale 或重新生成。

## 文件

| 文件 | 角色 |
| --- | --- |
| `evidence_units.jsonl` | 原子视觉/OCR/ASR 证据 |
| `commerce_cues.jsonl` | 商业表达线索节点 |
| `commercial_relations.jsonl` | 受控商业论证关系边 |
| `gold_proposals.jsonl` | 四任务候选及图引用 |
| `gold_reviews.jsonl` | Challenger 的逐项检查 |
| `human_review_queue.jsonl` | 低置信、冲突、歧义、解析失败和 abstention |
| `video_evidence_dataset.jsonl` | 自动规则通过的 candidate 主文件 |
| `video_evidence_dataset_reviewed.jsonl` | 仅在人审决定实际应用后产生 |
| `qa_specs.jsonl` | 任务、Capability、Operator 和图路径约束 |
| `qa_realizations.jsonl` | 自然英文问题实现 |
| `vqa_public.jsonl` | 被测模型输入，不含答案和私有依据 |
| `vqa_gold_private.jsonl` | Gold、英文 Evidence/Graph 摘要和溯源，供 Judge 使用 |
| `audit_translations.jsonl` | 独立中文审计 sidecar，不属于 benchmark |

## 输入、Gold 与私有分析边界

被测模型输入固定为采样帧、ASR/字幕和英文 question。标题、账号、粉丝数、互动量、商品后台元数据、C1/C5 派生分数、Gold、Evidence refs 和中文审计翻译不得进入公开输入。

Judge 可以读取英文 reference answer、Capability、Operator、Evidence Context 和 Commercial Graph Context，但不能读取 `source_text_native`、互动量、账号信息或翻译 sidecar。

点赞、评论、分享、收藏、粉丝数和发布日期只属于独立 `snapshot interaction profile`；只用于分层、盲化难例、评测后诊断和非因果关联分析，不进入主排行榜。

## 自动质量门

- Evidence 外键有效率：1.0；
- CommerceCue Evidence validity：1.0；
- CommercialRelation endpoint/evidence validity：1.0；
- Relation provenance validity：1.0；
- consumer-outcome causal relation count：0；
- canonical Chinese field count：0（`source_text_native` 除外）；
- 私有字段泄漏：0；
- unresolved conflict：0；
- Challenger 拒绝项重新进入 Gold：0；
- BP、CM、SS、AE 四任务均非空；
- 缺失模型答案确定性记 0；
- 主排行榜为四任务 macro-average，micro-average 仅为辅助指标。

## 人工质量门

Evidence 审核先于 QA 审核：

1. 检查 EvidenceUnit 是否准确、可定位且区分“主播声称”与“产品事实”。
2. 检查 CommerceCue 类型是否贴合内容，没有创造消费者结果。
3. 检查 CommercialRelation 的端点、时间关系和理论操作化是否成立。
4. 检查 GroundedAnnotation 的 target、Gold、Capability、Operator 与引用路径是否完整。
5. 再检查英文 QA 是否自然、不公式化、不泄漏答案、领域特异且可公平评分。

双语 HTML 默认显示中文审计翻译，同时必须展示英文 canonical 原文、原语言 ASR/OCR 和关联帧。翻译本身不是 Gold；任何语义分歧以英文 canonical 和视频证据为准。

## 已知局限

- 16 帧采样不能证明完整视频中某对象绝对不存在；NOT_DEMONSTRATED 需要完整观察窗口。
- ASR/OCR、帧定位和翻译都可能有误，必须在人工审核中回看来源。
- SS/AE 是内容约束的商业论证理解，不是真实消费者调查或因果效果估计。
- 单一 GPT-4o 生成器/被测模型/Judge 的 pilot 只验证流程和发现问题，不能替代多模型比较与 Judge 人工校准。
