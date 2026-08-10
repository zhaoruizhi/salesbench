# SalesBench 方案 B：商业论证图实施计划

> **供智能体执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项实施。所有执行步骤使用复选框（`- [ ]`）跟踪。

**目标：** 将 SalesBench 从通用的帧/ASR/OCR 问答生成框架，升级为面向主播带货短视频的 Evidence-First Benchmark，重点测量商品与报价还原、主张—证据验证、销售论证重建和需求—顾虑—商品匹配能力。

**架构：** 原始 `EvidenceUnit` 只保存可定位事实；在其上新增有营销理论依据的 `CommerceCue` 节点和可审计的 `CommercialRelation` 边；四个任务规划器基于同一张商业论证图生成语义 `QuestionSpec`。随后由独立英文 Question Realizer 生成自然问题，经本地规则、人工审核和冻结版 LLM-as-Judge 完成质量控制。运行数据和 Prompt 统一为英文；审计前单独生成中文翻译包，中文翻译只用于人工查看，不进入 Gold、模型输入或 Judge。

**技术栈：** Python 3.13、dataclass/Enum、JSON/JSONL、OpenAI-compatible 多模态与文本 API、pytest、本地 HTML/CSS/JavaScript 审计工作台、ffmpeg/ffprobe。

## 全局约束

- 公开任务仍固定为 `BP`、`CM`、`SS`、`AE`，兼容 ID 不改变。
- 四个公开名称分别为 Product & Offer Grounding、Claim–Evidence Verification、Persuasion & Sales Logic、Need–Objection–Offer Alignment。
- 所有运行时 System/User Prompt 必须为全英文，包括 Evidence、Cue、Relation、四任务、Question Realizer、Challenger、Adjudicator、Judge 和审计翻译 Prompt。
- `EvidenceUnit` 的规范语义字段使用英文；中文 ASR/OCR 必须在 `source_text_native` 中逐字保留。
- `CommerceCue`、`CommercialRelation`、`QuestionSpec`、问题、Gold Answer、Judge 原始输出均使用英文。
- 审计中文翻译保存在独立 `audit_translations.jsonl`，不得写回 EvidenceDataset 或 QA JSONL。
- 审计工作台默认显示中文翻译，同时必须提供英文原文和中文源文本的展开入口。
- 翻译结果不能成为 Evidence、关系判定、Gold、问题生成、Judge 或模型输入的依据。
- `EvidenceUnit` 只能包含可从帧、OCR 或 ASR 定位的事实；营销解释不能写回原子证据。
- `CommerceCue` 可以概括可观察的商业呈现内容，但必须引用同一视频的有效 `EvidenceUnit`。
- `CommercialRelation` 必须连接有效 Cue、引用底层 Evidence，并标记直接事实、受限推断或待审。
- 禁止 `INCREASES_TRUST`、`CAUSES_PURCHASE`、`IMPROVES_CONVERSION` 等消费者结果或因果关系。
- 互动量、粉丝数、标题、创作者元数据和私有分析字段不得进入 Evidence、QA、Gold、模型或 Judge payload。
- 不强制每个视频都生成四个任务；只在 cohort 层面平衡任务与能力。
- v9 主评分继续使用冻结的五档 LLM-as-Judge，不增加独立确定性评分器或 key-fact scorer。
- Smoke 与正式结果必须物理分离，不覆盖 v6/v7/v8 产物。
- 新版本固定为 `evidence-dataset-schema-v3`、`evidence-prompt-v9`、`evidence-first-pipeline-v8`、`evidence-qa-compiler-v5`、`judge-prompt-v4`、`audit-translation-prompt-v1`。
- 每个实施任务都必须包含失败测试、最小实现、通过测试和独立 Git 提交。

---

## 1. 语言与审计边界

### 1.1 各产物的规范语言

| 产物 | 运行/发布语言 | 审计显示 | 是否可参与 Gold/评分 |
| --- | --- | --- | --- |
| System/User Prompt | 英文 | 英文原文 + 中文翻译 | 英文原文参与运行；中文翻译不参与 |
| Visual Evidence 语义 | 英文 | 中文翻译为主 + 英文原文 | 英文参与 |
| ASR/OCR 原文 | 视频原语言，通常中文 | 中文原文 + 英文规范语义 | 原文用于定位，英文语义参与 |
| CommerceCue | 英文 | 中文翻译 + 英文原文 | 英文参与 |
| CommercialRelation | 英文 | 中文翻译 + 英文原文 | 英文参与 |
| QuestionSpec | 英文 | 中文翻译 + 英文原文 | 英文参与 |
| Public Question | 英文 | 中文问题 + 英文原题 | 英文参与模型评测 |
| Gold Answer | 英文 | 中文答案 + 英文原答案 | 英文参与 Judge |
| Judge 原始结果 | 英文 | 中文理由 + 英文原理由 | 英文分数/标签参与统计 |
| Audit translation | 中文 | 中文 | 永远不参与 |

### 1.2 为什么翻译必须独立保存

如果把中文翻译直接写入 EvidenceDataset 或 QA，后续很容易出现三个问题：

1. 翻译模型把“声称有效”改写成“产品有效”，导致事实边界污染。
2. 中英文数字、数量、优惠条件或否定词不一致，Gold 来源不唯一。
3. 被测模型或 Judge 可能意外看到中文翻译，形成额外输入或泄漏。

因此翻译采用只读旁路：

```text
canonical English artifacts
  -> audit translation runner
  -> audit_translations.jsonl
  -> audit workbench join by object_id + source_field + source_sha256
```

规范数据发生变化时，`source_sha256` 不再匹配，旧翻译必须显示为 stale，不能继续使用。

### 1.3 审计翻译记录格式

```json
{
  "translation_id": "audit_translation::evidence_unit::E001::content_en",
  "object_type": "evidence_unit",
  "object_id": "E001",
  "source_field": "content_en",
  "source_language": "en",
  "target_language": "zh-CN",
  "source_sha256": "6e2f5d8b4bd1",
  "translated_text": "主播将清洁剂喷在污渍上并进行擦拭。",
  "translation_method": "gpt-4o",
  "prompt_version": "audit-translation-prompt-v1",
  "audit_only": true
}
```

中文 ASR/OCR 原文直接展示，不需要再次翻译；其英文规范语义仍需要中文审计翻译，以便核对两者是否一致。

---

## 2. 文献来源与关系标签的性质

商业本体是基于文献进行的 benchmark 操作化，不是从某一篇论文逐字复制的标签集合。每个 Cue 和 Relation 必须记录以下来源类型之一：

- `THEORY_DIRECT`：文献直接研究该内容构念，例如商品描述、商品演示、替代试用、过程呈现、结果呈现和商品不确定性。
- `THEORY_OPERATIONALIZED`：SalesBench 将理论构念转为可审计的视频关系，例如“可见特征被口播解释为消费者利益”。
- `BENCHMARK_OPERATIONAL`：为测试跨模态或时间推理而设计，例如区分跨模态重复与独立视觉证明。

### 2.1 强相关领域文献

1. Guo 等，*Analyzing and Predicting Consumer Response to Short Videos in E-Commerce*，ACM TMIS 2024：在 23,001 个淘宝电商短视频中研究 product description、product demonstration、pleasure 和 aesthetics。<https://doi.org/10.1145/3690393>
2. *Process Reveal or Product Display?*，Journal of Retailing and Consumer Services 2026：区分短视频中的过程导向和结果导向商品呈现。<https://doi.org/10.1016/j.jretconser.2026.104827>
3. *The Effects of Mini-detail Short Videos on Consumer Purchase Intention on Taobao*，Entertainment Computing 2024：支持商品细节、功能、具体用法、应用场景、虚拟体验等标注。<https://doi.org/10.1016/j.entcom.2024.100745>
4. Lu 和 Chen，*Live Streaming Commerce and Consumers' Purchase Intention: An Uncertainty Reduction Perspective*，Information & Management 2021：提供替代试用、商品适配不确定性和商品/社交信号区分。<https://doi.org/10.1016/j.im.2021.103509>
5. *What Reduces Product Uncertainty in Live Streaming E-Commerce?*，Journal of Retailing and Consumer Services 2023：支持主播—商品、内容—商品信号一致性的理论来源。<https://doi.org/10.1016/j.jretconser.2023.103441>
6. *What Drives Taobao Live Streaming Commerce?*，Heliyon 2022：提供来源可信度、主播—商品契合和拟社会关系理论；SalesBench 只使用视频内可观察信号。<https://doi.org/10.1016/j.heliyon.2022.e09676>
7. *Let TikTokers Talk About Products*，Journal of Interactive Advertising 2026：区分短视频主播的体验口述和视觉商品演示。<https://doi.org/10.1080/15252019.2026.2642022>
8. E-VAds：提供电商短视频 BP、跨模态检测、营销逻辑、消费者洞察和证据链 QA 的直接 benchmark 参照。<https://arxiv.org/html/2602.08355>

### 2.2 Cue 与 Relation 的边界修正

早期草案把“呈现内容”和“内容之间的关系”混在了一起，v9 做如下修正：

| 早期标签 | v9 表达 | 修正原因 |
| --- | --- | --- |
| `DESCRIBES_PRODUCT` | `PRODUCT_DESCRIPTION` Cue | 商品描述本身是节点。 |
| `DEMONSTRATES_CLAIM` | `CLAIM_SUPPORTED_BY_DEMONSTRATION` Relation | 需要连接 Claim Cue 与独立 Demo Cue。 |
| `REPEATS_CLAIM` | `CLAIM_REPEATED_ACROSS_MODALITIES` Relation | 重复不能等同于证明。 |
| `PARTIALLY_SUPPORTS` | `CLAIM_PARTIALLY_SUPPORTED` Relation | 只支持复合主张的一部分。 |
| `CONTRADICTS` | `CLAIM_CONTRADICTED` Relation | 关联证据与主张冲突。 |
| `FRAMES_FEATURE_AS_BENEFIT` | `FEATURE_FRAMED_AS_BENEFIT` Relation | 连接具体特征与口播利益。 |
| `PRESENTS_PROBLEM` | `PAIN_POINT` Cue | 痛点是节点。 |
| `ADDRESSES_PROBLEM` | `PROBLEM_ADDRESSED_BY_SOLUTION` Relation | 连接痛点与商品/演示回应。 |
| `RAISES_OBJECTION` | `OBJECTION` Cue | 顾虑是节点。 |
| `RESPONDS_TO_OBJECTION` | `OBJECTION_RESPONDED_BY_CUE` Relation | 连接顾虑与解释、演示或保证。 |
| `REDUCES_FIT_UNCERTAINTY` | `CONTENT_ADDRESSES_FIT_UNCERTAINTY` Relation | 视频只能回应不确定性，不能证明观众心理变化。 |
| `REDUCES_USAGE_UNCERTAINTY` | `CONTENT_ADDRESSES_USAGE_UNCERTAINTY` Relation | 避免消费者结果因果断言。 |
| `REDUCES_PRICE_UNCERTAINTY` | `CONTENT_ADDRESSES_PRICE_UNCERTAINTY` Relation | 避免消费者结果因果断言。 |
| `ESTABLISHES_OFFER_CONDITION` | `OFFER_REQUIRES_CONDITION` Relation | 连接优惠与数量、时间、优惠券或操作条件。 |
| `BUILDS_CREDIBILITY_SIGNAL` | `CREDIBILITY_SIGNAL` Cue | 只能观察信号，不能判断真实可信度。 |
| `PRECEDES_ACTION_PROMPT` | `CONTENT_PRECEDES_CTA` Relation | 这是时间顺序边，不是营销效果。 |

### 2.3 最终 Relation 目录

| Relation | 含义与接受条件 | 来源类别 | 主要任务 |
| --- | --- | --- | --- |
| `DESCRIPTION_REFERS_TO_PRODUCT` | 描述/属性明确指向已定位商品或变体。 | `THEORY_DIRECT` | BP |
| `CLAIM_SUPPORTED_BY_DEMONSTRATION` | 独立视觉演示显示了口播/OCR 主张的实质部分；文案重复不算。 | `THEORY_OPERATIONALIZED` | CM |
| `CLAIM_REPEATED_ACROSS_MODALITIES` | ASR 与 OCR/画面文字重复同一宣传表述，但没有独立演示。 | `BENCHMARK_OPERATIONAL` | CM |
| `CLAIM_PARTIALLY_SUPPORTED` | 复合主张只有可分割的一部分被显示。 | `BENCHMARK_OPERATIONAL` | CM |
| `CLAIM_CONTRADICTED` | 同一商品、条件和时间窗口内的证据与主张不兼容。 | `BENCHMARK_OPERATIONAL` | CM |
| `CLAIM_TEMPORALLY_MISALIGNED` | 主张和候选证据对应不同阶段或不同商品状态。 | `BENCHMARK_OPERATIONAL` | CM |
| `FEATURE_FRAMED_AS_BENEFIT` | ASR/OCR 明确把可见/已描述特征连接为消费者利益。 | `THEORY_OPERATIONALIZED` | SS |
| `DEMONSTRATION_SHOWS_STATE_CHANGE` | 前、中、后证据建立可观察状态变化。 | `THEORY_DIRECT` | BP/SS |
| `PROBLEM_ADDRESSED_BY_SOLUTION` | 内容先呈现问题，再把商品、特征或演示作为回应。 | `THEORY_OPERATIONALIZED` | SS |
| `OBJECTION_RESPONDED_BY_CUE` | 明确顾虑后出现相关解释、演示、比较、保证或服务。 | `THEORY_OPERATIONALIZED` | SS/AE |
| `CONTENT_ADDRESSES_FIT_UNCERTAINTY` | 试穿、尺寸、物理比较或限制条件说明适配问题。 | `THEORY_OPERATIONALIZED` | AE |
| `CONTENT_ADDRESSES_USAGE_UNCERTAINTY` | 教程或过程演示说明如何使用、在哪使用或操作难度。 | `THEORY_OPERATIONALIZED` | AE |
| `CONTENT_ADDRESSES_PRICE_UNCERTAINTY` | 价格组成、套装数量、折扣比较或条件说明买到什么、付多少。 | `THEORY_OPERATIONALIZED` | BP/AE |
| `OFFER_REQUIRES_CONDITION` | 价格、赠品或折扣依赖明确可定位的条件。 | `THEORY_OPERATIONALIZED` | BP/AE |
| `CONTENT_PRECEDES_CTA` | Hook、问题、演示、报价或保证发生在行动提示之前。 | `BENCHMARK_OPERATIONAL` | SS |

Schema 必须拒绝把“内容回应某顾虑”改写成“消费者顾虑降低”，也必须拒绝“消费者信任主播”或“消费者购买”。

---

## 3. CommerceCue 目录

`CommerceCue` 是有 Evidence 支撑的语义节点，不是消费者效果标签。

| 家族 | Cue 类型 | 可观察边界 |
| --- | --- | --- |
| 商品 | `PRODUCT_IDENTITY`、`PRODUCT_ATTRIBUTE`、`PRODUCT_VARIANT`、`QUANTITY`、`BUNDLE` | 视频中可见或明确说出的商品、变体、属性、数量和套装。 |
| 报价 | `PRICE`、`DISCOUNT`、`GIFT`、`OFFER_CONDITION`、`SERVICE_GUARANTEE` | 明确报价及条件，不推断市场价值。 |
| 呈现 | `PRODUCT_DESCRIPTION`、`PROCESS_DEMONSTRATION`、`OUTCOME_DISPLAY`、`BEFORE_AFTER`、`VICARIOUS_TRIAL`、`USAGE_SCENARIO` | 主播实际说/做了什么，是展示过程还是结果。 |
| 主张 | `FUNCTION_CLAIM`、`EFFECT_CLAIM`、`PRICE_CLAIM`、`FIT_CLAIM`、`EXPERIENCE_REVIEW` | 在被独立演示前始终保持 Claim 身份。 |
| 需求与障碍 | `PAIN_POINT`、`NEED`、`OBJECTION`、`FIT_CONSTRAINT`、`USAGE_DIFFICULTY`、`PRICE_CONCERN`、`RISK_CONCERN` | 视频明确呈现的问题、限制或顾虑。 |
| 销售信号 | `BENEFIT`、`COMPARISON_ANCHOR`、`CREDIBILITY_SIGNAL`、`SOCIAL_PROOF`、`SCARCITY`、`URGENCY`、`CTA` | 只描述可见信号，不声称其改变行为。 |

每个 Cue 保存 `content_en`、可选 `source_text_native`、`evidence_refs`、`attributes`、`directness`、`confidence` 和 `theory_tags`。

---

## 4. 四任务如何使用 Evidence、Cue 和 Relation

四个任务不重复提取证据，它们消费同一层级结构：

```text
BP: EvidenceUnit -> CommerceCue，可选一条 Relation
CM: 至少两个 EvidenceUnit -> 至少两个 CommerceCue -> 一条 Claim Relation
SS: 多个 CommerceCue -> 一条 Relation 或有序 Relation Path
AE: Need/Constraint/Objection Cue -> 受限的 Alignment Relation Path
```

### 4.1 BP — Product & Offer Grounding

| 子能力 | 具体测量 | 必要来源 |
| --- | --- | --- |
| `PRODUCT_IDENTITY` | 视频正在呈现什么商品。 | Product Cue + visual/ASR/OCR。 |
| `ATTRIBUTE_AND_VARIANT` | 材质、尺寸、口味、型号、颜色或变体。 | Attribute/Variant Cue。 |
| `QUANTITY_AND_BUNDLE` | 报价包含多少件、什么组合。 | Quantity/Bundle Cue；多模态存在时需对齐。 |
| `PRICE_AND_DISCOUNT` | 明确价格或折扣。 | Price/Discount Cue；无显式比较不计算节省。 |
| `OFFER_CONDITION` | 获得优惠需要满足什么条件。 | Offer Cue + `OFFER_REQUIRES_CONDITION`。 |
| `USAGE_STEP` | 主播执行了什么具体步骤。 | Process Demo Cue + 定位帧。 |
| `DEMONSTRATED_STATE_CHANGE` | 演示前后发生什么可见变化。 | `DEMONSTRATION_SHOWS_STATE_CHANGE`。 |
| `USAGE_SCENARIO` | 视频呈现了什么使用场景。 | Usage Scenario Cue，不推断真实用户。 |

BP 主要是 Cue 层任务，不再把每条原始 Evidence 自动编成泛化事实问题。

### 4.2 CM — Claim–Evidence Verification

| 子能力 | 具体测量 | 必要 Relation |
| --- | --- | --- |
| `SPEECH_VISUAL_COREFERENCE` | 口播与画面是否指向同一商品、部件或动作。 | 同一观察窗口内的 Cue 引用。 |
| `OCR_SPEECH_OFFER_ALIGNMENT` | 口播报价和画面报价是否一致。 | 两种模态的 Offer Cue。 |
| `CLAIM_DEMONSTRATION_STATUS` | 主张是被演示、重复、部分支持、冲突还是未演示。 | 一条受控 Claim Relation。 |
| `REPETITION_VS_INDEPENDENT_EVIDENCE` | 第二模态提供独立证明还是只重复宣传语。 | Support 或 Repetition Relation。 |
| `PARTIAL_SUPPORT` | 复合主张哪部分有证据、哪部分没有。 | `CLAIM_PARTIALLY_SUPPORTED`。 |
| `CONTRADICTION` | 什么局部证据与主张冲突。 | `CLAIM_CONTRADICTED`。 |
| `TEMPORAL_MISALIGNMENT` | 主张与证据是否对应不同阶段。 | `CLAIM_TEMPORALLY_MISALIGNED`。 |
| `NOT_DEMONSTRATED` | 完整观察窗口内哪些内容只被说出而未展示。 | 完整窗口元数据；抽样帧缺失不能当全视频未出现。 |

CM 是 Relation 层任务，必须有真实跨模态信息差。

### 4.3 SS — Persuasion & Sales Logic

| 子能力 | 具体测量 | Cue/Relation Path |
| --- | --- | --- |
| `PROBLEM_SOLUTION` | 视频如何从问题过渡到商品回应。 | Pain Point -> Solution Relation。 |
| `FEATURE_BENEFIT` | 商品特征如何被解释为用户利益。 | Attribute -> Benefit Relation。 |
| `PROCESS_DEMONSTRATION` | 过程展示在销售解释中承担什么信息作用。 | Process Cue + Steps + Result。 |
| `OUTCOME_DISPLAY` | 视频突出什么结果，如何连接主张或报价。 | Outcome Cue + Claim/Sequence。 |
| `BEFORE_AFTER_COMPARISON` | 前后变化及视频要求观众比较什么。 | Before/After + State Change。 |
| `VICARIOUS_TRIAL` | 主播代替观众试穿、试用或试吃什么。 | Vicarious Trial + Fit/Use Evidence。 |
| `PRICE_VALUE_FRAMING` | 数量、功能或服务如何为价格提供上下文。 | Price + Bundle/Benefit/Service。 |
| `REFERENCE_PRICE_ANCHORING` | 哪个显式比较价格与当前报价同时或先出现。 | Anchor + Price Cue。 |
| `CREDIBILITY_SIGNAL` | 视频呈现了什么专业解释、亲测、保证或限制说明。 | Credibility Cue；不判断真实可信。 |
| `SOCIAL_PROOF` | 视频内部展示了什么评价、证言或用户内容。 | Social Proof Cue；私有互动快照禁止使用。 |
| `LIMITATION_DISCLOSURE` | 主播承认了什么适用限制。 | Objection/Constraint + Response。 |
| `OBJECTION_HANDLING` | 明确购买顾虑如何被回应。 | `OBJECTION_RESPONDED_BY_CUE`。 |
| `SCARCITY_AND_URGENCY` | 视频呈现什么时间、库存或价格窗口。 | Scarcity/Urgency + Offer。 |
| `CTA_SEQUENCE` | 哪些内容发生在行动提示之前。 | `CONTENT_PRECEDES_CTA` Path。 |

SS 是关系/路径任务。问题必须指向具体主张、演示、比较或顺序，不能直接问抽象的 “What mechanism...”。

### 4.4 AE — Need–Objection–Offer Alignment

| 子能力 | 受限推断内容 | Cue/Relation Path |
| --- | --- | --- |
| `CONTENT_IMPLIED_NEED` | 视频呈现了什么需求，不声称真实人群画像。 | Need/Pain Point + Product Response。 |
| `USAGE_CONTEXT` | 视频在何处、何时使用商品。 | Usage Scenario Cue。 |
| `FIT_CONSTRAINT` | 尺寸、体型、兼容性或环境限制。 | Fit Constraint Cue。 |
| `QUALITY_UNCERTAINTY` | 演示/比较试图回应什么质量顾虑。 | Objection + Response。 |
| `USAGE_UNCERTAINTY` | 过程演示澄清什么使用方式或操作难点。 | Usage Uncertainty Relation。 |
| `PRICE_UNCERTAINTY` | 价格、套装或条件澄清什么报价疑问。 | Price Uncertainty Relation。 |
| `SERVICE_OR_RISK_CONCERN` | 视频回应什么保证、退换、安全或服务顾虑。 | Risk Concern + Guarantee/Response。 |
| `DECISION_BARRIER` | 视频明确构造了什么考虑障碍。 | Objection/Constraint + Response。 |
| `OFFER_NEED_ALIGNMENT` | 报价条件或套装如何对应视频呈现的需求。 | Need -> Offer/Condition Path。 |

AE 不能推断真实人口统计、购买意愿、转化或流行程度。

---

## 5. 目标产物与数据流

```text
frames + native ASR + native OCR
  -> evidence_units.jsonl                 # English semantics + native source spans
  -> commerce_cues.jsonl                  # English
  -> commercial_relations.jsonl           # English
  -> video_evidence_dataset.jsonl          # English canonical Gold source
  -> Evidence/Cue/Relation human review
  -> video_evidence_dataset_reviewed.jsonl
  -> qa_specs.jsonl                        # English
  -> qa_realizations.jsonl                 # English questions
  -> QA human review
  -> vqa_gold_private.jsonl + vqa_public.jsonl
  -> predictions.jsonl
  -> Judge details                         # English
  -> audit_translations.jsonl              # Chinese, audit-only sidecar
  -> bilingual audit workbench
```

正式交付包含：

1. EvidenceDataset v3：Atomic Evidence、CommerceCue、CommercialRelation 和人工冻结的 GroundedAnnotation。
2. Prompt Manifest：所有全英文 Prompt 原文、版本、模型和配置指纹。
3. QA：全英文公开问题和私有 Gold/evidence refs。
4. Evaluation：predictions、Judge 明细、错误标签、四任务 macro-average 和人工校准摘要。
5. Audit Sidecar：独立中文翻译包与双语审计 HTML，不属于公开 benchmark 数据。

---

## 6. 文件结构

### 新建文件

- `src/salesbench/goldbank/commerce_schema.py`：Cue/Relation Enum、dataclass、解析器和稳定 ID。
- `src/salesbench/goldbank/commerce_ontology.py`：Cue/Relation 目录、理论来源和允许端点。
- `src/salesbench/vqa/specs.py`：`QuestionSpec`、`QuestionRealization` 数据契约。
- `src/salesbench/vqa/prompts.py`：全英文 Question Realizer Prompt。
- `src/salesbench/vqa/realizer.py`：可断点续跑的 QA 实现 runner。
- `src/salesbench/audit_translation_prompts.py`：全英文审计翻译 Prompt。
- `src/salesbench/audit_translation.py`：翻译任务收集、缓存、来源哈希和 JSONL runner。
- `tests/test_commerce_schema.py`。
- `tests/test_commerce_ontology.py`。
- `tests/test_vqa_specs.py`。
- `tests/test_vqa_realizer.py`。
- `tests/test_audit_translation.py`。
- `configs/evidence_smoke_v9_5videos.json`。
- `configs/evidence_pilot_v9_64videos.json`。
- `configs/pilot64_gpt4o_v9_delivery.json`：正式、smoke、QA、Evaluation 和审计翻译 sidecar 的路径清单。
- `docs/data/EvidenceDataset_Data_Card_v3.md`。

### 修改文件

- `src/salesbench/goldbank/schema.py`：v3 record 和四任务语义字段。
- `src/salesbench/goldbank/ontology.py`：替换 v8 泛化 subtype。
- `src/salesbench/goldbank/prompts.py`：全英文 Cue/Relation/Task Prompt v9。
- `src/salesbench/goldbank/parsing.py`、`normalizer.py`、`validators.py`。
- `src/salesbench/goldbank/pipeline.py`、`runner.py`、`review_io.py`、`audit.py`。
- `src/salesbench/vqa/question_programs.py`：只保留 legacy compatibility helper。
- `src/salesbench/vqa/compiler.py`：读取审核后的 Question Realization 并执行英文/多样性/隐私校验。
- `src/salesbench/vqa_evaluate/context.py`、`prompts.py`、`judge.py`、`runner.py`。
- `src/salesbench/cli.py`：新增 `realize-qa` 和 `build-audit-translations`。
- `tools/audit_workbench/build.py`：双语 Prompt、Evidence、Cue、Relation、QA 和 Judge 审计。
- `tools/audit_workbench/evidence_assets.py`：关联帧懒加载。
- `tools/audit_workbench/review_queue.py`：标准化审计队列与翻译状态。
- 对应现有测试和端到端测试。
- `docs/Pilot_64_Video_Execution_Guide.md`。

---

## Task 1：新增 CommerceCue 与 CommercialRelation 契约

**文件：**
- 新建：`src/salesbench/goldbank/commerce_schema.py`
- 新建：`tests/test_commerce_schema.py`
- 修改：`src/salesbench/goldbank/schema.py`

**接口：**
- 输入：`EvidenceUnit` ID、`stable_digest`、`QualityStatus`。
- 输出：`CommerceCue`、`CommercialRelation`、`parse_commerce_cue()`、`parse_commercial_relation()`、`make_cue_id()`、`make_relation_id()`。

- [ ] **Step 1：先写失败测试**

```python
def test_commerce_contract_uses_english_semantics_and_stable_ids():
    cue_id = make_cue_id("v1", CueType.PRICE, ("e1",), "the displayed price is 9.9 yuan")
    cue = CommerceCue(
        cue_id=cue_id,
        video_id="v1",
        cue_type=CueType.PRICE,
        content_en="The displayed price is 9.9 yuan.",
        source_text_native="9.9元",
        evidence_ids=("e1",),
        attributes={"currency": "CNY", "amount": 9.9},
        directness="DIRECT",
        theory_tags=("offer_information",),
        extractor="fake",
        confidence=0.95,
    )
    assert parse_commerce_cue(cue.to_dict()) == cue
    assert "content_zh" not in cue.to_dict()


def test_v3_evidence_separates_english_semantics_from_native_source():
    unit = parse_evidence_unit(
        {
            "evidence_id": "e_asr",
            "video_id": "v1",
            "modality": "asr",
            "content_en": "The speaker claims that the product removes stains.",
            "source_text_native": "这个产品可以去除污渍",
            "subject": "speaker",
            "predicate": "claims",
            "value": "the product removes stains",
            "confidence": 0.9,
        }
    )
    assert unit.content_en.startswith("The speaker claims")
    assert unit.source_text_native == "这个产品可以去除污渍"
    assert "content_zh" not in unit.to_dict()
```

- [ ] **Step 2：确认测试失败**

运行：`pytest tests/test_commerce_schema.py -q`

预期：因模块尚不存在而 FAIL。

- [ ] **Step 3：实现 Enum、dataclass、parser 和稳定 ID**

```python
class RelationProvenance(str, Enum):
    THEORY_DIRECT = "THEORY_DIRECT"
    THEORY_OPERATIONALIZED = "THEORY_OPERATIONALIZED"
    BENCHMARK_OPERATIONAL = "BENCHMARK_OPERATIONAL"


@dataclass(frozen=True)
class CommerceCue:
    cue_id: str
    video_id: str
    cue_type: CueType
    content_en: str
    source_text_native: str
    evidence_ids: tuple[str, ...]
    attributes: dict[str, object]
    directness: str
    theory_tags: tuple[str, ...]
    extractor: str
    confidence: float
```

按第 2.3 和第 3 节实现全部 Enum，并设置 `SCHEMA_VERSION = "evidence-dataset-schema-v3"`。同时为 `EvidenceUnit` 增加显式 `content_en` 和 `source_text_native`：v3 writer 只写新字段；v2 reader 将旧 `text_span` 映射为 `source_text_native`，并从英文 `subject/predicate/value` 生成 `content_en`，以便旧结果只读审计。Canonical schema 中不得出现 `content_zh` 或 `translated_text`。

- [ ] **Step 4：运行契约测试**

运行：`pytest tests/test_commerce_schema.py tests/test_goldbank_schema.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/goldbank/commerce_schema.py src/salesbench/goldbank/schema.py tests/test_commerce_schema.py tests/test_goldbank_schema.py
git commit -m "feat: add commerce cue and relation contracts"
```

## Task 2：实现本体来源和确定性校验

**文件：**
- 新建：`src/salesbench/goldbank/commerce_ontology.py`
- 新建：`tests/test_commerce_ontology.py`
- 修改：`src/salesbench/goldbank/validators.py`

**接口：**
- 输入：Task 1 的 Cue/Relation。
- 输出：`validate_commerce_cue()`、`validate_commercial_relation()`。

- [ ] **Step 1：写因果越界和端点失败测试**

```python
def test_relation_catalog_excludes_consumer_outcomes():
    values = {item.value for item in RelationType}
    assert "INCREASES_TRUST" not in values
    assert "CAUSES_PURCHASE" not in values
    assert "IMPROVES_CONVERSION" not in values
    assert "CONTENT_ADDRESSES_FIT_UNCERTAINTY" in values
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_commerce_ontology.py -q`

预期：FAIL。

- [ ] **Step 3：实现 `RELATION_RULES` 和本地规则**

```python
RELATION_RULES = {
    RelationType.FEATURE_FRAMED_AS_BENEFIT: {
        "source": {CueType.PRODUCT_ATTRIBUTE, CueType.PRODUCT_DESCRIPTION},
        "target": {CueType.BENEFIT},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION: {
        "source": {CueType.FUNCTION_CLAIM, CueType.EFFECT_CLAIM, CueType.FIT_CLAIM},
        "target": {
            CueType.PROCESS_DEMONSTRATION,
            CueType.OUTCOME_DISPLAY,
            CueType.BEFORE_AFTER,
            CueType.VICARIOUS_TRIAL,
        },
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
}
```

校验必须覆盖：ID 存在、同视频、端点类型、英文规范字段、原文定位、私有字段、置信度、因果词和完整观察窗口。

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_commerce_ontology.py tests/test_goldbank_validators.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/goldbank/commerce_ontology.py src/salesbench/goldbank/validators.py tests/test_commerce_ontology.py tests/test_goldbank_validators.py
git commit -m "feat: validate commerce graph semantics"
```

## Task 3：实现全英文 Prompt v9

**文件：**
- 修改：`src/salesbench/goldbank/prompts.py`
- 修改：`src/salesbench/goldbank/parsing.py`
- 修改：`src/salesbench/goldbank/normalizer.py`
- 修改：`tests/test_goldbank_prompts.py`
- 修改：`tests/test_goldbank_normalizer.py`

**接口：**
- 输入：英文规范 Evidence + 原语言 source span。
- 输出：`build_commerce_cue_prompt()`、`build_commercial_relation_prompt()` 和归一化结果。

- [ ] **Step 1：写全英文和边界测试**

```python
def test_v9_prompts_are_english_and_forbid_outcome_claims():
    cue_system, cue_user = build_commerce_cue_prompt("v1", [evidence_fixture()])
    relation_system, relation_user = build_commercial_relation_prompt(
        "v1", [evidence_fixture()], [cue_fixture()]
    )
    combined = cue_system + cue_user + relation_system + relation_user
    assert PROMPT_VERSION == "evidence-prompt-v9"
    assert not contains_cjk(combined)
    assert "Never claim that a viewer trusted, purchased, converted, or became less uncertain" in combined
```

- [ ] **Step 2：确认测试失败**

运行：`pytest tests/test_goldbank_prompts.py -q`

预期：FAIL。

- [ ] **Step 3：实现严格 JSON Prompt**

Cue 输出只允许：

```json
{
  "commerce_cues": [
    {
      "cue_type": "PROCESS_DEMONSTRATION",
      "content_en": "The host applies the cleaner and wipes the surface.",
      "source_text_native": "",
      "evidence_ids": ["existing_visual_id"],
      "attributes": {"presentation_stage": "process"},
      "directness": "DIRECT",
      "theory_tags": ["product_demonstration"],
      "confidence": 0.92
    }
  ],
  "abstentions": []
}
```

Relation 输出只允许第 2.3 节的 Enum。所有解释字段使用英文；本地代码生成 ID，并用 ontology 覆盖模型自报 provenance。

- [ ] **Step 4：运行 Prompt/Parser/Normalizer 测试**

运行：`pytest tests/test_goldbank_prompts.py tests/test_goldbank_normalizer.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/goldbank/prompts.py src/salesbench/goldbank/parsing.py src/salesbench/goldbank/normalizer.py tests/test_goldbank_prompts.py tests/test_goldbank_normalizer.py
git commit -m "feat: add all-English v9 commerce prompts"
```

## Task 4：在 EvidenceDataset Pipeline 中持久化商业论证图

**文件：**
- 修改：`src/salesbench/goldbank/pipeline.py`
- 修改：`src/salesbench/goldbank/runner.py`
- 修改：`src/salesbench/goldbank/review_io.py`
- 修改：`tests/test_goldbank_pipeline.py`
- 修改：`tests/test_goldbank_runner.py`
- 修改：`tests/test_goldbank_review_io.py`

**接口：**
- 输入：Evidence、Cue、Relation、Validators。
- 输出：新增 `GoldBankResult.commerce_cues`、`commercial_relations` 和对应 JSONL。

- [ ] **Step 1：写阶段顺序失败测试**

```python
def test_v9_stage_order_is_evidence_then_cue_then_relation_then_task():
    result = pipeline.run_video(bundle(), frames_b64=["frame"])
    stages = [trace["stage"] for trace in result.agent_traces]
    assert stages.index("evidence_extraction") < stages.index("commerce_cue_extraction")
    assert stages.index("commerce_cue_extraction") < stages.index("commercial_relation_building")
    assert stages.index("commercial_relation_building") < stages.index("task_proposal")
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_goldbank_pipeline.py -q`

预期：FAIL。

- [ ] **Step 3：实现固定阶段顺序和持久化**

```python
evidence_units = self._extract_evidence(bundle, frames_b64, traces)
commerce_cues = self._extract_commerce_cues(bundle.video_id, evidence_units, traces)
commercial_relations = self._build_commercial_relations(
    bundle.video_id, evidence_units, commerce_cues, traces
)
proposals = self._build_task_proposals(
    bundle.video_id, evidence_units, commerce_cues, commercial_relations, traces
)
```

Runner 写出 `commerce_cues.jsonl`、`commercial_relations.jsonl`，并设置 `PIPELINE_VERSION = "evidence-first-pipeline-v8"`。Resume 指纹包含视频哈希、帧策略、模型、schema、prompt 和 pipeline 版本。

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_goldbank_pipeline.py tests/test_goldbank_runner.py tests/test_goldbank_review_io.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/goldbank/pipeline.py src/salesbench/goldbank/runner.py src/salesbench/goldbank/review_io.py tests/test_goldbank_pipeline.py tests/test_goldbank_runner.py tests/test_goldbank_review_io.py
git commit -m "feat: persist v3 commerce graph outputs"
```

## Task 5：将四任务改为方案 B 能力体系

**文件：**
- 修改：`src/salesbench/goldbank/ontology.py`
- 修改：`src/salesbench/goldbank/schema.py`
- 修改：`src/salesbench/goldbank/prompts.py`
- 修改：`src/salesbench/goldbank/pipeline.py`
- 修改对应 schema/prompt/pipeline 测试。

**接口：**
- 输入：商业论证图。
- 输出：包含 `capability`、`reasoning_operator`、Cue/Relation refs、Gold 边界和 question intent 的四任务候选。

- [ ] **Step 1：写任务层级失败测试**

```python
def test_plan_b_capabilities_use_expected_graph_levels():
    assert capability_level("BP", "OFFER_CONDITION") == "RELATION_OPTIONAL"
    assert capability_level("CM", "CLAIM_DEMONSTRATION_STATUS") == "RELATION_REQUIRED"
    assert capability_level("SS", "PROBLEM_SOLUTION") == "RELATION_PATH_REQUIRED"
    assert capability_level("AE", "FIT_CONSTRAINT") == "CUE_OR_RELATION"
    assert "TRUST_MECHANISM" not in allowed_subtypes("SS")
    assert "AUDIENCE_NEED_FIT" not in allowed_subtypes("AE")
```

- [ ] **Step 2：运行并确认失败**

运行：`pytest tests/test_goldbank_schema.py tests/test_goldbank_prompts.py -q`

预期：FAIL。

- [ ] **Step 3：扩展 Proposal/GoldItem**

```python
capability: str
reasoning_operator: str
commerce_cue_ids: tuple[str, ...]
commercial_relation_ids: tuple[str, ...]
question_intent: str
forbidden_inferences: tuple[str, ...]
```

BP planner 仍可确定性生成，但只消费商品、报价和演示 Cue；CM/SS/AE 使用各自全英文 proposer。所有 planner 都允许 abstain，不要求每视频四任务齐全。

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_goldbank_schema.py tests/test_goldbank_prompts.py tests/test_goldbank_pipeline.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/goldbank/ontology.py src/salesbench/goldbank/schema.py src/salesbench/goldbank/prompts.py src/salesbench/goldbank/pipeline.py tests/test_goldbank_schema.py tests/test_goldbank_prompts.py tests/test_goldbank_pipeline.py
git commit -m "feat: align four tasks with commerce graph capabilities"
```

## Task 6：新增 QuestionSpec 和自然英文问题实现

**文件：**
- 新建：`src/salesbench/vqa/specs.py`
- 新建：`src/salesbench/vqa/prompts.py`
- 新建：`src/salesbench/vqa/realizer.py`
- 新建：`tests/test_vqa_specs.py`
- 新建：`tests/test_vqa_realizer.py`
- 修改：`src/salesbench/cli.py`

**接口：**
- 输入：审核后的 GroundedAnnotation 和证据图。
- 输出：`qa_specs.jsonl`、`qa_realizations.jsonl`。

- [ ] **Step 1：写自然化失败测试**

```python
def test_realized_question_is_specific_english_without_template_jargon():
    question = "What is shown after the stain-removal claim, and what remains only stated?"
    assert not contains_cjk(question)
    assert "mechanism" not in question.lower()
    assert "support the answer" not in question.lower()
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_vqa_specs.py tests/test_vqa_realizer.py -q`

预期：FAIL。

- [ ] **Step 3：实现全英文 Realizer Contract**

```json
{
  "spec_id": "qs_v1_cm_001",
  "question": "What is shown after the stain-removal claim, and what remains only stated?"
}
```

Prompt 禁止 `What mechanism`、`What strategy`、`What audience` 和 `Support the answer with evidence`，要求使用具体商品、动作、数字或时间关系。新增 CLI：

```bash
python salesbench.py realize-qa \
  --evidence-dir <reviewed-evidence-dir> \
  --output-dir <qa-dir> \
  --text-model gpt-4o \
  --max-workers 2
```

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_vqa_specs.py tests/test_vqa_realizer.py tests/test_goldbank_cli.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/vqa/specs.py src/salesbench/vqa/prompts.py src/salesbench/vqa/realizer.py src/salesbench/cli.py tests/test_vqa_specs.py tests/test_vqa_realizer.py tests/test_goldbank_cli.py
git commit -m "feat: realize evidence-specific English questions"
```

## Task 7：编译审核后的英文 QA 并控制多样性

**文件：**
- 修改：`src/salesbench/vqa/question_programs.py`
- 修改：`src/salesbench/vqa/compiler.py`
- 修改：`src/salesbench/cli.py`
- 修改对应 compiler/question/CLI 测试。

**接口：**
- 输入：GroundedAnnotation、`qa_specs.jsonl`、审核后 `qa_realizations.jsonl`。
- 输出：全英文 private/public VQA、diversity report 和 rejection reasons。

- [ ] **Step 1：写失败测试**

```python
def test_compiler_keeps_english_and_does_not_force_all_tasks_per_video():
    assert {row["task_type"] for row in records} == {"CM", "SS"}
    assert all(not contains_cjk((row["question"], row["gold_answer"])) for row in records)
    assert not any(row.get("reason") == "missing_bp_for_video" for row in validation)
```

- [ ] **Step 2：确认失败**

运行：`pytest tests/test_goldbank_qa_compiler.py -q`

预期：FAIL。

- [ ] **Step 3：实现 compiler v5**

设置 `COMPILER_VERSION = "evidence-qa-compiler-v5"`，新增必填 `--realizations`。输出 `qa_diversity.json`，至少包含 exact duplicate、normalized stem cluster、within-video semantic duplicate、cross-task evidence reuse 和 non-marginal reuse。Canonical QA 不得包含中文审计翻译。

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_vqa_question_programs.py tests/test_goldbank_qa_compiler.py tests/test_goldbank_cli.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/vqa/question_programs.py src/salesbench/vqa/compiler.py src/salesbench/cli.py tests/test_vqa_question_programs.py tests/test_goldbank_qa_compiler.py tests/test_goldbank_cli.py
git commit -m "feat: compile diverse English QA realizations"
```

## Task 8：新增独立审计中文翻译层

**文件：**
- 新建：`src/salesbench/audit_translation_prompts.py`
- 新建：`src/salesbench/audit_translation.py`
- 新建：`tests/test_audit_translation.py`
- 修改：`src/salesbench/cli.py`
- 修改：`tests/test_goldbank_cli.py`

**接口：**
- 输入：Prompt、Evidence、Cue、Relation、QuestionSpec、QA、Judge 中允许审计的英文字段。
- 输出：`collect_audit_translation_jobs()`、`run_audit_translations()`、`audit_translations.jsonl`。

- [ ] **Step 1：写隔离、哈希和忠实翻译测试**

```python
def test_audit_translation_is_separate_versioned_and_hash_bound():
    job = build_translation_job(
        object_type="qa",
        object_id="q1",
        source_field="question",
        source_text="What condition is required for the discount?",
    )
    assert job.source_sha256
    assert job.target_language == "zh-CN"
    assert job.audit_only is True
    assert "question_zh" not in public_vqa_item({"question": job.source_text})
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_audit_translation.py -q`

预期：FAIL，因为翻译模块尚不存在。

- [ ] **Step 3：实现全英文翻译 Prompt**

```python
AUDIT_TRANSLATION_PROMPT_VERSION = "audit-translation-prompt-v1"

AUDIT_TRANSLATION_SYSTEM_PROMPT = """You are the Simplified Chinese audit translation layer for SalesBench. Translate the supplied English field faithfully for human review. Preserve all numbers, units, negation, modality distinctions, claim-versus-fact boundaries, enum tokens, JSON keys, evidence IDs, and uncertainty wording. Do not summarize, correct, strengthen, weaken, explain, or add marketing interpretations. Return JSON only."""
```

模型输出：

```json
{
  "translation_id": "audit_translation::qa::q1::question",
  "translated_text": "获得该折扣需要满足什么条件？"
}
```

本地代码负责 object metadata、source hash、model、prompt version 和 `audit_only=true`。相同 source hash 复用缓存；不同 hash 强制重译。翻译任务必须过滤私有互动字段。

- [ ] **Step 4：新增 CLI 并运行测试**

```bash
python salesbench.py build-audit-translations \
  --manifest configs/pilot64_gpt4o_v9_delivery.json \
  --output outputs/audit/translations/v9/audit_translations.jsonl \
  --base-url "$OPENAI_BASE_URL" \
  --model gpt-4o \
  --max-workers 2
```

运行：`pytest tests/test_audit_translation.py tests/test_goldbank_cli.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/audit_translation_prompts.py src/salesbench/audit_translation.py src/salesbench/cli.py tests/test_audit_translation.py tests/test_goldbank_cli.py
git commit -m "feat: add audit-only Chinese translations"
```

## Task 9：将审计工作台升级为双语懒加载视图

**文件：**
- 修改：`tools/audit_workbench/build.py`
- 修改：`tools/audit_workbench/evidence_assets.py`
- 修改：`tools/audit_workbench/review_queue.py`
- 修改：`tests/test_audit_workbench.py`
- 修改：`tests/test_audit_review_queue.py`

**接口：**
- 输入：Canonical 英文产物、中文原文、`audit_translations.jsonl` 和关联帧。
- 输出：默认中文、可展开英文的双语审计 HTML。

- [ ] **Step 1：写双语显示失败测试**

```python
def test_workbench_renders_chinese_translation_and_english_source():
    html = render_workbench(workbench_v9_fixture(), collect_prompt_snapshot())
    assert "获得该折扣需要满足什么条件" in html
    assert "What condition is required for the discount?" in html
    assert "中文审计翻译" in html
    assert "英文规范原文" in html
    assert 'loading="lazy"' in html
    assert "likes" not in html
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_audit_workbench.py tests/test_audit_review_queue.py -q`

预期：FAIL。

- [ ] **Step 3：实现双语审计卡片**

Prompt 页面：英文原 Prompt 为唯一规范版本，中文翻译用于阅读，并显示 source hash、translation model 和 stale 状态。

Evidence/Cue/Relation 页面：默认显示中文翻译；显示中文 ASR/OCR 原文；英文 `content_en` 可展开；始终展示 evidence refs 和关联帧。

QA 页面：默认显示 `question_zh`/`gold_answer_zh` 审计翻译，同时展示英文原题/Gold、Capability、Operator、Relation Path、Evidence 内容与帧。

Judge 页面：显示英文分数/错误标签、中文理由翻译、英文原理由和 Judge 实际依据。

缺失或 stale 翻译必须显示黄色警告，但不能阻断对英文规范数据的审核。

在 `tools/audit_workbench/build.py` 的参数解析中新增必填 `--translations`；读取后按 `(object_type, object_id, source_field, source_sha256)` 建立索引。发现同一字段多个有效翻译、hash 不匹配、`audit_only != true` 或私有字段时立即失败，不生成 HTML。

- [ ] **Step 4：运行审计测试**

运行：`pytest tests/test_audit_workbench.py tests/test_audit_review_queue.py tests/test_audit_translation.py -q`

预期：PASS，HTML 不包含私有互动字段。

- [ ] **Step 5：提交**

```bash
git add tools/audit_workbench/build.py tools/audit_workbench/evidence_assets.py tools/audit_workbench/review_queue.py tests/test_audit_workbench.py tests/test_audit_review_queue.py tests/test_audit_translation.py
git commit -m "feat: render bilingual commerce audit views"
```

## Task 10：保留 LLM-as-Judge 主评分并增加诊断标签

**文件：**
- 修改：`src/salesbench/vqa_evaluate/context.py`
- 修改：`src/salesbench/vqa_evaluate/prompts.py`
- 修改：`src/salesbench/vqa_evaluate/judge.py`
- 修改：`src/salesbench/vqa_evaluate/runner.py`
- 修改：`tests/test_vqa_evaluate.py`

**接口：**
- 输入：英文问题、Capability、Reference Answer、Evidence/Graph Summary 和模型英文答案。
- 输出：五档主分数和英文错误标签；中文理由由 Task 8 另行生成。

- [ ] **Step 1：写 Judge v4 失败测试**

```python
def test_judge_v4_keeps_english_output_and_error_tags():
    parsed = parse_judge_response(
        '{"score":0.5,"correctness":0.5,"grounding":0.5,"completeness":0.5,'
        '"error_tags":["CLAIM_EVIDENCE_CONFUSION"],'
        '"reason":"The answer treats repeated text as visual proof.",'
        '"evidence_alignment":"The frames do not independently demonstrate the claim."}'
    )
    assert parsed["score"] == 0.5
    assert parsed["error_tags"] == ["CLAIM_EVIDENCE_CONFUSION"]
    assert not contains_cjk((parsed["reason"], parsed["evidence_alignment"]))
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_vqa_evaluate.py -q`

预期：FAIL。

- [ ] **Step 3：实现 task-aware Judge v4**

保留 `{0, 0.25, 0.5, 0.75, 1}`。错误标签固定为：

```python
JUDGE_ERROR_TAGS = {
    "FACTUAL_ERROR",
    "UNSUPPORTED_INFERENCE",
    "MISSING_KEY_INFORMATION",
    "CLAIM_EVIDENCE_CONFUSION",
    "OFFER_CONDITION_MISSING",
    "TEMPORAL_ERROR",
    "TASK_MISUNDERSTANDING",
    "UNANSWERED",
}
```

Judge payload 只含英文规范数据；缺失答案计 0；不新增独立 key-fact scorer。

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_vqa_evaluate.py tests/test_evidence_vqa_e2e.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/vqa_evaluate/context.py src/salesbench/vqa_evaluate/prompts.py src/salesbench/vqa_evaluate/judge.py src/salesbench/vqa_evaluate/runner.py tests/test_vqa_evaluate.py tests/test_evidence_vqa_e2e.py
git commit -m "feat: add commerce-aware Judge diagnostics"
```

## Task 11：接入 v9 CLI、配置、元数据和端到端测试

**文件：**
- 修改：`src/salesbench/cli.py`
- 新建：`configs/evidence_smoke_v9_5videos.json`
- 新建：`configs/evidence_pilot_v9_64videos.json`
- 新建：`configs/pilot64_gpt4o_v9_delivery.json`
- 修改 CLI/E2E/Convergence 测试。

- [ ] **Step 1：写全链路语言与隐私失败测试**

```python
def test_v9_canonical_payloads_are_english_and_audit_translation_is_isolated():
    canonical = json.dumps(outputs["canonical_payloads"], ensure_ascii=False)
    audit = json.dumps(outputs["audit_payloads"], ensure_ascii=False)
    assert "private_analysis_metadata" not in canonical
    assert "translated_text" not in canonical
    assert "translated_text" in audit
    assert outputs["task_counts"].keys() == {"BP", "CM", "SS", "AE"}
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_evidence_vqa_e2e.py tests/test_benchmark_convergence.py -q`

预期：FAIL。

- [ ] **Step 3：接入版本配置**

```json
{
  "prompt_version": "evidence-prompt-v9",
  "schema_version": "evidence-dataset-schema-v3",
  "pipeline_version": "evidence-first-pipeline-v8",
  "frame_strategy": "hook_plus_uniform",
  "frames_per_video": 16,
  "min_confidence": 0.7
}
```

v9 smoke 和 pilot 复用 v8 的 5/64 个 video ID，只改变 ontology/prompt/pipeline，避免 cohort 变化干扰比较。`configs/pilot64_gpt4o_v9_delivery.json` 必须分别声明 smoke/formal Evidence、QA、模型结果、Evaluation、帧 manifest 和 `audit_translations` 路径，且不得引用任何 v6/v7/v8 运行目录。

- [ ] **Step 4：运行集成测试**

运行：`pytest tests/test_goldbank_cli.py tests/test_evidence_vqa_e2e.py tests/test_benchmark_convergence.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/cli.py configs/evidence_smoke_v9_5videos.json configs/evidence_pilot_v9_64videos.json configs/pilot64_gpt4o_v9_delivery.json tests/test_goldbank_cli.py tests/test_evidence_vqa_e2e.py tests/test_benchmark_convergence.py
git commit -m "feat: wire the v9 SalesBench pilot flow"
```

## Task 12：更新审计指标、Data Card 和执行指南

**文件：**
- 修改：`src/salesbench/goldbank/audit.py`
- 修改：`docs/Pilot_64_Video_Execution_Guide.md`
- 新建：`docs/data/EvidenceDataset_Data_Card_v3.md`
- 修改：`tests/test_goldbank_audit.py`

- [ ] **Step 1：写审计指标失败测试**

```python
def test_audit_reports_graph_language_and_translation_separation():
    report = audit_gold_bank(v3_records(), v3_evidence(), v3_cues(), v3_relations())
    assert report["commerce_cue_coverage"] == 1.0
    assert report["commercial_relation_validity"] == 1.0
    assert report["causal_outcome_relation_count"] == 0
    assert report["canonical_chinese_field_count"] == 0
    assert report["audit_translation_stale_count"] == 0
```

- [ ] **Step 2：运行失败测试**

运行：`pytest tests/test_goldbank_audit.py -q`

预期：FAIL。

- [ ] **Step 3：实现指标与文档**

增加 Cue evidence validity、Relation endpoint validity、provenance、causal leakage、capability/operator、cross-task reuse、canonical language、translation coverage/stale 等指标。新建 v3 Data Card，保留 v2 文件作为历史契约。

- [ ] **Step 4：运行测试**

运行：`pytest tests/test_goldbank_audit.py tests/test_goldbank_cli.py -q`

预期：PASS。

- [ ] **Step 5：提交**

```bash
git add src/salesbench/goldbank/audit.py docs/Pilot_64_Video_Execution_Guide.md docs/data/EvidenceDataset_Data_Card_v3.md tests/test_goldbank_audit.py
git commit -m "docs: document and audit the v9 commerce benchmark"
```

## Task 13：API Smoke 前完整验证

**文件：** 只验证；发现缺陷时回到所属 Task 增加回归测试和最小修复。

- [ ] **Step 1：运行聚焦测试**

```bash
pytest tests/test_commerce_schema.py tests/test_commerce_ontology.py \
  tests/test_goldbank_prompts.py tests/test_goldbank_pipeline.py \
  tests/test_vqa_specs.py tests/test_vqa_realizer.py \
  tests/test_audit_translation.py tests/test_goldbank_qa_compiler.py \
  tests/test_audit_workbench.py tests/test_vqa_evaluate.py \
  tests/test_evidence_vqa_e2e.py -q
```

- [ ] **Step 2：运行完整测试和静态检查**

```bash
pytest -q
python -m compileall -q src tools
git diff --check
```

预期：测试全部通过；compileall 返回 0；`git diff --check` 无输出。

- [ ] **Step 3：核对全部版本常量**

```bash
rg -n "evidence-dataset-schema-v3|evidence-prompt-v9|evidence-first-pipeline-v8|evidence-qa-compiler-v5|judge-prompt-v4|audit-translation-prompt-v1" src configs docs
```

- [ ] **Step 4：处理验证缺陷**

如果验证失败，在负责该组件的 Task 中添加聚焦回归测试，只提交该 Task 明确列出的文件。若验证没有产生文件变化，不创建空提交或汇总提交。

---

## 7. 实施后的完整运行手册

### 7.1 加载本地中转 API 凭证

```bash
set -a
source .env
set +a
```

不得打印 key，也不得将 `.env` 加入 Git。

### 7.2 运行 5 视频 v9 smoke EvidenceDataset

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_smoke_v9_5videos.json \
  --output-dir outputs/evidence/v9_smoke5_gpt4o_yunwu \
  --vision-base-url "$OPENAI_BASE_URL" \
  --vision-model gpt-4o \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 1
```

Smoke 通过条件：

- 5/5 视频无 schema stage failure。
- Evidence、Cue 和 Relation 文件非空。
- Canonical 语义字段全部英文；中文只存在于 source-native 字段。
- 不存在消费者结果 Relation。
- 文案重复没有被误判为视觉证明。
- 5 视频集合层面至少覆盖 BP/CM/SS/AE；单视频缺任务允许。
- 审计能看到关联帧和原语言文本。

### 7.3 生成 smoke QA

```bash
python salesbench.py realize-qa \
  --evidence-dir outputs/evidence/v9_smoke5_gpt4o_yunwu \
  --output-dir outputs/vqa/v9_smoke5_gpt4o_yunwu \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 1
```

确认问题和 Gold 全英文，不包含公式化后缀。

### 7.4 生成中文审计翻译包

```bash
python salesbench.py build-audit-translations \
  --manifest configs/pilot64_gpt4o_v9_delivery.json \
  --output outputs/audit/translations/v9_smoke/audit_translations.jsonl \
  --base-url "$OPENAI_BASE_URL" \
  --model gpt-4o \
  --max-workers 2
```

翻译审计重点：

- 数字、单位、价格和数量必须一致。
- `claim` 不能被翻译成已验证事实。
- `not shown`、`partially supported`、否定词和条件词不能丢失。
- Enum、ID、JSON key 不翻译。
- 翻译缺失或 stale 时使用英文原文审核，不能自动接受。

### 7.5 构建双语 smoke 审计 HTML

```bash
python -m tools.audit_workbench.build \
  --manifest configs/pilot64_gpt4o_v9_delivery.json \
  --translations outputs/audit/translations/v9_smoke/audit_translations.jsonl \
  --output outputs/audit/SalesBench_Prompt_Audit_Workbench_v9_smoke.html \
  --skip-organize
```

审核 Prompt、Evidence、Cue、Relation、QuestionSpec 和 QA。默认看中文，同时用英文原文、中文 ASR/OCR 和关联帧核对。

### 7.6 运行正式 64 视频 EvidenceDataset

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_pilot_v9_64videos.json \
  --output-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --vision-base-url "$OPENAI_BASE_URL" \
  --vision-model gpt-4o \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 2
```

### 7.7 人工审核并冻结 EvidenceDataset

审核顺序：

1. 原子 Evidence 的定位、原文和英文规范语义。
2. CommerceCue 类型、英文语义、原文和 evidence refs。
3. CommercialRelation 端点、provenance、rationale 和因果边界。
4. GroundedAnnotation 的 Capability、Operator、Gold 和 forbidden inference。

```bash
python salesbench.py apply-evidence-reviews \
  --evidence-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --decisions outputs/reviews/v9_pilot64_evidence_decisions.json \
  --output outputs/evidence/v9_pilot64_gpt4o_yunwu/video_evidence_dataset_reviewed.jsonl
```

审核修改必须写回英文 canonical 字段；中文翻译只能作为理解辅助，不能直接成为 Gold。

### 7.8 生成、翻译并审核正式 QA

```bash
python salesbench.py realize-qa \
  --evidence-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 2

python salesbench.py build-audit-translations \
  --manifest configs/pilot64_gpt4o_v9_delivery.json \
  --output outputs/audit/translations/v9_pilot64/audit_translations.jsonl \
  --base-url "$OPENAI_BASE_URL" \
  --model gpt-4o \
  --max-workers 2
```

64 视频 pilot 的 QA 全量人工审核。中文翻译帮助阅读，最终修改对象仍是英文 Question 和 Gold。

### 7.9 编译公开 QA

```bash
python salesbench.py compile-vqa \
  --evidence-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --dataset-file video_evidence_dataset_reviewed.jsonl \
  --realizations outputs/vqa/v9_pilot64_gpt4o_yunwu/qa_realizations_reviewed.jsonl \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu \
  --max-questions-per-video 6 \
  --max-per-task 3
```

目标约 192–384 条 QA，以有效证据决定，不固定每视频题数。

### 7.10 跑模型和 LLM-as-Judge

```bash
python salesbench.py run-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --vqa outputs/vqa/v9_pilot64_gpt4o_yunwu/vqa_public.jsonl \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu/run_gpt4o \
  --base-url "$OPENAI_BASE_URL" \
  --model gpt-4o \
  --max-workers 2

python salesbench.py evaluate-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --gold outputs/vqa/v9_pilot64_gpt4o_yunwu/vqa_gold_private.jsonl \
  --predictions outputs/vqa/v9_pilot64_gpt4o_yunwu/run_gpt4o/predictions.jsonl \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu/evaluation_gpt4o \
  --base-url "$OPENAI_BASE_URL" \
  --judge-model gpt-4o \
  --max-workers 2
```

主排行榜是 BP/CM/SS/AE Judge score 的 macro-average。中文 Judge 理由只在评测完成后通过 audit translation 生成。

---

## 8. 最终验收标准

- `pytest -q`、compileall、`git diff --check` 全部通过。
- Prompt 全英文，Prompt audit 同时提供中文翻译。
- Canonical Evidence/Cue/Relation/Question/Gold/Judge 自然语言字段全英文。
- 中文 ASR/OCR 原文逐字保留并可定位。
- 中文审计翻译只存在于独立 sidecar 和 HTML，不进入 canonical JSONL。
- 每条翻译绑定 object ID、source field 和 source hash；stale 翻译不能静默使用。
- 中文翻译保留数字、单位、条件、否定、claim/fact 和 relation status。
- 公共 payload 不包含私有互动、画像、标题或审计翻译。
- 引用的 Evidence、Cue、Relation 同视频且可解析。
- 不存在消费者结果因果 Relation。
- Pilot 中保留的 Evidence/Cue/Relation/QA 全部有人工作出决定。
- 每条 QA 对应一个主要 Capability 和 Reasoning Operator。
- 64 视频 pilot 完全重复问题为 0；视频内语义重复为 0；最大标准化句式簇不超过 5%。
- CM 100% 具有真实跨模态信息差。
- SS/AE 不包含真实用户画像、信任、购买、互动或转化结论。
- 四任务在 cohort 层面非空，不强制单视频四任务齐全。
- 缺失模型答案计 0。
- Judge model/version/prompt 冻结并记录。
- 人工校准样本覆盖四任务；条件允许时覆盖五个 Judge 分档。
- Smoke、formal、translation sidecar 和 legacy 目录物理隔离。
