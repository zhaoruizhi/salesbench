# SalesBench Gold-Bank-First QA 设计说明

> 状态：待用户评审
>
> 日期：2026-07-15
>
> 范围：先完成 5 条视频的 Gold Bank pilot，再从 Gold Bank 派生 QA；不修改和覆盖现有 v9 产物。

## 1. 决策摘要

SalesBench-QA 的生成主流程从“问题优先”调整为“Gold 优先”：

```text
旧流程：
视频/上下文
  -> 多视角自由提出问题
  -> 筛选问题
  -> 临时生成答案
  -> LLM 验证答案
  -> QA Gold

新流程：
视频/上下文
  -> 抽取可定位的 Evidence Units
  -> 多 Agent 提出结构化 Gold Proposals
  -> Challenger 审查
  -> Adjudicator 合并与定级
  -> 确定性规则验证
  -> Video Gold Bank
  -> Task Planner 选择 Gold Items
  -> QA Compiler 派生问题和答案
  -> QA Validator
  -> QA Gold
```

现有 Multi-Agent 框架不是废弃，而是改变职责：

- 原来的 Proposer 从“问题生成者”变为“Gold 候选标注者”。
- 原来的 Challenger 从“问题质量审查者”变为“证据、标签和推理关系审查者”。
- 原来的 Synthesizer 改为 Adjudicator，负责合并同义 Gold、解决冲突和分配 Gold Tier。
- 原来的 Answerer 不再创造 Gold 答案；QA 答案由结构化 Gold Item 确定性渲染。
- 第五类 IP 的标签由真实互动数据确定性构建，不由 Agent 生成或校准。

## 2. 目标与非目标

### 2.1 目标

1. 每条视频尽可能完整地保存所有可靠、可追溯的 Gold 信息。
2. 观察事实和营销推理分层存储，不在同一自然语言答案中混杂。
3. 一个 Gold Item 可以派生多个题型，且所有题型共享相同事实和答案原子。
4. 后续可以改变任务配额、题型和措辞，而无需再次观看视频或重新调用 Gold Agent。
5. 保留每条候选、审查、拒绝和人工待审记录，便于解释数据构建过程。
6. 物理隔离点赞、评论、转发、收藏等私有结果，防止内容任务和被测模型看到效果标签。

### 2.2 非目标

1. Pilot 阶段不追求 1,200 条视频全量运行。
2. 不强制每条视频都具备 BP、CM、SS、AE、IP 五类 Gold。
3. 不把收藏称为购买转化，不生成因果归因 Gold。
4. 不将单个 Agent 的开放式解释直接标记为 Gold。
5. 不覆盖 `outputs/vqa/v1_0_multiagent_prompt_probe_v9_gpt54_20videos/`。
6. Pilot 阶段不正式生成 IP-Pair；IP-Pair 至少需要 20 条视频后再构建。

## 3. “尽可能完整可靠”的具体含义

Gold Bank 使用三层记录保证完整性和可靠性不冲突：

| 层级 | 文件 | 保存内容 | 是否可直接生成 QA |
| --- | --- | --- | --- |
| 候选层 | `gold_proposals.jsonl` | 所有 Agent 提出的候选，包括后来被拒绝的候选 | 否 |
| 待审层 | `human_review_queue.jsonl` | 证据存在但解释有歧义、Agent 不一致或需要人工判断的候选 | 否 |
| 规范层 | `video_gold_bank.jsonl` | 已通过证据、冲突、引用完整性和 Tier 规则的 Gold Items | 是 |

因此，“完整”不是把所有推测都放进正式 Gold Bank，而是所有候选都有去向且可追溯；“可靠”意味着只有通过规则的项目才可供 QA Compiler 使用。

Gold Bank 不设置“每类至少一条”的硬约束。每个视频的 `coverage` 字段必须记录：

```json
{
  "BP": {"eligible_count": 8, "status": "covered"},
  "CM": {"eligible_count": 3, "status": "covered"},
  "SS": {"eligible_count": 2, "status": "covered"},
  "AE": {"eligible_count": 0, "status": "insufficient_evidence"},
  "IP": {"eligible_count": 4, "status": "private_labels_available"}
}
```

## 4. 数据对象

### 4.1 EvidenceUnit

Evidence Unit 是视频中一个可引用的观察事实、文本片段或元数据事实。建议字段：

```json
{
  "evidence_id": "7365054535036849420_visual_0008_0011_001",
  "video_id": "7365054535036849420",
  "modality": "visual_ocr",
  "start_s": 8.0,
  "end_s": 11.0,
  "frame_indices": [3, 4],
  "text_span": "净含量450g",
  "subject": "面包包装",
  "predicate": "显示净含量",
  "value": "450g",
  "attributes": {},
  "source_domains": ["C6_raw_video"],
  "extractor": "objective_evidence_extractor",
  "confidence": 0.96,
  "timestamp_status": "available"
}
```

约束：

- `evidence_id` 在整个输出目录中唯一。
- 可见事实和推断结论不能写在同一 Evidence Unit 中。
- 没有时间戳时允许 `start_s/end_s=null`，同时写 `timestamp_status=unavailable`。
- 不能为了满足 schema 伪造时间戳、OCR 或 ASR 位置。
- 结构化 C1-C5 特征也可成为 evidence，但必须保留 `source_domains` 和原字段名。

### 4.2 GoldProposal

Gold Proposal 是尚未裁决的任务标签候选：

```json
{
  "proposal_id": "7365054535036849420_operator_cm_003",
  "video_id": "7365054535036849420",
  "source_agent": "operator",
  "task_type": "CM",
  "task_subtype": "CLAIM_PARTIAL_SUPPORT",
  "target": {"claim": "9个450克"},
  "proposed_gold": {
    "relation": "PARTIALLY_SUPPORTED",
    "supported_atoms": ["450g"],
    "unsupported_atoms": ["9个"]
  },
  "evidence_ids": [
    "7365054535036849420_asr_0007_0010_001",
    "7365054535036849420_visual_0008_0011_001"
  ],
  "reasoning_edges": [
    ["口播450克", "包装450g", "SUPPORTED"],
    ["口播9个", "未找到数量画面", "NOT_SHOWN"]
  ],
  "proposal_confidence": 0.91
}
```

### 4.3 GoldReview

Gold Review 保存 Challenger 对单个 proposal 的审查：

```json
{
  "review_id": "review_7365054535036849420_operator_cm_003",
  "proposal_id": "7365054535036849420_operator_cm_003",
  "video_id": "7365054535036849420",
  "reviewer": "gold_challenger",
  "verdict": "PASS",
  "checks": {
    "evidence_exists": true,
    "evidence_supports_gold": true,
    "fact_inference_separated": true,
    "task_type_valid": true,
    "alternative_interpretation_risk": "low",
    "duplicate_risk": "low"
  },
  "issues": [],
  "suggested_revision": null
}
```

`verdict` 只能为：

```text
PASS
REVISE
HUMAN_REVIEW
REJECT
```

### 4.4 GoldItem

Gold Item 是进入规范 Gold Bank 的最小监督单元：

```json
{
  "gold_id": "7365054535036849420_cm_003",
  "video_id": "7365054535036849420",
  "task_type": "CM",
  "task_subtype": "CLAIM_PARTIAL_SUPPORT",
  "target": {"claim": "9个450克"},
  "gold_value": {
    "relation": "PARTIALLY_SUPPORTED",
    "supported_atoms": ["450g"],
    "unsupported_atoms": ["9个"]
  },
  "evidence_ids": [
    "7365054535036849420_asr_0007_0010_001",
    "7365054535036849420_visual_0008_0011_001"
  ],
  "reasoning_edges": [
    ["口播450克", "包装450g", "SUPPORTED"],
    ["口播9个", "无数量证据", "NOT_SHOWN"]
  ],
  "eligible_question_formats": [
    "relation_choice",
    "supported_missing",
    "evidence_selection"
  ],
  "source_proposal_ids": ["7365054535036849420_operator_cm_003"],
  "gold_tier": "Gold-A",
  "review_status": "verified",
  "confidence": 0.94
}
```

### 4.5 VideoGoldRecord

`video_gold_bank.jsonl` 一行对应一条视频：

```json
{
  "video_id": "7365054535036849420",
  "schema_version": "goldbank_v1",
  "evidence_unit_ids": [
    "7365054535036849420_asr_0007_0010_001",
    "7365054535036849420_visual_0008_0011_001"
  ],
  "gold_items": [
    {"gold_id": "7365054535036849420_bp_001", "task_type": "BP"},
    {"gold_id": "7365054535036849420_cm_003", "task_type": "CM"}
  ],
  "private_interaction_ref": "interaction_gold/7365054535036849420",
  "coverage": {},
  "quality_summary": {
    "gold_a_count": 7,
    "gold_b_count": 3,
    "silver_count": 2,
    "rejected_count": 4
  },
  "observation_scope": {
    "frame_strategy": "hook_plus_uniform",
    "sampled_frame_count": 16,
    "asr_timestamp_available": false,
    "known_limitations": ["ASR仅有全文，无词级时间戳"]
  }
}
```

正式实现中 `gold_items` 保存完整 Gold Item；以上仅为缩略示例。

## 5. 五类任务在 Gold Bank 中的规范

### 5.1 BP

BP Gold 只保存可直接观察的事实：

```text
ENTITY_ATTRIBUTE
COUNT_SPATIAL
ACTION
STATE_CHANGE
TEMPORAL_ORDER
OCR_FACT
ASR_FACT
```

示例：

```json
{
  "task_type": "BP",
  "task_subtype": "STATE_CHANGE",
  "gold_value": {
    "subject": "手机支架",
    "before": "横屏",
    "after": "竖屏",
    "action": "旋转"
  }
}
```

### 5.2 CM

CM Gold 保存 Claim 与多模态证据的关系：

```text
SUPPORTED
PARTIALLY_SUPPORTED
CONTRADICTED
NOT_SHOWN
TEMPORALLY_MISALIGNED
```

一条 claim 必须先拆成原子，再分别判断证据关系，不能只保存“一致/不一致”的自然语言。

### 5.3 SS

SS Gold 使用受控营销机制本体：

```text
content_anchor
  -> mechanism
  -> target_barrier
  -> funnel_stage
```

核心本体：

```text
Hook: pain/problem, result_first, question, contrast, surprise
Value: function, price_value, convenience, health, aesthetics, education
Trust: demonstration, comparison, authority, social_proof, process, guarantee
Objection: price, authenticity, effectiveness, difficulty, risk
Urgency: time_limit, stock_limit, price_window
CTA: purchase, comment, collect, share
Funnel: attention, comprehension, trust, action
```

SS 只有在证据锚点、机制和影响对象均明确时才能成为 Gold-B；否则进入 Silver。

### 5.4 AE

AE Gold 使用受众—需求—情境结构：

```text
audience_need
usage_context
decision_state
content_motivation
supporting_evidence
uncertainty
```

受众推断至少需要两类独立证据；仅有商品品类或标题时不能标记高置信 Gold。

### 5.5 IP

IP 不由 Multi-Agent Proposer 生成。私有标签仅支持：

```text
IP_LEVEL: 点赞/评论/转发/收藏相对低中高档
IP_PROFILE: 四项分别标准化后的相对突出指标
IP_PAIR: 严格匹配视频的单指标成对比较
```

不支持：

```text
因果归因
购买转化
销量提升
唯一正确的优化建议
```

IP 的原始计数和派生标签必须与内容 Gold 物理隔离。

## 6. Multi-Agent 新架构

### 6.1 Stage 0：Context Assembly

复用现有 `SalesBenchContextStore`，组装 C1-C6 和私有 performance 数据。内容 Agent 永远只接收内容域。

Gold Bank pilot 将当前 8 帧采样改为可配置，建议首批使用 16 帧。`observation_scope` 必须记录抽帧策略，避免把“抽帧内未看到”表述为“完整视频不存在”。

### 6.2 Stage 1：Objective Evidence Extraction

`objective_evidence_extractor` 使用 VLM：

- 抽取人物、商品、道具、颜色、数量、位置。
- 抽取动作、状态变化和前后关系。
- 从可见画面抽取 OCR。
- 将 ASR/标题/商品标题拆成可引用 claim。
- 输出 EvidenceUnit，不输出问题。

本地 Normalizer 负责：

- ID 生成。
- 枚举和值域校验。
- 完全重复合并。
- 时间范围合法性检查。
- 保留原始 Agent 输出用于审计。

### 6.3 Stage 2：Gold Proposal

Gold Proposal 分为两类：

1. 任务构造器：
   - BP Builder 从 Evidence Units 确定性构造 BP 候选。
   - CM Agent 将 ASR/OCR/title claim 与视觉证据建立关系。
2. 多视角推理：
   - Consumer 提出需求、疑虑、场景和受众候选。
   - Operator 提出内容结构、跨模态关系和 CTA 候选。
   - Strategist 提出 Hook、信任、风险消除、紧迫感和漏斗候选。

所有 proposal 必须引用已有 `evidence_id`。无法引用证据时输出 `insufficient_evidence`，不能补写想象证据。

### 6.4 Stage 3：Gold Challenge

Challenger 对 Gold 而不是问题执行检查：

- 引用的 Evidence Unit 是否存在。
- Gold 是否被这些证据支持。
- 是否把观察事实写成心理或效果结论。
- task/subtype 是否正确。
- 是否存在更合理的替代解释。
- 是否与同视频其他 Gold 冲突。
- 是否与已有候选同义重复。
- 是否应该进入人工复核。

### 6.5 Stage 4：Adjudication

Adjudicator：

- 合并同义 proposals。
- 保留不同但不冲突的解释。
- 对冲突 proposal 选择、降级或送人工。
- 生成规范 Gold Item。
- 生成 `coverage` 和 `quality_summary`。

Adjudicator 不能新增 proposal 中不存在的证据。

### 6.6 Stage 5：Deterministic Validation

本地代码最终验证：

- Schema。
- Evidence 外键。
- ID 唯一性。
- 时间范围。
- 任务本体和值域。
- Gold Tier 条件。
- 同 anchor 重复。
- 互斥关系冲突。
- 私有 performance 泄漏。

任何确定性检查失败的 Gold 不能因为 LLM 说 PASS 而进入规范 Bank。

### 6.7 Stage 6：IP Private Label Builder

独立规则模块从真实互动数据生成私有 IP 标签：

- 数据划分优先于阈值拟合。
- 阈值只使用 train split。
- 默认三档，不继续使用五档。
- `conversion_score` 改名为收藏响应相关名称。
- 不向任何内容 Agent 发送原始值或标签。

## 7. Gold Tier

| Tier | 条件 | 是否可自动编译 QA |
| --- | --- | --- |
| Gold-A | 直接观察事实、明确 CM 关系或确定性 IP 标签；证据引用完整；规则验证通过 | 是 |
| Gold-B | SS/AE 受控本体推理；至少两个独立 proposal 一致或人工确认；证据完整 | 是 |
| Silver | 单 Agent 推理、存在替代解释、证据不足以唯一确定或待人工审核 | 否 |
| Rejected | 无证据、冲突、重复、越界或标签类型错误 | 否 |

`auto_generated_pending_review` 不能直接使用 Gold-A/B 名称。Gold Tier 必须由 provenance 和验证条件决定。

## 8. 输出目录与每个文件的含义

Pilot 输出目录：

```text
outputs/goldbank/v1_pilot_5videos/
├── video_samples.jsonl
├── evidence_units.jsonl
├── gold_proposals.jsonl
├── gold_reviews.jsonl
├── video_gold_bank.jsonl
├── interaction_gold_private.jsonl
├── human_review_queue.jsonl
├── agent_traces.jsonl
└── generation_meta.json
```

### 8.1 `video_samples.jsonl`

**粒度：** 一行一条入选视频。

**作用：** 固定 pilot 样本，记录视频路径、品类、达人层级、时长和采样原因，使后续重复运行使用相同视频。

**产生者：** Gold Bank Runner。

**消费者：** 抽帧器、Context Store、复现实验脚本。

**注意：** 该文件包含本地资源路径，属于内部构建产物，不直接公开发布。

示例：

```json
{
  "video_id": "7365054535036849420",
  "primary_video_path": "videos/...mp4",
  "product_bucket": "食品",
  "fan_segment": "头腰",
  "video_duration_s": 53,
  "selection_reason": "CM部分支撑案例"
}
```

### 8.2 `evidence_units.jsonl`

**粒度：** 一行一个 Evidence Unit，而不是一行一个视频。

**作用：** 保存所有规范化、可引用的事实证据，是 Gold Bank 的外键表。

**产生者：** Objective Evidence Extractor + Normalizer。

**消费者：** Gold Proposer、Challenger、Adjudicator、QA Compiler、人工审核界面。

**为什么单独保存：** 同一证据可被多个 BP/CM/SS/AE Gold 引用，避免复制长文本和重复画面描述。

### 8.3 `gold_proposals.jsonl`

**粒度：** 一行一个 Agent Gold 候选。

**作用：** 保存所有原始候选，包括 PASS、待修订和最终被拒绝的内容，是“候选完整性”和审计依据。

**产生者：** BP Builder、CM Agent、Consumer、Operator、Strategist。

**消费者：** Challenger、Adjudicator、质量分析脚本。

**注意：** proposal 不等于 Gold，不能直接用于 benchmark。

### 8.4 `gold_reviews.jsonl`

**粒度：** 一行一次对某个 proposal 的审查。

**作用：** 保存 Challenger 的 verdict、逐项检查、问题和修改意见。

**产生者：** Gold Challenger。

**消费者：** Adjudicator、人工审核队列生成器、质控报告。

一个 proposal 在重试或人工复核后可以有多条 review，通过 `review_id` 和 `proposal_id` 关联。

### 8.5 `video_gold_bank.jsonl`

**粒度：** 一行一条视频。

**作用：** 规范 Gold Bank 主文件，保存该视频所有已接受的 Gold-A/Gold-B Items、证据引用、任务覆盖和质量摘要。

**产生者：** Adjudicator + Deterministic Validator。

**消费者：** QA Compiler、数据统计、人工 Gold 抽检。

**准入规则：** Silver 和 Rejected 不进入 `gold_items`。它们分别保存在 review queue 和 proposals/reviews 中。

### 8.6 `interaction_gold_private.jsonl`

**粒度：** 一行一条视频。

**作用：** 保存 IP 所需的真实点赞/评论/转发/收藏、训练集拟合的相对分数和离散标签。

**产生者：** IP Private Label Builder，不调用 LLM。

**消费者：** IP-Level/IP-Profile/IP-Pair 构造器和评估器。

**安全边界：**

- 不发送给 Evidence Extractor、Gold Proposer、Challenger、Adjudicator。
- 不进入 BP/CM/SS/AE QA。
- 不进入公开模型输入。
- 公开数据导出时默认排除整个文件。

### 8.7 `human_review_queue.jsonl`

**粒度：** 一行一个待人工决定的 Gold 候选。

**作用：** 将 Silver、Agent 冲突、替代解释风险高、证据边界不清的条目整理成人工可操作任务。

示例：

```json
{
  "review_item_id": "hr_736059_ae_002",
  "proposal_id": "736059_consumer_ae_002",
  "video_id": "7360594186715909416",
  "task_type": "AE",
  "question_for_reviewer": "证据是否足以推断为低饱和裸唇需求人群？",
  "evidence_ids": ["..."],
  "candidate_gold": {},
  "conflict_summary": "Consumer与Strategist对受众范围不一致",
  "allowed_decisions": ["ACCEPT_GOLD_B", "REVISE", "REJECT"]
}
```

人工决定另存为 `human_review_decisions.jsonl`，之后由导入命令提升、修订或拒绝对应候选。

### 8.8 `agent_traces.jsonl`

**粒度：** 一行一条视频的完整 Agent 调用轨迹。

**作用：** 保存模型、耗时、token、费用、原始响应、解析结果和错误，复用现有 trace 机制。

**消费者：** 成本统计、错误诊断和复现实验。

### 8.9 `generation_meta.json`

**粒度：** 整个运行一个 JSON 对象。

**作用：** 记录运行配置和总览：

```json
{
  "schema_version": "goldbank_v1",
  "model": "...",
  "seed": 42,
  "max_videos": 5,
  "video_ids": [],
  "frame_strategy": "hook_plus_uniform",
  "frames_per_video": 16,
  "prompt_versions": {},
  "input_paths": {},
  "input_checksums": {},
  "counts": {
    "videos": 5,
    "evidence_units": 0,
    "proposals": 0,
    "accepted_gold_items": 0,
    "human_review_items": 0,
    "rejected_items": 0
  },
  "task_distribution": {},
  "gold_tier_distribution": {},
  "failures": [],
  "total_cost_usd": 0.0,
  "total_elapsed_s": 0.0
}
```

### 8.10 文件之间如何关联

所有文件通过稳定 ID 连接，不依赖数组位置或文件行号：

```text
video_samples.video_id
  ├── evidence_units.video_id
  │     └── evidence_units.evidence_id
  │             ├── gold_proposals.evidence_ids[]
  │             ├── video_gold_bank.gold_items[].evidence_ids[]
  │             └── human_review_queue.evidence_ids[]
  ├── gold_proposals.video_id
  │     └── gold_proposals.proposal_id
  │             ├── gold_reviews.proposal_id
  │             ├── video_gold_bank.gold_items[].source_proposal_ids[]
  │             └── human_review_queue.proposal_id
  ├── video_gold_bank.video_id
  │     └── video_gold_bank.gold_items[].gold_id
  │             └── qa_plan.gold_id
  └── interaction_gold_private.video_id
        └── video_gold_bank.private_interaction_ref
```

隐私和发布边界：

| 文件 | 内部构建 | 人工审核 | QA 编译 | 可公开发布 |
| --- | --- | --- | --- | --- |
| `video_samples.jsonl` | 是 | 可选 | 否 | 否，含本地路径 |
| `evidence_units.jsonl` | 是 | 是 | 是 | 可生成脱敏版本 |
| `gold_proposals.jsonl` | 是 | 可选 | 否 | 否，属于审计过程 |
| `gold_reviews.jsonl` | 是 | 是 | 否 | 通常不公开 |
| `video_gold_bank.jsonl` | 是 | 是 | 是 | 仅 Gold 私有版本 |
| `interaction_gold_private.jsonl` | 是 | 否 | 仅 IP 编译器 | 否 |
| `human_review_queue.jsonl` | 是 | 是 | 否 | 否 |
| `agent_traces.jsonl` | 是 | 否 | 否 | 否 |
| `generation_meta.json` | 是 | 可选 | 可选 | 可发布脱敏统计 |

## 9. 从 Gold Bank 生成 QA

QA 编译是下游独立流程：

```text
video_gold_bank.jsonl + evidence_units.jsonl
  -> Task Planner
  -> Question Program
  -> Deterministic Answer Renderer
  -> Optional Paraphraser
  -> QA Validator
  -> vqa_gold.jsonl
```

### 9.1 Task Planner

- 只选择 `Gold-A/Gold-B + review_status=verified`。
- 数据集级控制任务比例，不强制每视频五类齐全。
- 同一 `gold_id` 默认只派生一道正式测试题；数据增强集可以派生多种问法。
- 控制同 anchor、同对象、同属性重复。

输出：`qa_plan.jsonl`。

```json
{
  "qa_plan_id": "plan_736505_cm_003_01",
  "video_id": "7365054535036849420",
  "gold_id": "7365054535036849420_cm_003",
  "task_type": "CM",
  "question_format": "supported_missing"
}
```

### 9.2 Question Renderer

Renderer 将 `task_subtype + question_format + gold_value` 转换为问题和标准答案，不重新推断事实。

同一 Gold Item 可支持：

```text
relation_choice
supported_missing
evidence_selection
```

答案从 `gold_value` 渲染；禁止 Answerer 自由创造新 Gold。

### 9.3 Optional Paraphraser

Paraphraser 只能改变措辞，必须保持：

```text
gold_id 不变
task_subtype 不变
required answer atoms 不变
evidence_ids 不变
question scope 不变
```

无法通过回译/结构检查的改写退回模板版本。

### 9.4 QA Validator

- 问题可回答性。
- 问题是否泄露答案。
- answer atoms 是否等于 Gold Item。
- Evidence IDs 是否完整。
- 同义重复与同 anchor 重复。
- 任务类型和题型是否匹配。
- IP 标签是否从 private ref 正确复制且未泄漏数值。

建议 QA 输出：

```text
outputs/vqa/v2_goldbank_pilot_5videos/
├── qa_plan.jsonl
├── qa_candidates.jsonl
├── qa_validation.jsonl
├── vqa_gold_private.jsonl
├── vqa_public.jsonl
└── generation_meta.json
```

`vqa_gold_private.jsonl` 用于评估；`vqa_public.jsonl` 删除答案、Evidence Gold 和私有互动标签，作为被测模型输入。

## 10. 代码结构调整

### 10.1 保留的旧代码

第一阶段不删除、不覆盖：

```text
src/salesbench/multiagent/pipeline.py
src/salesbench/multiagent/runner.py
src/salesbench/multiagent/prompts.py
outputs/vqa/v1_0_multiagent_prompt_probe_v9_gpt54_20videos/
```

这些作为 question-first 基线，便于与 v10/v2 gold-first 对比。

### 10.2 新增 Gold Bank 包

```text
src/salesbench/goldbank/
├── __init__.py
├── schema.py
├── ontology.py
├── prompts.py
├── normalizer.py
├── validators.py
├── pipeline.py
├── interaction_labels.py
└── runner.py
```

职责：

| 文件 | 责任 |
| --- | --- |
| `schema.py` | EvidenceUnit、GoldProposal、GoldReview、GoldItem、VideoGoldRecord、GoldBankResult |
| `ontology.py` | 五类任务 subtype、关系值域、SS/AE 本体、Tier 规则 |
| `prompts.py` | Evidence Extractor、CM、三视角 Proposer、Challenger、Adjudicator prompts |
| `normalizer.py` | LLM JSON 解析后的 ID、枚举、文本、重复规范化 |
| `validators.py` | 外键、冲突、重复、时间、Tier、性能泄漏的确定性校验 |
| `pipeline.py` | 单视频 Stage 0-5 编排，不负责写文件 |
| `interaction_labels.py` | 私有 IP-Level/IP-Profile 标签，后续扩展 IP-Pair |
| `runner.py` | 视频抽样、并发、断点、文件汇总、meta 和 trace 写出 |

### 10.3 新增 QA Compiler

```text
src/salesbench/vqa/compiler.py
src/salesbench/vqa/question_programs.py
src/salesbench/vqa/goldbank_loader.py
```

| 文件 | 责任 |
| --- | --- |
| `goldbank_loader.py` | 加载 Evidence 外键表、Video Gold Bank 和私有 IP refs |
| `question_programs.py` | 各 task/subtype 可用题型及确定性 answer renderer |
| `compiler.py` | 数据集级任务规划、QA 渲染、可选改写、验证和 public/private 导出 |

### 10.4 修改共享代码

```text
src/salesbench/cli.py
src/salesbench/multiagent/context.py
src/salesbench/vqa/schema.py
```

修改内容：

- `cli.py` 增加 Gold Bank 和 QA Compiler 命令。
- `multiagent/context.py` 暂时复用 `SalesBenchContextStore`；新增 Gold-specific 可见域方法，但不破坏旧路由。
- `vqa/schema.py` 增加 `gold_id`、`task_subtype`、`question_format` 和 public/private 字段过滤。

### 10.5 新增测试

```text
tests/test_goldbank_schema.py
tests/test_goldbank_pipeline.py
tests/test_goldbank_outputs.py
tests/test_goldbank_privacy.py
tests/test_vqa_goldbank_compiler.py
```

核心测试：

1. Evidence 外键不存在时 Gold 被拒绝。
2. 同一 Gold 的多种题型答案原子一致。
3. Silver 不进入 QA Plan。
4. IP 原始数值不会出现在内容 Agent prompt、video Gold 或 public QA。
5. Gold Bank 不强制五类覆盖。
6. Challenger 失败或返回非法 JSON 时不得默认 PASS。
7. 同 anchor 重复 Gold 被合并或拒绝。
8. 冲突 Gold 进入人工队列。
9. 输出文件行数、ID 和关联关系满足 contract。
10. 固定 seed 和固定 mock responses 时输出可复现。

### 10.6 推荐代码实施顺序

实施顺序按“每一步都有可测试产物”划分：

1. **Schema 与本体**
   - 新建 `goldbank/schema.py`、`goldbank/ontology.py`。
   - 完成 JSON round-trip、枚举和值域测试。
2. **Normalizer 与确定性 Validator**
   - 新建 `normalizer.py`、`validators.py`。
   - 先用手工 fixture 验证 Evidence 外键、重复、冲突、Tier 和隐私规则。
3. **Evidence Extraction**
   - 新建 Gold prompts 和 `GoldBankPipeline._extract_evidence()`。
   - 先用一个视频和 mock client 输出 `evidence_units.jsonl`。
4. **Gold Proposal、Challenge、Adjudication**
   - 依次增加三个阶段。
   - 每增加一个阶段就扩充 `GoldBankResult` 和 contract test。
5. **Runner 与完整输出集**
   - 完成固定 video IDs、16 帧、并发、trace、failure 和 meta。
   - 使用 fake client 运行 1 视频集成测试，确认九个输出文件的关联完整。
6. **IP Private Label Builder**
   - 从 Agent 流程外部构建私有标签。
   - 完成 train-only 阈值和 performance leakage 测试。
7. **QA Compiler**
   - 先实现 BP/CM 的确定性 question programs，再实现 SS/AE，最后接 IP。
   - 默认关闭 paraphraser，确保相同 Bank 和 seed 输出字节级稳定。
8. **CLI 与 5 视频 Pilot**
   - 注册 `build-gold-bank`、`apply-gold-reviews`、`compile-vqa-from-gold`。
   - 运行固定 5 视频并进行人工结构审查。
9. **20 视频与 IP-Pair**
   - 只有 5 视频 schema 和 QA 派生通过验收后再实施。

## 11. CLI 与后续使用方式

### 11.1 构建 5 视频 Gold Bank

计划新增：

```bash
python salesbench.py build-gold-bank \
  --model gpt-5.4 \
  --max-videos 5 \
  --frames-per-video 16 \
  --max-workers 2 \
  --seed 42 \
  --output-dir outputs/goldbank/v1_pilot_5videos
```

为保证每次 pilot 使用同一组视频，命令还应支持：

```bash
--video-ids-file configs/goldbank_pilot_5_video_ids.txt
```

建议首批固定：

```text
7362856271365606683
7360594186715909416
7360565413341613348
7359538830657064192
7365054535036849420
```

### 11.2 导入人工决定

计划新增：

```bash
python salesbench.py apply-gold-reviews \
  --gold-bank-dir outputs/goldbank/v1_pilot_5videos \
  --decisions outputs/goldbank/v1_pilot_5videos/human_review_decisions.jsonl
```

该命令不修改原 proposals/reviews，只生成新的 canonical bank 版本和 review audit。

### 11.3 从 Gold Bank 编译 QA

计划新增：

```bash
python salesbench.py compile-vqa-from-gold \
  --gold-bank-dir outputs/goldbank/v1_pilot_5videos \
  --questions-per-video 5 \
  --seed 42 \
  --output-dir outputs/vqa/v2_goldbank_pilot_5videos
```

QA 生成不再要求调用 VLM；仅当开启可选 paraphraser 时需要模型：

```bash
--paraphrase-model gpt-5.4
```

关闭 paraphraser 时应完全确定性、低成本且可复现。

### 11.4 后续 20 视频 IP-Pair

Gold Bank schema 和 5 视频 pilot 稳定后：

```bash
python salesbench.py build-gold-bank \
  --video-ids-file configs/goldbank_pilot_20_video_ids.txt \
  --output-dir outputs/goldbank/v1_pilot_20videos

python salesbench.py build-ip-pairs \
  --gold-bank-dir outputs/goldbank/v1_pilot_20videos \
  --output outputs/goldbank/v1_pilot_20videos/interaction_pair_gold_private.jsonl
```

IP-Pair 使用细粒度 product family、同 fan segment、相近粉丝量和时长、足够指标差距，不复用当前粗粒度宽松匹配规则。

## 12. 错误处理与恢复

### 12.1 Agent 调用失败

- 保存失败 trace。
- 对应阶段标记 `failed`。
- 不允许像旧 Challenger fallback 一样自动将所有候选标为 PASS。
- Runner 支持按 `video_id + stage` 断点重试。

### 12.2 非法 JSON

- 保存原始响应。
- 允许一次结构修复调用。
- 再次失败则进入 failure，不产生规范 Gold。

### 12.3 Evidence 缺失

- Gold Proposal 保留在 audit 文件。
- verdict 为 `REJECT` 或 `HUMAN_REVIEW`。
- 不进入 Video Gold Bank。

### 12.4 Agent 冲突

- 不使用简单多数票覆盖差异。
- 若本体标签互斥，送人工队列。
- 若是可并存的不同解释，可分别保留，但必须各自有证据和明确 scope。

## 13. Pilot 验收标准

5 视频 pilot 的第一目标是验证结构，不以 QA 数量为成功标准。

### 13.1 数据完整性

- `evidence_id/proposal_id/review_id/gold_id` 唯一率 100%。
- Gold Item Evidence 外键完整率 100%。
- 每个 proposal 至少有一条 review。
- 每个被拒绝或待审 proposal 有明确原因。
- performance 私有字段泄漏数为 0。

### 13.2 Gold 质量

- BP/CM Gold 人工抽检事实冲突数为 0。
- 同视频同 anchor 重复 Gold 为 0。
- Gold-A 全部满足确定性准入规则。
- Gold-B 全部具有至少两个独立 evidence domains。
- SS/AE 无证据推断不得进入规范 Bank。

### 13.3 QA 派生

- 每个 QA 都能回溯到唯一 `gold_id`。
- QA 答案原子与 Gold Item 一致率 100%。
- 同一 Gold 派生不同问法时不存在答案冲突。
- 关闭 paraphraser 时重复运行结果完全一致。

## 14. 迁移策略

1. 冻结 v9，不修改旧 output。
2. 新建独立 Gold Bank 包和命令。
3. 使用固定 5 视频生成 Gold Bank。
4. 人工阅读 `video_gold_bank.jsonl`、`human_review_queue.jsonl` 和证据引用。
5. 从同一 Bank 编译第一版 QA。
6. 比较 v9 与 gold-first：重复率、证据完整率、Gold 冲突率、人工通过率。
7. 结构通过后扩展到 20 视频，并构建 IP-Pair。
8. 20 视频稳定后再决定是否全量运行 1,200 视频。

## 15. 预期改进

| 当前问题 | Gold-first 对应改进 |
| --- | --- |
| 问题模板压缩候选空间 | 先完整抽取 Gold，再从 task program 选题 |
| 同一事实出现多个近义题 | 一个事实一个 gold_id，QA 层控制复用 |
| 答案临时生成并互相冲突 | 答案由 gold_value 确定性渲染 |
| Evidence 是长段自然语言 | Evidence Unit 可定位、可引用、可复用 |
| SS/AE 推断标准不清 | 受控本体、证据要求和 Tier 准入 |
| PA 为 0 或被结果过滤 | IP 私有标签确定性构建，错误预测不删题 |
| 修改 prompt 后需要重新观看视频 | 可反复从同一 Gold Bank 编译不同 QA 版本 |

## 16. 已确认的实施决策

根据后续讨论，以下设计作为实现基线：

1. 新建 `src/salesbench/goldbank/`，不在旧 `multiagent/pipeline.py` 上直接重写。
2. `video_gold_bank.jsonl` 只收录通过验证的 Gold-A/Gold-B；所有原始候选仍保留在 proposals/reviews。
3. IP 原始结果独立保存在 `interaction_gold_private.jsonl`。
4. 第一批固定 5 条视频、每条 16 帧，不设置五类覆盖硬指标。
5. QA Compiler 默认确定性运行，模型 paraphraser 是可选能力。
