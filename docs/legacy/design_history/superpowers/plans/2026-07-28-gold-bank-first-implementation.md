# SalesBench Gold-Bank-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` while implementing each task and `superpowers:verification-before-completion` before reporting completion.

**Goal:** 在不破坏现有 question-first 实验的前提下，实现“证据抽取 → Gold 提议 → 质疑与裁决 → 规则验证 → Gold Bank → QA 编译”的可复现流水线，并用固定 5 条视频完成首轮质量验收。

**Architecture:** 新增独立的 `salesbench.goldbank` 包，复用现有 `SalesBenchContextStore`、抽帧器和 `VLMClient`，但不复用旧 `MultiAgentQAPipeline` 的 question-first 状态机。Gold Bank 是唯一事实与受控解释来源；QA 是 Gold 的派生视图。BP/CM/SS/AE 使用内容证据，IP 使用隔离的互动标签，并通过 public/private 输出防火墙避免把真实互动数据泄漏给作答模型。

**Tech Stack:** Python 3.13、标准库 `dataclasses`/`enum`/`json`/`concurrent.futures`、现有 OpenAI-compatible `VLMClient`、现有帧采样器、`unittest`。

---

## 0. 实施边界与完成定义

### 0.1 本计划包含

- 五类任务统一 Gold 数据模型：BP、CM、SS、AE、IP。
- Evidence Extractor、Gold Proposer、Challenger、Adjudicator 四阶段编排。
- 证据引用、Gold tier、冲突、泄漏、重复和完整性验证。
- 互动指标的私有标签构造，阈值只在训练集上拟合。
- 固定 5 视频 pilot 的可复现运行入口和九类输出文件。
- 从 Gold Bank 编译 QA 的确定性程序。
- 人工复核结果回写和版本追踪。
- 单元测试、集成测试、dry-run 和 pilot 验收。

### 0.2 本计划不包含

- 不删除或改写 `salesbench.multiagent` 旧流水线；已有 v9 结果保持可复现。
- 不在首个 5 视频 pilot 中构造 IP pairwise 题；pairwise 在 Gold Bank 通过 20 视频验收后单独实施。
- 不把互动标签解释成营销因果归因，也不声称点赞/收藏/转发代表真实销售转化。
- 不立即把 1200 条扩展到 2000+；是否扩容由 100 条质控阶段的数据饱和度和置信区间决定。

### 0.3 Definition of Done

以下条件必须同时满足，才算首轮实现完成：

1. 全量单元测试通过，旧测试无回归。
2. `build-gold-bank` 对固定 5 个视频产生九个规定文件且可断点续跑。
3. 每个 accepted Gold Item 的 `evidence_ids` 都能解析到本视频 Evidence Unit。
4. public Gold/QA 文件中不存在 `likes`、`collects`、`shares`、`comments`、原始互动分数和阈值。
5. 无法解析、无证据或存在冲突的项目进入复核队列，不能静默视为 PASS。
6. `compile-vqa-from-gold` 只消费 Gold-A/Gold-B，重复运行字节级一致。
7. 5 个视频每个至少具备 BP、CM；有充分证据时再具备 SS、AE；IP 标签独立存放。

### 0.4 Git 说明

当前工作目录没有 `.git`，实施时不能执行提交。每个任务末尾仍保留“验证检查点”；若之后恢复 Git 仓库，可在对应检查点按任务边界提交，不能把无关用户改动混入提交。

---

## Task 1：建立 Gold Bank 数据契约与任务本体

**Files:**

- Create: `src/salesbench/goldbank/__init__.py`
- Create: `src/salesbench/goldbank/schema.py`
- Create: `src/salesbench/goldbank/ontology.py`
- Test: `tests/test_goldbank_schema.py`

### Step 1.1：先写失败测试

在 `tests/test_goldbank_schema.py` 覆盖以下行为：

```python
class GoldBankSchemaTest(unittest.TestCase):
    def test_gold_item_serializes_enum_values(self): ...
    def test_every_task_has_allowed_subtypes_and_question_formats(self): ...
    def test_ip_is_not_named_attribution_or_conversion(self): ...
    def test_gold_id_is_stable_for_same_semantics(self): ...
    def test_evidence_id_is_video_scoped(self): ...
```

运行：

```bash
python -m unittest tests.test_goldbank_schema -v
```

预期：因 `salesbench.goldbank` 尚不存在而失败。

### Step 1.2：实现枚举和 dataclass

`schema.py` 必须定义下列公开接口：

```python
class GoldTaskType(str, Enum):
    BP = "BP"
    CM = "CM"
    SS = "SS"
    AE = "AE"
    IP = "IP"

class EvidenceModality(str, Enum):
    VISUAL = "visual"
    ASR = "asr"
    OCR = "ocr"
    TITLE = "title"
    METADATA = "metadata"
    CROSS_MODAL = "cross_modal"

class GoldTier(str, Enum):
    GOLD_A = "Gold-A"
    GOLD_B = "Gold-B"
    SILVER = "Silver"
    REJECTED = "Rejected"

class ReviewVerdict(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REJECT = "REJECT"

@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    video_id: str
    modality: EvidenceModality
    start_s: float | None
    end_s: float | None
    frame_indices: tuple[int, ...]
    text_span: str
    subject: str
    predicate: str
    value: object
    attributes: dict[str, object]
    source_domains: tuple[str, ...]
    extractor: str
    confidence: float
    timestamp_status: str

@dataclass(frozen=True)
class GoldProposal:
    proposal_id: str
    video_id: str
    source_agent: str
    task_type: GoldTaskType
    task_subtype: str
    target: dict[str, object]
    proposed_gold: dict[str, object]
    evidence_ids: tuple[str, ...]
    reasoning_edges: tuple[tuple[str, str, str], ...]
    proposal_confidence: float

@dataclass(frozen=True)
class GoldReview:
    review_id: str
    proposal_id: str
    video_id: str
    reviewer: str
    verdict: ReviewVerdict
    checks: dict[str, object]
    issues: tuple[str, ...]
    suggested_revision: dict[str, object] | None

@dataclass(frozen=True)
class GoldItem:
    gold_id: str
    video_id: str
    task_type: GoldTaskType
    task_subtype: str
    target: dict[str, object]
    gold_value: dict[str, object]
    evidence_ids: tuple[str, ...]
    reasoning_edges: tuple[tuple[str, str, str], ...]
    eligible_question_formats: tuple[str, ...]
    source_proposal_ids: tuple[str, ...]
    gold_tier: GoldTier
    review_status: str
    confidence: float

@dataclass(frozen=True)
class VideoGoldRecord:
    video_id: str
    schema_version: str
    evidence_unit_ids: tuple[str, ...]
    gold_items: tuple[GoldItem, ...]
    private_interaction_ref: str
    coverage: dict[str, object]
    quality_summary: dict[str, int]
    observation_scope: dict[str, object]
```

每个 dataclass 提供 `to_dict()`；枚举序列化为 `.value`，tuple 序列化为 list。提供：

```python
def make_evidence_id(video_id: str, modality: EvidenceModality, ordinal: int) -> str
def make_gold_id(
    video_id: str,
    task_type: GoldTaskType,
    task_subtype: str,
    target: dict[str, object],
    gold_value: dict[str, object],
) -> str
def parse_evidence_unit(record: dict[str, object]) -> EvidenceUnit
def parse_gold_item(record: dict[str, object]) -> GoldItem
def parse_video_gold_record(record: dict[str, object]) -> VideoGoldRecord
```

ID 使用规范化内容的 SHA-256 前 12 位，不依赖运行顺序。`confidence` 必须位于 `[0, 1]`，非法输入直接抛 `ValueError`。`reasoning_edges` 只保存可审计的“证据原子—结论原子—关系”三元组，不保存模型隐藏思维链。

### Step 1.3：定义五任务本体

`ontology.py` 定义不可由 prompt 随意扩展的 subtype、关系与本体：

```python
BP_SUBTYPES = {
    "ENTITY_ATTRIBUTE", "COUNT_SPATIAL", "ACTION", "STATE_CHANGE",
    "TEMPORAL_ORDER", "OCR_FACT", "ASR_FACT",
}

CM_RELATIONS = {
    "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED",
    "NOT_SHOWN", "TEMPORALLY_MISALIGNED",
}

SS_ONTOLOGY = {
    "hook": {"pain_problem", "result_first", "question", "contrast", "surprise"},
    "value": {"function", "price_value", "convenience", "health", "aesthetics", "education"},
    "trust": {"demonstration", "comparison", "authority", "social_proof", "process", "guarantee"},
    "objection": {"price", "authenticity", "effectiveness", "difficulty", "risk"},
    "urgency": {"time_limit", "stock_limit", "price_window"},
    "cta": {"purchase", "comment", "collect", "share"},
    "funnel": {"attention", "comprehension", "trust", "action"},
}

AE_FIELDS = {
    "audience_need", "usage_context", "decision_state",
    "content_motivation", "supporting_evidence", "uncertainty",
}

IP_SUBTYPES = {
    "LIKE_RESPONSE_LEVEL", "COLLECT_RESPONSE_LEVEL",
    "SHARE_RESPONSE_LEVEL", "COMMENT_RESPONSE_LEVEL",
}

TASK_MIN_EVIDENCE = {
    GoldTaskType.BP: 1,
    GoldTaskType.CM: 2,
    GoldTaskType.SS: 2,
    GoldTaskType.AE: 2,
    GoldTaskType.IP: 0,
}
```

`IP` 的模块说明必须使用“互动响应模式/水平”，禁止“归因、转化、购买、销售效果”命名。

### Step 1.4：验证检查点

```bash
python -m unittest tests.test_goldbank_schema -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

---

## Task 2：实现规范化、规则验证与信息防火墙

**Files:**

- Create: `src/salesbench/goldbank/normalizer.py`
- Create: `src/salesbench/goldbank/validators.py`
- Test: `tests/test_goldbank_validators.py`

### Step 2.1：先写失败测试

覆盖：

```python
class GoldBankValidatorTest(unittest.TestCase):
    def test_missing_evidence_reference_is_error(self): ...
    def test_cross_video_evidence_reference_is_error(self): ...
    def test_ss_single_evidence_cannot_be_gold_b(self): ...
    def test_bp_direct_observation_can_be_gold_a(self): ...
    def test_inference_language_downgrades_bp_to_silver(self): ...
    def test_private_performance_fields_never_enter_public_record(self): ...
    def test_duplicate_semantics_are_reported(self): ...
    def test_conflicting_values_are_reported(self): ...
```

### Step 2.2：实现标准化

`normalizer.py` 提供：

```python
def normalize_text(value: object) -> str
def normalize_task_subtype(task_type: GoldTaskType, value: object) -> str
def normalize_evidence_units(video_id: str, raw_units: list[dict[str, object]]) -> list[EvidenceUnit]
def normalize_proposals(video_id: str, proposer: str, raw: list[dict[str, object]]) -> list[GoldProposal]
def semantic_key(item: GoldProposal | GoldItem) -> tuple[str, str, str, str]
```

行为要求：

- 去除空白和不可见字符，但不能改写原意。
- 无效枚举、未知谓词、空 subject/value、越界时间戳必须抛出或转成验证错误。
- LLM 自报的 `evidence_id` 若不存在，禁止自动猜测映射。
- 保留原始模型响应在 trace；规范化结果中不保留长 chain-of-thought，只保留可由 evidence 核验的 `reasoning_edges`。

`semantic_key` 由 `task_type + task_subtype + canonical_target + canonical_gold_value` 组成。

### Step 2.3：实现验证器

`validators.py` 提供：

```python
@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str       # ERROR | WARNING
    item_id: str
    message: str

def validate_evidence_unit(unit: EvidenceUnit) -> list[ValidationIssue]
def validate_gold_proposal(proposal: GoldProposal, evidence: dict[str, EvidenceUnit]) -> list[ValidationIssue]
def validate_gold_item(item: GoldItem, evidence: dict[str, EvidenceUnit]) -> list[ValidationIssue]
def find_duplicate_and_conflicting_items(items: list[GoldItem]) -> list[ValidationIssue]
def public_gold_record(item: GoldItem) -> dict[str, object]
```

固定错误码：

- `EVIDENCE_NOT_FOUND`
- `EVIDENCE_VIDEO_MISMATCH`
- `INSUFFICIENT_EVIDENCE`
- `UNKNOWN_TASK_SUBTYPE`
- `INVALID_GOLD_VALUE`
- `OBSERVATION_INFERENCE_MIXED`
- `UNSUPPORTED_CAUSAL_CLAIM`
- `DUPLICATE_SEMANTICS`
- `CONFLICTING_VALUE`
- `PRIVATE_FIELD_LEAK`
- `INVALID_TIER`

Public 防火墙递归检查以下 key：

```python
PRIVATE_KEYS = {
    "likes", "collects", "shares", "comments",
    "like_response_score", "collect_response_score",
    "share_response_score", "comment_response_score",
    "raw_value", "thresholds", "performance_data",
}
```

Gold tier 的确定规则只由本地代码执行：

- BP 直接观察、证据明确、无冲突：允许 Gold-A。
- CM 有两个模态证据且关系唯一：允许 Gold-A；否则最高 Gold-B。
- SS/AE 至少两个不同 evidence，且两个独立 proposer 结论一致或已经人工确认、无替代解释冲突：允许 Gold-B；不满足进入 Silver。
- IP 由确定性私有标签程序产生：私有记录可标 Gold-A，但 public QA 只暴露等级答案，不暴露原始值。
- 任何 ERROR：Rejected 或 human review queue，不能进入可编译 Gold。

### Step 2.4：验证检查点

```bash
python -m unittest tests.test_goldbank_validators -v
python -m unittest tests.test_goldbank_schema tests.test_goldbank_validators -v
```

---

## Task 3：实现结构化 prompt 和严格解析契约

**Files:**

- Create: `src/salesbench/goldbank/prompts.py`
- Create: `src/salesbench/goldbank/parsing.py`
- Test: `tests/test_goldbank_prompts.py`

### Step 3.1：先写失败测试

```python
class GoldBankPromptTest(unittest.TestCase):
    def test_evidence_extractor_requests_units_not_questions(self): ...
    def test_all_proposers_must_reference_existing_evidence_ids(self): ...
    def test_challenger_can_request_human_review_and_cannot_default_pass(self): ...
    def test_adjudicator_returns_items_and_review_queue(self): ...
    def test_prompt_payload_never_contains_performance_data(self): ...
    def test_parser_rejects_markdown_or_missing_top_level_key(self): ...
```

### Step 3.2：实现四类 prompt builder

`prompts.py` 提供：

```python
def build_evidence_extractor_prompt(video_id: str, content_context: dict[str, object]) -> tuple[str, list[dict]]
def build_proposer_prompt(
    perspective: str,
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]
def build_challenger_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]
def build_adjudicator_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    reviews: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]
```

硬约束：

- Extractor 只输出可定位证据，不输出问题、策略结论或受众结论。
- Consumer proposer 只覆盖 AE 和与需求相关的 SS。
- Operator proposer 只覆盖 CM、内容结构和 CTA 相关 SS。
- Strategist proposer 只覆盖价值主张、信任、风险、紧迫感和漏斗角色相关 SS。
- BP 由 Evidence Unit 的确定性 builder 产生，不让三个 proposer 重复造 BP。
- 提议必须引用给定 `evidence_id`；证据不足时返回 `abstentions`。
- Challenger 必须分别检查事实/推断混合、替代解释、重复、矛盾和因果越界。
- Adjudicator 不自行新增 evidence；新增事实必须拒绝。

### Step 3.3：实现严格 JSON 解析

`parsing.py` 提供：

```python
class ModelOutputError(ValueError): ...

def parse_json_object(raw_response: str, required_key: str) -> dict[str, object]
def parse_evidence_response(raw_response: str) -> list[dict[str, object]]
def parse_proposal_response(raw_response: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]
def parse_review_response(raw_response: str) -> list[dict[str, object]]
def parse_adjudication_response(raw_response: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]
```

不再使用“解析失败则构造默认 PASS/默认答案”的 fallback。失败必须写 trace，并将本视频标记为 `needs_retry` 或进入复核队列。

### Step 3.4：验证检查点

```bash
python -m unittest tests.test_goldbank_prompts -v
```

---

## Task 4：实现 GoldBankPipeline 多 Agent 状态机

**Files:**

- Create: `src/salesbench/goldbank/pipeline.py`
- Test: `tests/test_goldbank_pipeline.py`
- Reuse: `src/salesbench/vlm/api_client.py`
- Reuse: `src/salesbench/multiagent/context.py`

### Step 4.1：先构造 Fake Client 和失败测试

复用 `tests/test_multiagent_qa.py` 的 `FakeJSONClient` 思路，但测试独立定义。至少覆盖：

```python
class GoldBankPipelineTest(unittest.TestCase):
    def test_pipeline_builds_evidence_before_proposals(self): ...
    def test_bp_builder_does_not_call_proposer(self): ...
    def test_three_perspectives_only_propose_allowed_tasks(self): ...
    def test_rejected_proposal_never_enters_gold_items(self): ...
    def test_parse_failure_is_visible_and_not_passed(self): ...
    def test_conflict_enters_human_review_queue(self): ...
    def test_agent_prompt_never_receives_performance_data(self): ...
```

### Step 4.2：实现结果对象和依赖注入

`pipeline.py` 定义：

```python
@dataclass
class GoldBankResult:
    video_id: str
    evidence_units: list[dict[str, object]]
    gold_proposals: list[dict[str, object]]
    gold_reviews: list[dict[str, object]]
    video_gold_record: dict[str, object] | None
    human_review_queue: list[dict[str, object]]
    agent_traces: list[dict[str, object]]
    status: str

class GoldBankPipeline:
    def __init__(
        self,
        vlm_client: VLMClient,
        llm_client: VLMClient,
        min_confidence: float = 0.70,
    ): ...

    def run_video(
        self,
        bundle: VideoContextBundle,
        frames_b64: list[str] | None = None,
    ) -> GoldBankResult: ...
```

### Step 4.3：实现固定阶段顺序

`run_video` 必须严格执行：

1. `all_content_context(bundle)` 取得无 PD 内容。
2. Extractor 用视频帧和结构化内容产生 Evidence Units。
3. 本地 `build_bp_proposals_from_evidence()` 产生高确定性 BP。
4. Consumer、Operator、Strategist 分别提议 SS/AE/CM Gold。
5. Challenger 对全部 proposal 一次性审查。
6. Adjudicator 合并同义、保留不冲突多解释、输出裁决建议。
7. 本地 validator 重新验证，代码裁定 tier；LLM 不能最终决定 tier。
8. accepted Gold-A/Gold-B 项进入本视频 `video_gold_record.gold_items`；Silver/冲突/解析失败只进入 `human_review_queue`。

每次模型调用写 `AgentCallTrace`，`stage` 固定为：

```text
evidence_extraction
consumer_proposal
operator_proposal
strategist_proposal
challenge
adjudication
```

### Step 4.4：定义失败和重试语义

- 单个 proposer 失败：继续其他 proposer，但本视频状态为 `partial`。
- Extractor 失败：本视频状态为 `failed`，不继续提议。
- Challenger 或 Adjudicator 失败：proposal 全部进入复核队列，不进入 Gold。
- API 级重试仍由 `VLMClient` 负责；pipeline 不无限重试。
- 不保存模型隐藏思维链；仅保存 raw response、短 reason、token、latency 和 error。

### Step 4.5：验证检查点

```bash
python -m unittest tests.test_goldbank_pipeline -v
python -m unittest tests.test_goldbank_schema tests.test_goldbank_validators tests.test_goldbank_prompts tests.test_goldbank_pipeline -v
```

---

## Task 5：重构任务五为独立、私有的互动响应标签

**Files:**

- Create: `src/salesbench/goldbank/interaction_labels.py`
- Create: `src/salesbench/goldbank/splits.py`
- Test: `tests/test_interaction_gold.py`
- Reference only: `src/salesbench/vqa/effect_labels.py`
- Reference only: `src/salesbench/vqa/sampler.py`

### Step 5.1：先写泄漏测试

```python
class InteractionGoldTest(unittest.TestCase):
    def test_thresholds_are_fit_on_train_ids_only(self): ...
    def test_val_outlier_does_not_change_train_thresholds(self): ...
    def test_three_levels_are_low_medium_high(self): ...
    def test_raw_interactions_exist_only_in_private_record(self): ...
    def test_missing_metric_abstains_instead_of_inventing_label(self): ...
    def test_same_creator_never_crosses_group_split(self): ...
```

### Step 5.2：实现 group split manifest

`splits.py` 提供：

```python
@dataclass(frozen=True)
class SplitManifest:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    grouping_key: str
    seed: int

def build_group_split_manifest(
    records: list[dict[str, object]],
    group_key: str = "douyin_handle",
    seed: int = 42,
) -> SplitManifest
```

当前 `outputs/processed/videos_v1.jsonl` 没有 `author_id`，但有 `douyin_handle` 与 `author_name`。因此优先使用 `douyin_handle`，缺失时回退到 `author_name`；两者都缺失时使用 `video_id`，并在 manifest 中记录 `grouping_key="video_id_fallback"` 和高风险 warning，不能假装是 creator-disjoint split。

### Step 5.3：实现三档 train-only 标签

`interaction_labels.py` 提供：

```python
INTERACTION_METRICS = {
    "like_response_level": "engagement_score",
    "collect_response_level": "conversion_score",
    "share_response_level": "virality_score",
    "comment_response_level": "discussion_score",
}

def fit_tertile_thresholds(
    dimension_records: list[dict[str, object]],
    train_ids: set[str],
) -> dict[str, dict[str, float]]

def label_tertile(value: float, thresholds: dict[str, float]) -> str

def build_interaction_private_records(
    dimension_records: list[dict[str, object]],
    split_manifest: SplitManifest,
) -> tuple[list[dict[str, object]], dict[str, object]]
```

只使用全量数据预先划定的训练集计算 `q33`、`q67`；不能在 5 视频 pilot 内临时拟合阈值。标签为 `low`、`medium`、`high`。私有记录包含原始 metric、threshold version、split；public Gold Item 只保留等级答案和非数值 provenance token。Runner 完成全量标签后再按 5 个 pilot ID 取子集。

不要复用现有 `compute_global_quintile_thresholds()`；该函数使用全数据五分位，存在泄漏且增加回答边界模糊性。

### Step 5.4：验证检查点

```bash
python -m unittest tests.test_interaction_gold -v
```

---

## Task 6：实现 runner、固定采样和九类产物

**Files:**

- Create: `src/salesbench/goldbank/runner.py`
- Create: `configs/goldbank_pilot_5videos.json`
- Create: `tests/test_goldbank_runner.py`
- Modify: `src/salesbench/multiagent/context.py`
- Reuse: `src/salesbench/vlm/frame_sampler.py`
- Reuse: `src/salesbench/io_utils.py`

### Step 6.1：先写 runner 集成测试

测试临时目录输出必须恰好包含：

```text
video_samples.jsonl
evidence_units.jsonl
gold_proposals.jsonl
gold_reviews.jsonl
video_gold_bank.jsonl
interaction_gold_private.jsonl
human_review_queue.jsonl
agent_traces.jsonl
generation_meta.json
```

并覆盖：

```python
class GoldBankRunnerTest(unittest.TestCase):
    def test_explicit_video_ids_preserve_requested_cohort(self): ...
    def test_runner_writes_all_contract_files(self): ...
    def test_records_are_deterministically_sorted(self): ...
    def test_resume_skips_completed_video_ids(self): ...
    def test_generation_meta_records_prompt_and_schema_versions(self): ...
```

### Step 6.2：增加精确 video ID 选择

在 `SalesBenchContextStore` 新增：

```python
def raw_records_for_ids(self, video_ids: list[str]) -> list[dict[str, Any]]:
    missing = [video_id for video_id in video_ids if video_id not in self.raw_lookup]
    if missing:
        raise ValueError(f"Unknown video_ids: {missing}")
    return [self.raw_lookup[video_id] for video_id in video_ids]
```

不能用随机 sampler 近似代替指定 cohort。

### Step 6.3：写固定 pilot 配置

`configs/goldbank_pilot_5videos.json`：

```json
{
  "version": "goldbank-pilot-v1",
  "video_ids": [
    "7362856271365606683",
    "7360594186715909416",
    "7360565413341613348",
    "7359538830657064192",
    "7365054535036849420"
  ],
  "frame_strategy": "hook_plus_uniform",
  "frames_per_video": 16,
  "prompt_version": "goldbank-prompt-v1",
  "schema_version": "goldbank-schema-v1",
  "min_confidence": 0.70
}
```

### Step 6.4：实现 runner 入口

`runner.py` 公开：

```python
def build_gold_bank_dataset(
    config: BenchmarkConfig,
    pilot_config_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    base_url: str | None = None,
    max_workers: int = 1,
    resume: bool = True,
) -> dict[str, object]
```

输出约定：

- `video_samples.jsonl`：本轮视频 cohort 和资产路径/可用性快照。
- `evidence_units.jsonl`：所有可定位的画面、ASR、OCR、标题、跨模态证据。
- `gold_proposals.jsonl`：本地 BP builder 与三视角 Agent 的候选事实/解释。
- `gold_reviews.jsonl`：Challenger/Adjudicator 对候选的逐项裁决记录。
- `video_gold_bank.jsonl`：一行一个视频，内部 `gold_items` 只保存通过代码验证的 Gold-A/Gold-B；同时保存覆盖、质量摘要与 observation scope。
- `interaction_gold_private.jsonl`：IP 原始分数、训练阈值和等级；不得作为模型输入或公开发布。
- `human_review_queue.jsonl`：Silver、冲突、证据不足、解析失败及建议复核动作。
- `agent_traces.jsonl`：各阶段调用、成本、延迟、解析状态；不保存 API key。
- `generation_meta.json`：模型、版本、随机种子、输入哈希、计数、失败、成本和各输出路径。

为支持 resume，每处理完一个视频就以临时分片写入 `output_dir/.parts/<video_id>/`；最终合并时按 `video_id + record_id` 排序。不得在进程中断时生成看似完整的总文件。

### Step 6.5：验证检查点

```bash
python -m unittest tests.test_goldbank_runner -v
```

---

## Task 7：接入 CLI 和人工复核回写

**Files:**

- Modify: `src/salesbench/cli.py`
- Create: `src/salesbench/goldbank/review_io.py`
- Test: `tests/test_goldbank_cli.py`
- Test: `tests/test_goldbank_review_io.py`

### Step 7.1：先写 CLI 解析测试

验证以下命令及默认值：

```text
build-gold-bank
apply-gold-reviews
```

### Step 7.2：增加 CLI 参数

`build-gold-bank`：

```text
--config configs/benchmark_v1.json
--pilot-config configs/goldbank_pilot_5videos.json
--output-dir outputs/goldbank/v1_pilot_5videos
--api-key / OPENAI_API_KEY
--base-url / OPENAI_BASE_URL
--model gpt-4o
--max-workers 1
--no-resume
```

`apply-gold-reviews`：

```text
--gold-bank-dir outputs/goldbank/v1_pilot_5videos
--decisions outputs/goldbank/v1_pilot_5videos/human_review_decisions.jsonl
--output outputs/goldbank/v1_pilot_5videos/video_gold_bank_reviewed.jsonl
```

### Step 7.3：实现人工复核格式和回写

`review_io.py`：

```python
def validate_human_review_decisions(records: list[dict[str, object]]) -> list[str]
def apply_human_reviews(
    video_gold_records: list[dict[str, object]],
    review_queue: list[dict[str, object]],
    decisions: list[dict[str, object]],
) -> list[dict[str, object]]
```

每条 decision 必须包含：

```json
{
  "review_item_id": "hr_736059_ae_002",
  "proposal_id": "736059_consumer_ae_002",
  "decision": "ACCEPT_GOLD_B|REVISE|REJECT",
  "reviewer_id": "annotator_01",
  "reviewed_at": "ISO-8601",
  "reason_code": "...",
  "revised_value": null,
  "revised_evidence_ids": []
}
```

`review_item_id` 必须存在于 `human_review_queue.jsonl`。REVISE 必须提供修改字段；ACCEPT_GOLD_B/REVISE 产生带 `human_accepted` 状态的新 Gold Item 并写入相应视频的 reviewed bank；REJECT 保留在 proposal/review lineage 中，不进入 Gold Bank。

### Step 7.4：验证检查点

```bash
python -m unittest tests.test_goldbank_cli tests.test_goldbank_review_io -v
python salesbench.py --help
```

---

## Task 8：实现 Gold Bank → QA 的确定性编译器

**Files:**

- Create: `src/salesbench/vqa/goldbank_loader.py`
- Create: `src/salesbench/vqa/question_programs.py`
- Create: `src/salesbench/vqa/compiler.py`
- Create: `tests/test_goldbank_qa_compiler.py`
- Modify: `src/salesbench/cli.py`

### Step 8.1：先写 QA 编译测试

```python
class GoldBankQACompilerTest(unittest.TestCase):
    def test_only_gold_a_and_gold_b_are_compiled(self): ...
    def test_each_qa_has_source_gold_ids_and_evidence_ids(self): ...
    def test_same_gold_compiles_deterministically(self): ...
    def test_answer_is_derived_from_value_not_generated_by_llm(self): ...
    def test_public_qa_contains_no_private_interaction_fields(self): ...
    def test_question_does_not_leak_answer(self): ...
    def test_per_video_task_quota_is_enforced(self): ...
```

### Step 8.2：实现 loader

```python
def load_compilable_gold(path: Path) -> list[GoldItem]
```

读取一行一个视频的 `VideoGoldRecord` 并展开其 `gold_items`；只返回 Gold-A/Gold-B 且 `review_status` 为 `verified` 或 `human_accepted` 的记录。Silver、Rejected、冲突项不会存在于主文件，若误混入则 loader 直接报错。

### Step 8.3：定义受控 question programs

`question_programs.py` 使用 `task_type + task_subtype + question_format` 到模板的显式映射，不让 LLM 临场生成答案。例如：

```python
QUESTION_PROGRAMS = {
    ("BP", "COUNT_SPATIAL", "direct_question"): "视频中出现了多少个{subject}？",
    ("BP", "ACTION", "direct_question"): "视频中人物对{subject}做了什么？",
    ("CM", "CLAIM_EVIDENCE_RELATION", "supported_missing"): "画面中的哪些证据支持或不支持口播关于{claim}的说法？",
    ("SS", "HOOK_MECHANISM", "mechanism_with_evidence"): "视频开场使用了什么吸引注意的机制？请结合证据回答。",
    ("AE", "AUDIENCE_NEED_FIT", "need_with_evidence"): "该内容主要满足哪类受众的什么需求？请说明视频证据。",
    ("IP", "COLLECT_RESPONSE_LEVEL", "ordinal_3"): "该视频的收藏响应水平属于低、中还是高？",
}
```

每个可编译 subtype 至少有一个模板；若没有模板，编译器抛 `UnsupportedQuestionProgramError`，不能回退成泛问题。

### Step 8.4：实现编译器

```python
@dataclass(frozen=True)
class CompilePolicy:
    max_questions_per_video: int = 8
    task_priority: tuple[str, ...] = ("BP", "CM", "SS", "AE", "IP")
    max_per_task: int = 2
    include_tiers: tuple[str, ...] = ("Gold-A", "Gold-B")

def compile_qa_records(
    gold_items: list[GoldItem],
    policy: CompilePolicy,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]

def compile_vqa_from_gold(
    gold_bank_dir: Path,
    output_dir: Path,
    policy: CompilePolicy,
    bank_filename: str = "video_gold_bank_reviewed.jsonl",
) -> dict[str, object]
```

函数从 `gold_bank_dir` 显式读取 `bank_filename`、`evidence_units.jsonl` 和可选的 `interaction_gold_private.jsonl`。默认要求人工复核版存在；若要编译纯自动 pilot，调用方必须显式传入 `bank_filename="video_gold_bank.jsonl"`，不能静默回退。

输出：

```text
qa_plan.jsonl
qa_candidates.jsonl
qa_validation.jsonl
vqa_gold_private.jsonl
vqa_public.jsonl
generation_meta.json
```

`vqa_gold_private` 包含标准答案；`vqa_public` 隐藏答案、Evidence Gold 及 IP 私有元数据。未通过项及原因写入 `qa_validation.jsonl`，但不进入 private/public 正式 QA。每个 QA 都带 `source_gold_ids`、`evidence_ids`、`question_program_id` 和版本。

### Step 8.5：新增 CLI

```bash
python salesbench.py compile-vqa-from-gold \
  --gold-bank-dir outputs/goldbank/v1_pilot_5videos \
  --bank-file video_gold_bank_reviewed.jsonl \
  --output-dir outputs/vqa/v2_goldbank_pilot_5videos \
  --max-questions-per-video 8 \
  --max-per-task 2
```

### Step 8.6：验证检查点

```bash
python -m unittest tests.test_goldbank_qa_compiler -v
```

---

## Task 9：增加 Gold/QA 质量审计和任务特定评测入口

**Files:**

- Create: `src/salesbench/goldbank/audit.py`
- Create: `src/salesbench/vqa/task_evaluator.py`
- Create: `tests/test_goldbank_audit.py`
- Create: `tests/test_vqa_task_evaluator.py`
- Modify: `src/salesbench/cli.py`

### Step 9.1：先写审计与评分测试

覆盖：

- evidence coverage 分母为 accepted Gold 数，不以有 evidence 的项目为分母。
- SS/AE 允许受控同义答案，不能只做精确字符串匹配。
- BP 数字/布尔/短文本采用确定性评分。
- CM 关系标签与证据定位分别评分。
- IP ordinal_3 报 accuracy、macro-F1、QWK。
- 无效输出和拒答明确计 0，同时单独报告 invalid rate。
- `LLM-as-Judge` 只作为 SS/AE 辅助指标，不能作为唯一主指标。

### Step 9.2：实现 Gold 审计

`audit_gold_bank(records, evidence)` 返回：

```python
{
    "video_count": int,
    "item_count_by_task": dict,
    "item_count_by_tier": dict,
    "evidence_coverage": float,
    "cross_modal_coverage": float,
    "duplicate_rate": float,
    "conflict_rate": float,
    "silver_rate": float,
    "human_review_rate": float,
    "abstention_count": int,
    "videos_missing_required_tasks": list,
}
```

固定 5 视频 pilot gate：

- evidence reference validity = 100%。
- private-field leakage = 0。
- duplicate rate ≤ 5%。
- unresolved conflict rate = 0（允许进入 review queue，不允许进入 compiled Gold）。
- BP、CM 每视频至少 1 条。
- SS、AE 若证据不足允许 abstain，不允许为凑配额造题。
- 人工抽检 accepted items 的 factual precision ≥ 90%。

### Step 9.3：实现任务特定评分

`task_evaluator.py` 提供：

```python
def evaluate_goldbank_qa(
    gold_records: list[dict[str, object]],
    answer_records: list[dict[str, object]],
) -> dict[str, object]
```

主指标按任务分开报告，不先混成单一总分：

- BP：exact/normalized accuracy。
- CM：relation accuracy、evidence F1。
- SS：controlled-label macro-F1、evidence-grounded score。
- AE：audience-label macro-F1、evidence-grounded score。
- IP：accuracy、macro-F1、QWK。

### Step 9.4：新增 audit CLI

```bash
python salesbench.py audit-gold-bank \
  --gold-bank outputs/goldbank/v1_pilot_5videos/video_gold_bank.jsonl \
  --evidence outputs/goldbank/v1_pilot_5videos/evidence_units.jsonl \
  --output outputs/goldbank/v1_pilot_5videos/gold_audit.json
```

### Step 9.5：验证检查点

```bash
python -m unittest tests.test_goldbank_audit tests.test_vqa_task_evaluator -v
```

---

## Task 10：更新研究文档、运行全套验证并执行 5 视频 pilot

**Files:**

- Modify: `README.md`
- Modify: `docs/SalesBench-QA_Design.md`
- Create: `docs/annotation/GoldBank_Annotation_Guide_v1.md`
- Create: `docs/annotation/GoldBank_Review_Form_v1.jsonl`
- Create: `docs/data/GoldBank_Data_Card_v1.md`
- Verify: `docs/SalesBench-QA_Thesis_Roadmap_v2.md`

### Step 10.1：更新主说明，明确 legacy 与新流程

README 和主设计文档必须并列说明：

- `build-multiagent-qa` 是旧 question-first 实验，只用于历史复现。
- `build-gold-bank` 是新主线。
- `compile-vqa-from-gold` 是 QA 生成唯一正式入口。
- 五类任务的定义、证据要求、答案类型和评价指标。
- public/private 数据边界。
- IP 是互动响应预测，不是销售转化或因果归因。

### Step 10.2：写人工标注手册

标注手册必须给出每类至少 2 个正例、2 个反例，并定义：

- 如何核对时间戳、ASR/OCR 和画面证据。
- “观察事实”和“营销解释”的边界。
- SS/AE 多个合理答案的处理。
- Challenger 何时 PASS、REVISE、HUMAN_REVIEW、REJECT，以及 proposer 何时输出 insufficient evidence。
- 冲突处理和 reviewer 不确定性记录。
- 双人独立标注 + 仲裁流程。

### Step 10.3：运行静态和回归验证

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -p 'test_*.py' -v
python salesbench.py --help
python salesbench.py build-gold-bank --help
python salesbench.py compile-vqa-from-gold --help
python salesbench.py audit-gold-bank --help
```

预期：全部退出码为 0，旧测试不回归。

### Step 10.4：执行固定 5 视频 pilot

需要有效 API key 和网络时运行：

```bash
python salesbench.py build-gold-bank \
  --config configs/benchmark_v1.json \
  --pilot-config configs/goldbank_pilot_5videos.json \
  --output-dir outputs/goldbank/v1_pilot_5videos \
  --model gpt-4o \
  --max-workers 1
```

然后：

```bash
python salesbench.py audit-gold-bank \
  --gold-bank outputs/goldbank/v1_pilot_5videos/video_gold_bank.jsonl \
  --evidence outputs/goldbank/v1_pilot_5videos/evidence_units.jsonl \
  --output outputs/goldbank/v1_pilot_5videos/gold_audit.json
```

人工完成 `human_review_decisions.jsonl` 后：

```bash
python salesbench.py apply-gold-reviews \
  --gold-bank-dir outputs/goldbank/v1_pilot_5videos \
  --decisions outputs/goldbank/v1_pilot_5videos/human_review_decisions.jsonl \
  --output outputs/goldbank/v1_pilot_5videos/video_gold_bank_reviewed.jsonl

python salesbench.py compile-vqa-from-gold \
  --gold-bank-dir outputs/goldbank/v1_pilot_5videos \
  --bank-file video_gold_bank_reviewed.jsonl \
  --output-dir outputs/vqa/v2_goldbank_pilot_5videos \
  --max-questions-per-video 8 \
  --max-per-task 2
```

### Step 10.5：生成首轮决策报告

将以下结果追加到 `docs/data/GoldBank_Data_Card_v1.md`：

- 每视频 Evidence/Gold/QA 数量。
- 五任务与 tier 分布。
- 各 Agent abstain/reject/revise 比例。
- 人工一致率和仲裁率。
- 重复、冲突、泄漏和证据覆盖。
- 单视频平均成本和耗时。
- 是否进入 20 视频阶段的结论。

进入 20 视频阶段的最低条件：

1. 5 视频 pilot 所有硬 gate 通过。
2. 人工抽检 precision ≥ 90%。
3. 每视频成本和人工复核时间可接受。
4. 至少完成一次从 Gold Bank 到 public/private QA 的全链闭环。

---

## 实施后的扩展顺序

### A. 20 视频 Alpha

- 使用与 v9 相同 20 视频 cohort 做成对比较。
- 对比 question-first 与 Gold-first 的有效题率、证据覆盖、重复率、人工修订率和成本。
- 执行三项消融：无 Challenger、无 Adjudicator、无多视角 proposer。
- 通过后冻结 `goldbank-schema-v1` 与 `goldbank-prompt-v1`。

### B. 100 视频质控集

- 双人独立标注 20%，其余单标 + 仲裁抽检。
- 报告 Cohen's kappa/加权 kappa、任务级 precision 和 95% bootstrap CI。
- 测量新增视频的 Gold 谓词/答案覆盖增益，即数据饱和曲线。

### C. 1200 全量与 2000+ 扩容决策

只有在以下任一条件成立时扩容：

- 关键品类、受众、策略或创作者层存在显著覆盖缺口；
- 学习曲线在 1200 条仍未趋于平台；
- 任务级置信区间过宽，无法支撑论文结论；
- 2000+ 数据能保持同等 Gold 质控，而不是仅增加原始视频数量。

若 1200 条已经达到覆盖与统计功效要求，应优先增加双标比例、基线模型、消融和误差分析，而不是机械扩容。

---

## 推荐实施顺序总结

```text
Schema/Ontology
  → Validator/Firewall
  → Prompt/Parser
  → GoldBankPipeline
  → IP Private Labels
  → Runner/Outputs
  → CLI/Review
  → QA Compiler
  → Audit/Evaluator
  → 5-video Pilot
  → 20-video Alpha
  → 100-video Quality Set
  → 1200/full benchmark
```

这个顺序保证“数据契约和正确性规则”先于大模型调用，避免在 prompt 还不稳定、标准尚不明确时继续扩大生成规模。
