# EvidenceDataset 标注与复核指南 v2

## 1. 复核单位

复核对象是 `human_review_queue.jsonl` 中的 proposal 或 GroundedAnnotation，不是自然语言 QA。QA 会在复核完成后由确定性程序编译。

核心原则：只有当答案值能够回指到列出的 visual frame、in-frame OCR 或 ASR evidence 时才可接受。合理但无证据的猜测必须 `REJECT` 或继续 `NEEDS_REVIEW`，不得静默补全。

## 2. 任务规则

### BP

允许：出现的产品、人物、动作、可见属性、数量与空间、状态变化、画面文字、口播事实。

拒绝：高赞、销量、传播或转化判断；“建立信任”“适合某受众”等推断性答案。

BP 至少引用 1 个直接证据。

### CM

允许：ASR 与画面/OCR 是否 `SUPPORTED`、`PARTIALLY_SUPPORTED`、`CONTRADICTED`、`NOT_SHOWN` 或 `TEMPORALLY_MISALIGNED`。

拒绝：只凭标题、粉丝画像或互动量判断口播真假；没有明确 claim 与两侧证据的笼统一致性判断。

CM 至少引用 2 个不同 evidence。

### SS

允许：视频如何使用 Hook、卖点、演示、信任、顾虑处理、紧迫性、CTA 或漏斗结构。

拒绝：“为什么一定高赞”“预计收藏量”“该策略导致销量”等效果推断。

SS 至少引用 2 个 evidence，结论应描述可观察内容如何组成策略。

### AE

允许：内容面向的需求、使用场景、决策状态和障碍。

拒绝：依据粉丝画像、账号层级或互动量推断真实受众、购买率或转化。

AE 至少引用 2 个 evidence，并在答案中保留不确定性边界；它描述内容定位，不宣称真实用户画像。

## 3. Verdict

- `PASS`：证据存在且支持答案，无冲突、无泄漏、无重复。
- `REVISE`：修改明确的结构化值或 evidence refs 后可用。
- `HUMAN_REVIEW`：多个解释均合理，自动流程不能裁决。
- `REJECT`：证据缺失、跨视频、低于可接受标准、重复、冲突、因果过度或私有泄漏。

模型给出的 `quality_status` 和 tier 一律忽略；本地规则根据任务、证据和人工决定重新赋值。

## 4. 人工决定格式

每行 JSONL 至少包含：

```json
{
  "review_item_id": "hr_v1_p1",
  "proposal_id": "p1",
  "decision": "ACCEPT_INFERRED",
  "reviewer_id": "annotator_01",
  "reviewed_at": "2026-08-04T00:00:00Z",
  "reason_code": "evidence_complete",
  "revised_value": null,
  "revised_evidence_ids": []
}
```

可用 decision：`ACCEPT_INFERRED`、兼容值 `ACCEPT_GOLD_B`、`REVISE`、`REJECT`。`REVISE` 必须给出 `revised_value` 和 `revised_evidence_ids`。

人工接受项统一写为 `quality_status=INFERRED`、`review_status=human_accepted`。自动流程不能将 SS/AE 提升为 DIRECT。

## 5. 一致性流程

- 5 视频 pilot：所有自动接受项由两名复核员独立检查。
- 20/100 视频阶段：test 核心项与 alpha anchor 双人复核。
- 分歧由第三人裁决，并保留最终 reason code。
- 记录 factual precision、evidence validity、duplicate rate、conflict rate 和任务边界错误率。

互动量、粉丝量和发布情境不得显示给内容标注员；互动分析由独立脚本在标注完成后运行。
