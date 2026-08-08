# SalesBench 64 视频 GPT-4o Pilot 运行报告（v6）

运行日期：2026-08-08

## 1. 运行范围

本次使用 `configs/evidence_pilot_64videos.json` 固定 64 个视频，Evidence 提取、CM/SS/AE 提案、Challenger、Adjudicator、被测 VLM 和 LLM-as-Judge 均使用中转 API 提供的 `gpt-4o`。API key 仅从项目根目录的 `.env` 注入，未写入配置、输出或 Git。

版本：

- schema：`evidence-dataset-schema-v2`
- prompt：`evidence-prompt-v6`
- pipeline：`evidence-first-pipeline-v5`
- QA compiler：`evidence-qa-compiler-v3`
- cohort：64 个视频，16 帧，`hook_plus_uniform`

## 2. EvidenceDataset 生成

输出目录：`outputs/evidence/v6_pilot64_gpt4o_yunwu`

首次运行有 1 个 `partial` 和 1 个 `failed`，原因均为中转站瞬时 `Connection error`。断点恢复逻辑修复为只复用 `ok` part 后，仅重试这两个视频，最终 64/64 均为 `ok`。

最终产出：

| 项目 | 数量 |
| --- | ---: |
| 视频主记录 | 64 |
| EvidenceUnit | 609 |
| Gold proposal | 946 |
| Challenger review | 936 |
| 自动接受 GroundedAnnotation | 732 |
| Human review queue | 296 |
| Agent trace | 384 |

自动接受 annotation 的任务分布：

| 任务 | 数量 |
| --- | ---: |
| BP | 443 |
| CM | 68 |
| SS | 132 |
| AE | 89 |

最终 trace 记录的 Evidence 生成估算成本为 8.362355 美元。该值使用项目内置 token 单价估算，不等同于中转站账单，并且不包含 smoke、失败调用和重试产生的全部实际费用。

## 3. 自动审计

审计文件：`outputs/evidence/v6_pilot64_gpt4o_yunwu/audit_before_review.json`

| 指标 | 结果 |
| --- | ---: |
| Evidence 外键覆盖 | 100% |
| CM 跨模态 Evidence 覆盖 | 100% |
| 重复率 | 0% |
| 冲突率 | 0% |
| 私有字段泄漏 | 0 |
| 缺 BP 或 CM 的视频 | 21/64 |

结构审计通过，但该数据仍是自动接受版本。296 个 review queue 项尚未人工处理，21 个缺 BP 或 CM 的视频应作为优先复核对象。

## 4. QA 编译

输出目录：`outputs/vqa/v6_pilot64_gpt4o_yunwu`

本次直接从 `video_evidence_dataset.jsonl` 编译 provisional QA，没有创建或伪造 `reviewed` 文件。

| 任务 | QA 数量 |
| --- | ---: |
| BP | 124 |
| CM | 66 |
| SS | 106 |
| AE | 88 |
| 合计 | 384 |

公开 QA 与私有 gold 各 384 条，四任务均非空。每个视频最多 8 题、每任务最多 2 题。

v6 相比早期 smoke 的文本质量有明显改善：BP 问题显式包含对象和属性；中文视频的 Evidence、SS 和 AE answer 使用中文；SS/AE 优先编译完整答案，不再只使用 `urgency` 等短标签。

## 5. GPT-4o 回答结果

回答目录：`outputs/vqa/v6_pilot64_gpt4o_yunwu/run_gpt4o`

- 384/384 调用成功；
- 空答案 0；
- 重复 `vqa_id` 0；
- 记录成本 1.892784 美元。

## 6. LLM-as-Judge 结果

Judge 目录：`outputs/vqa/v6_pilot64_gpt4o_yunwu/evaluation_gpt4o`

- `gold_count = 384`
- `matched_answer_count = 384`
- `missing_answer_count = 0`
- `scored_count = 384`
- `judge_failed_count = 0`
- Judge 记录成本 1.232605 美元

`strict_accuracy` 表示得分恰好为 1.0 的比例；`relaxed_accuracy` 是 `0/0.25/0.5/0.75/1.0` 五档分数的均值。

| 任务 | 题数 | Strict | Relaxed |
| --- | ---: | ---: | ---: |
| BP | 124 | 0.4758 | 0.7177 |
| CM | 66 | 0.4091 | 0.7765 |
| SS | 106 | 0.1887 | 0.6604 |
| AE | 88 | 0.1591 | 0.6705 |
| Micro overall | 384 | 0.3125 | 0.7012 |
| Macro average | 4 tasks | 0.3082 | 0.7063 |

分数分布显示 BP 有 59 道满分、CM 有 27 道满分、SS 有 20 道满分、AE 有 14 道满分。BP 的 7 道零分主要来自数量、价格或 OCR 字段读取错误。SS/AE 很少出现零分，但较多集中在 0.5 和 0.75，常见原因是模型回答覆盖了合理内容，却没有对齐当前 gold 选择的具体证据焦点。

这些结果不能直接作为正式 benchmark 结论，原因是 Gold 生成模型、被测模型和 Judge 都是 `gpt-4o`，存在同模型偏差；同时 Gold 与 Judge 尚未完成双人人工校准。

## 7. 独立互动分析

输出目录：`outputs/analysis/interactions_v6_pilot64_gpt4o`

互动分析使用点赞、评论、分享和收藏的 `log1p` 三分位分层，输出 48 个“任务 × 互动指标 × 分层”诊断切片和 1000 次 bootstrap 区间。

该实验满足：

- `included_in_vqa = false`
- `included_in_leaderboard = false`
- `diagnostic_only`
- `noncausal_association`

不同互动层的得分没有稳定单调关系，多个置信区间明显重叠。结果只能描述为样本内关联或共现，不能解释为互动量导致模型表现，也不能声称模型可以预测互动、传播或销量。

## 8. 当前结论与下一步

本次已经验证 64 视频的技术全流程可以完整运行：Evidence 生成、自动审计、四任务 QA 编译、多模态回答、LLM-as-Judge 和独立互动分析均有可检查产出，且最终 API/解析失败为 0。

当前数据应标记为 `provisional_auto_accepted`，不能直接发布为正式 benchmark。扩到 128 或更大规模前应完成：

1. 处理 296 个 review queue 项，优先补审 21 个缺 BP 或 CM 的视频。
2. 按任务分层抽取至少 64 个 QA，每任务至少 16 个，进行双人人工事实与可回答性检查。
3. 用人工裁决分数校准 Judge，报告 exact agreement、相邻一档 agreement、weighted kappa 和 MAE。
4. 增加与生成/Judge 不同家族的第二个多模态被测模型；正式 Judge 也应使用异构模型或双 Judge 仲裁。
5. 重点修订过长 OCR gold、泛化过度的 SS/AE gold，以及问题与 gold 证据焦点不一致的样本。

详细执行步骤见 `docs/Pilot_64_Video_Execution_Guide.md`。
