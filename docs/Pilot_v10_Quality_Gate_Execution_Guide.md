# SalesBench v10 质量门禁执行指南

> v10 不覆盖任何 v9 产物。先跑 5 视频 smoke；只有生产门禁、审计视图和 QA 候选质量通过后，才允许重新运行 64 视频 pilot。

## 1. v10 解决什么问题

v9 的 440 条“人工审核”中，大部分其实是低置信度、校验失败、解析失败、重复、Abstention 或明确拒绝。v10 将生成结果固定路由到：

```text
ACCEPT       -> 下一生产阶段
REPAIR       -> 最多一次自动修复，再走全部本地规则
REJECT       -> rejected_candidates.jsonl
HUMAN_REVIEW -> 仅限无法由帧和证据唯一判定的语义歧义
```

人工歧义队列目标不超过生成候选的 5%；超过 10% 时阻断该次 pilot，先修复生产链路，不能把问题转嫁给标注员。

## 2. 固定版本与输出目录

| 环节 | v10 契约 |
| --- | --- |
| Evidence schema | `evidence-dataset-schema-v4` |
| Evidence/Commerce prompt | `evidence-prompt-v10.2` |
| Pipeline | `evidence-first-pipeline-v10.3` |
| Semantic quality prompt | `quality-gate-prompt-v2` |
| QA compiler | `evidence-qa-compiler-v8` |
| Question realizer | `question-realizer-prompt-v2` |
| Judge | `judge-prompt-v5` |

Smoke 和正式目录必须分开。当前官方 API 运行使用以下不含密钥的命名：

```text
outputs/evidence/v10_smoke5_qwen_deepseek_official/
outputs/vqa/v10_smoke5_qwen_deepseek_official/
outputs/evidence/v10_pilot64_qwen_deepseek_official/
outputs/vqa/v10_pilot64_qwen_deepseek_official/
```

本地 `.env` 只保存 provider 配置，不进入 Git。视觉 Evidence 和 VQA 答题使用 Qwen，文本生成、语义门禁、审计翻译和 Judge 使用 DeepSeek：

```dotenv
QWEN_API_KEY=<local-secret>
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VISION_MODEL=qwen3-vl-plus
DEEPSEEK_API_KEY=<local-secret>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

运行命令前加载 `.env`：

```bash
set -a
source .env
set +a
```

## 3. 先运行 5 视频严格 smoke

凭证仍通过本地 `.env` 或环境变量注入，不写入配置、日志或 Git：

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_smoke_v10_5videos.json \
  --output-dir outputs/evidence/v10_smoke5_qwen_deepseek_official \
  --vision-model qwen3-vl-plus \
  --text-model deepseek-v4-pro \
  --max-workers 2
```

配置中的 `strict_semantic_verification=true` 会让商业关系在进入任务生成器前接受引用帧复核。它不会使用标题、互动量、粉丝量或私有分析字段。

每个 Evidence 目录必须包含：

- `evidence_units.jsonl`：通过 schema、语言、定位、断言类型、时间范围和置信度门禁的证据；
- `commerce_cues.jsonl`、`commercial_relations.jsonl`：通过本地合同和严格语义验证的商业图；
- `video_evidence_dataset.jsonl`：自动通过的候选 annotation，不等于正式 Gold；
- `human_review_queue.jsonl`：真正的语义歧义；
- `repaired_candidates.jsonl`：一次修复成功的候选；
- `rejected_candidates.jsonl`：确定性淘汰项；
- `pipeline_diagnostics.jsonl`：API、解析、Abstention 和阶段故障；
- `quality_decisions.jsonl`：所有质量路由决定；
- `generation_meta.json`：视频、配置、模型和质量 prompt 指纹。

## 4. 先做候选 QA 审计，不发布正式集

为了查看自动候选质量，需要显式声明候选模式：

```bash
python salesbench.py realize-qa \
  --evidence-dir outputs/evidence/v10_smoke5_qwen_deepseek_official \
  --dataset-file video_evidence_dataset.jsonl \
  --output-dir outputs/vqa/v10_smoke5_qwen_deepseek_official \
  --text-model deepseek-v4-pro \
  --strict-semantic-verification \
  --allow-auto-candidates

python salesbench.py compile-vqa \
  --evidence-dir outputs/evidence/v10_smoke5_qwen_deepseek_official \
  --dataset-file video_evidence_dataset.jsonl \
  --realizations outputs/vqa/v10_smoke5_qwen_deepseek_official/qa_realizations.jsonl \
  --output-dir outputs/vqa/v10_smoke5_qwen_deepseek_official \
  --allow-auto-candidates \
  --allow-missing-tasks
```

`--allow-auto-candidates` 只用于 smoke/pilot 阶段读取尚未冻结的 Evidence annotation。Question Realizer 没有该参数时只读取 `human_accepted`。Compiler v7 在正式模式下还有一条受限自动路径：只有同目录 `qa_realizer_meta.json` 明确启用当前版本的严格 QA 门禁，并且相同 `spec_id` 在 `qa_semantic_verifications.jsonl` 中为 `PASS`，`auto_accepted_candidate` 才能进入编译；旧版 `verified`、缺少门禁元数据或没有 PASS 记录的候选仍会失败关闭。这样不需要逐题人工确认全部 PASS QA，同时也不能把普通候选误发为正式集。

`--strict-semantic-verification` 会在英文问题实现后增加独立 QA 语义门禁：明确缺证据、Gold 过度推断、任务错位、答案不唯一或领域性不足的候选写入 `qa_rejected_candidates.jsonl`；API/解析故障写入 `qa_pipeline_diagnostics.jsonl`；只有真正存在两种合理解释的候选进入 `qa_human_review_queue.jsonl`。只有 `PASS` 项写入 `qa_realizations.jsonl`，因此坏 QA 不会再依赖人工审核阶段淘汰。

QA 输出目录的质量文件含义如下：

- `qa_realizations.jsonl`：通过本地表面规则和严格语义门禁的 QA；
- `qa_semantic_verifications.jsonl`：每题五个语义质量维度与英文理由；
- `qa_rejected_candidates.jsonl`：确定性或语义上明确不合格的 QA，只读；
- `qa_human_review_queue.jsonl`：仅保存无法由现有证据唯一消歧的 QA；
- `qa_pipeline_diagnostics.jsonl`：API、解析或执行故障，不交给内容审核员；
- `qa_accepted_sample.jsonl`：按任务稳定抽取约 10% 的 PASS QA，用来估计自动通过精度，不要求全量人工逐题确认。

## 5. 使用 Qwen 答题并由 DeepSeek Judge

Qwen runner 只接收视频帧、ASR/字幕和英文问题；DeepSeek Judge 接收题目、Gold、模型答案及已公开的证据上下文：

```bash
python salesbench.py run-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --vqa outputs/vqa/v10_smoke5_qwen_deepseek_official/vqa_gold_private.jsonl \
  --output-dir outputs/evaluation/v10_smoke5_qwen_deepseek_official/predictions_qwen3_vl_plus \
  --model qwen3-vl-plus \
  --max-workers 2

python salesbench.py evaluate-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --gold outputs/vqa/v10_smoke5_qwen_deepseek_official/vqa_gold_private.jsonl \
  --predictions outputs/evaluation/v10_smoke5_qwen_deepseek_official/predictions_qwen3_vl_plus/predictions.jsonl \
  --output-dir outputs/evaluation/v10_smoke5_qwen_deepseek_official/judge_deepseek_v4_pro \
  --judge-model deepseek-v4-pro \
  --max-workers 2
```

缺失模型答案仍由本地规则计 0；Judge API 或解析失败必须作为失败显式保留，不能伪造评分。

## 6. 构建翻页式审计 HTML

中文仅作为审计 sidecar；Evidence、Prompt、QA、Gold、预测和 Judge canonical 内容仍为英文。

```bash
python salesbench.py build-audit-translations \
  --manifest configs/pilot64_qwen_deepseek_v10_delivery.json \
  --output outputs/audit/translations/v10_smoke_qwen_deepseek/audit_translations.jsonl \
  --model deepseek-v4-pro

python -m tools.audit_workbench.build \
  --manifest configs/pilot64_qwen_deepseek_v10_delivery.json \
  --translations outputs/audit/translations/v10_smoke_qwen_deepseek/audit_translations.jsonl \
  --output outputs/audit/SalesBench_Quality_Audit_v10_smoke.html \
  --fragment outputs/audit/SalesBench_Quality_Audit_v10_smoke_fragment.html \
  --group smoke \
  --skip-organize
```

审计页每次只展示一个案例，可用上一条、下一条、序号跳转、键盘方向键和筛选器导航。视图分为：

1. 人工语义审核：唯一有 Accept/Edit/Reject 控件的 Evidence/Graph 队列；
2. 自动通过抽样：每任务约 10% 的漏检监控样本；
3. 自动拒绝：只读，检查生产门禁是否按预期淘汰；
4. 流水线诊断：只读，供修代码而不是标内容；
5. Abstention/补采样：决定是否补尾帧、重采样或替换视频；
6. QA：再分为人工语义审核、自动通过抽样、自动拒绝、流水线诊断和自动通过全集只读五个翻页队列；
7. Judge：逐题翻页查看问题、Gold、模型输出、评分理由和实际使用的 Evidence。

## 7. Smoke 验收后再启动 64 视频

必须同时满足：

- 明显错误均出现在 `rejected_candidates.jsonl` 或 `pipeline_diagnostics.jsonl`，不在人工队列；
- `human_review_queue` 比例不超过 10%，目标不超过 5%；
- 自动通过的分层抽样人工精度至少 95%；
- 没有口播效果主张被写成观察事实；
- 没有短时演示被当成长效、心理、因果或转化证明；
- QA 无答案泄漏、泛化空问、明显公式模板和缺证据问题；
- 公开、模型和 Judge payload 不含互动、粉丝、标题或私有分析字段。

满足后，将 cohort 改为 `configs/evidence_pilot_v10_64videos.json`、输出改到 `v10_pilot64_qwen_deepseek_official`，执行相同命令。64 视频仍是 candidate pilot；完成质量校准与人工接受之前不能称为正式 Gold 或公开 benchmark。
