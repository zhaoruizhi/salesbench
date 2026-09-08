# SalesBench v10 Candidate.2 质量链路执行指南

本指南只描述新的不可变运行链路。旧版 v10 smoke 仍可作为历史对照，但不能与本次产物混合，也不能直接升级为正式 Gold。

`v10c2-smoke5-qwen-deepseek-20260907-001` 是字段契约诊断运行：模型曾把
`assertion_scope/action_role` 输出为自由文本，因而在生产门禁被大量自动拒绝。该目录永久保留，
但不作为 QA 输入。`-002` 使用封闭枚举 prompt 和本地安全回退重新生成全部 Evidence。

`v10c2-smoke5-qwen-deepseek-20260907-002` 验证了封闭枚举修复，但仍暴露出 ASR
事实范围升级、语言阶段无自动修复，以及同模态“重复”关系误入人工队列的问题。`-003` 增加
了 ASR 一次性修复、模态语义覆盖和跨模态关系硬门禁。

`v10c2-smoke5-qwen-deepseek-20260907-003` 验证了语言修复并补回全部五个视频的 ASR，
但审查发现单帧结果被错误编译为“状态变化”，以及模拟品尝被误当作结果演示。`-004` 在
Evidence/Cue 阶段拒绝模拟使用，并且只有真正的 before/after Cue 才能生成状态变化 BP。

完整历史运行 `v10c2-smoke5-qwen-deepseek-20260907-004` 随后的重跑暴露了一个独立工程问题：
16 张 Base64 JPEG 在部分 HTTPS 会话中于服务端返回
token 之前发生 SSL EOF、broken pipe 或 remote protocol error。Qwen 支持多图/视频，早期同样
16 帧的运行也曾成功；因此不能把该故障解释成模型不支持视频。`-005` 保持 16 帧不变，先将帧
上传到 DashScope 临时 OSS，再使用带帧号和时间戳的 `oss://` URL 调用 Qwen。

本轮固定身份：

```text
run_id: v10c2-smoke5-qwen37-deepseek-20260908-005
benchmark_release: salesbench-v10-candidate.2
run_root: outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/
```

## 1. 本轮质量路由

质量问题必须在生产阶段分流，而不是全部交给人工：

```text
PASS         -> 允许进入下一生产阶段
REPAIR       -> 最多自动修复一次，再执行全部规则
REJECT       -> 明确错误，写入 rejected_candidates.jsonl
HUMAN_REVIEW -> 只有证据支持两种合理解释时才进入人工队列
DIAGNOSTIC   -> API、解析或程序故障，写入 pipeline_diagnostics.jsonl
```

人工歧义队列目标不超过候选的 5%，超过 10% 时阻断 candidate。自动通过项只做按任务分层的约 10% 抽样审核，用于估计门禁漏检率，不是全量人审。

## 2. 固定版本

| 环节 | 固定版本 |
| --- | --- |
| Evidence schema | `evidence-dataset-schema-v4` |
| Evidence/Commerce prompt | `evidence-prompt-v10.6` |
| Evidence pipeline | `evidence-first-pipeline-v10.8` |
| Evidence quality prompt | `quality-gate-prompt-v3` |
| Question realizer | `question-realizer-prompt-v3` |
| QA quality prompt | `qa-quality-prompt-v2` |
| QA compiler | `evidence-qa-compiler-v9` |
| Judge | `judge-prompt-v5` |
| Audit translation | `audit-translation-prompt-v2` |
| Audit workbench | `audit-workbench-v11` |

组件版本和 benchmark release 是两个概念。例如 `compiler-v9` 只是编译器版本，不表示 benchmark 是 v9。

## 3. API 与模型职责

密钥只保存在本地 `.env`，禁止复制到配置、文档、命令输出和 Git。加载时不打印变量值：

```bash
set -a
source .env
set +a
```

沿用已有变量：

```text
QWEN_API_KEY, QWEN_BASE_URL, QWEN_VISION_MODEL
DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
```

模型职责固定如下：

- Qwen：视觉/OCR Evidence、视觉商业线索、跨模态关系复核、QA 独立语义质量复核；
- DeepSeek：文本商业线索、CM/SS/AE 候选、英文问题实现、中文审计翻译；
- 正式评测时由 Qwen benchmark runner 回答视频问题，由 DeepSeek Judge 评分。

本轮视觉模型固定为 `qwen3.7-plus`。每个视频仍采用完整 16 帧；改变的只是传输方式，不改变
抽帧位置、Evidence 定义或题目难度。HTTP 推理请求自动携带
`X-DashScope-OssResourceResolve: enable`。临时 OSS URL 只允许用于 smoke/开发，有效期约
48 小时且与上传时的模型和账号绑定；正式 64 视频 release 必须迁移到长期 OSS，不能依赖临时
存储，也不能把临时 URL 写入公开 Evidence 或 QA。

QA 生成模型和质量复核模型是两个独立 client：DeepSeek 生成，Qwen 复核，避免生成器自评直接放行。

## 4. 不可变目录与指纹

本轮所有产物位于同一 run root：

```text
outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/
  run_manifest.json
  evidence/
  qa/
    realizations/
    compiled/
  evaluation/
  translations/
  audit/
```

完整性链路：

1. Evidence 对视频、配置、帧策略、模型和组件版本生成 `source_fingerprint`，再对产物生成 `evidence_fingerprint`。
2. QA realization 必须读取该 Evidence 指纹；`.parts` 按 Evidence 指纹隔离。
3. Compiler 必须匹配 Evidence 与 QA realization 指纹，并验证 Annotation、Evidence、Cue、Relation 四类引用 100% 闭合。
4. HTML 再执行一次指纹和引用闭包检查；不一致时隐藏最终 QA 并显示阻断原因。

即使目录名都包含 v10，只要指纹不同，也不能把旧 Evidence 与新 QA 拼接。

## 5. 生成五视频 Evidence

固定 cohort：`configs/evidence_smoke_v10_candidate2_qwen37_5videos.json`。

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_smoke_v10_candidate2_qwen37_5videos.json \
  --output-dir outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/evidence \
  --run-id v10c2-smoke5-qwen37-deepseek-20260908-005 \
  --benchmark-release salesbench-v10-candidate.2 \
  --vision-transport dashscope_temporary_oss \
  --vision-preflight \
  --max-workers 1
```

命令先抽取并核对第一个视频的 16 帧，将 16 帧上传后连续执行三次完整输入健康检查；三次均
确认收到 16 个带标签图像块才进入 Evidence 生产。任一门禁不通过都会在写入视频 part 之前
停止，并将不含 URL/密钥的诊断写到 `evidence/.parts/vision_preflight.json`。生产阶段的网络
失败只进入 diagnostic 和网络重试，不再错误触发 Evidence 内容 repair。

该命令从现有 processed dataset 读取指定五个视频，重新运行 Evidence、商业图和四任务 Annotation。它不重新下载视频，也不重建 C1-C6。C1-C6 只通过 Context Store 管理内部来源，公开 Evidence 仍只允许帧、OCR 和 ASR。

主要产物：

- `evidence_units.jsonl`：可直接引用的帧、OCR、ASR 事实；
- `commerce_cues.jsonl`：产品、报价、演示、情景、问题、异议、CTA 等商业线索；
- `commercial_relations.jsonl`：主张—演示、问题—解决、异议—回应、优惠条件等证据关系；
- `video_evidence_dataset.jsonl`：通过门禁的 BP/CM/SS/AE Annotation；
- `human_review_queue.jsonl`：真正语义歧义；
- `rejected_candidates.jsonl`：过度推断、背景动作、主张当事实、内部 ID 泄漏等明确拒绝项；
- `pipeline_diagnostics.jsonl`：API 或解析故障；
- `quality_decisions.jsonl`：所有质量路由；
- `generation_meta.json` 与 `../run_manifest.json`：模型、计数、组件和指纹。

### ASR 时间戳不可用

当前 ASR 可能只有全文，没有可靠片段时间。此时 `start_s/end_s=null`、`timestamp_status=unavailable`。ASR 时间戳不可用是输入能力边界，不是 Evidence 缺陷，也不进入人工审核。

- 仍可生成口播事实和不依赖顺序的视觉—语音一致性任务；
- 不生成依赖时序定位的任务或关系，例如“主张是否早于 CTA”或“某一口播片段是否与画面错位”；
- HTML 用中性提示展示代表帧，并说明无法精确定位语音片段。

## 6. 生成英文 QA，并由 Qwen 独立复核

```bash
python salesbench.py realize-qa \
  --evidence-dir outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/evidence \
  --dataset-file video_evidence_dataset.jsonl \
  --output-dir outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/qa/realizations \
  --allow-auto-candidates \
  --strict-semantic-verification \
  --max-workers 2
```

DeepSeek 只实现自然英文问题；Gold 来自 EvidenceDataset，不允许模型改写事实范围。Qwen 检查 12 项：证据可回答、Gold 受支持、答案唯一、任务对齐、内容具体、商业相关、问法自然、非琐碎、具有商业诊断价值、断言范围不升级、确实需要预期模态、引用闭合。

只有 12 项全部为真才进入 `qa_realizations.jsonl`。明确不合格进入 `qa_rejected_candidates.jsonl`，真正歧义进入 `qa_human_review_queue.jsonl`，模型/API 故障进入 `qa_pipeline_diagnostics.jsonl`。

## 7. 编译最终候选 QA

```bash
python salesbench.py compile-vqa \
  --evidence-dir outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/evidence \
  --dataset-file video_evidence_dataset.jsonl \
  --realizations outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/qa/realizations/qa_realizations.jsonl \
  --output-dir outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/qa/compiled \
  --allow-auto-candidates \
  --allow-missing-tasks
```

Compiler v9 先失败关闭地检查指纹和引用，再按 Evidence 置信度、商业图强度、推理价值、12 项验证、内容具体性，以及背景动作、啰嗦答案和能力重复惩罚排序。

重点检查：

- `vqa_gold_private.jsonl`：最终接受的英文问题与私有 Gold；
- `vqa_public.jsonl`：不含 Gold 的公开输入；
- `qa_selection.jsonl`：选中/跳过原因与分项分数；
- `reference_closure.json`：四类引用必须 100%；
- `generation_meta.json`：Evidence、QA realization、compile 三个指纹。

## 8. 生成审计中文版与翻页 HTML

中文只作为审计 sidecar，不修改英文 canonical Evidence、QA 或 Gold：

```bash
python salesbench.py build-audit-translations \
  --manifest configs/v10_candidate2_qwen37_smoke_delivery.json \
  --output outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/translations/audit_translations.jsonl \
  --batch-size 20

python -m tools.audit_workbench.build \
  --manifest configs/v10_candidate2_qwen37_smoke_delivery.json \
  --translations outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/translations/audit_translations.jsonl \
  --output outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/audit/SalesBench_v10c2_Smoke_Audit.html \
  --fragment outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/audit/SalesBench_v10c2_Smoke_Audit_fragment.html \
  --group smoke \
  --skip-organize
```

工作台含六页：运行概览、当前 Prompt、Evidence 审计、QA 审计、最终 QA 浏览、评估 / Judge。Evidence/QA 审计每页一个案例，只对真正歧义和约 10% 监控样本提供按钮。最终 QA 浏览只读取 `qa/compiled/vqa_gold_private.jsonl`，不读取 `.parts`、未筛选 QuestionSpec、拒绝项或诊断；中文问题/答案优先，英文可展开，并展示可读 Cue、Relation、Evidence 和懒加载关联帧。

## 9. Smoke 验收顺序

1. 五个视频都已处理，Evidence fingerprint 非空；
2. 人工歧义率不超过 10%，理想值不超过 5%；
3. 没有背景 handling 被当作产品身份或功能演示；
4. 没有卖家效果主张被升级为观察事实；
5. 没有依赖 unavailable ASR 时间的任务；
6. QA 生成器/复核器分别是 DeepSeek/Qwen；
7. QA 的 12 项质量全部 PASS；
8. `reference_closure.json` 为 100%；
9. 最终 QA 不含空泛、琐碎、模板化问法；
10. HTML 不含互动量、粉丝量、标题或私有分析元数据。

只有本轮 smoke 经人工确认后，才使用新的 run ID 执行 64 视频。不能在当前 smoke run root 中覆盖或追加 64 视频。

## 10. 后续评测

Smoke QA 通过抽检后，再用 Qwen 生成 `predictions.jsonl`，用 DeepSeek Judge 评分。缺失答案由本地规则计 0；Judge API/解析失败必须保留为失败。主排行榜仍使用 BP、CM、SS、AE 四任务 macro-average；互动分析始终是私有独立实验，不进入 QA、Gold、模型输入或 Judge。
