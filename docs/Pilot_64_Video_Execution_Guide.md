# SalesBench 64 视频 Pilot 完整执行方案

本文档用于用固定 64 个视频跑通 SalesBench 的完整链路，验证 EvidenceDataset、QA 对、模型作答和 LLM-as-Judge 的质量。它是 pilot 执行手册，不是正式大规模 benchmark 发布协议。

当前项目中的“4 个公开数据集”应准确表述为同一 SalesBench 数据集下的 4 个公开任务集：BP、CM、SS、AE。互动分析是独立附加实验，不是第五个 VQA 任务，也不进入主榜。

> 版本说明（2026-08-10）：现有 64 视频 Evidence/QA/Evaluation 是 `evidence-prompt-v6` 运行快照。Prompt v8、英文 QA/Judge 和审计队列归一化只改变当前代码，不会把旧结果自动升级为 v8。下一轮必须先使用 `configs/evidence_smoke_v8_5videos.json` 写入 `outputs/evidence/v8_smoke5_gpt4o_yunwu/`；5/5 全链路与人工抽检通过后，才使用 `configs/evidence_pilot_v8_64videos.json` 写入 `outputs/evidence/v8_pilot64_gpt4o_yunwu/`。smoke 与正式目录禁止复用，v8 也不得覆盖或重标 v6/v7 产物。

## 1. Pilot 的目标和完成标准

64 视频 pilot 需要回答以下问题：

1. 原始 Excel、视频文件和内部 C1-C6 资产能否稳定连接。
2. 每个视频能否提取可定位的 visual、OCR、ASR EvidenceUnit。
3. BP、CM、SS、AE GroundedAnnotation 是否有足够证据且没有越界推断。
4. 确定性编译后的问题是否可回答，标准答案是否正确、充分且不泄漏。
5. 多模态模型能否完成全部 QA，失败率、耗时和成本是多少。
6. LLM-as-Judge 是否与人工评审基本一致，主指标是否可信。
7. 互动分析能否在不污染 VQA 标注和 Judge 的情况下独立运行。

完成 pilot 时至少应得到：

- 64 个固定 `video_id` 的 cohort 配置；
- 一版自动生成及人工复核后的 EvidenceDataset；
- 约 256-512 个 QA，实际数量由有效证据决定，每视频最多 8 题；
- 至少一个多模态被测模型的 `predictions.jsonl`；
- Judge 逐题结果和四任务汇总结果；
- QA 人工抽检记录、Judge 校准结果和问题清单；
- 一份独立互动分析报告。

## 2. 目录和数据安全边界

默认配置位于 `configs/benchmark_v1.json`。关键输入与输出如下：

| 类型 | 默认路径 | 读写行为 |
| --- | --- | --- |
| 研究 Excel | `videos/研究数据.xlsx` | 只读，不会被命令修改 |
| 原始视频 | `videos/raw_data/video/` | 只读，不会被命令修改 |
| 销售截图 | `videos/raw_data/sales/` | 只读，不会被命令修改 |
| 规范化主表 | `outputs/processed/videos_v1.jsonl` | `prepare` 重新生成并覆盖 |
| 资产索引 | `outputs/manifests/assets_v1.json` | `prepare` 重新生成并覆盖 |
| 数据概况 | `outputs/reports/data_profile_v1.json` | `prepare` 重新生成并覆盖 |
| C1-C6 派生资产 | `input/` | `build-inputs` 重新生成并覆盖 |
| 5 视频 v8 smoke EvidenceDataset | `outputs/evidence/v8_smoke5_gpt4o_yunwu/` | 仅用于结构和质量预检 |
| 64 视频 v8 EvidenceDataset | `outputs/evidence/v8_pilot64_gpt4o_yunwu/` | 正式 pilot 目录，支持断点续跑 |
| 64 视频 v8 VQA | `outputs/vqa/v8_pilot64_gpt4o_yunwu/` | 由审核冻结后的 v8 EvidenceDataset 编译 |

因此，本文中的“重建”不是重新制作或删除原始数据，而是从现有 Excel 和素材目录重新计算可再生的 JSON/JSONL 派生文件。原始 Excel、MP4 和 PNG 始终只读；会被覆盖的是 `outputs/processed`、`outputs/manifests`、`outputs/reports` 和 `input/` 中的派生文件。

只有在 Excel、视频目录、字段清洗逻辑或 C1-C6 构建逻辑发生变化时，才必须重新运行 `prepare` 和 `build-inputs`。如果这些输入和代码都没有变化，可以复用现有派生文件，但第一次正式跑 64 视频时建议重新生成一次，确保所有文件来自同一代码版本。

## 3. Step 0：环境和输入预检

### 3.1 安装依赖

项目要求 Python 3.13+，视频采帧要求系统可执行 `ffmpeg` 和 `ffprobe`。

```bash
cd /Users/zhaoruizhi/Desktop/code/SalesBench

python --version
ffmpeg -version
ffprobe -version
python -m pip install -e '.[providers,data]'
pytest -q
```

预期结果：

- Python 版本不低于 3.13；
- `ffmpeg`、`ffprobe` 能输出版本；
- 项目可以导入 OpenAI-compatible provider；
- 测试全部通过。

### 3.2 确认原始输入

运行前人工确认：

- `videos/研究数据.xlsx` 存在且能读取；
- `videos/raw_data/video/` 下视频以数字 `video_id.mp4` 命名；
- Excel 中的 `video_id` 能与视频文件名对应；
- 不把 API key 写入 JSON、Markdown 或 Git。

## 4. Step 1：重建规范化数据和 C1-C6 派生资产

```bash
python salesbench.py prepare --config configs/benchmark_v1.json
python salesbench.py build-inputs --config configs/benchmark_v1.json
```

### 4.1 `prepare` 具体做什么

`prepare` 读取研究 Excel，并扫描 `videos/raw_data/video/` 和 `videos/raw_data/sales/`：

1. 以文件名中的数字 ID 建立视频和截图索引；
2. 规范化日期、时间、粉丝量、产品类别等字段；
3. 将 Excel 记录与本地 MP4/PNG 路径按 `video_id` 连接；
4. 计算 `has_video_asset`、`primary_video_path`、产品分桶和粉丝分层；
5. 写出新的规范化主表、资产 manifest 和数据概况。

主要产出：

| 文件 | 用途 |
| --- | --- |
| `outputs/processed/videos_v1.jsonl` | 后续 cohort 选择和互动分析的规范化主表 |
| `outputs/manifests/assets_v1.json` | 视频/截图文件数量、匹配情况和未匹配文件 |
| `outputs/reports/data_profile_v1.json` | 样本量、视频覆盖率、品类与粉丝层分布 |

验收重点：`record_count` 大于 0、`records_with_video_asset` 足够覆盖 64 个视频、未匹配视频文件数量没有异常增长。

### 4.2 `build-inputs` 具体做什么

`build-inputs` 从 `outputs/processed/videos_v1.jsonl` 重新组织六类内部资产：

| 内部层 | 输出 | 在正式 VQA 中的作用 |
| --- | --- | --- |
| C1 视觉特征 | `input/visual_features/visual_features.jsonl` | 候选发现；只有能回指帧的内容可成为 Evidence |
| C2 语音/ASR | `input/audio_speech/audio_speech_features.jsonl` | 口播和字幕证据 |
| C3 文本语言 | `input/text_language/text_language_features.jsonl` | 内部检索和候选发现；标题不能直接作为标准答案 |
| C4 发布情境 | `input/publish_context/publish_context_features.jsonl` | cohort 分层和私有分析，不进入公开问答 |
| C5 跨模态派生 | `input/cross_modal_consistency/cross_modal_consistency_features.jsonl` | 候选发现和审计，派生分数不是标准答案 |
| C6 原始视频索引 | `input/raw_video/video_index.jsonl` | 视频路径和采样帧入口 |

另有汇总文件 `input/input_summary.json`。C1-C6 是内部资产目录，不是 Gold Answer；EvidenceDataset 只保留能够回指 sampled frame、OCR 或 ASR 的证据和受控推理结论。

当前 v2 代码在真正调用 Evidence 生成模型时，只公开 C2 中的 ASR/字幕和 C6 中的 16 张采样帧。C1、C3、C4、C5 会被整理为内部资产，但不会直接进入 Evidence Extractor、Proposer 或公开被测模型的 prompt。这是防止标题、账号、互动量和派生特征变成答题捷径的隔离边界。

## 5. Step 2：固定 64 视频 cohort

### 5.1 cohort 是什么

cohort 是一次实验使用的固定视频清单和采样配置。它不会复制视频，也不是 train/test split；它只保存 64 个 `video_id`、随机种子、采帧策略、帧数、提示词版本和置信度阈值。

固定 cohort 的作用：

- 确保多次重跑、不同生成模型和不同被测模型使用同一批视频；
- 避免每次随机抽样导致 QA 质量和模型分数不可比较；
- 让人工复核、错误追踪和论文实验都能指向明确的视频集合；
- 让断点续跑的缓存指纹稳定可追溯。

### 5.2 选择方式

```bash
python salesbench.py select-evidence-cohort \
  --config configs/benchmark_v1.json \
  --total 64 \
  --seed 42 \
  --output configs/evidence_pilot_v8_64videos.json

python -m json.tool configs/evidence_pilot_v8_64videos.json
```

选择器先保留默认 anchor 视频，再在“产品类别 × 粉丝层级”单元格之间轮转抽样，并优先选择不同 creator。只有 creator 数量不足时才允许重复 creator。

产出 `configs/evidence_pilot_v8_64videos.json`，重点检查：

- `selection.actual_total == 64`；
- `video_ids` 没有重复；
- `selection.warnings` 中没有无法解释的 `missing_anchor`；
- `selection.cell_counts` 没有被单一品类或粉丝层完全主导；
- `frames_per_video == 16`；
- `frame_strategy == "hook_plus_uniform"`。

64 视频 pilot 主要用于流程和质量验证，可以暂不切 train/validation/test。正式 benchmark 发布前仍需按 creator 生成互斥 split，当前 CLI 尚未把 split helper 接入主流程。

## 6. Step 3：配置视觉模型和文本模型

### 6.1 两个模型分别做什么

EvidenceDataset 生成过程使用两个 provider：

| 参数 | 输入 | 负责阶段 |
| --- | --- | --- |
| `--vision-model` | 16 张采样帧 + ASR/字幕 | 客观 visual/OCR/ASR EvidenceUnit 提取 |
| `--text-model` | 已提取的结构化 EvidenceUnit | CM/SS/AE proposal、Challenger 和 Adjudicator |

本轮按已确认的实验条件，两个阶段都固定使用中转站提供的 `gpt-4o`。这样 5 视频 smoke 与 64 视频正式运行只改变数据规模，不额外引入模型差异。BP 不调用 proposer，而是由本地确定性编译器从已验证 EvidenceUnit 生成。

### 6.2 本轮固定模型

| 用途 | 视觉模型 | 文本模型 | 说明 |
| --- | --- | --- | --- |
| v8 smoke 与 64 视频 pilot | `gpt-4o` | `gpt-4o` | 共用同一中转 base URL 和凭证；不同运行目录严格隔离 |

本轮不使用 DeepSeek。后续若比较其他 proposer 或 Judge，只能新建实验目录，并在 metadata 中记录模型、Prompt、pipeline 和 compiler 指纹，不能与本轮结果混写。

### 6.3 设置 API key

不要把 key 直接放到命令参数、配置或文档中。项目根目录已有被 Git 忽略的 `.env`，当前 CLI 不会自动读取它，因此在当前终端只加载变量，不打印密钥：

```bash
set -a
source .env
set +a

export VISION_API_KEY="$OPENAI_API_KEY"
export VISION_BASE_URL="$OPENAI_BASE_URL"
export VISION_MODEL="gpt-4o"
export TEXT_API_KEY="$OPENAI_API_KEY"
export TEXT_BASE_URL="$OPENAI_BASE_URL"
export TEXT_MODEL="gpt-4o"
```

运行前只确认变量非空，不回显其值。若 smoke 出现 401、404、图片输入不支持或 JSON schema 持续失败，不得继续启动 64 视频正式目录。

### 6.4 先跑 5 视频 smoke test

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_smoke_v8_5videos.json \
  --output-dir outputs/evidence/v8_smoke5_gpt4o_yunwu \
  --vision-base-url "$VISION_BASE_URL" \
  --vision-model "$VISION_MODEL" \
  --text-base-url "$TEXT_BASE_URL" \
  --text-model "$TEXT_MODEL" \
  --max-workers 1
```

smoke test 通过标准：

- `outputs/evidence/v8_smoke5_gpt4o_yunwu/generation_meta.json` 中 5 个视频都有状态；
- 没有大面积 `failed` 或 `review_only`；
- `evidence_units.jsonl` 非空；
- `agent_traces.jsonl` 中没有持续的 401、404、模型不支持图片、JSON 解析失败；
- 抽看 visual/OCR/ASR Evidence 与原视频一致。

如果更换任何模型、提示词版本、采帧数或原始视频，使用新的输出目录做 smoke test。不要把不同配置生成的结果合并成同一个 pilot 版本。

## 7. Step 4：生成 64 视频 EvidenceDataset

smoke test 通过后运行：

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_pilot_v8_64videos.json \
  --output-dir outputs/evidence/v8_pilot64_gpt4o_yunwu \
  --vision-base-url "$VISION_BASE_URL" \
  --vision-model "$VISION_MODEL" \
  --text-base-url "$TEXT_BASE_URL" \
  --text-model "$TEXT_MODEL" \
  --max-workers 2
```

单视频内部流程为：

```text
16 帧 + ASR/字幕
  -> 视觉模型提取 EvidenceUnit
  -> 本地 BP compiler 确定性生成 BP candidate
  -> CM proposer 只生成 CM candidate
  -> SS proposer 只生成 SS candidate
  -> AE proposer 只生成 AE candidate
  -> Challenger 检查证据、冲突、重复和因果越界
  -> Adjudicator 合并或送人工复核
  -> 本地验证、去重、质量状态赋值
```

`--max-workers 2` 表示同时处理两个视频；每个视频内部仍包含多次 API 调用。首次运行建议从 1 或 2 开始，确认 provider 限流后再增加并发。

主要产出：

| 文件 | 内容 |
| --- | --- |
| `video_samples.jsonl` | 实际视频路径和采帧配置快照 |
| `evidence_units.jsonl` | visual、OCR、ASR 最小证据单元 |
| `gold_proposals.jsonl` | 各任务候选 GroundedAnnotation |
| `gold_reviews.jsonl` | Challenger 对 proposal 的逐项意见 |
| `video_evidence_dataset.jsonl` | 自动规则通过的 EvidenceDataset 主文件 |
| `human_review_queue.jsonl` | 低置信、冲突、重复、歧义和解析失败项 |
| `agent_traces.jsonl` | 调用阶段、模型响应、token、成本和错误追踪 |
| `generation_meta.json` | 版本、数量、状态、fingerprint、耗时和文件路径汇总 |

### 7.1 断点续跑规则

每个完成的视频会写入 `outputs/evidence/v8_pilot64_gpt4o_yunwu/.parts/<video_id>/result.json`。同一命令再次运行时会复用 fingerprint 一致的已完成结果，只处理未完成或配置已变化的视频。

fingerprint 包含视频 SHA256、视觉模型、文本模型、采帧策略、帧数、schema、prompt 和置信度阈值。修改任一项后，对应视频会重新计算。需要完全强制重跑时可以使用 `--no-resume`，但更推荐为新模型配置使用新的版本目录，保留可比较的旧结果。

## 8. Step 5：自动审计 EvidenceDataset

```bash
python salesbench.py audit-evidence-dataset \
  --dataset outputs/evidence/v8_pilot64_gpt4o_yunwu/video_evidence_dataset.jsonl \
  --evidence outputs/evidence/v8_pilot64_gpt4o_yunwu/evidence_units.jsonl \
  --output outputs/evidence/v8_pilot64_gpt4o_yunwu/audit_before_review.json
```

重点指标与 pilot 建议阈值：

| 指标 | 建议阈值 | 解释 |
| --- | --- | --- |
| `video_count` | 64 | 64 个视频都有主记录 |
| `evidence_coverage` | 1.0 | 每个接受项的 Evidence 外键有效且属于同视频 |
| `private_field_leakage` | 0 | 没有互动量、粉丝量等私有字段进入 EvidenceDataset |
| `conflict_rate` | 0 | 接受项中没有 unresolved conflict |
| `duplicate_rate` | 尽量为 0，最高不超过 0.05 | 语义重复受控 |
| `videos_missing_required_tasks` | 空列表为理想目标 | 每视频至少有 BP 和 CM；不足项需人工判断 |
| 四任务总数 | 均大于 0 | 正式编译的硬要求 |

自动 audit 只检查结构、引用、重复、冲突和泄漏，不能替代人工判断“问题是否自然、答案是否正确”。

### 8.1 生成可交互审计 HTML

当前工作台用 v8 代码读取既有 v6 历史结果，因此页面必须同时显示 `runtime_prompt_version=evidence-prompt-v6` 和 `current_prompt_version=evidence-prompt-v8`。它不会修改 Gold，只把历史 review queue 归一化为可读卡片，并按需加载关联帧缩略图：

```bash
python -m tools.audit_workbench.build \
  --manifest configs/pilot64_gpt4o_v6_delivery.json \
  --output outputs/audit/SalesBench_Prompt_Audit_Workbench_v8.html \
  --skip-organize
```

页面中候选项必须直接显示 target、Candidate Gold、Evidence 内容和关联帧；abstention 必须显示 `No candidate generated` 及具体原因；无法恢复的历史 ID 必须标记为 unresolved，不能显示空白 Gold 或笼统的 `UNKNOWN`。

## 9. Step 6：人工复核 Evidence

复核对象是 `human_review_queue.jsonl` 中的 proposal/annotation，不是先修改自然语言 QA。每个待审项应同时查看：

- 对应原视频或 16 张采样帧；
- `evidence_units.jsonl` 中被引用的证据；
- `gold_proposals.jsonl` 和 `gold_reviews.jsonl` 中的上下文；
- `docs/annotation/Evidence_Annotation_Guide_v2.md` 的任务边界。

创建 `outputs/evidence/v8_pilot64_gpt4o_yunwu/human_review_decisions.jsonl`。字段格式参考 `docs/annotation/Evidence_Review_Form_v2.jsonl`，每个 review item 使用以下决定之一：

- `ACCEPT_INFERRED`：证据足够，接受为人工确认的 INFERRED 项；
- `REVISE`：修改结构化答案和 evidence refs 后接受；
- `REJECT`：证据不足、跨视频、重复、冲突或任务越界。

建议覆盖方式：

1. 64 个视频的 review queue 全部作出决定；
2. 按品类和任务分层抽取 16 个视频，由两名评审员独立检查所有自动接受项；
3. 其余 48 个视频抽检至少 10% 自动接受项；
4. 双人分歧由第三人裁决并保留 reason code。

应用人工决定：

```bash
python salesbench.py apply-evidence-reviews \
  --evidence-dir outputs/evidence/v8_pilot64_gpt4o_yunwu \
  --decisions outputs/evidence/v8_pilot64_gpt4o_yunwu/human_review_decisions.jsonl \
  --output outputs/evidence/v8_pilot64_gpt4o_yunwu/video_evidence_dataset_reviewed.jsonl
```

产出 `video_evidence_dataset_reviewed.jsonl`。`REJECT` 项不会进入该文件；`ACCEPT_INFERRED` 和 `REVISE` 项会带有 reviewer、时间和 reason code。

随后对 reviewed 文件再执行一次 audit：

```bash
python salesbench.py audit-evidence-dataset \
  --dataset outputs/evidence/v8_pilot64_gpt4o_yunwu/video_evidence_dataset_reviewed.jsonl \
  --evidence outputs/evidence/v8_pilot64_gpt4o_yunwu/evidence_units.jsonl \
  --output outputs/evidence/v8_pilot64_gpt4o_yunwu/audit_after_review.json
```

将 `audit_before_review.json` 与 `audit_after_review.json` 一起保留，检查人工接受项没有引入跨视频 Evidence、重复、冲突或私有字段。

## 10. Step 7：确定性编译 BP/CM/SS/AE QA

```bash
python salesbench.py compile-vqa \
  --evidence-dir outputs/evidence/v8_pilot64_gpt4o_yunwu \
  --dataset-file video_evidence_dataset_reviewed.jsonl \
  --output-dir outputs/vqa/v8_pilot64_gpt4o_yunwu \
  --max-questions-per-video 8 \
  --max-per-task 2
```

编译器不是再次调用 LLM 自由生成问题，而是从已审核 GroundedAnnotation 选择固定 question program，因此同一 EvidenceDataset 和编译策略应产生确定性结果。

主要产出：

| 文件 | 是否私有 | 用途 |
| --- | --- | --- |
| `qa_plan.jsonl` | 私有 | GroundedAnnotation 到 question program 的计划 |
| `qa_candidates.jsonl` | 私有 | 编译后的完整候选 QA |
| `qa_validation.jsonl` | 私有 | accepted、quota skipped、leak rejected 等记录 |
| `vqa_gold_private.jsonl` | 私有 | 标准答案、Evidence Context 和溯源信息，供 Judge 使用 |
| `vqa_public.jsonl` | 可公开 | 被测模型输入，不含答案和私有字段 |
| `public/bp.jsonl` 等 | 可公开 | 四任务拆分文件 |
| `generation_meta.json` | 内部 | QA 总数、任务分布和编译版本 |

编译验收：

- `generation_meta.json` 中 BP、CM、SS、AE 均非空；
- `compiler_version == "evidence-qa-compiler-v4"`；
- `question` 与 `gold_answer` 的规范化自然语言全部为英文；中文视频中的 OCR/ASR 原文只保留在 EvidenceDataset 的 `text_span`；
- `vqa_public.jsonl` 和 `vqa_gold_private.jsonl` 行数一致；
- 每视频不超过 8 题，每任务每视频不超过 2 题；
- `qa_validation.jsonl` 中的 rejected 原因可解释；
- 公开文件不包含 `gold_answer`、`evidence_refs`、互动量、粉丝量或内部特征。

不要为绕过空任务而在正式 pilot 使用 `--allow-missing-tasks`。该参数只用于定位流水线错误。

## 11. Step 8：人工检查最终 QA 对质量

Evidence 复核和 QA 复核解决不同问题。Evidence 复核回答“事实和受控推理是否成立”，QA 复核回答“问题与答案组合是否自然、明确且能公平评分”。

建议至少抽取 64 个 QA，每任务至少 16 个，并由两人独立标注以下字段：

| 字段 | 判断内容 |
| --- | --- |
| `answerable` | 仅根据公开给模型的帧、ASR/字幕能否回答 |
| `gold_correct` | 标准答案是否事实正确或推理合理 |
| `gold_complete` | 是否覆盖问题要求的关键点 |
| `evidence_valid` | 私有 Evidence Context 是否真正支持答案 |
| `task_boundary_valid` | 是否符合 BP/CM/SS/AE 的允许和禁止范围 |
| `question_clear` | 是否无歧义、无答案泄漏、无异常模板表达 |
| `duplicate` | 是否与同视频其他 QA 语义重复 |

推荐 pilot 质量门：

- Evidence ref 有效率 100%；
- 私有字段泄漏 0；
- BP/CM 人工事实准确率至少 95%；
- SS/AE 人工可接受率至少 85%；
- 明显不可回答 QA 为 0；
- 语义重复率低于 5%。

当前项目尚无统一的 QA 质量报告 CLI，因此这一步需要保留人工表格或 JSONL，并在正式扩量前补充自动汇总工具。

## 12. Step 9：运行多模态模型回答 QA

被测模型必须能够直接接收图片输入。本轮先固定为 `gpt-4o`，使 Evidence 生成、模型作答和 Judge 都能使用已经验证过的同一中转 API；这只用于 pilot 流程和质量检查，不用于宣称无偏的正式榜单比较。

先用少量题验证模型和 API：

```bash
# 复用 Step 3 已从 .env 加载的 OPENAI_API_KEY / OPENAI_BASE_URL，
# 不要在命令、配置或文档中写入明文密钥。
python salesbench.py run-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --vqa outputs/vqa/v8_pilot64_gpt4o_yunwu/vqa_public.jsonl \
  --output-dir outputs/vqa/v8_pilot64_gpt4o_yunwu/run_smoke_gpt4o \
  --model gpt-4o \
  --max-samples 8 \
  --max-workers 1
```

确认 8 道题成功后运行全量，并为每个模型使用独立输出目录：

```bash
python salesbench.py run-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --vqa outputs/vqa/v8_pilot64_gpt4o_yunwu/vqa_public.jsonl \
  --output-dir outputs/vqa/v8_pilot64_gpt4o_yunwu/run_gpt4o \
  --model gpt-4o \
  --max-workers 2
```

产出：

- `predictions.jsonl`：每题答案、调用状态、原始响应、token、成本和错误；
- `model_answers/<model>_run_meta.json`：题目总数、失败数和总成本。

当前 baseline runner 没有逐题断点续跑；全量运行前必须先完成 8 题 smoke test。本轮只用 `gpt-4o` 检查流程和数据质量；以后若发布正式模型比较，应另建运行目录并加入至少一个不同模型家族，不能把本轮单模型 pilot 当成完整排行榜。

## 13. Step 10：运行并校准 LLM-as-Judge

### 13.1 运行当前 Judge

Judge 最好与 QA 生成模型和被测模型使用不同模型或至少不同模型家族。Judge 只接收问题、任务类型、参考答案、模型答案和私有 Evidence Context，不接收互动量或账号信息。

```bash
# 复用 Step 3 已从 .env 加载的中转 API 环境变量。
python salesbench.py evaluate-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --gold outputs/vqa/v8_pilot64_gpt4o_yunwu/vqa_gold_private.jsonl \
  --predictions outputs/vqa/v8_pilot64_gpt4o_yunwu/run_gpt4o/predictions.jsonl \
  --output-dir outputs/vqa/v8_pilot64_gpt4o_yunwu/evaluation_gpt4o \
  --judge-model gpt-4o \
  --max-workers 4
```

产出：

- `predictions_judge_details.jsonl`：逐题五档分数、原因、Evidence 对齐说明、错误和成本；
- `predictions_salesbench_qa_eval.json`：BP/CM/SS/AE、宏平均、微平均、缺失答案和 Judge 失败数。

当前实现中缺失模型答案本地记 0 分；Judge 调用或解析失败会计入 `judge_failed_count`，但不进入成功样本均值。因此 pilot 报告必须同时写出 `gold_count`、`scored_count` 和 `judge_failed_count`，不能只报告平均分。

v8 Judge 的 system/user prompt 和 `reason`、`evidence_alignment` 输出统一为英文，报告记录 `judge_prompt_version=judge-prompt-v3`。原始中文 OCR/ASR 仍可作为 Evidence Context 提供给 Judge，但 Judge 不得把它复制成中英混杂的评语。

### 13.2 Pilot 阶段的 Judge 校准

使用 Step 8 的至少 64 个双人评审 QA 作为 Judge 校准集：

1. 两名人工评审独立给出 `0/0.25/0.5/0.75/1.0`；
2. 分歧由第三人裁决，形成 adjudicated human score；
3. Judge 对相同 QA 盲评，不提供被测模型名称；
4. 报告 exact agreement、相邻一档 agreement、weighted Cohen's kappa 和 Judge-human MAE；
5. 低置信、两 Judge 分差至少 0.5、解析失败项进入仲裁；
6. 正式结果按 video/creator 聚类 bootstrap，报告 95% CI。

建议将 Judge 优化为分任务 rubric：

| 任务 | 优先自动检查 | LLM Judge 主要判断 |
| --- | --- | --- |
| BP | 数字、数量、OCR、短文本归一化 | 同义表达、事实覆盖、幻觉 |
| CM | 关系标签和两侧 evidence refs | 解释是否覆盖一致、互补、冲突或未展示 |
| SS | 禁止效果/因果词、Evidence 引用 | 策略识别、说服结构、证据连接和完整性 |
| AE | 禁止真实转化/粉丝画像推断 | 需求、场景、决策障碍、不确定性边界 |

最终 Judge JSON 应增加 criterion scores、unsupported claims、confidence 和 needs review。Judge 失败经过重试仍未解决时，应阻止正式总分发布，或同时报告失败计 0 的下界和失败计 1 的上界，不能静默排除。

## 14. Step 11：独立互动分析

互动分析只在 VQA Judge 完成后读取逐题分数，再与私有互动快照做描述性关联和分层诊断：

```bash
python salesbench.py analyze-interactions \
  --config configs/benchmark_v1.json \
  --judge-details outputs/vqa/v8_pilot64_gpt4o_yunwu/evaluation_gpt4o/predictions_judge_details.jsonl \
  --output-dir outputs/analysis/interactions_pilot64 \
  --bootstrap-samples 1000
```

产出：

- `private_analysis_metadata.jsonl`：私有互动分层元数据；
- `interaction_analysis.json`：描述性关联、诊断切片和 bootstrap 区间。

报告必须标记 `diagnostic_only`、`non_causal` 和 `included_in_leaderboard=false`。不能写“互动量导致模型表现”或“模型能够预测点赞/销量”。

## 15. 最终验收清单

### 数据和 Evidence

- [ ] `prepare`、`build-inputs` 完成，原始 Excel/MP4/PNG 未被修改。
- [ ] cohort 的 `actual_total` 为 64，随机种子已记录；实际模型 ID 可从 `agent_traces.jsonl` 核验。
- [ ] `generation_meta.json` 中 64 个视频状态均有解释。
- [ ] Evidence 外键有效率 100%，私有字段泄漏 0。
- [ ] review queue 已全部处理，双人复核分歧已裁决。

### QA

- [ ] BP、CM、SS、AE 均非空。
- [ ] 公开/私有 QA 行数一致，公开文件无答案或私有字段。
- [ ] 至少 64 个 QA 完成双人人工质量检查。
- [ ] BP/CM 与 SS/AE 达到约定人工质量门。

### 模型和 Judge

- [ ] 至少两个多模态被测模型完成全部题目。
- [ ] 缺失答案、模型失败和 Judge 失败均单独报告。
- [ ] Judge 与人工评分完成一致性校准。
- [ ] 四任务分数、宏平均、微平均和 95% CI 齐全。

### 独立互动实验

- [ ] 互动字段未进入 Evidence、公开 QA、模型 prompt 或 Judge prompt。
- [ ] 互动分析与 VQA 主榜分开保存和报告。
- [ ] 所有结论保持描述性、非因果表述。

## 16. 推荐的实际执行顺序

```text
环境与输入预检
  -> prepare + build-inputs
  -> 固定 64 视频 cohort
  -> 配置视觉/文本 provider
  -> 5 视频 smoke test
  -> 64 视频 EvidenceDataset
  -> 自动 audit
  -> Evidence 人工复核并合并
  -> reviewed audit
  -> 确定性编译 QA
  -> 最终 QA 双人抽检
  -> 被测 VLM 8 题 smoke test
  -> 被测 VLM 全量回答
  -> LLM-as-Judge
  -> Judge-human 校准和置信区间
  -> 独立互动分析
  -> pilot 质量报告与扩量决策
```

只有当 Evidence、QA 和 Judge 三个质量门都通过后，才进入 128 视频或更大规模的数据扩展。不要在 pilot 期间同时修改 cohort、生成模型、prompt、Judge rubric 和编译策略；每次只改变一个因素，并使用新的版本目录保留可比较结果。
