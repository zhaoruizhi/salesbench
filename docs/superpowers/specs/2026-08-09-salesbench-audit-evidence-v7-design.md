# SalesBench 可视证据审计与 Prompt v7 设计

## 1. 目标

改造现有 Prompt/审计工作台，使人工审核者无需根据不可读的 ID 反查 JSONL，即可在同一条 Evidence、QA 或 Judge 记录中看到：

- 结构化 EvidenceUnit 内容；
- OCR/ASR 原文；
- 视觉/OCR Evidence 引用的精确采样帧；
- 帧索引与视频时间戳；
- QA 的问题、Gold、来源 Annotation、证据内容和证据帧；
- Judge 的参考答案、模型回答、评分理由、证据内容和证据帧。

同时形成 `evidence-prompt-v7`，修正 v6 中已经观察到的 schema 漂移、跨模态约束不足、销售声明事实化、中英自由文本混杂和 Adjudicator 职责过重问题。

## 2. 运行版本边界

- 现有 64 视频产物由 `evidence-prompt-v6` 生成，工作台必须继续把它标为 v6 运行快照。
- 代码中的新 Prompt 版本为 `evidence-prompt-v7`，工作台以“下一次运行 Prompt”展示。
- 工作台不得暗示现有 EvidenceDataset 已经由 v7 重新生成。
- v7 的应用效果需要新的 smoke 与正式运行验证，不在本次 HTML 重建中伪造。

## 3. 帧资产方案

### 3.1 数据来源

帧只允许从已有缓存读取：

```text
outputs/cache/frames/<video_id>/manifest.json
outputs/cache/frames/<video_id>/frame_<index>.jpg
```

`manifest.json` 提供 `frame_index`、`timestamp_s` 和原图路径。EvidenceUnit 的 `frame_indices` 与其建立确定性关联。

### 3.2 缩略图生成

- 审计生成器建立所有审计记录实际需要的唯一 `(video_id, frame_index)` 集合；
- 使用 macOS `sips` 将原帧压缩为最长边 360px、JPEG quality 60 的缩略图；
- 输出到 `outputs/audit/assets/frames/<video_id>/frame_<index>.jpg`；
- 不复制完整视频，不把原始帧 Base64 嵌入完整 HTML；
- HTML 使用相对路径和 `loading="lazy"`，只有滚动到当前卡片时才加载；
- 若 `sips` 不可用，复制原图并记录 `thumbnail_status=source_copy`，不能静默丢失图像；
- 会话内可视化预览只复制其采样记录所需缩略图到可视化目录下的同级 `assets/`。

### 3.3 Evidence 到帧的关联

- `visual`/`ocr`：显示 EvidenceUnit 的全部 `frame_indices`；
- `asr`：若有 `start_s/end_s`，选择时间范围内最近采样帧；
- `asr` 无时间戳：显示帧 0、中间帧、最后一帧作为代表上下文，并醒目标注“无法精确定位到该语音片段”；
- 队列中的 abstention/stage 项若没有 evidence refs，同样显示代表帧并标注“无直接 Evidence 引用”。

## 4. 审计记录的数据契约

每条 Evidence/QA/Judge 审计记录新增：

```json
{
  "evidence_items": [
    {
      "evidence_id": "...",
      "modality": "visual|ocr|asr",
      "subject": "...",
      "predicate": "...",
      "value": "...",
      "text_span": "...",
      "start_s": null,
      "end_s": null,
      "confidence": 0.9,
      "frames": [
        {
          "frame_index": 3,
          "timestamp_s": 3.48,
          "thumbnail_src": "assets/frames/.../frame_003.jpg",
          "relation": "direct|temporal_nearest|representative"
        }
      ],
      "localization_note": ""
    }
  ]
}
```

QA 通过 `evidence_refs` 直接关联 EvidenceUnit；Judge 通过 `vqa_id` 关联同一份 QA evidence items。页面不再以裸 ID 作为主要展示信息。

## 5. HTML 交互

- 帧以响应式缩略图网格显示，带帧号、时间戳和 direct/representative 标签；
- 点击缩略图打开原比例灯箱，不预加载完整视频；
- Evidence 内容默认直接可见，完整 JSON 放在 `<details>`；
- 任务、风险、视频 ID、关键词过滤保持可用；
- `localStorage` 审核决定与 JSON 导出保持不变；
- 图片缺失时显示可读占位，不导致整条记录无法审核；
- 完整 HTML 可通过本地静态服务器或常规浏览器打开，HTML 与 `assets/` 必须一起保留。

## 6. Prompt v7 语言策略

不采用“全部英文”。统一规则为：

- Prompt 指令使用中文；
- `task_type`、`task_subtype`、JSON keys、关系枚举保持英文；
- `subject/predicate/value/answer/reason/issues/evidence_alignment/reasoning_edges` 等自然语言统一中文；
- OCR/ASR `text_span` 原样保留，不翻译；
- 原视频出现的品牌或英文术语允许原样保留；
- 禁止在中文自然语言字段中用英文句子解释中文 Evidence。

## 7. Prompt v7 结构优化

### 7.1 Evidence Extractor

- 将卖方口播效果声明编码为“口播/声称/具体原文”，不得编码为已经验证的产品效果；
- OCR `text_span` 必须是逐字原文，禁止 `[0,50]` 坐标式值；
- 要求 ASR 使用输入提供的 `start_s/end_s`，不得自行猜测；
- 输出中文自然语言字段。

### 7.2 Proposer

- 每个 subtype 明确独立 `target` 和 `proposed_gold` schema；
- AE 禁止 `target.claim` 和 `proposed_gold.relation`；
- CM 必须引用至少两种不同模态，并输出 `modality_pair`；
- `NOT_SHOWN` 只有在定义了完整观察窗口时允许；
- 模型不再负责 proposal ID，本地代码统一生成 canonical ID；
- Consumer 不再生成 SS，Operator 负责 CM，Strategist 负责 SS，Consumer 只负责 AE，消除职责重叠。

### 7.3 Challenger

- 针对 BP、CM、SS、AE 提供不同审核 rubric；
- 中文输出 `issues`、`suggested_revision` 中的自然语言；
- 检查 claim/fact 分离、subtype schema、CM 模态多样性和可定位性；
- 所有非 PASS 项必须进入可追踪队列，最终仍由本地规则控制。

### 7.4 Adjudicator

- 只输出 `accepted_groups` 和 `human_review_queue`；
- 不再重写完整 GroundedAnnotation；
- 本地代码根据被接受 proposal 确定性重建 annotation；
- 只允许合并同任务、同 subtype、语义同义且不冲突的 proposal。

### 7.5 Judge

- Judge Prompt 改为中文，并要求 `reason`、`evidence_alignment` 为中文；
- BP、CM、SS、AE 使用 task-specific rubric；
- 分别输出 `correctness`、`grounding`、`completeness` 三项五档分数；
- 本地代码将三项分数确定性映射到最终 `{0,0.25,0.5,0.75,1}`；
- 当前评估兼容旧 `score` 输出，v7 新格式的聚合变更留给独立评估升级，避免在本次修改中重算既有结果。

## 8. 验收

- 合成测试证明 visual/OCR Evidence 关联正确帧与时间戳；
- 合成测试证明无时间戳 ASR 使用代表帧并带警告；
- QA 与 Judge 记录包含可读 Evidence 内容，不只包含 ID；
- 生成 HTML 使用 `loading="lazy"`，图片路径为相对路径；
- 缩略图资产唯一、可打开且不复制完整视频；
- v6 runtime 与 v7 current prompt 在页面中明确区分；
- Prompt 测试验证自然语言输出中文、subtype schema、角色边界、CM 多模态约束和精简 Adjudicator 契约；
- `pytest -q`、`compileall`、`git diff --check` 通过；
- 在 736px 与 360px 宽度检查证据内容、缩略图、灯箱和筛选交互。
