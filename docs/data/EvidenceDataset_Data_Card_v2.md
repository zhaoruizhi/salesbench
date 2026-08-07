# EvidenceDataset Data Card v2

## 状态

当前契约版本为 `evidence-dataset-schema-v2`，提示词版本为 `evidence-prompt-v6`。固定 5 视频 pilot、64 视频 pilot 与 128 视频 alpha cohort 配置已准备，真实生成需要由环境变量注入模型凭证。

## 用途

EvidenceDataset 是 SalesBench-QA 的私有权威标注源。公开 QA 是从已验证的 GroundedAnnotation 确定性编译得到的派生视图。

允许用途：四任务 VQA 构建、证据审计、人工复核、公开/私有评测文件生成。

禁止用途：销量或转化预测、互动因果归因、使用互动量修正标准答案、把内部六维特征作为公开答题捷径。

## 文件

- `video_samples.jsonl`：固定 cohort 与视频资产快照。
- `evidence_units.jsonl`：visual/OCR/ASR 直接证据。
- `gold_proposals.jsonl`：兼容命名的内部候选标注记录。
- `gold_reviews.jsonl`：Challenger 审核记录。
- `video_evidence_dataset.jsonl`：自动验证后的主数据集。
- `human_review_queue.jsonl`：低置信度、冲突、歧义、重复和解析失败项。
- `agent_traces.jsonl`：不含 API key 的调用追踪。
- `generation_meta.json`：版本、计数、指纹和输出位置。

内部 Python 包仍暂用 `salesbench.goldbank` 作为兼容路径；这不是公开数据产品名称。

## 公开与私有边界

公开答题输入只包含 16 帧、ASR/字幕和 question。以下内容不得出现在公开 QA、模型 prompt 或 Judge prompt：

- likes、comments、shares、collects；
- followers、达人层级和发布情境；
- 标题、商品元数据和内部结构化特征；
- `gold_answer`、`evidence_refs`、`source_annotation_ids`；
- 私有互动 profile、阈值和分层标签。

Judge 私有输入可包含参考答案和解析后的 direct evidence context，但仍不能包含互动与账号信息。

## 质量门

- Evidence 外键有效率：100%。
- 私有字段泄漏率：0。
- Challenger 拒绝项自动接受率：0。
- 未送审或低置信度项自动接受率：0。
- 已接受 unresolved conflict：0。
- 语义重复项：确定性移除并进入审计队列。
- 正式公开版本：BP、CM、SS、AE 均非空。
- pilot 人工事实准确率目标：至少 90%。

## 数据划分

train/validation/test 按 creator 分组约为 80/10/10。同一创作者不得跨 split。互动分析使用的私有快照和匹配变量不随公开 VQA 发布。

## 已知局限

- 16 帧采样不能证明完整视频中某对象绝对不存在。
- ASR 与 OCR 可能存在识别错误。
- SS/AE 属于证据约束推理，不等同于真实消费者调查。
- 互动字段缺少曝光、观看时长、统一窗口和快照时间，只能做非因果诊断。
