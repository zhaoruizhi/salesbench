# 六维上下文、证据与公开输入边界

## 1. 三层分工

SalesBench 不把 C1-C6、EvidenceDataset 和 QA 当作三份平行数据集。它们是顺序明确的三层：

| 层 | 回答的问题 | 使用者 |
| --- | --- | --- |
| C1-C6 inventory | 当前项目有哪些可检索、可审计的内部资产 | 数据生产流水线 |
| EvidenceDataset | 哪些事实与受控推理有可定位证据并通过质量审核 | 标注员、Compiler、Judge |
| public VQA | 在统一输入条件下，模型能否回答 BP/CM/SS/AE 问题 | 被测模型 |

六维上下文本身不是 standard answer。EvidenceDataset 也不是把六维字段重新复制一遍，而是把可用的原始事实压缩为带外键的 `EvidenceUnit`，再组织成任务化 `GroundedAnnotation`。

## 2. C1-C6 的内部角色

- C1 视觉：内部视觉特征、帧定位和质量检查。
- C2 语音：ASR、语速等；只有 ASR 原文与时间段可作为直接证据。
- C3 文本：标题与语言特征用于检索/审计；标题不能作为直接证据。
- C4 发布情境：品类、账号和发布时间用于 cohort/split/私有切片。
- C5 跨模态：派生一致性特征用于发现候选；最终 CM 必须回到 ASR/OCR/画面两侧。
- C6 原始视频：公开模型采样 16 帧，也是 visual evidence 的来源。

任何由互动量、粉丝量、标题分数或派生商业特征得到的结论，都不能进入公开答案。

## 3. 不同阶段可见信息

| 阶段 | 16 帧 | ASR/OCR | C1-C6 派生特征 | 标题/发布信息 | 互动量 | 标准答案 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Evidence 生产 | 是 | 是 | 内部候选/审计 | 仅内部，非直接证据 | 否 | 否 |
| 内容人工复核 | 是 | 是 | 必要审计信息 | 默认否 | 否 | 候选值 |
| Public model | 是 | 是 | 否 | 否 | 否 | 否 |
| Judge | 否 | 证据上下文 | 否 | 否 | 否 | 是 |
| Interaction analysis | 否 | 否 | 分层变量可选 | 私有控制变量 | 是 | 可用 VQA 得分切片 |

## 4. 防重合原则

1. C1-C6 只负责信息库存，不定义 task answer。
2. EvidenceUnit 只保存原子事实，不保存 QA 文本。
3. GroundedAnnotation 保存结构化答案和 evidence refs，不保存公开 prompt。
4. QA Compiler 只把已验证 annotation 映射为问题和答案。
5. Public QA 删除答案、证据 refs、来源 annotation 与私有元数据。
6. Interaction analysis 物理和逻辑上独立，不回写 annotation 或 QA。

这一分层解决了原方案中的重合：内部上下文可以丰富标注发现和审计，但不会给被测模型额外信息，也不会让互动结果影响正确答案。
