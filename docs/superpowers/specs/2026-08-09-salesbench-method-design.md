# SalesBench 中文 Method 章节设计说明

## 1. 交付目标

本任务将先形成一个可交互审阅的中文算法工作台，再在同一工作台中形成可直接用于 ACL、EMNLP 等计算语言学会议论文的中文 `Method` 视图。算法审阅先于论文润色，避免在算法定义尚未稳定时反复修改成稿。内容必须完整覆盖：

1. 数据集构建与 cohort 选择；
2. 多模态观测、证据提取及 EvidenceDataset 构建；
3. BP、CM、SS、AE 四类任务设计；
4. Evidence-First 多智能体标注与 QA 生成框架；
5. 确定性 QA 编译、公开/私有数据隔离；
6. LLM-as-Judge 评估、聚合指标及独立互动诊断；
7. 质量控制、复现机制、实现边界与局限。

最终交付物不使用 Markdown 作为用户审阅载体，而采用两层 HTML：

- 会话内 `salesbench-algorithm-workbench.html`：位于当前任务专属可视化目录，用于逐节审阅、展开代码依据、切换算法/论文视图并向 Codex 发回修改意见；
- `deliverables/SalesBench_Algorithm_Method_Workbench.html`：算法冻结后导出的独立单文件 HTML，可脱离 Codex 浏览，内含完整算法表述、论文 Method、数学定义、伪代码、任务表和版本记录；
- `deliverables/figures/salesbench_method_framework.png`：算法冻结后使用内置 ImageGen 生成并视觉复核的论文框架图，嵌入或链接于独立 HTML。

当前 `docs/superpowers/specs/` 下的 Markdown 仅作为内部设计与审计记录，不属于最终交付物。

HTML 工作台采用中文表达，保留 `EvidenceUnit`、`GroundedAnnotation`、`BP/CM/SS/AE` 等实现术语及英文缩写。不会引入代码中不存在的模型训练目标、损失函数、检索器、视频编码器或智能体通信机制。

## 1.1 交互轮询工作流

工作台为每个方法模块分配稳定编号 `M1`–`M8`，并提供两种同步视图：

- **算法视图**：以输入、输出、数据结构、状态转移、质量门和实现边界为核心；
- **论文视图**：把已经确认的算法转换为 Method 论文叙事、公式、表格和伪代码。

每个模块包含可展开的“代码依据”和“当前实现边界”，并提供以下反馈路径：

1. 用户选择一个模块并填写修改意见；
2. 会话内版本通过 `window.openai.sendFollowUpMessage` 把模块编号、当前版本和意见发送回 Codex；
3. Codex重新核对代码、修改算法基线与论文表述，并把版本从 `v0.x` 递增；
4. 工作台的 changelog 记录“修改内容—代码依据—受影响章节”；
5. 用户明确确认“算法冻结”后，才生成最终论文框架图并导出独立 HTML。

独立 HTML 不依赖服务器或外部 API；展示状态和本地草稿可使用浏览器 `localStorage`，同时支持将反馈导出为 JSON。由于独立文件无法自动把信息发回 Codex，用户可将导出的 JSON 重新附加到任务中。会话内版本则可直接触发后续消息。

工作台的主要模块为：总体数据流、数据构建、证据构建、任务本体、多智能体状态机、确定性 QA 编译、评估聚合、复现与实现边界。初始版本优先展示算法视图；论文视图在相应模块得到确认后同步更新。

## 2. 写作原则与证据优先级

所有方法描述以当前代码和配置为第一事实来源，优先级如下：

1. `src/salesbench/` 中的实际执行逻辑；
2. `configs/*.json` 中的固定参数与 cohort；
3. `outputs/reports/data_profile_v1.json`、`input/input_summary.json` 等现有数据产物；
4. 测试所证明的公开/私有隔离和端到端契约；
5. README 和设计文档仅用于解释代码意图，不用于覆盖代码事实。

正文会严格区分三类陈述：

- **已实现协议**：当前代码真实执行的行为；
- **当前数据实例**：现有产物中的规模或统计量；
- **发布建议或尚未接入主流程的能力**：例如 creator-disjoint split helper 已实现，但尚未被 CLI 主流程调用，因此不能写成已经完成的数据划分。

## 3. 拟定 Method 结构

### 3.1 问题定义与总体框架

给定营销短视频集合 \(\mathcal{V}=\{v_i\}_{i=1}^{N}\)，系统将每个视频映射为公开观测 \(O_i=(F_i,A_i)\)，其中 \(F_i\) 为采样帧集合，\(A_i\) 为 ASR/字幕文本。构建过程首先产生可定位证据集合 \(E_i\)，再产生证据约束的标注集合 \(G_i\)，最后由确定性程序 \(\mathcal{C}\) 编译为开放式 VQA 样本：

\[
O_i \xrightarrow{\mathcal{E}} E_i
\xrightarrow{\mathcal{M}} G_i
\xrightarrow{\mathcal{C}} Q_i.
\]

其中 \(\mathcal{E}\) 是客观证据提取器，\(\mathcal{M}\) 是多智能体候选生成、挑战和裁决流程，\(\mathcal{C}\) 不再调用生成模型。该形式化突出本项目的关键事实：智能体生成的是证据约束候选标注，而不是直接自由生成最终 QA。

### 3.2 数据集构建与 cohort 采样

本节将覆盖以下代码事实：

- 研究工作簿经字段清洗、数值解析、日期时间转换和资产索引形成内部视频记录；
- 当前数据产物含 1,200 条记录，其中 1,196 条匹配视频资产、1,158 条匹配商品图像；商品图像并不进入公开 VQA；
- 内部资产按 C1 视觉、C2 语音、C3 文本、C4 发布情境、C5 跨模态派生特征、C6 原始视频六域组织；
- 实际 Evidence 生成只暴露 C2 中的 ASR/字幕和 C6 中的采样帧。C1、C3、C4、C5 不进入 Evidence Extractor 或 Proposer；
- cohort 候选需具有 `video_id`、产品桶、粉丝层级、可用视频以及标题或视频文本。标题只用于 cohort 资格判断，不能成为直接证据；
- 粉丝层级由全体记录的粉丝数四分位点确定，当前产物阈值为 15,000、58,000 和 350,250；
- 选择器首先保留最多 20 个固定 anchor，然后在 5 个产品类别与 4 个粉丝层级构成的单元之间按固定顺序轮转抽样；第一次优先 creator 唯一，样本不足时允许 creator 重复；随机种子为 42；
- 当前配置包含 64-video pilot 和 128-video alpha cohort。正文不会把 128 配置误写成已经完成人工审校的正式 benchmark；
- creator-disjoint 划分函数按 creator group 随机打乱并近似分配 80%/10%/10%。由于该 helper 尚未接入主 CLI，正文将其写为“实现提供的数据划分规则”，而不是“当前产物已经完成划分”。

### 3.3 多模态观测与证据构建

对时长为 \(T_i\) 的视频，配置目标为 \(K=16\) 帧，前三帧覆盖开场 \(h_i=\min(3,T_i)\)，其余十三帧覆盖剩余时段。时间戳按代码写为：

\[
t^{\mathrm{hook}}_r=\frac{h_i(r+0.5)}{3},\quad r\in\{0,1,2\},
\]

\[
t^{\mathrm{uni}}_s=h_i+\frac{(T_i-h_i)(s+0.5)}{13},\quad s\in\{0,\ldots,12\}.
\]

第二式仅在 \(T_i>h_i\) 时执行，且时间戳上界裁剪至 \(T_i-0.1\)。因此严谨表述应为“目标采样 16 帧”；极短视频或抽帧失败时实际帧数可少于 16。平均灰度低于 20 的黑帧会尝试向后偏移 0.5 秒重新抽取。

每个证据单元形式化为：

\[
e=(\mathrm{id},v,m,\tau,F,x,s,p,o,a,c),
\]

其中 \(m\in\{\texttt{visual},\texttt{ocr},\texttt{asr}\}\)，\(\tau\) 为时间区间，\(F\) 为帧索引，\(x\) 为 OCR/ASR 原文片段，\((s,p,o)\) 为主语—谓词—值结构，\(a\) 为属性，\(c\in[0,1]\) 为置信度。视觉证据必须引用帧；OCR/ASR 必须保留 `text_span`；标题、元数据、互动量和派生跨模态分数被验证器拒绝为直接证据。

证据 ID 由规范化后的 `video_id`、模态和当前输出序号经 SHA-256 截断摘要确定。因而它在证据单元顺序不变时可复现，但不是由证据语义内容独立寻址；正文不会将其误写成跨不同生成顺序仍保持不变的内容哈希。证据输出经过 JSON 解析、模态规范化、字段验证和直接证据模态过滤。

### 3.4 层次化任务设计

任务由低层感知到高层受控推理构成，但不宣称四者形成因果链：

- **BP (Basic Perception)**：产品、人物、动作、数量、画面文字和口播事实。当前本地自动 BP builder 实际产生 `ENTITY_ATTRIBUTE`、`COUNT_SPATIAL`、`ACTION`、`OCR_FACT` 和 `ASR_FACT`；本体和问题模板还定义了 `STATE_CHANGE` 与 `TEMPORAL_ORDER`，但当前自动映射逻辑不会直接产生这两类；
- **CM (Cross-Modal Verification)**：`SUPPORTED`、`PARTIALLY_SUPPORTED`、`CONTRADICTED`、`NOT_SHOWN` 和 `TEMPORALLY_MISALIGNED` 五类关系；
- **SS (Selling Strategy Reasoning)**：Hook、价值主张、信任、异议处理、紧迫性/CTA 和漏斗作用；
- **AE (Audience-Need Alignment)**：需求匹配、使用场景、决策状态和内容动机。AE 只解释内容所回应的需求，不推断真实用户画像或转化。

BP 至少需要 1 个证据引用，CM/SS/AE 至少需要 2 个不同证据引用。确定性质量映射将 BP/CM 标为 `DIRECT`，SS/AE 标为 `INFERRED`。

### 3.5 Evidence-First 多智能体标注框架

算法按以下状态机书写：

1. **Objective Evidence Extractor** 接收采样帧与 ASR/字幕，只输出 visual/OCR/ASR `EvidenceUnit`；
2. **Local BP Builder** 从每个有效证据单元确定性地产生 BP proposal；
3. 三个文本智能体只接收结构化证据：
   - Consumer：AE 全部子任务，以及 SS 的价值主张和异议处理；
   - Operator：CM 全部子任务，以及 SS 的 Hook、紧迫性/CTA 和漏斗作用；
   - Strategist：SS 全部子任务；
4. 每个 proposer 输出 0–3 个候选及 abstentions，候选置信度低于 \(\gamma=0.70\) 时不进入 Challenger；
5. **Challenger** 检查证据存在性、证据支持、事实/推理分离、替代解释、重复、冲突和因果越界，只允许 `PASS` 项继续；`REVISE`、`HUMAN_REVIEW` 和 `REJECT` 均不会自动进入裁决接受路径；
6. **Adjudicator** 只接收通过 Challenger 的非 BP proposal，负责合并同义候选、保留非冲突候选并输出 `GroundedAnnotation` 或人工复核项；BP 由本地代码直接转换，但同样必须先获得 Challenger 的 `PASS`；
7. 本地验证器重新绑定唯一来源 proposal，忽略模型自报质量层级，检查证据外键、同视频约束、最小证据数、CM 枚举、私有字段、因果语言、低置信度、语义重复和相同目标冲突；
8. 低置信度、proposal 解析错误、abstention、Adjudicator 主动退回或遗漏、最终验证失败、语义重复和目标冲突会进入 `human_review_queue`。Challenger 的非 `PASS` 项会被排除在自动接受路径之外，但当前代码不会把每一条 `REVISE`、`HUMAN_REVIEW` 或 `REJECT` review 自动物化为 queue item；正文会把这一点写成当前实现边界。对已经进入队列的 proposal，人工决定支持接受、修订或拒绝，并以 `human_accepted` 状态合并回 EvidenceDataset。

算法伪代码会同时表达失败语义：Evidence 提取失败导致该视频失败；单个 proposer 失败产生 `partial`；Challenger 或 Adjudicator 失败会返回 review-only 结果，不能产生自动接受的主记录。

### 3.6 确定性 QA 编译

编译器只处理 `Gold-A`/`Gold-B`，即 `DIRECT`/`INFERRED` 标注。对每个视频，候选按 BP、CM、SS、AE 优先级及稳定 ID 排序；每视频最多 8 题，每任务最多 2 题。问题由 `(task, subtype, format)` 索引的固定中文模板渲染，答案从任务特定字段按固定优先级导出；不调用生成模型。

编译器拒绝无模板标注、答案直接出现在问题中的样本及私有字段泄漏样本。正式编译要求整个输出中的 BP、CM、SS、AE 均非空，而不是要求每个视频都有四任务。

公开记录保留问题、任务、子任务和编译器元数据，但移除 `gold_answer`、`evidence_refs`、`evidence_context`、来源标注和质量状态。私有 Gold 文件保留参考答案和证据上下文供评估使用。

### 3.7 被测模型与评估协议

被测 VLM 对每题只接收同一 `hook_plus_uniform` 帧集合、`video_text` 形式的 ASR/字幕以及当前问题。公开 runner 不把任务类型、标准答案、标题、商品元数据、结构化特征或互动量传给模型。

Judge 接收问题、任务类型、参考答案、模型答案和去除私有字段的 Evidence Context，不接收互动/账号信息。允许分数为：

\[
\mathcal{S}=\{0,0.25,0.5,0.75,1\}.
\]

对任务 \(t\) 的成功评分集合 \(D_t\)，定义：

\[
\mathrm{StrictAcc}_t=\frac{1}{|D_t|}\sum_{j\in D_t}\mathbf{1}[s_j=1],
\qquad
\mathrm{RelaxedAcc}_t=\frac{1}{|D_t|}\sum_{j\in D_t}s_j.
\]

主指标为四个非空任务的 Relaxed Accuracy 宏平均：

\[
\mathrm{MacroRA}=\frac{1}{|\mathcal{T}'|}\sum_{t\in\mathcal{T}'}\mathrm{RelaxedAcc}_t,
\]

其中 \(\mathcal{T}'\) 是至少存在一个成功评分样本的任务集合。微平均仅作辅助。缺失模型回答由本地规则直接赋 0，且进入指标分母；Judge API/解析失败被记录为失败，但当前聚合实现不把无分数的 Judge 失败项放入准确率分母，因此正文必须同时报告 `judge_failed_count`，不能把这一点写成“所有失败均计 0”。

### 3.8 质量保证、复现与独立互动分析

质量审计包括：证据外键覆盖、CM 多证据覆盖、语义重复率、冲突率、人工复核率、私有字段泄漏和任务缺失。帧缓存使用视频 SHA-256、采样策略、帧数、Hook 参数和缓存版本作为失效条件；流水线 part 只在 fingerprint 相同且状态为 `ok` 时复用。

点赞、评论、分享和收藏仅进入独立私有分析。实现对计数做 `log1p`，按全局三分位构造层级，使用产品桶、粉丝层级、发布周和时长层级作匹配控制；报告 Spearman 相关和任务得分分层的 bootstrap 95% 区间。所有输出标记为 `diagnostic_only`、`non_causal` 且不进入排行榜。

## 4. 框架图设计

框架图采用横向三阶段布局：

1. **Data and Observation**：研究工作簿、原始视频、内部 C1–C6 资产；强调只有 16 帧目标采样和 ASR/字幕越过公开观测边界；
2. **Evidence-First Annotation**：Evidence Extractor；并行的 Local BP、Consumer、Operator、Strategist；随后 Challenger、Adjudicator、本地验证和 Human Review Queue；
3. **Benchmark and Evaluation**：GroundedAnnotation、Deterministic QA Compiler、Public QA/Private Gold、Tested VLM、Evidence-aware Judge 和宏平均指标。

图中使用红色虚线隔离 `title / metadata / creator / engagement`，标注其不能进入 Evidence、公开 QA、被测模型或 Judge。所有文字使用简短、可辨认的英文标签，以降低生成式图像中的中文排版错误；中文图注在正文中解释各模块。

ImageGen 提示词会要求：ACL/EMNLP 论文风格、白色背景、扁平矢量式信息图、清晰箭头、蓝灰主色、橙色质量门、红色隐私防火墙、16:9 横向、无装饰性人物、无水印、严格使用给定标签。生成后必须人工视觉检查文本、箭头方向和模块遗漏；如文本错误，正文中的 Mermaid 图作为权威逻辑版本。

## 5. 验收标准

完整交付前必须满足：

- HTML 工作台包含用户要求的五个核心部分，并补足 QA 编译、质量控制和复现协议；
- 工作台默认首先展示当前项目算法，而不是直接展示润色后的论文成稿；
- 每个模块具有稳定编号、算法表述、代码依据、实现边界和反馈入口；
- 会话内反馈能携带模块编号和版本信息发回 Codex，独立 HTML 能导出结构化反馈；
- 算法视图与论文视图可以切换，并共享同一份已确认的方法事实；
- 所有数量、阈值、角色、子任务、评分公式和数据边界均可回指当前代码或现有产物；
- 明确区分 `EvidenceUnit`、proposal、`GroundedAnnotation` 和最终 QA；
- 不把 C1–C6 全部写成公开模型输入；
- 不把智能体写成直接自由生成 QA；
- 不把互动指标写成答案、模型输入、Judge 输入或排行榜指标；
- 不声称 creator split 已接入当前主流程；
- 不声称 Judge 调用失败会自动计 0；
- 包含至少一套数学形式化、一段端到端算法伪代码、一张任务本体表和一张总体框架图；
- 在 736px 和 360px 宽度验证布局、键盘操作、视图切换和反馈发送/导出行为；
- 对内容执行关键词和结构检查，并运行项目现有测试以确认交付工作未破坏代码；
- 对生成 PNG 进行视觉检查，并在最终答复中给出独立 HTML 和图片的绝对路径。
