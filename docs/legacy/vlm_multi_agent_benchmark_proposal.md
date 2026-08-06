# VLM 与 Multi-Agent 营销视频效果预测方案

> 硕士毕业设计：基于 VLM 和 Multi-Agent 的抖音营销短视频效果预测研究
> 在已有 LightGBM baseline 基础上，新增 VLM 和 Multi-Agent 两条预测线路

---

## 0. 项目定位

### 0.1 研究问题

**核心问题**：能否利用视觉语言模型（VLM）和多智能体（Multi-Agent）方法，通过分析营销视频的多模态内容（画面、文案、商品信息、账号上下文等），预测视频的营销效果（点赞、评论、转发、收藏）？

### 0.2 研究路径

```
Phase 1（已完成）: 建立 Benchmark + LightGBM 结构化 baseline
Phase 2（本方案）: 新增 VLM baseline + Multi-Agent baseline
Phase 3（后续）  : 基于 benchmark 结果改进方法，提炼经济学/营销学结论
```

### 0.3 预期贡献

1. **方法层面**：对比传统 ML / VLM / Multi-Agent 三类方法在营销视频效果预测上的能力差异
2. **发现层面**：通过 baseline 对比发现哪些因素对营销效果最重要（如 LightGBM 已发现 publish_context 贡献最大）
3. **应用层面**：为短视频营销效果预估提供可落地的技术方案参考

---

## 1. 当前项目现状

### 1.1 数据资产

| 资产 | 路径 | 规模 |
|------|------|------|
| 研究主表 | `videos/研究数据.xlsx` | 1200 条视频 |
| 原始视频 | `videos/raw_data/video/{日期}/*.mp4` | 1196 条有视频 |
| 销售截图 | `videos/raw_data/sales/{日期}/*.png` | ~1158 条 |
| 标签文件 | `outputs/processed/labels_v1.jsonl` | 1200 条（含 likes/comments/shares/collects + final_score） |
| Pairwise 对 | `outputs/processed/pairs_v1.jsonl` | 5000 对 |
| 六类结构化输入 | `input/` 下 6 个 jsonl | 各 1200 行 |

### 1.2 已有评估管线

任何模型只需产出标准格式的 `predictions.jsonl`：
```json
{"video_id": "xxx", "final_pred_score": 0.73, "confidence": 0.84}
```
然后调用 `salesbench.py evaluate --predictions xxx.jsonl` 即可得到完整评估报告。

已有评估指标：

| 指标 | 权重(agent_score) | 说明 |
|------|------------------|------|
| spearman_rho | 0.30 | 排序能力 |
| 1 - nmae | 0.20 | 分数精度 |
| auc_top30 | 0.30 | Top30 筛选能力 |
| 1 - ece | 0.15 | 置信度校准 |
| macro_f1 | 0.05 | 等级分类 |
| agent_score | — | 综合分 |

### 1.3 LightGBM Baseline 关键结论

| 指标 | 数值 |
|------|------|
| Spearman ρ | 0.4641 |
| AUC Top30 | 0.7216 |
| Agent Score | 0.6530 |
| MAE | 0.1749 |

**关键发现**（已有经济学意义的结论）：

1. `publish_context` 贡献 30.2% 的特征重要性 → **账号基础盘和商品上下文对视频表现影响最大**
2. `video_sales_power` 单字段去掉后 Spearman 从 0.4641 降到 0.4295 → **"视频带货力"这个平台指标高度预测性**
3. 纯内容特征 Spearman 只有 0.3525 → **仅靠视频内容本身很难准确预测效果，上下文很关键**
4. 预测分数严重压缩（IQR 0.16 vs 真实 0.36） → **A/E 两端极端表现难以预测**

### 1.4 当前 `final_score` 的构造与可比性问题

```
原始数据: likes=69000, comments=2326, shares=33000, collects=68000
    ↓ Step 1: log1p
likes_log = log(1+69000) = 11.14
    ↓ Step 2: robust normalization（全量样本 p5-p95 截断归一化）
likes_norm = clip((likes_log - p5) / (p95 - p5), 0, 1)
    ↓ Step 3: 加权求和
final_score = 0.30 × likes_norm + 0.20 × comments_norm + 0.25 × shares_norm + 0.25 × collects_norm
    → 结果: [0, 1] 连续值
```

**已知问题 1**：权重 0.3/0.2/0.25/0.25 是人工设定，缺乏理论依据。

**已知问题 2（可比性问题，核心）**：

`final_score` 的本质是"真实互动量在全量样本中的分位数位置"。而 VLM/Agent 输出的分数是"LLM 对视频效果的主观评估"。两边的 0-1 分数**语义完全不同**：

| 维度 | LightGBM 的 pred_score | VLM/Agent 的 pred_score |
|------|----------------------|------------------------|
| 本质 | 拟合真实互动的归一化值 | LLM 主观打分 |
| 0.8 意味着 | 互动量处于全量 p5-p95 的 80% 位置 | "我觉得效果很好" |
| 锚定 | 锚定在 1200 条视频的分布上 | 锚定在 LLM 内部标准上 |

**这意味着**：
- ✅ **排序类指标可以直接对比**：Spearman ρ、AUC Top30、Pairwise Accuracy——它们只关心排序，不关心绝对值
- ⚠️ **绝对值类指标不可直接对比**：MAE、nMAE——VLM 的 0.8 和 ground truth 的 0.8 不是同一回事
- ⚠️ **阈值类指标需要校准后对比**：Macro F1、ECE——依赖分数的绝对切分

本方案通过以下方式解决这个问题（详见 §2.3 评估方案）。

---

## 2. 方案设计

### 2.1 整体架构

```
                        ┌─────────────────────┐
                        │   SalesBench V1      │
                        │   1200 条视频样本     │
                        │   统一评估管线        │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
     │   LightGBM      │ │   VLM           │ │  Multi-Agent    │
     │   (已完成)       │ │   Baseline      │ │  Baseline       │
     │                  │ │                  │ │                  │
     │ 128 结构化特征   │ │ 视频帧+文案     │ │ 4 Agent 协作    │
     │ → 回归预测       │ │ → VLM 直接评分  │ │ → 分工+验证     │
     └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  predictions.jsonl   │
                        │  统一评估 + 对比分析  │
                        └─────────────────────┘
```

### 2.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **评估管线不动** | 三类模型都产出 `predictions.jsonl`，复用已有 `evaluate` |
| **同一 1200 条** | 不换数据集，保证可比 |
| **先跑通再优化** | 先用最简方案跑出结果，后续再迭代改进 |
| **Track 对齐** | Content-only 和 Full-context 两个赛道分别对比 |
| **增量扩展** | 新增文件独立，不修改已有代码 |

### 2.3 评估方案：解决可比性问题

#### 核心思路：排序为主，校准为辅

由于 `final_score`（真实互动归一化）与 VLM/Agent 输出（主观评分）的绝对值语义不同，评估方案分两层：

**第一层：排序类指标（直接可比，核心指标）**

| 指标 | 为什么可比 |
|------|----------|
| **Spearman ρ** | 只看排序是否一致，不看绝对值 |
| **AUC Top30 / Top10** | 只看"高分视频是否排在前面" |
| **Pairwise Accuracy** | 只看"给定两个视频，谁排前面" |

这些指标对三类方法完全公平——不管你的分数是 0-1 还是 0-100，排序对了就行。

**第二层：校准后的绝对值指标（辅助参考）**

对 VLM/Agent 的原始输出分数做后处理校准后，才计算 MAE / Macro F1 / ECE：

```python
def calibrate_predictions(pred_scores, true_scores):
    """
    方案 A：分位数映射（Quantile Mapping）
      将 VLM 输出的分数按排名映射到 true_scores 的同排名分位数
      效果：保持排序不变，但绝对值对齐到 ground truth 的分布
    
    方案 B：Isotonic Regression
      学习一个单调映射 f，使得 f(pred) ≈ true
      效果：保序的同时最小化 MSE
    """
```

校准后再算 MAE / F1 / ECE，就能公平对比。**但报告中需要明确标注"校准前"和"校准后"的区别。**

**第三层：agent_score 综合分的计算**

当前 `agent_score` 公式中包含 nmae 和 ece 这两个对绝对值敏感的指标。解决方案：

```
agent_score_rank = 0.40 × Spearman_normalized   （排序能力，权重提升）
                + 0.30 × AUC_Top30              （筛选能力）
                + 0.15 × Pairwise_Accuracy_norm  （排序对准确率）
                + 0.15 × (1 - ECE_calibrated)    （校准后的置信度）
```

这个版本的综合分以排序指标为主，对三类方法公平。原始 `agent_score` 保留用于向后兼容。

#### 实际操作

评估报告中同时输出两组结果：

```json
{
    "metrics_raw": {
        "spearman_rho": 0.48,
        "auc_top30": 0.75,
        "pairwise_accuracy": 0.68,
        "mae_raw": 0.22,
        "note": "MAE 基于原始分数，VLM/LightGBM 分数语义不同，不可直接对比"
    },
    "metrics_calibrated": {
        "mae_calibrated": 0.15,
        "macro_f1_calibrated": 0.35,
        "ece_calibrated": 0.12,
        "note": "校准后指标，通过 quantile mapping 对齐分数分布"
    },
    "agent_score_rank": 0.72,
    "agent_score_v1": 0.65
}
```

### 2.4 新增评估维度：4 个子维度分数

在现有 `final_score`（综合分）基础上，新增 4 个子维度预测目标：

| 子维度 | 对应互动指标 | 含义 | 构造方式 |
|--------|------------|------|---------|
| **engagement_score** | likes | 吸引力/点赞潜力 | `likes_norm`（已有） |
| **discussion_score** | comments | 讨论性/评论潜力 | `comments_norm`（已有） |
| **virality_score** | shares | 传播性/转发潜力 | `shares_norm`（已有） |
| **conversion_score** | collects | 转化力/收藏潜力 | `collects_norm`（已有） |

这 4 个子维度分数已经在 `videos_v1.jsonl` 中存在（`likes_norm` 等字段），只需在标签文件中新增输出。

**子维度评估同样采用 Spearman 为核心**——VLM 预测的 `engagement_score` 与真实的 `likes_norm` 之间算 Spearman，衡量的是"VLM 对点赞量高低的排序判断是否准确"，完全不受绝对值语义差异影响。

**与 V1 final_score 的关系**：
- `final_score` = 加权综合（保持不变，向后兼容）
- 4 个子维度 = 独立评估各互动维度的排序预测能力
- 对比分析时可以看：VLM 在哪个维度排序最准？Agent 在哪个维度改善最大？

这样可以回答更细粒度的研究问题，例如：
- "VLM 能更好地预测哪种互动？是点赞还是转发？"
- "Multi-Agent 的 Verifier 对哪个维度的校准效果最好？"
- "不同品类下各子维度的可预测性差异有多大？"

---

## 3. VLM Baseline

### 3.1 目标

让 VLM 直接"看视频帧 + 读原始文案 + 可选上下文"，输出效果预测分数，验证"看懂视频内容"是否能超越"只用结构化数字"。

### 3.2 候选模型

| 模型 | Provider | 成本/条 | 说明 |
|------|----------|---------|------|
| **GPT-4o** | OpenAI | ~$0.02 | 当前最强多模态，优先跑 |
| **GPT-4o-mini** | OpenAI | ~$0.002 | 开发调试用，性价比高 |
| **Claude 3.5 Sonnet** | Anthropic | ~$0.02 | 推理能力强 |
| **Qwen2.5-VL-72B** | 阿里 DashScope | ~$0.01 | 开源最强，中文好 |

**建议首批只跑 GPT-4o**，验证方案可行后再扩展其他模型。

### 3.3 输入构造

#### 视频抽帧

```python
# src/salesbench/vlm/frame_sampler.py

def sample_frames(video_path: str, total_frames: int = 8) -> list[str]:
    """
    从 mp4 中用 ffmpeg 采样关键帧，返回 base64 编码的图片列表。
    
    默认策略 'hook_plus_uniform'：
    - 首 3 秒取 3 帧（测 hook 质量）
    - 后续均匀取 5 帧（覆盖全视频）
    
    帧缓存到 outputs/cache/frames/{video_id}/ 避免重复抽帧。
    """
```

实现方式：调用系统 `ffmpeg` 命令，不引入 opencv。

#### 文本输入

直接从 `raw_video/video_index.jsonl`（或 `videos_v1.jsonl`）读取：
- `title`：视频标题
- `video_text`：口播/字幕文本
- `product_title`：商品标题

#### 上下文输入（Pre-publish Track）

从 `publish_context/` 读取，拼成文本描述：
```
达人类型：美食 | 粉丝规模：391万（头部）| 带货口碑：4.8 | 商品品类：食品 | ...
```

### 3.4 Prompt 设计

先做一种最直接的 prompt，跑通后再做消融：

**主 Prompt（Structured Scoring）**：

```
你是一位抖音营销视频效果评估专家。

请分析这条营销短视频，从多个维度预测其营销效果。
评分范围 0-1，0 代表效果极差，1 代表爆款级别。

【视频画面】
（附上 8 帧关键帧图片）

【视频标题】
{title}

【口播/字幕文本】
{video_text}

【商品信息】
{product_title}

{可选：上下文信息}

请评估以下维度并给出分数：

1. engagement_score（吸引力/点赞潜力）：画面和内容是否有吸引力？能否让用户停留并点赞？
2. discussion_score（讨论性/评论潜力）：是否能引发用户评论、提问或互动？
3. virality_score（传播性/转发潜力）：用户是否有意愿分享给他人？
4. conversion_score（转化力/收藏潜力）：用户是否会产生购买或收藏意愿？
5. overall_score（综合效果）：整体营销效果如何？

请以 JSON 格式输出：
{
    "engagement_score": 0.xx,
    "discussion_score": 0.xx,
    "virality_score": 0.xx,
    "conversion_score": 0.xx,
    "overall_score": 0.xx,
    "reasoning": "简要说明评分理由",
    "confidence": 0.xx
}
```

### 3.5 Response 解析与分数处理

```python
# src/salesbench/vlm/response_parser.py

def parse_vlm_response(raw_response: str) -> dict:
    """
    从 VLM 回复中提取分数。
    
    解析优先级：
    1. 尝试 JSON 解析（正常情况）
    2. JSON 失败 → 正则提取 key: value 模式
    3. 完全失败 → 返回 None（标记为 unparseable）
    
    所有分数 clip 到 [0, 1]。
    overall_score 映射为 final_pred_score。
    """
```

**关于 VLM 输出的分数与 ground truth 的关系**：

VLM 输出的 `overall_score` 是主观评分（"我觉得效果有多好"），而 `final_score` 是真实互动量的归一化值（"互动量在所有视频中处于什么位置"）。两者的绝对值语义不同，但我们的核心评估指标 Spearman 只关心排序——即 **VLM 认为好的视频，实际互动量是否也高**。

这个设定在方法论上是合理的：我们要求 VLM 做的是"判断哪些视频效果更好"（排序任务），而不是"精确预测会有多少点赞"（绝对值回归任务）。

### 3.6 输出与评估

VLM 预测结果写入标准格式：

```json
// outputs/baselines/vlm/gpt4o_content_only/predictions.jsonl
{"video_id": "7354336973860900159", "final_pred_score": 0.82, "confidence": 0.75}
```

直接调用已有评估管线：
```bash
python3 salesbench.py evaluate \
    --predictions outputs/baselines/vlm/gpt4o_content_only/predictions.jsonl
```

额外输出 `raw_responses.jsonl`（完整 VLM 回复，含子维度分数和 reasoning，用于后续分析）和 `run_meta.json`（耗时、费用等运行时信息）。

### 3.7 Track 对齐

| Track | VLM 输入 | 对标 LightGBM |
|-------|---------|--------------|
| **Content-only** | 视频帧 + title + video_text + product_title | Content-only LightGBM（Spearman 0.3525） |
| **Full-context** | 上述 + 达人/粉丝/品类等上下文文本 | Full LightGBM（Spearman 0.4641） |

---

## 4. Multi-Agent Baseline

### 4.1 目标

验证多个专业化 Agent 分工协作是否能超越单 VLM 直接回答。核心假设：
- 单 VLM 一次性处理太多信息容易遗漏细节
- 分工后每个 Agent 专注一个子任务，输出更精确
- Verifier Agent 回看原始证据，减少幻觉和分数漂移

### 4.2 架构：4 Agent 流水线

```
视频帧 ──→ ① Perception Agent ──→ 结构化视觉描述
              （VLM，只看不评）          │
                                        ▼
文本输入 ──→ ② Content Analyst ──→ 营销策略分析
              （LLM，分析文案+感知输出）  │
                                        ▼
             ③ Scoring Agent ───→ 多维度评分 + 综合分
              （LLM，基于前两步推理）     │
                                        ▼
视频帧 ──→ ④ Verifier Agent ───→ 校准后最终分数
              （VLM，回看帧+核查前序）
```

### 4.3 各 Agent 定义

#### Agent 1：Perception Agent（视觉感知）

**职责**：只负责"看"，不负责"评"。输出客观的视觉描述。

**输入**：8 帧视频关键帧

**Prompt 核心**：
```
你是一位视频内容分析师。请只描述你在画面中观察到的客观事实，不要做效果评价。

请输出 JSON：
{
    "scene_description": "画面整体描述",
    "presenter": {
        "appears": true/false,
        "expression": "微笑/严肃/自然...",
        "eye_contact": true/false
    },
    "product": {
        "visible": true/false,
        "appears_in_first_3s": true/false,
        "demonstration": "有演示/仅展示/未出现"
    },
    "visual_quality": {
        "clarity": "清晰/模糊",
        "lighting": "明亮/偏暗",
        "text_overlay": "有字幕/有价格/无"
    },
    "hook_first_3s": "首3秒画面描述"
}
```

**模型**：VLM（需要看图）

#### Agent 2：Content Analyst Agent（内容分析）

**职责**：分析文案策略，结合视觉感知输出做内容评价。

**输入**：title + video_text + product_title + Perception 输出 + (可选)上下文

**Prompt 核心**：
```
你是一位营销内容策略分析师。基于视觉观察报告和文案内容，分析营销策略。

【视觉观察】{perception_output}
【标题】{title}
【口播文本】{video_text}
【商品】{product_title}

请输出 JSON：
{
    "selling_points": ["卖点1", "卖点2"],
    "cta_present": true/false,
    "marketing_mechanisms": {
        "social_proof": true/false,
        "urgency": true/false,
        "benefit_highlight": true/false,
        "risk_reduction": true/false,
        "authority": true/false
    },
    "text_visual_consistency": "一致/不一致/部分一致",
    "strategy_summary": "一句话总结策略"
}
```

**模型**：LLM 即可（不需要看图，已有感知输出）

#### Agent 3：Scoring Agent（评分推理）

**职责**：基于前两步的分析输出，做多维度评分。

**输入**：Perception 输出 + Content Analyst 输出 + (可选)上下文

**Prompt 核心**：
```
你是一位营销视频效果评分专家。基于以下分析，预测视频的营销效果。

评分标准：
- 0.8-1.0（爆款）：各维度都优秀
- 0.6-0.8（优秀）：整体执行良好
- 0.4-0.6（中等）：有亮点也有不足
- 0.2-0.4（较弱）：缺乏吸引力
- 0.0-0.2（极弱）：质量差

【视觉观察】{perception_output}
【内容分析】{content_analysis_output}

请输出 JSON：
{
    "engagement_score": 0.xx,
    "discussion_score": 0.xx,
    "virality_score": 0.xx,
    "conversion_score": 0.xx,
    "overall_score": 0.xx,
    "reasoning": "评分理由",
    "confidence": 0.xx
}
```

**模型**：LLM 即可

#### Agent 4：Verifier Agent（验证校准）

**职责**：回看原始帧，核查前序 Agent 的描述是否准确，校准分数。

**输入**：原始 8 帧 + 全部前序 Agent 输出

**Prompt 核心**：
```
你是一位独立审核专家。请重新看视频帧，核查分析报告并校准分数。

特别注意：
- 感知描述是否与画面一致？
- 分析结论是否有依据？
- 分数是否合理？

【视频帧】（8帧图片）
【感知报告】{perception_output}
【内容分析】{content_analysis_output}
【评分结果】{scoring_output}

请输出 JSON：
{
    "perception_accurate": true/false,
    "corrections": ["纠正1", ...],
    "score_adjustment": 0.xx,
    "final_scores": {
        "engagement_score": 0.xx,
        "discussion_score": 0.xx,
        "virality_score": 0.xx,
        "conversion_score": 0.xx,
        "overall_score": 0.xx
    },
    "confidence": 0.xx
}
```

**模型**：VLM（需回看原始帧）

### 4.4 Orchestrator（编排器）

```python
# src/salesbench/agents/orchestrator.py

class Orchestrator:
    """Agent 调用编排器"""

    def run(self, video_id, frames, text_inputs, context=None, mode="full"):
        """
        mode:
        - "full":      Perception → Content → Scoring → Verifier（完整流水线）
        - "no_verify": Perception → Content → Scoring（无验证）
        - "simple":    Perception → Scoring（跳过内容分析）
        """
        trace = AgentTrace(video_id=video_id)
        
        # Step 1: Perception（必须，VLM 看帧）
        perception_out = self.perception_agent.run(frames)
        trace.add("perception", perception_out)
        
        # Step 2: Content Analysis（full 和 no_verify 模式）
        if mode in ("full", "no_verify"):
            content_out = self.content_agent.run(text_inputs, perception_out, context)
            trace.add("content_analyst", content_out)
        else:
            content_out = None
        
        # Step 3: Scoring（必须）
        scoring_out = self.scoring_agent.run(perception_out, content_out, context)
        trace.add("scoring", scoring_out)
        
        # Step 4: Verification（仅 full 模式，VLM 回看帧）
        if mode == "full":
            verify_out = self.verifier_agent.run(frames, trace.all_outputs())
            trace.add("verifier", verify_out)
            final_score = verify_out["final_scores"]["overall_score"]
            confidence = verify_out["confidence"]
        else:
            final_score = scoring_out["overall_score"]
            confidence = scoring_out["confidence"]
        
        trace.set_final(final_score, confidence)
        return trace
```

### 4.5 消融变体

| 变体 | Agent 链 | API 调用数/条 | 目的 |
|------|---------|-------------|------|
| **MA-Full** | Perception→Content→Scoring→Verifier | 4 | 完整流水线（主方案） |
| **MA-NoVerify** | Perception→Content→Scoring | 3 | 量化 Verifier 价值 |
| **MA-Simple** | Perception→Scoring | 2 | 量化 Content Analyst 价值 |

后续可选扩展：MA-Debate（3 个独立 VLM 投票）

### 4.6 输出

```
outputs/baselines/multi_agent/
├── ma_full_gpt4o_content_only/
│   ├── predictions.jsonl          # 标准格式，喂 evaluate
│   ├── evaluation.json            # 评估结果
│   ├── agent_traces.jsonl         # 完整 Agent 调用链（每条视频的全部 Agent 输出）
│   └── run_meta.json              # 运行元信息
├── ma_no_verify_gpt4o_content_only/
│   └── ...
└── ma_simple_gpt4o_content_only/
    └── ...
```

---

## 5. 新增代码结构

```
src/salesbench/
├── (已有模块全部不动)
│
├── vlm/                          # 🆕 VLM 基础设施
│   ├── __init__.py
│   ├── api_client.py             # 统一的 VLM/LLM API 封装
│   ├── frame_sampler.py          # ffmpeg 抽帧
│   ├── prompt_builder.py         # Prompt 模板
│   ├── response_parser.py        # 回复解析
│   └── cache.py                  # 响应缓存
│
├── agents/                       # 🆕 Agent 模块
│   ├── __init__.py
│   ├── base_agent.py             # Agent 基类 + AgentTrace
│   ├── perception_agent.py
│   ├── content_analyst_agent.py
│   ├── scoring_agent.py
│   ├── verifier_agent.py
│   └── orchestrator.py
│
├── baseline_vlm.py               # 🆕 VLM baseline runner
├── baseline_multi_agent.py       # 🆕 Multi-Agent baseline runner
└── label_utils.py                # 🆕 子维度标签工具

configs/
├── benchmark_v1.json             # (已有)
├── vlm_baseline.json             # 🆕
└── multi_agent_baseline.json     # 🆕
```

CLI 新增两个命令（在 `cli.py` 中扩展）：
```bash
python3 salesbench.py baseline-vlm [--config configs/vlm_baseline.json]
python3 salesbench.py baseline-multi-agent [--config configs/multi_agent_baseline.json]
```

---

## 6. 子维度评估方案

### 6.1 标签扩展

在现有 `labels_v1.jsonl` 基础上，生成补充标签文件 `labels_v1_dimensions.jsonl`：

```json
{
    "video_id": "7354336973860900159",
    "final_score": 1.0,
    "engagement_score": 1.0,
    "discussion_score": 1.0,
    "virality_score": 1.0,
    "conversion_score": 1.0
}
```

这些值直接来自 `videos_v1.jsonl` 中已有的 `likes_norm` / `comments_norm` / `shares_norm` / `collects_norm`，只是重新命名并独立输出。

### 6.2 子维度评估

VLM 和 Agent 的 `raw_responses.jsonl` / `agent_traces.jsonl` 中已包含各子维度分数。

**核心指标统一用 Spearman**（排序指标），避免绝对值不可比的问题：

```python
# 子维度评估：分别计算每个子维度的 Spearman
for dim in ["engagement", "discussion", "virality", "conversion"]:
    # VLM 的 engagement_score vs 真实的 likes_norm → Spearman
    # 衡量的是：VLM 认为点赞潜力高的视频，实际点赞量是否也高？
    spearman = compute_spearman(true_dim_scores, pred_dim_scores)
    print(f"{dim}: Spearman = {spearman:.4f}")
```

### 6.3 对比分析维度

| 分析问题 | 方法 | 备注 |
|---------|------|------|
| VLM 在哪个子维度排序最准？ | 4 维 Spearman 对比 | 纯排序指标，公平可比 |
| Agent vs VLM 在哪个子维度改善最大？ | Spearman 差值分析 | 纯排序指标，公平可比 |
| 不同品类下各子维度的可预测性 | 按 product_bucket 分组 Spearman | 纯排序指标，公平可比 |
| final_score 与子维度的一致性 | 相关性矩阵 | 分析 V1 权重合理性 |
| VLM 分数分布 vs 真实分布 | 直方图叠加对比 | 看 VLM 是否也有分数压缩 |
| 校准前后指标差异 | MAE_raw vs MAE_calibrated | 量化校准的必要性 |

---

## 7. 公平对比：Track 设计

### 7.1 两个 Track

| Track | LightGBM 输入 | VLM / Agent 输入 |
|-------|--------------|-----------------|
| **Content-only** | visual + audio + text + cross_modal（110 特征） | 视频帧 + title + video_text + product_title |
| **Full-context** | 全量 128 特征 | 上述 + 达人/粉丝/品类等上下文描述 |

### 7.2 对比矩阵（核心指标：Spearman ρ，排序指标，三方公平可比）

| Track | LightGBM | VLM (GPT-4o) | MA-Full | MA-NoVerify | MA-Simple |
|-------|----------|-------------|---------|-------------|-----------|
| Content-only | 0.3525* | ? | ? | ? | ? |
| Full-context | 0.4641 | ? | ? | ? | ? |

*LightGBM ablation 结果：去掉全部 publish_context 后的 Spearman

**为什么 Spearman 是核心对比指标**：
- LightGBM 拟合的是真实互动归一化分数，VLM/Agent 输出的是主观评分
- 两者的绝对值语义不同（详见 §1.4），但排序语义一致
- Spearman 只比较排序，对三类方法完全公平
- MAE 等绝对值指标需要校准后才能辅助对比

---

## 8. 实施路线图

### Phase 1：基础设施（3-4 天）

| 任务 | 产出 |
|------|------|
| 实现 `frame_sampler.py`，全量抽帧并缓存 | 1200 条视频 × 8 帧 |
| 实现 `api_client.py`，支持 OpenAI API | 能调通 GPT-4o |
| 实现 `response_parser.py` | 能解析 JSON 回复 |
| 实现 `cache.py` | 响应缓存可用 |
| 生成 `labels_v1_dimensions.jsonl` | 子维度标签就绪 |

### Phase 2：VLM Baseline 首跑（3-4 天）

| 任务 | 产出 |
|------|------|
| 实现 `prompt_builder.py` + `baseline_vlm.py` | VLM runner 就绪 |
| GPT-4o-mini dry run（5 条） | 确认端到端可行 |
| GPT-4o Content-only 全量跑 | 首批 predictions + evaluation |
| GPT-4o Full-context 全量跑 | 两个 Track 完成 |
| VLM 初步分析 | 与 LightGBM 对比 |

### Phase 3：Multi-Agent 实现 + 评测（5-7 天）

| 任务 | 产出 |
|------|------|
| 实现 4 个 Agent + Orchestrator | Agent 框架就绪 |
| 实现 `baseline_multi_agent.py` | Agent runner 就绪 |
| MA-Full dry run（5 条） | 端到端可行 |
| MA-Full Content-only 全量跑 | Agent 首批结果 |
| MA-NoVerify + MA-Simple 消融 | 消融数据完成 |
| Full-context Track 补跑 | 全 Track 数据完成 |

### Phase 4：分析与报告（3-5 天）

| 任务 | 产出 |
|------|------|
| 三方对比分析（Spearman / AUC / MAE / agent_score） | 核心对比表 |
| 子维度分析（4 维 × 3 方法 × 2 Track） | 子维度对比 |
| 按品类分组分析 | 品类差异发现 |
| 按等级分析（A/E 两端） | 极端样本诊断 |
| Agent 消融分析（Verifier 价值 / Content Analyst 价值） | 消融结论 |
| 撰写分析报告 | `docs/vlm_baseline_analysis_v1.md` + `docs/multi_agent_baseline_analysis_v1.md` |

**总计约 2-3 周可完成首轮全部实验。**

---

## 9. 预期结果与研究价值

### 9.1 预期数据结果

| 方法 | Content-only Spearman | Full-context Spearman |
|------|----------------------|----------------------|
| LightGBM | 0.35 | 0.46 |
| VLM (GPT-4o) | **0.38-0.48** | **0.48-0.58** |
| MA-Full | **0.40-0.52** | **0.50-0.62** |

### 9.2 可以得出的研究结论

#### 方法对比层面

| 预期结论 | 支撑数据 |
|---------|---------|
| VLM 在 Content-only 上显著优于 LightGBM | 因为 VLM 能看视频、理解文案 |
| Full-context 差距缩小 | LightGBM 的上下文特征很强 |
| Multi-Agent > VLM 的增益来自 Verifier | 消融实验对比 |
| VLM/Agent 的分数分布更接近真实 | 解决 LightGBM 的分数压缩问题 |

#### 经济学/营销学层面

| 预期结论 | 支撑数据 |
|---------|---------|
| 账号基础盘比内容本身更能预测效果 | Content-only vs Full 的差距 |
| 不同品类的关键成功因素不同 | 按品类分组的子维度分析 |
| VLM 对"吸引力"维度预测最准，对"讨论性"最难 | 子维度 Spearman 对比 |
| "看懂视频"能额外解释 X% 的效果方差 | VLM vs LightGBM 的 Spearman 差 |
| 多步推理有助于更准确的效果预估 | MA-Full vs VLM 的差异 |

### 9.3 后续改进方向（Phase 3 之后）

| 方向 | 说明 |
|------|------|
| **VLM fine-tuning** | 用 1200 条视频微调 VLM，看是否超越 zero-shot |
| **混合模型** | LightGBM 上下文分数 + VLM 内容分数 → ensemble |
| **Agent prompt 优化** | 根据错误分析优化各 Agent 的 prompt |
| **扩展更多 VLM** | Claude / Qwen / Gemini 横向对比 |
| **分数校准** | 对 VLM/Agent 输出做 isotonic calibration |
| **特征重要性对比** | VLM 的 reasoning 与 LightGBM 的 feature importance 做交叉分析 |

---

## 10. 依赖与预算

### 10.1 新增依赖

```toml
# pyproject.toml 新增
[project.optional-dependencies]
vlm = [
    "openai>=1.40",
    "Pillow>=10.0",
]
```

系统依赖：`ffmpeg`（`brew install ffmpeg`）

后续需要时再加 `anthropic` / `dashscope`。

### 10.2 费用预估（首轮最小实验）

| 实验 | 调用数 | 费用 |
|------|-------|------|
| VLM GPT-4o × 2 Track × 1200 条 | 2,400 | ~$48 |
| MA-Full × 2 Track × 1200 条 × 4 Agent（其中 2 个 VLM + 2 个 LLM） | 9,600 | ~$100 |
| MA 消融 × 2 变体 × 1 Track × 1200 条 | ~4,800 | ~$50 |
| 调试（mini 模型 + dry run） | ~200 | ~$2 |
| **合计** | ~17,000 | **~$200** |

**成本控制**：
- 开发调试全用 GPT-4o-mini（便宜 10 倍）
- 响应缓存，避免重复调用
- 先跑 Content-only Track，确认可行再跑 Full-context
- Agent 的 Content Analyst 和 Scoring Agent 用 LLM（不用 VLM，省图片费）

---

## 11. 风险与缓解

| 风险 | 缓解 |
|------|------|
| VLM 分数也出现压缩 | 用 Comparative Anchoring prompt + 后处理校准 |
| API 调用失败/超时 | 重试 3 次 + 响应缓存 + 断点续跑 |
| Response 解析失败 | JSON mode + 正则 fallback + 标记 unparseable |
| 抽帧质量差（黑屏） | 跳过纯黑帧 + 备选帧 |
| Agent 链路错误传播 | Verifier 回看原图纠错 + 消融实验定位问题 |
| 费用超预算 | 先小样本验证，再按优先级全量跑 |

---

## 12. 总结

本方案的核心思路是**简单可执行，逐步迭代**：

1. **先跑通**：VLM 一条 prompt 跑 1200 条 → 拿到 predictions → 和 LightGBM 对比
2. **再加 Agent**：4 Agent 流水线跑通 → 消融实验 → 分析各 Agent 贡献
3. **然后分析**：三方对比 + 子维度分析 + 品类分析 → 提炼研究结论
4. **后续改进**：根据分析结果做 prompt 优化 / ensemble / fine-tuning

整个过程**评估管线不变**，新增代码完全独立，随时可以回退或扩展。