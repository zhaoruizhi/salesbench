# Multi-Agent 营销视频效果预测框架

## 1. 架构概述

```
原始视频帧 ──→ ① Perception Agent ──→ 结构化视觉描述
                 （VLM，只看不评）         │
                                          ▼
文本 + 上下文 ──→ ② Content Analyst ──→ 营销策略分析报告
                  （LLM，分析文案策略）     │
                                          ▼
                 ③ Scoring Agent ───→ 多维度评分 + 综合分
                  （LLM，基于前两步推理）    │
                                          ▼
原始视频帧 ──→ ④ Verifier Agent ───→ 校准后最终分数
                  （VLM，回看帧+核查前序）
```

## 2. 各 Agent 详细规格

### Agent 1: Perception Agent（视觉感知代理）

| 属性 | 规格 |
|------|------|
| **模型类型** | VLM（需要处理图像） |
| **输入** | 8 帧视频关键帧（base64 JPEG） |
| **输出** | 结构化视觉描述 JSON |
| **原则** | 只描述，不评价 |
| **代码** | `src/salesbench/agents/perception_agent.py` |

**输出字段**：
- `scene_description`: 画面整体内容概述
- `presenter`: 出镜人物特征（是否出镜、表情、眼神接触、手势）
- `product`: 产品展示特征（是否可见、首 3 秒是否出现、是否有演示、是否有特写）
- `visual_quality`: 视觉质量（清晰度、光线、构图、文字覆盖）
- `hook_first_3s`: 首 3 帧画面描述
- `scene_changes`: 场景切换情况
- `notable_elements`: 其他值得注意的视觉元素

### Agent 2: Content Analyst Agent（内容策略分析代理）

| 属性 | 规格 |
|------|------|
| **模型类型** | LLM（纯文本即可） |
| **输入** | Perception 输出 + 标题 + 口播文本 + 商品标题 + (可选)上下文 |
| **输出** | 营销策略分析 JSON |
| **代码** | `src/salesbench/agents/content_analyst_agent.py` |

**输出字段**：
- `selling_points`: 核心卖点列表
- `selling_point_clarity`: 卖点清晰度
- `hook_analysis`: Hook 有效性分析
- `cta_present` / `cta_type`: 行动号召识别
- `marketing_mechanisms`: 6 种营销机制识别（社会证明、紧迫感、利益凸显、风险消除、权威背书、价格锚定）
- `text_visual_consistency`: 文本-视觉一致性判断
- `content_type`: 内容类型（教程/测评/种草/剧情等）
- `emotional_trigger`: 情绪触发类型
- `strategy_summary`: 一句话策略总结

### Agent 3: Scoring Agent（评分推理代理）

| 属性 | 规格 |
|------|------|
| **模型类型** | LLM（纯文本即可） |
| **输入** | Perception 输出 + Content Analyst 输出 + (可选)上下文 |
| **输出** | 5 维评分 + 理由 JSON |
| **代码** | `src/salesbench/agents/scoring_agent.py` |

**输出字段**：
- `engagement_score`: 点赞潜力 (0-1)
- `discussion_score`: 评论潜力 (0-1)
- `virality_score`: 转发潜力 (0-1)
- `conversion_score`: 收藏/购买潜力 (0-1)
- `overall_score`: 综合效果 (0-1)
- `key_strengths` / `key_weaknesses`: 优势与不足
- `scoring_reasoning`: 评分依据（需引用前序 Agent 的发现）
- `confidence`: 置信度

### Agent 4: Verifier Agent（校验代理）

| 属性 | 规格 |
|------|------|
| **模型类型** | VLM（需要回看原始帧） |
| **输入** | 原始 8 帧 + 全部前序 Agent 输出 |
| **输出** | 校验结论 + 校准后分数 JSON |
| **代码** | `src/salesbench/agents/verifier_agent.py` |

**输出字段**：
- `perception_accuracy`: 感知描述准确度
- `perception_corrections`: 纠正列表
- `analysis_grounded`: 分析是否有证据支持
- `analysis_issues`: 分析问题列表
- `score_assessment`: 分数评价（合理/偏高/偏低）
- `score_adjustment_reason`: 调整原因
- `adjusted_scores`: 校准后的 5 维分数
- `adjusted_confidence`: 校准后置信度
- `verification_summary`: 校验结论总结

## 3. 编排器（Orchestrator）

**代码**: `src/salesbench/agents/orchestrator.py`

### 运行模式

| 模式 | Agent 链 | API 调用数/条 | 用途 |
|------|---------|-------------|------|
| `full` | Perception → Content → Scoring → Verifier | 4 | 完整流水线（主方案） |
| `no_verify` | Perception → Content → Scoring | 3 | 消融：量化 Verifier 价值 |
| `simple` | Perception → Scoring | 2 | 消融：量化 Content Analyst 价值 |

### 分数确定逻辑

- **full 模式**: 使用 Verifier 的 `adjusted_scores`；如 Verifier 失败则 fallback 到 Scoring 的分数
- **no_verify / simple 模式**: 直接使用 Scoring Agent 的分数

## 4. API 模型需求

### 最小配置（当前实现）

| Agent | 模型需求 | 当前使用 |
|-------|---------|---------|
| Perception | VLM（需要看图） | gpt-4o |
| Content Analyst | LLM（纯文本） | gpt-4o |
| Scoring | LLM（纯文本） | gpt-4o |
| Verifier | VLM（需要看图） | gpt-4o |

当前实现中 4 个 Agent 共用同一个 gpt-4o 模型，通过 `VLMClient` 统一调用。Content Analyst 和 Scoring Agent 虽然只需要纯文本 LLM，但为简化实现也使用 VLM 模型（只是不传图片）。

### 费用结构（基于 dry run 5 条）

| Agent | 平均 tokens | 平均费用/条 |
|-------|------------|-----------|
| Perception | ~950 in + ~300 out | ~$0.0054 |
| Content Analyst | ~1000 in + ~400 out | ~$0.0066 |
| Scoring | ~1500 in + ~200 out | ~$0.0059 |
| Verifier | ~2000 in + ~400 out | ~$0.0090 |
| **Full 合计** | ~5450 in + ~1300 out | **~$0.0269/条** |

### 全量预估（1200 条）

| 配置 | API 调用数 | 费用预估 | 时间预估(5 workers) |
|------|----------|---------|-------------------|
| MA-Full × content_only | 4800 | ~$32 | ~60 min |
| MA-Full × full_context | 4800 | ~$34 | ~60 min |
| MA-NoVerify × content_only | 3600 | ~$22 | ~45 min |
| MA-Simple × content_only | 2400 | ~$14 | ~30 min |
| **全部消融实验** | ~15600 | **~$102** | ~3.5 h |

### 可选优化：使用更便宜的 LLM

Content Analyst 和 Scoring Agent 只需要文本理解能力，可以替换为更便宜的模型：

| 方案 | Perception + Verifier | Content + Scoring | 费用降幅 |
|------|----------------------|------------------|---------|
| 全 gpt-4o | gpt-4o | gpt-4o | — |
| 混合 A | gpt-4o | gpt-4o-mini | ~40% |
| 混合 B | gpt-4o | gpt-4.1-mini | ~35% |

在 `Orchestrator.__init__` 中已支持传入不同的 `vlm_client` 和 `llm_client`。

## 5. CLI 使用

```bash
# Full 模式，5 条 dry run
python3 salesbench.py baseline-multi-agent \
    --api-key YOUR_KEY \
    --base-url https://yunwu.ai/v1 \
    --model gpt-4o \
    --track content_only \
    --mode full \
    --dry-run

# Full 模式，全量
python3 salesbench.py baseline-multi-agent \
    --api-key YOUR_KEY \
    --base-url https://yunwu.ai/v1 \
    --model gpt-4o \
    --track content_only \
    --mode full \
    --max-workers 5

# 消融：无 Verifier
python3 salesbench.py baseline-multi-agent \
    --api-key YOUR_KEY \
    --base-url https://yunwu.ai/v1 \
    --mode no_verify --track content_only

# 消融：简单模式
python3 salesbench.py baseline-multi-agent \
    --api-key YOUR_KEY \
    --base-url https://yunwu.ai/v1 \
    --mode simple --track content_only
```

## 6. 输出文件

```
outputs/baselines/multi_agent/
├── ma_full_gpt-4o_content_only/
│   ├── predictions.jsonl       # 标准格式预测 (video_id, final_pred_score, confidence)
│   ├── agent_traces.jsonl      # 完整 Agent 调用链 (每条视频的 4 个 Agent 输出)
│   ├── evaluation.json         # 评估结果 (Spearman, AUC, etc.)
│   └── run_meta.json           # 运行元信息 (费用, 耗时, 成功率)
├── ma_no_verify_gpt-4o_content_only/
│   └── ...
└── ma_simple_gpt-4o_content_only/
    └── ...
```

## 7. 消融实验设计

| 对比 | 变量 | 研究问题 |
|------|------|---------|
| MA-Full vs MA-NoVerify | 是否有 Verifier | Verifier 的证据回溯和分数校准贡献多大？ |
| MA-Full vs MA-Simple | 是否有 Content Analyst | 营销策略分析对评分推理有多大帮助？ |
| MA-Full vs VLM-single | 单 VLM vs 4 Agent | 多 Agent 分工是否优于单次推理？ |
| MA-Full (content) vs MA-Full (full_ctx) | 有无上下文 | 上下文对 Multi-Agent 的帮助有多大？ |

## 8. 代码文件结构

```
src/salesbench/
├── agents/
│   ├── __init__.py               # 模块入口
│   ├── base_agent.py             # AgentOutput + AgentTrace 数据结构
│   ├── perception_agent.py       # Agent 1: 视觉感知
│   ├── content_analyst_agent.py  # Agent 2: 内容策略分析
│   ├── scoring_agent.py          # Agent 3: 评分推理
│   ├── verifier_agent.py         # Agent 4: 校验代理
│   └── orchestrator.py           # 编排器（3 种模式）
├── baseline_multi_agent.py       # Multi-Agent runner (并发 + 缓存)
└── vlm/                          # 共用基础设施 (API client, frame sampler, cache)
```