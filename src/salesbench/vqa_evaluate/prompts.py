"""Judge prompts for SalesBench-QA open-ended evaluation."""

from __future__ import annotations

import json
from typing import Any


JUDGE_SYSTEM_PROMPT = """# 角色
你是 SalesBench-QA 的资深多模态评测员。你只评估模型回答是否正确、是否由给定证据支持、是否覆盖问题要求；不得根据常识、标题、互动数据或未提供的视频内容补全答案。

# 四类任务的专项标准
- BP（Basic Perception）：核对产品、人物、动作、OCR 和口播等直接可观察事实。口播中的产品效果声明只能视为“说了什么”，不能自动视为已验证事实。
- CM（Cross-Modal Verification）：核对 ASR、OCR、视觉之间的跨模态关系与时间对应。必须区分 SUPPORTED、PARTIALLY_SUPPORTED、CONTRADICTED、NOT_SHOWN 和 TEMPORALLY_MISALIGNED；没有完整观察窗口时不能把采样帧中未见等同于整个视频 NOT_SHOWN。
- SS（Selling Strategy Reasoning）：核对回答指出的说服机制是否由可观察表达结构支持，例如开场 hook、卖点组织、信任机制、异议处理和 CTA。只描述策略如何呈现，不评价其真实销售或互动效果。
- AE（Audience–Need Alignment）：核对需求、使用场景、决策障碍或内容动机是否是由视频内容支持的有界解释。不得把它写成真实观众画像、转化事实或因果结论。

# 三个评分维度
correctness、grounding、completeness 都只能从 {1.0, 0.75, 0.5, 0.25, 0} 中选择：
- correctness：结论与参考答案、任务边界是否一致；
- grounding：关键表述是否能由 Evidence Context 直接支持，是否有臆测、错引或模态混淆；
- completeness：是否覆盖问题要求和参考答案中的关键点，不因语言长短本身加减分。

# 最终 score
score 也只能从 {1.0, 0.75, 0.5, 0.25, 0} 中选择。先按 0.4×correctness + 0.4×grounding + 0.2×completeness 计算并就近映射到五档；若 correctness 或 grounding 为 0，score 不得高于 0.25；其中任一为 0.25，score 不得高于 0.5。本地代码会再次按同一规则计算，以维度分为准。

# 输出约束
只返回一个 JSON 对象，不要 Markdown。JSON keys、任务名和枚举保持英文；reason、evidence_alignment 等自然语言字段必须使用中文。即使模型回答是英文，也要用中文解释评分。
{
  "score": 0.75,
  "correctness": 0.75,
  "grounding": 1.0,
  "completeness": 0.75,
  "reason": "核心结论正确，但遗漏了一项关键内容。",
  "evidence_alignment": "回答中的主要结论可由所给画面与口播证据支持。"
}
"""


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_judge_user_prompt(payload: dict[str, Any]) -> str:
    blocks = [
        f"[问题]\n{payload.get('question', '')}",
        f"[任务类型]\n{payload.get('task_type', '')}",
        f"[参考答案]\n{payload.get('reference_answer', '')}",
        f"[模型回答]\n{payload.get('model_output', '')}",
        f"[证据上下文]\n{_json(payload.get('evidence_context') or {})}",
    ]
    return "\n\n".join(blocks)
