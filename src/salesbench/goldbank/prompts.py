"""Prompt builders for Evidence-First multi-agent annotation stages."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .validators import PRIVATE_KEYS


PROMPT_VERSION = "evidence-prompt-v7"


def _strip_private(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _strip_private(value)
            for key, value in payload.items()
            if str(key) not in PRIVATE_KEYS and str(key) != "performance_data"
        }
    if isinstance(payload, list):
        return [_strip_private(value) for value in payload]
    return payload


def _json(payload: object) -> str:
    return json.dumps(_strip_private(deepcopy(payload)), ensure_ascii=False, sort_keys=True)


def build_evidence_extractor_prompt(video_id: str, content_context: dict[str, object]) -> tuple[str, list[dict]]:
    system = (
        "你是 SalesBench-QA 的客观 Evidence Extractor。只提取能够在输入帧、画面文字或 ASR/字幕中定位的 EvidenceUnit，"
        "不要生成问题、销售策略、受众结论、互动效果或隐藏推理。只返回严格 JSON，顶层只能包含 evidence_units。"
        "每个单元只能使用一种 modality：visual 表示可见的非文字事实，ocr 表示画面内文字，asr 表示口播或字幕；"
        "不得输出 image、text 或 metadata 作为 modality。visual 必须包含 frame_indices；ocr 必须包含 frame_indices 和逐字 text_span；"
        "asr 必须包含输入中实际提供的逐字 text_span。OCR 的 text_span 只能是画面原文，禁止把坐标、边界框或类似 [0,50] 的值写入 text_span。"
        "若输入明确提供 ASR 的 start_s/end_s，必须原样使用；若没有提供则填 null，不得根据语义或帧位置猜测时间。"
        "每个单元必须包含 subject、predicate、value。自然语言字段必须使用中文，包括 subject、predicate、value 和 attributes 中的说明；"
        "OCR/ASR 的 text_span 必须保留原文，品牌名和原有英文术语可以保留。"
        "卖方对效果、性能或用途的口播声明不是已验证事实：应编码为 subject=口播者、predicate=声称、value=具体声明，"
        "并保留原始 text_span；除非画面本身直接验证，否则不得把声明写成产品已经具有该效果。"
        "confidence 必须是 0 到 1 的 JSON 数字，不能使用 high/medium/low。evidence_id 可以省略；"
        "即使输出也只能作为 frame_006、asr_subtitles 一类来源定位符，canonical ID 由本地代码生成。"
        "无法定位、含糊或需要外部知识的内容必须省略。输出示例："
        '{"evidence_units":[{"modality":"visual","start_s":null,"end_s":null,"frame_indices":[0],'
        '"text_span":"","subject":"产品包装","predicate":"颜色","value":"红色","attributes":{},"confidence":0.9}]}'
    )
    user_text = _json({"video_id": video_id, "content_context": content_context})
    return system, [{"type": "text", "text": user_text}]


def build_proposer_prompt(
    perspective: str,
    video_id: str,
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    perspective = perspective.lower()
    contracts = {
        "consumer": (
            "你只负责 AE，不得生成 BP、CM 或 SS。允许 subtype：AUDIENCE_NEED_FIT、USAGE_CONTEXT、DECISION_STATE、CONTENT_MOTIVATION。"
            "各 subtype 必须使用以下结构："
            "AUDIENCE_NEED_FIT: target={need}, proposed_gold={audience_need,answer}; "
            "USAGE_CONTEXT: target={scenario}, proposed_gold={usage_context,answer}; "
            "DECISION_STATE: target={decision_barrier}, proposed_gold={decision_state,answer}; "
            "CONTENT_MOTIVATION: target={motivation_cue}, proposed_gold={content_motivation,answer}。"
            "AE 禁止使用 claim、relation 或把推断写成真实观众画像、真实需求、转化结论。"
        ),
        "operator": (
            "你只负责 CM，不得生成 BP、AE 或 SS。允许 subtype：CLAIM_EVIDENCE_RELATION、CLAIM_PARTIAL_SUPPORT、TEXT_VISUAL_CONSISTENCY。"
            "CLAIM_EVIDENCE_RELATION/CLAIM_PARTIAL_SUPPORT: target={claim,observation_window}, "
            "proposed_gold={relation,modality_pair,answer}; TEXT_VISUAL_CONSISTENCY: target={text_claim,visual_target}, "
            "proposed_gold={relation,modality_pair,answer}。relation 只能是 SUPPORTED、PARTIALLY_SUPPORTED、CONTRADICTED、"
            "NOT_SHOWN、TEMPORALLY_MISALIGNED。每条 CM 必须引用至少两种不同模态，modality_pair 必须列出实际模态。"
            "只有输入定义了覆盖完整视频的完整观察窗口时才能使用 NOT_SHOWN；只有采样帧时应 abstain，不能声称整个视频未展示。"
        ),
        "strategist": (
            "你只负责 SS，不得生成 BP、CM 或 AE。允许 subtype：HOOK_MECHANISM、VALUE_PROPOSITION、TRUST_MECHANISM、"
            "OBJECTION_HANDLING、URGENCY_CTA、FUNNEL_ROLE。每个 target 必须包含 segment 和 mechanism；"
            "proposed_gold 必须包含 label 和 answer。answer 应说明可观察内容如何构成该机制，不得预测销售、转化或互动效果。"
        ),
    }
    examples = {
        "consumer": (
            '{"task_type":"AE","task_subtype":"USAGE_CONTEXT",'
            '"target":{"scenario":"家庭收纳"},"proposed_gold":{"usage_context":"厨房和卫生间收纳",'
            '"answer":"内容通过展示置物架并在口播中列出厨房和卫生间用途，对应家庭收纳场景。"},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","画面展示厨房使用场景","SUPPORTED"],'
            '["existing_id_2","口播提到卫生间用途","SUPPORTED"]],"proposal_confidence":0.85}'
        ),
        "operator": (
            '{"task_type":"CM","task_subtype":"CLAIM_EVIDENCE_RELATION",'
            '"target":{"claim":"充电线采用编织材质","observation_window":"引用的口播与画面帧"},'
            '"proposed_gold":{"relation":"SUPPORTED","modality_pair":["asr","visual"],"answer":"口播提出编织材质，画面中的编织外层提供视觉支持。"},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","口播提出编织材质声明","SUPPORTED"],'
            '["existing_id_2","画面显示编织外层","SUPPORTED"]],"proposal_confidence":0.85}'
        ),
        "strategist": (
            '{"task_type":"SS","task_subtype":"HOOK_MECHANISM",'
            '"target":{"segment":"开场","mechanism":"结果前置"},"proposed_gold":{"label":"结果前置",'
            '"answer":"开场先给出预期结果，再展示对应产品，以结果前置方式吸引注意。"},'
            '"evidence_ids":["existing_id_1","existing_id_2"],'
            '"reasoning_edges":[["existing_id_1","开场口播先描述预期结果","SUPPORTED"],'
            '["existing_id_2","画面展示对应产品","SUPPORTED"]],"proposal_confidence":0.85}'
        ),
    }
    scope = contracts.get(perspective, contracts["consumer"])
    example = examples.get(perspective, examples["consumer"])
    system = (
        f"你是 SalesBench-QA 的 {perspective} Gold Proposer。{scope}"
        "只返回严格 JSON，顶层必须且只能包含 proposals 和 abstentions，二者都是数组；最多输出三条高质量 proposal。"
        f"本角色合法示例：{example}。task_type/task_subtype 必须使用上述英文受控值；target/proposed_gold 必须是非空对象。"
        "不要输出 proposal_id，canonical proposal ID 由本地代码确定性生成。每条 proposal 必须引用至少两个不同的 evidence_ids，"
        "且必须逐字复制输入中的 ID；每条 reasoning_edges 的首项必须是已引用 ID。"
        "target、proposed_gold、reasoning_edges 和 abstentions.reason 中的自然语言必须使用中文；JSON keys、task/subtype、"
        "relation 与模态枚举保持英文，OCR/ASR 原文、品牌名和原有英文术语可以保留。"
        "proposal_confidence 必须是 0 到 1 的 JSON 数字。不得使用 annotation、proposal_type、hook、value、trust、strategy、"
        "content_structure 等非约定 wrapper，不得发明证据或引用输入外的 ID。"
        "不得根据标题、粉丝数、互动指标或外部知识生成内容；不得讨论热度、销量、转化、传播效果或因果表现。"
        "如果不足两个不同 EvidenceUnit、schema 不完整、结论存在合理替代解释或超出可观察证据，应在 abstentions 中输出"
        " task_type、task_subtype 和中文 reason，而不是勉强生成 proposal。"
    )
    user = _json({"video_id": video_id, "evidence_units": evidence_units})
    return system, user


def build_challenger_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "你是 SalesBench-QA 的 Gold Challenger。只返回严格 JSON，顶层只能包含 reviews。必须为每个输入 proposal 返回且只返回一条 review，"
        "并逐字复制 proposal_id；证据缺失、含糊或 schema 不符时不得默认 PASS。"
        "通用检查：Evidence ID 是否存在、论断是否被引用证据支持、事实与推断是否分离、是否有合理替代解释、是否重复/冲突、"
        "是否越界使用标题/互动/外部知识、是否声称销售或互动因果。任务专项 rubric："
        "BP 只能是直接可定位事实，口播效果声明不能当作已验证产品事实；"
        "CM 必须包含两种不同模态，逐项检查 claim、relation、modality_pair、观察窗口和时间对齐，NOT_SHOWN 必须有完整观察窗口；"
        "SS 必须由至少两条可观察证据支持具体说服机制，不得把一般产品描述拔高为策略或效果预测；"
        "AE 必须是对内容所对应需求/场景/决策障碍的有界解释，不得冒充真实观众画像或转化结论。"
        "每条 review 使用以下 schema："
        '{"review_id":"unique string","proposal_id":"exact input proposal_id","video_id":"exact input video_id",'
        '"reviewer":"gold_challenger","verdict":"PASS|REVISE|HUMAN_REVIEW|REJECT",'
        '"checks":{"evidence_exists":true,"evidence_support":true,"fact_inference_separated":true,'
        '"schema_valid":true,"modality_diverse":true,"localizable":true,"no_duplicate":true,'
        '"no_conflict":true,"no_causal_overreach":true},'
        '"issues":[],"suggested_revision":null}. '
        "verdict 只能是 PASS、REVISE、HUMAN_REVIEW 或 REJECT。issues 中的自然语言必须使用中文；suggested_revision 为对象或 null，"
        "其中的自然语言也必须使用中文。只指出问题，不新增 Evidence，不直接生成 GroundedAnnotation。"
    )
    user = _json({"video_id": video_id, "proposals": proposals, "evidence_units": evidence_units})
    return system, user


def build_adjudicator_prompt(
    video_id: str,
    proposals: list[dict[str, object]],
    reviews: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        "你是 SalesBench-QA 的 Evidence Adjudicator。输入只包含 Challenger 判定为 PASS 的非 BP proposals。"
        "你只负责决定哪些 proposal 可接受、哪些严格同义 proposal 可合并、哪些仍需人工审核；不得重写 proposal 内容，"
        "不得生成 GroundedAnnotation、gold_value、evidence_refs、tier、quality_status 或新证据。"
        "只返回严格 JSON，顶层必须且只能包含 accepted_groups 和 human_review_queue。"
        "accepted_groups 的每项结构为 "
        '{"source_proposal_ids":["exact input proposal_id"],"reason":"中文接受或合并理由"}。'
        "一个 proposal_id 只能出现一次。只有 task_type、task_subtype、target 语义和 proposed_gold 结论均相同、彼此不冲突的 proposal 才能合并；"
        "否则分别接受，或放入 human_review_queue。human_review_queue 每项结构为 "
        '{"source_proposal_ids":["exact input proposal_id"],"reason":"中文审核原因"}。'
        "所有 ID 必须逐字复制输入，reason 自然语言必须使用中文。不得遗漏输入 proposal；本地代码将根据接受组确定性重建 annotation 并执行最终质量规则。"
    )
    user = _json({"video_id": video_id, "proposals": proposals, "reviews": reviews, "evidence_units": evidence_units})
    return system, user
