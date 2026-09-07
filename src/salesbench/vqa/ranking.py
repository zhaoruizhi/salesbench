"""Auditable deterministic quality ranking for compile-time QA selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..goldbank.schema import GoldItem, GoldTaskType
from ..utils import clean_text


_HIGH_VALUE_BP = {
    "PRODUCT_IDENTITY",
    "ATTRIBUTE_AND_VARIANT",
    "QUANTITY_AND_BUNDLE",
    "PRICE_AND_DISCOUNT",
    "OFFER_CONDITION",
    "DEMONSTRATED_STATE_CHANGE",
    "USAGE_SCENARIO",
}
_TRIVIAL_ACTIONS = (
    "flips through",
    "turns the pages",
    "holds up",
    "points to",
    "shows the product",
    "rotates the product",
)


@dataclass(frozen=True)
class CandidateScore:
    total: float
    components: dict[str, float]
    penalties: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "components": dict(self.components),
            "penalties": dict(self.penalties),
        }


def score_qa_candidate(
    item: GoldItem,
    *,
    verification: dict[str, object] | None = None,
) -> CandidateScore:
    answer = " ".join(clean_text(value) for value in item.gold_value.values()).lower()
    capability = clean_text(item.capability or item.task_subtype).upper()
    components = {
        "evidence_confidence": round(item.confidence * 50.0, 6),
        "commerce_graph": float(min(18, len(item.commerce_cue_ids) * 3 + len(item.commercial_relation_ids) * 8)),
        "reasoning_value": 10.0 if item.task_type != GoldTaskType.BP else 0.0,
        "bp_calibration_value": 8.0 if item.task_type == GoldTaskType.BP and capability in _HIGH_VALUE_BP else 0.0,
        "semantic_verification": 0.0,
    }
    if verification:
        quality_values = [
            value for key, value in verification.items() if key not in {"spec_id", "verdict", "reason"} and isinstance(value, bool)
        ]
        components["semantic_verification"] = float(sum(quality_values) * 2)
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", answer))
    penalties = {
        "trivial_background_action": 45.0 if any(marker in answer for marker in _TRIVIAL_ACTIONS) else 0.0,
        "answer_verbosity": float(max(0, word_count - 30)) * 0.5,
    }
    total = round(sum(components.values()) - sum(penalties.values()), 6)
    return CandidateScore(total=total, components=components, penalties=penalties)
