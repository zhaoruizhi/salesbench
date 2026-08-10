from __future__ import annotations

import re
import sys

sys.path.insert(0, "src")

from salesbench.goldbank.schema import parse_gold_item  # noqa: E402
from salesbench.vqa.question_programs import render_question  # noqa: E402


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _item(task: str, subtype: str, target: dict[str, object], gold: dict[str, object], fmt: str):
    return parse_gold_item(
        {
            "annotation_id": f"g_{task.lower()}",
            "video_id": "v1",
            "task_type": task,
            "task_subtype": subtype,
            "target": target,
            "gold_value": gold,
            "evidence_refs": ["e1", "e2"],
            "reasoning_edges": [],
            "eligible_question_formats": [fmt],
            "source_proposal_ids": ["p1"],
            "quality_status": "DIRECT" if task in {"BP", "CM"} else "INFERRED",
            "review_status": "verified",
            "confidence": 0.9,
        }
    )


def test_all_four_task_programs_render_english_questions_and_answers() -> None:
    fixtures = [
        _item("BP", "OCR_FACT", {"subject": "package", "predicate": "certification"}, {"value": "national patent"}, "direct_question"),
        _item(
            "CM",
            "CLAIM_EVIDENCE_RELATION",
            {"claim": "the cable uses braided material"},
            {"relation": "SUPPORTED", "answer": "The visible weave supports the spoken claim."},
            "relation_choice",
        ),
        _item(
            "SS",
            "TRUST_MECHANISM",
            {"segment": "demonstration", "mechanism": "product demonstration"},
            {"label": "demonstration", "answer": "The product is demonstrated in use."},
            "mechanism_with_evidence",
        ),
        _item(
            "AE",
            "USAGE_CONTEXT",
            {"scenario": "commuting"},
            {"usage_context": "commuting", "answer": "The compact setup is presented for commuting."},
            "need_with_evidence",
        ),
    ]

    rendered = [render_question(item, item.eligible_question_formats[0]) for item in fixtures]

    assert all(not CJK_RE.search(result.question) for result in rendered)
    assert all(not CJK_RE.search(result.answer) for result in rendered)
    assert all("?" in result.question for result in rendered)
