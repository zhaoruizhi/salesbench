from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tools.audit_workbench.review_queue import normalize_review_queue_row


PROPOSAL = {
    "proposal_id": "p1",
    "video_id": "v1",
    "task_type": "CM",
    "task_subtype": "CLAIM_EVIDENCE_RELATION",
    "target": {"claim": "the cable uses braided material"},
    "proposed_gold": {"relation": "SUPPORTED", "answer": "The visible weave supports the claim."},
    "evidence_ids": ["e1", "e2"],
    "commerce_cue_ids": ["c1", "c2"],
    "commercial_relation_ids": ["r1"],
    "capability": "CLAIM_DEMONSTRATION_STATUS",
    "reasoning_operator": "CLASSIFY_RELATION_STATUS",
    "question_intent": "Ask whether the spoken claim is independently demonstrated.",
    "proposal_confidence": 0.9,
}


def _index():
    return {("v1", "p1"): PROPOSAL}


def test_source_proposal_ids_resolve_readable_candidate_content() -> None:
    normalized = normalize_review_queue_row(
        {
            "review_item_id": "r1",
            "video_id": "v1",
            "source_proposal_ids": ["p1"],
            "reason": "The relation needs review.",
        },
        _index(),
    )

    assert normalized["item_type"] == "candidate"
    assert normalized["task_type"] == "CM"
    assert normalized["target"] == PROPOSAL["target"]
    assert normalized["candidate_gold"] == PROPOSAL["proposed_gold"]
    assert normalized["evidence_refs"] == ["e1", "e2"]
    assert normalized["commerce_cue_ids"] == ["c1", "c2"]
    assert normalized["commercial_relation_ids"] == ["r1"]
    assert normalized["capability"] == "CLAIM_DEMONSTRATION_STATUS"
    assert normalized["resolution_status"] == "resolved"


def test_nested_gold_item_aliases_are_normalized() -> None:
    normalized = normalize_review_queue_row(
        {
            "review_item_id": "r2",
            "video_id": "v1",
            "reason": "validation_failed",
            "issues": [{"code": "INSUFFICIENT_EVIDENCE", "message": "Two refs required."}],
            "proposal": {
                "annotation_id": "g1",
                "task_type": "SS",
                "task_subtype": "TRUST_MECHANISM",
                "target": {"segment": "demo", "mechanism": "demonstration"},
                "gold_value": {"label": "demonstration", "answer": "The product is shown in use."},
                "evidence_refs": ["e1"],
                "source_proposal_ids": ["p_ss"],
            },
        },
        {},
    )

    assert normalized["stage"] == "validation"
    assert normalized["task_type"] == "SS"
    assert normalized["candidate_gold"]["answer"] == "The product is shown in use."
    assert normalized["evidence_refs"] == ["e1"]
    assert normalized["source_proposal_ids"] == ["p_ss"]


def test_nested_proposer_record_is_normalized_without_index() -> None:
    normalized = normalize_review_queue_row(
        {"review_item_id": "r3", "video_id": "v1", "reason": "below_min_confidence", "proposal": PROPOSAL},
        {},
    )

    assert normalized["task_type"] == "CM"
    assert normalized["candidate_gold"] == PROPOSAL["proposed_gold"]
    assert normalized["evidence_refs"] == ["e1", "e2"]
    assert normalized["source_proposal_ids"] == ["p1"]


def test_abstention_is_explicit_and_does_not_fake_empty_gold() -> None:
    normalized = normalize_review_queue_row(
        {
            "review_item_id": "r4",
            "video_id": "v1",
            "reason": "ABSTENTION",
            "abstention": {
                "task_type": "AE",
                "task_subtype": "USAGE_CONTEXT",
                "reason": "Fewer than two evidence units support a bounded answer.",
            },
        },
        {},
    )

    assert normalized["item_type"] == "abstention"
    assert normalized["task_type"] == "AE"
    assert normalized["reason"] == "Fewer than two evidence units support a bounded answer."
    assert normalized["candidate_gold"] == {}
    assert normalized["display_summary"] == "No candidate generated"


def test_unresolved_source_id_is_labeled_instead_of_becoming_unknown_content() -> None:
    normalized = normalize_review_queue_row(
        {"review_item_id": "r5", "video_id": "v1", "source_proposal_ids": ["missing"], "reason": "review"},
        {},
    )

    assert normalized["resolution_status"] == "unresolved"
    assert normalized["unresolved_proposal_ids"] == ["missing"]
    assert normalized["display_summary"] == "Candidate source could not be resolved"


def test_canonical_v8_row_round_trips() -> None:
    row = {
        "review_item_id": "r6",
        "video_id": "v1",
        "stage": "challenge",
        "item_type": "candidate",
        "task_type": "CM",
        "task_subtype": "CLAIM_EVIDENCE_RELATION",
        "reason_code": "CHALLENGER_HUMAN_REVIEW",
        "reason": "The visual support is ambiguous.",
        "target": PROPOSAL["target"],
        "candidate_gold": PROPOSAL["proposed_gold"],
        "evidence_refs": ["e1", "e2"],
        "source_proposal_ids": ["p1"],
        "issues": ["Ambiguous weave."],
    }

    normalized = normalize_review_queue_row(row, _index())

    for key, value in row.items():
        assert normalized[key] == value
