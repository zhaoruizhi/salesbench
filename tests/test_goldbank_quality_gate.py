from __future__ import annotations

import sys

sys.path.insert(0, "src")

from salesbench.goldbank.quality_gate import (  # noqa: E402
    LifecycleStatus,
    QualityDisposition,
    route_quality_record,
)


def test_validation_failure_is_rejected_without_human_escalation():
    routed = route_quality_record(
        {
            "review_item_id": "r1",
            "stage": "validation",
            "item_type": "candidate",
            "reason_code": "PROPOSAL_GRAPH_VALIDATION_FAILED",
            "candidate_snapshot": {"proposal_id": "p1"},
        }
    )

    assert routed.disposition == QualityDisposition.REJECT
    assert routed.output_channel == "rejected_candidates"


def test_abstention_is_diagnostic_not_human_review():
    routed = route_quality_record(
        {
            "review_item_id": "r2",
            "stage": "proposal",
            "item_type": "abstention",
            "reason_code": "ABSTENTION",
        }
    )

    assert routed.disposition == QualityDisposition.REJECT
    assert routed.output_channel == "pipeline_diagnostics"


def test_challenger_revision_is_repairable_not_human_review():
    routed = route_quality_record(
        {
            "review_item_id": "r3",
            "stage": "challenge",
            "item_type": "candidate",
            "reason_code": "CHALLENGER_REVISE",
        }
    )

    assert routed.disposition == QualityDisposition.REPAIR
    assert routed.output_channel == "pipeline_diagnostics"


def test_only_explicit_semantic_ambiguity_reaches_human_review():
    routed = route_quality_record(
        {
            "review_item_id": "r4",
            "stage": "challenge",
            "item_type": "candidate",
            "reason_code": "CHALLENGER_HUMAN_REVIEW",
        }
    )

    assert routed.disposition == QualityDisposition.HUMAN_REVIEW
    assert routed.output_channel == "human_review_queue"


def test_lifecycle_status_does_not_use_legacy_verified_label():
    assert LifecycleStatus.AUTO_ACCEPTED_CANDIDATE.value == "auto_accepted_candidate"
    assert LifecycleStatus.HUMAN_ACCEPTED.value == "human_accepted"
