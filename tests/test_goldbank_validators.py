from __future__ import annotations

import unittest
import sys

sys.path.insert(0, "src")

from salesbench.goldbank.schema import EvidenceModality, EvidenceUnit, GoldItem, GoldProposal, GoldTaskType, GoldTier  # noqa: E402
from salesbench.goldbank.validators import (  # noqa: E402
    find_duplicate_and_conflicting_items,
    public_gold_record,
    validate_gold_item,
    validate_gold_proposal,
)


def evidence(eid: str = "v1_visual_000_abc", video_id: str = "v1") -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=eid,
        video_id=video_id,
        modality=EvidenceModality.VISUAL,
        start_s=0.0,
        end_s=1.0,
        frame_indices=(0,),
        text_span="",
        subject="product",
        predicate="visible",
        value=True,
        attributes={},
        source_domains=("C6_raw_video",),
        extractor="test",
        confidence=0.95,
        timestamp_status="available",
    )


def gold_item(
    *,
    gold_id: str = "g1",
    task_type: GoldTaskType = GoldTaskType.BP,
    subtype: str = "USAGE_STEP",
    value: dict[str, object] | None = None,
    evidence_ids: tuple[str, ...] = ("v1_visual_000_abc",),
    tier: GoldTier = GoldTier.GOLD_A,
    video_id: str = "v1",
) -> GoldItem:
    return GoldItem(
        gold_id=gold_id,
        video_id=video_id,
        task_type=task_type,
        task_subtype=subtype,
        target={"subject": "product"},
        gold_value=value or {"action": "shown"},
        evidence_ids=evidence_ids,
        reasoning_edges=(),
        eligible_question_formats=("direct_question",),
        source_proposal_ids=("p1",),
        gold_tier=tier,
        review_status="verified",
        confidence=0.9,
    )


class GoldBankValidatorTest(unittest.TestCase):
    def test_cm_proposal_requires_two_distinct_modalities(self):
        first = evidence()
        second = evidence("v1_visual_001_def")
        proposal = GoldProposal(
            proposal_id="p1",
            video_id="v1",
            source_agent="operator",
            task_type=GoldTaskType.CM,
            task_subtype="CLAIM_EVIDENCE_RELATION",
            target={"claim": "口播称产品防水"},
            proposed_gold={"relation": "SUPPORTED", "modality_pair": ["asr", "visual"]},
            evidence_ids=(first.evidence_id, second.evidence_id),
            reasoning_edges=(),
            proposal_confidence=0.9,
        )

        issues = validate_gold_proposal(proposal, {first.evidence_id: first, second.evidence_id: second})

        self.assertIn("CM_MODALITY_DIVERSITY", {issue.code for issue in issues})

    def test_missing_evidence_reference_is_error(self):
        issues = validate_gold_item(gold_item(evidence_ids=("missing",)), {})

        self.assertIn("EVIDENCE_NOT_FOUND", {issue.code for issue in issues})

    def test_cross_video_evidence_reference_is_error(self):
        other = evidence("other_visual_000_abc", video_id="other")
        issues = validate_gold_item(gold_item(evidence_ids=(other.evidence_id,)), {other.evidence_id: other})

        self.assertIn("EVIDENCE_VIDEO_MISMATCH", {issue.code for issue in issues})

    def test_ss_single_evidence_cannot_be_gold_b(self):
        item = gold_item(
            task_type=GoldTaskType.SS,
            subtype="PROBLEM_SOLUTION",
            value={"strategy": "question"},
            evidence_ids=("v1_visual_000_abc",),
            tier=GoldTier.GOLD_B,
        )

        issues = validate_gold_item(item, {evidence().evidence_id: evidence()})

        self.assertIn("INSUFFICIENT_EVIDENCE", {issue.code for issue in issues})

    def test_bp_direct_observation_can_be_gold_a(self):
        issues = validate_gold_item(gold_item(), {evidence().evidence_id: evidence()})

        self.assertEqual([issue for issue in issues if issue.severity == "ERROR"], [])

    def test_inference_language_downgrades_bp_to_silver(self):
        item = gold_item(value={"answer": "this probably creates trust"})

        issues = validate_gold_item(item, {evidence().evidence_id: evidence()})

        self.assertIn("OBSERVATION_INFERENCE_MIXED", {issue.code for issue in issues})

    def test_private_performance_fields_never_enter_public_record(self):
        item = gold_item(
            task_type=GoldTaskType.SS,
            subtype="PROBLEM_SOLUTION",
            value={"strategy": "question", "likes": 100, "thresholds": {"q33": 1}},
            evidence_ids=("v1_visual_000_abc", "v1_visual_001_def"),
            tier=GoldTier.GOLD_B,
        )

        public = public_gold_record(item)
        serialized = str(public)

        self.assertNotIn("likes", serialized)
        self.assertNotIn("thresholds", serialized)
        self.assertEqual(public["gold_value"]["strategy"], "question")

    def test_duplicate_semantics_are_reported(self):
        first = gold_item(gold_id="g1")
        second = gold_item(gold_id="g2")

        issues = find_duplicate_and_conflicting_items([first, second])

        self.assertIn("DUPLICATE_SEMANTICS", {issue.code for issue in issues})

    def test_conflicting_values_are_reported(self):
        first = gold_item(gold_id="g1", value={"count": 1})
        second = gold_item(gold_id="g2", value={"count": 2})

        issues = find_duplicate_and_conflicting_items([first, second])

        self.assertIn("CONFLICTING_VALUE", {issue.code for issue in issues})

    def test_same_target_in_different_videos_is_not_a_conflict(self):
        first = gold_item(gold_id="g1", video_id="v1", value={"count": 1})
        second = gold_item(gold_id="g2", video_id="v2", value={"count": 2})

        issues = find_duplicate_and_conflicting_items([first, second])

        self.assertFalse(issues)


if __name__ == "__main__":
    unittest.main()
