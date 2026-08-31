from __future__ import annotations

import sys
import unittest
from dataclasses import replace

sys.path.insert(0, "src")

from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CommerceCue,
    CommercialRelation,
    CueType,
    RelationProvenance,
    RelationType,
)
from salesbench.goldbank.schema import (  # noqa: E402
    EvidenceAssertionType,
    EvidenceModality,
    EvidenceTemporalScope,
    EvidenceUnit,
    GoldItem,
    GoldProposal,
    GoldTaskType,
    GoldTier,
)
from salesbench.goldbank.validators import (  # noqa: E402
    find_duplicate_and_conflicting_items,
    public_gold_record,
    validate_commerce_cue,
    validate_commercial_relation,
    validate_evidence_unit,
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


def cue_record(cue_type: CueType, evidence_ids: tuple[str, ...], cue_id: str) -> CommerceCue:
    return CommerceCue(
        cue_id=cue_id,
        video_id="v1",
        cue_type=cue_type,
        content_en="The cited content presents the cue.",
        source_text_native="",
        evidence_ids=evidence_ids,
        attributes={},
        directness="DIRECT",
        theory_tags=(),
        extractor="test",
        confidence=0.95,
    )


def relation_record(
    relation_type: RelationType,
    source: tuple[str, ...],
    target: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> CommercialRelation:
    return CommercialRelation(
        relation_id="r1",
        video_id="v1",
        relation_type=relation_type,
        source_cue_ids=source,
        target_cue_ids=target,
        evidence_ids=evidence_ids,
        status="SUPPORTED",
        rationale_en="The cited demonstration is presented alongside the claim.",
        provenance=RelationProvenance.THEORY_OPERATIONALIZED,
        directness="INFERRED",
        extractor="test",
        confidence=0.95,
    )


class GoldBankValidatorTest(unittest.TestCase):
    def test_asr_without_temporal_localization_is_rejected(self):
        unit = replace(
            evidence("asr_missing_time"),
            modality=EvidenceModality.ASR,
            start_s=None,
            end_s=None,
            frame_indices=(),
            text_span="主播说这是全天不鼓包",
            source_text_native="主播说这是全天不鼓包",
            assertion_type=EvidenceAssertionType.SPOKEN_CLAIM,
            temporal_scope=EvidenceTemporalScope.LONG_TERM_CLAIM,
        )

        issues = validate_evidence_unit(unit)

        self.assertIn("MISSING_TEMPORAL_LOCALIZATION", {issue.code for issue in issues})

    def test_asr_only_product_attribute_cue_is_rejected_as_claim_fact_mix(self):
        unit = replace(
            evidence("asr1"),
            modality=EvidenceModality.ASR,
            start_s=0.0,
            end_s=5.0,
            frame_indices=(),
            text_span="一整天也不鼓包",
            source_text_native="一整天也不鼓包",
            assertion_type=EvidenceAssertionType.SPOKEN_CLAIM,
            temporal_scope=EvidenceTemporalScope.LONG_TERM_CLAIM,
        )
        cue = cue_record(CueType.PRODUCT_ATTRIBUTE, (unit.evidence_id,), "c_attr")

        issues = validate_commerce_cue(cue, {unit.evidence_id: unit})

        self.assertIn("CLAIM_AS_PRODUCT_ATTRIBUTE", {issue.code for issue in issues})

    def test_unattributed_benefit_from_spoken_claim_is_rejected(self):
        unit = replace(
            evidence("asr_benefit"),
            modality=EvidenceModality.ASR,
            start_s=0.0,
            end_s=5.0,
            frame_indices=(),
            text_span="彩色标记帮助右脑记忆",
            source_text_native="彩色标记帮助右脑记忆",
            content_en="The speaker claims colorful markings assist right-brain memory.",
            assertion_type=EvidenceAssertionType.SPOKEN_CLAIM,
            temporal_scope=EvidenceTemporalScope.LONG_TERM_CLAIM,
        )
        cue = replace(
            cue_record(CueType.BENEFIT, (unit.evidence_id,), "c_benefit"),
            content_en="Colorful markings assist right-brain memory.",
        )

        issues = validate_commerce_cue(cue, {unit.evidence_id: unit})

        self.assertIn("UNATTRIBUTED_PROMOTIONAL_CLAIM", {issue.code for issue in issues})

    def test_short_visual_cannot_independently_support_long_term_claim(self):
        claim = replace(
            evidence("asr1"),
            modality=EvidenceModality.ASR,
            start_s=0.0,
            end_s=5.0,
            frame_indices=(),
            text_span="一整天也不鼓包",
            source_text_native="一整天也不鼓包",
            content_en="The speaker claims the pants do not bulge after a day of activity.",
            assertion_type=EvidenceAssertionType.SPOKEN_CLAIM,
            temporal_scope=EvidenceTemporalScope.LONG_TERM_CLAIM,
        )
        visual = replace(
            evidence("visual1"),
            content_en="The woman bends while wearing the pants.",
            assertion_type=EvidenceAssertionType.OBSERVED,
            temporal_scope=EvidenceTemporalScope.FRAME,
        )
        claim_cue = cue_record(CueType.FUNCTION_CLAIM, (claim.evidence_id,), "c_claim")
        demo_cue = cue_record(CueType.PROCESS_DEMONSTRATION, (visual.evidence_id,), "c_demo")
        relation = relation_record(
            RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION,
            (claim_cue.cue_id,),
            (demo_cue.cue_id,),
            (claim.evidence_id, visual.evidence_id),
        )

        issues = validate_commercial_relation(
            relation,
            {claim_cue.cue_id: claim_cue, demo_cue.cue_id: demo_cue},
            {claim.evidence_id: claim, visual.evidence_id: visual},
        )

        self.assertIn("UNSUPPORTED_CLAIM_SCOPE", {issue.code for issue in issues})
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
