from __future__ import annotations

import unittest
import sys

sys.path.insert(0, "src")

from salesbench.goldbank.ontology import (  # noqa: E402
    AE_SUBTYPES,
    BP_SUBTYPES,
    CM_SUBTYPES,
    SS_SUBTYPES,
    TASK_MIN_EVIDENCE,
    allowed_subtypes,
    capability_level,
    default_reasoning_operator,
)
from salesbench.goldbank.schema import (  # noqa: E402
    EvidenceModality,
    EvidenceUnit,
    GoldItem,
    GoldTaskType,
    GoldTier,
    QualityStatus,
    make_evidence_id,
    make_gold_id,
    parse_evidence_unit,
    parse_gold_item,
)


class GoldBankSchemaTest(unittest.TestCase):
    def test_gold_item_serializes_enum_values(self):
        item = GoldItem(
            gold_id="g1",
            video_id="v1",
            task_type=GoldTaskType.BP,
            task_subtype="USAGE_STEP",
            target={"subject": "hand"},
            gold_value={"action": "opens box"},
            evidence_ids=("v1_visual_000_abcdef123456",),
            reasoning_edges=(("evidence", "answer", "SUPPORTED"),),
            eligible_question_formats=("direct_question",),
            source_proposal_ids=("p1",),
            gold_tier=GoldTier.GOLD_A,
            review_status="verified",
            confidence=0.91,
            capability="USAGE_STEP",
            reasoning_operator="SEQUENCE_ACTION",
            commerce_cue_ids=("cue1",),
            commercial_relation_ids=(),
            question_intent="Identify the concrete action performed with the product.",
            forbidden_inferences=("Do not infer sales outcomes.",),
        )

        payload = item.to_dict()

        self.assertEqual(payload["task_type"], "BP")
        self.assertEqual(payload["quality_status"], QualityStatus.DIRECT.value)
        self.assertIsInstance(payload["evidence_refs"], list)
        self.assertEqual(payload["annotation_id"], "g1")
        self.assertEqual(payload["capability"], "USAGE_STEP")
        self.assertEqual(payload["commerce_cue_ids"], ["cue1"])

    def test_every_task_has_allowed_subtypes_and_question_formats(self):
        self.assertIn("USAGE_STEP", BP_SUBTYPES)
        self.assertIn("CLAIM_DEMONSTRATION_STATUS", CM_SUBTYPES)
        self.assertIn("CTA_SEQUENCE", SS_SUBTYPES)
        self.assertIn("DECISION_BARRIER", AE_SUBTYPES)
        self.assertEqual(set(TASK_MIN_EVIDENCE), set(GoldTaskType))

    def test_plan_b_capabilities_use_expected_graph_levels(self):
        self.assertEqual(capability_level("BP", "OFFER_CONDITION"), "RELATION_OPTIONAL")
        self.assertEqual(capability_level("CM", "CLAIM_DEMONSTRATION_STATUS"), "RELATION_REQUIRED")
        self.assertEqual(capability_level("SS", "PROBLEM_SOLUTION"), "RELATION_PATH_REQUIRED")
        self.assertEqual(capability_level("AE", "FIT_CONSTRAINT"), "CUE_OR_RELATION")
        self.assertNotIn("TRUST_MECHANISM", allowed_subtypes("SS"))
        self.assertNotIn("AUDIENCE_NEED_FIT", allowed_subtypes("AE"))
        self.assertEqual(default_reasoning_operator("CM", "PARTIAL_SUPPORT"), "DECOMPOSE_CLAIM")

    def test_gold_id_is_stable_for_same_semantics(self):
        first = make_gold_id(
            "v1",
            GoldTaskType.CM,
            "CLAIM_DEMONSTRATION_STATUS",
            {"claim": " 450g ", "atoms": ["9个", "450g"]},
            {"relation": "SUPPORTED"},
        )
        second = make_gold_id(
            "v1",
            GoldTaskType.CM,
            "CLAIM_DEMONSTRATION_STATUS",
            {"atoms": ["9个", "450g"], "claim": "450g"},
            {"relation": "SUPPORTED"},
        )

        self.assertEqual(first, second)

    def test_evidence_id_is_video_scoped(self):
        evidence_id = make_evidence_id("v1", EvidenceModality.VISUAL, 3)
        unit = EvidenceUnit(
            evidence_id=evidence_id,
            video_id="v1",
            modality=EvidenceModality.VISUAL,
            start_s=0.5,
            end_s=1.5,
            frame_indices=(3,),
            text_span="",
            subject="product",
            predicate="visible",
            value=True,
            attributes={},
            source_domains=("C6_raw_video",),
            extractor="test",
            confidence=1.0,
            timestamp_status="available",
        )

        parsed = parse_evidence_unit(unit.to_dict())

        self.assertTrue(evidence_id.startswith("v1_visual_"))
        self.assertEqual(parsed.evidence_id, evidence_id)

    def test_parse_gold_item_rejects_bad_confidence(self):
        payload = {
            "gold_id": "g1",
            "video_id": "v1",
            "task_type": "BP",
            "task_subtype": "USAGE_STEP",
            "target": {},
            "gold_value": {},
            "evidence_ids": [],
            "reasoning_edges": [],
            "eligible_question_formats": [],
            "source_proposal_ids": [],
            "gold_tier": "Gold-A",
            "review_status": "verified",
            "confidence": 1.3,
        }

        with self.assertRaises(ValueError):
            parse_gold_item(payload)


if __name__ == "__main__":
    unittest.main()
