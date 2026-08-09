from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.normalizer import normalize_evidence_units, normalize_proposals  # noqa: E402
from salesbench.goldbank.schema import EvidenceModality  # noqa: E402
from salesbench.goldbank.validators import validate_evidence_unit  # noqa: E402


class EvidenceNormalizerTest(unittest.TestCase):
    def test_normalizes_common_vlm_visual_aliases_and_frame_locator(self):
        unit = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "image",
                    "evidence_id": "frame_006.jpg",
                    "subject": "产品",
                    "predicate": "颜色",
                    "value": "红色",
                    "confidence": "high",
                }
            ],
        )[0]

        self.assertEqual(unit.modality, EvidenceModality.VISUAL)
        self.assertEqual(unit.frame_indices, (6,))
        self.assertEqual(unit.confidence, 0.9)
        self.assertEqual(unit.attributes["source_locator"], "frame_006.jpg")
        self.assertFalse(validate_evidence_unit(unit))

    def test_normalizes_asr_text_alias_and_creates_unique_local_id(self):
        units = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "text",
                    "evidence_id": "asr_subtitles",
                    "subject": "产品",
                    "predicate": "价格",
                    "value": "十几元",
                    "confidence": "medium",
                },
                {
                    "modality": "text",
                    "evidence_id": "asr_subtitles",
                    "subject": "产品",
                    "predicate": "用途",
                    "value": "厨房收纳",
                    "confidence": 90,
                },
            ],
        )

        self.assertTrue(all(unit.modality == EvidenceModality.ASR for unit in units))
        self.assertEqual([unit.text_span for unit in units], ["十几元", "厨房收纳"])
        self.assertEqual([unit.confidence for unit in units], [0.75, 0.9])
        self.assertEqual(len({unit.evidence_id for unit in units}), 2)
        self.assertTrue(all(not validate_evidence_unit(unit) for unit in units))

    def test_ambiguous_text_is_not_promoted_to_direct_evidence(self):
        unit = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "text",
                    "subject": "产品",
                    "predicate": "卖点",
                    "value": "耐用",
                    "confidence": "high",
                }
            ],
        )[0]

        self.assertEqual(unit.modality, EvidenceModality.METADATA)
        self.assertTrue(any(issue.code == "NON_DIRECT_EVIDENCE" for issue in validate_evidence_unit(unit)))

    def test_proposer_cannot_emit_another_perspectives_task_subtype(self):
        with self.assertRaisesRegex(ValueError, "consumer proposer cannot emit"):
            normalize_proposals(
                "v1",
                "consumer",
                [
                    {
                        "task_type": "CM",
                        "task_subtype": "CLAIM_EVIDENCE_RELATION",
                        "target": {"claim": "claim"},
                        "proposed_gold": {"relation": "SUPPORTED"},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )

        with self.assertRaisesRegex(ValueError, "consumer proposer cannot emit"):
            normalize_proposals(
                "v1",
                "consumer",
                [
                    {
                        "task_type": "SS",
                        "task_subtype": "VALUE_PROPOSITION",
                        "target": {"segment": "开场"},
                        "proposed_gold": {"label": "卖点"},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )

        with self.assertRaisesRegex(ValueError, "operator proposer cannot emit"):
            normalize_proposals(
                "v1",
                "operator",
                [
                    {
                        "task_type": "SS",
                        "task_subtype": "HOOK_MECHANISM",
                        "target": {"segment": "开场"},
                        "proposed_gold": {"label": "提问"},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )

    def test_placeholder_proposal_id_is_replaced_locally(self):
        proposal = normalize_proposals(
            "v1",
            "strategist",
            [
                {
                    "proposal_id": "optional",
                    "task_type": "SS",
                    "task_subtype": "HOOK_MECHANISM",
                    "target": {"segment": "opening"},
                    "proposed_gold": {"label": "result-first hook"},
                    "evidence_ids": ["e1", "e2"],
                    "reasoning_edges": [],
                    "proposal_confidence": 0.9,
                }
            ],
        )[0]

        self.assertTrue(proposal.proposal_id.startswith("v1_strategist_ss_000_"))


if __name__ == "__main__":
    unittest.main()
