from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.normalizer import normalize_evidence_units  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
