from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.audit import audit_gold_bank  # noqa: E402


class GoldBankAuditTest(unittest.TestCase):
    def test_evidence_coverage_denominator_is_accepted_gold_count(self):
        records = [
            {
                "video_id": "v1",
                "grounded_annotations": [
                    {
                        "annotation_id": "g1",
                        "video_id": "v1",
                        "task_type": "BP",
                        "task_subtype": "ACTION",
                        "target": {},
                        "gold_value": {"action": "show"},
                        "evidence_refs": ["e1"],
                        "reasoning_edges": [],
                        "eligible_question_formats": [],
                        "source_proposal_ids": [],
                        "quality_status": "DIRECT",
                        "review_status": "verified",
                        "confidence": 1.0,
                    },
                    {
                        "annotation_id": "g2",
                        "video_id": "v1",
                        "task_type": "CM",
                        "task_subtype": "CLAIM_EVIDENCE_RELATION",
                        "target": {},
                        "gold_value": {"relation": "SUPPORTED"},
                        "evidence_refs": ["missing"],
                        "reasoning_edges": [],
                        "eligible_question_formats": [],
                        "source_proposal_ids": [],
                        "quality_status": "DIRECT",
                        "review_status": "verified",
                        "confidence": 1.0,
                    },
                ],
            }
        ]
        evidence = [{"evidence_id": "e1", "video_id": "v1", "modality": "visual"}]

        audit = audit_gold_bank(records, evidence)

        self.assertEqual(audit["item_count"], 2)
        self.assertEqual(audit["evidence_coverage"], 0.5)

    def test_private_leakage_is_counted(self):
        records = [{"video_id": "v1", "grounded_annotations": [{"annotation_id": "g1", "task_type": "BP", "quality_status": "DIRECT", "likes": 1}]}]

        audit = audit_gold_bank(records, [])

        self.assertEqual(audit["private_field_leakage"], 1)


if __name__ == "__main__":
    unittest.main()
