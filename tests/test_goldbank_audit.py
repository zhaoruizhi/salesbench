from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.audit import audit_gold_bank  # noqa: E402


class GoldBankAuditTest(unittest.TestCase):
    def test_audit_reports_graph_language_and_translation_separation(self):
        records = [
            {
                "video_id": "v1",
                "schema_version": "evidence-dataset-schema-v3",
                "grounded_annotations": [
                    {
                        "annotation_id": "g1",
                        "video_id": "v1",
                        "task_type": "SS",
                        "task_subtype": "FEATURE_BENEFIT",
                        "capability": "FEATURE_BENEFIT",
                        "reasoning_operator": "FOLLOW_RELATION_PATH",
                        "target": {"specific_focus": "the visible product feature"},
                        "gold_value": {"answer": "The spoken explanation frames the visible feature as a practical benefit."},
                        "evidence_refs": ["e1", "e2"],
                        "commerce_cue_ids": ["c1", "c2"],
                        "commercial_relation_ids": ["r1"],
                        "reasoning_edges": [],
                        "eligible_question_formats": ["grounded_question"],
                        "source_proposal_ids": ["p1"],
                        "quality_status": "INFERRED",
                        "review_status": "verified",
                        "confidence": 0.9,
                    }
                ],
            }
        ]
        evidence = [
            {"evidence_id": "e1", "video_id": "v1", "modality": "visual", "content_en": "A textured handle is visible.", "source_text_native": ""},
            {"evidence_id": "e2", "video_id": "v1", "modality": "asr", "content_en": "The speaker says the texture improves grip.", "source_text_native": "防滑好握"},
        ]
        cues = [
            {"cue_id": "c1", "video_id": "v1", "cue_type": "PRODUCT_ATTRIBUTE", "content_en": "The handle has a visible texture.", "evidence_ids": ["e1"]},
            {"cue_id": "c2", "video_id": "v1", "cue_type": "BENEFIT", "content_en": "The speaker frames the texture as easier to grip.", "evidence_ids": ["e2"]},
        ]
        relations = [
            {
                "relation_id": "r1",
                "video_id": "v1",
                "relation_type": "FEATURE_FRAMED_AS_BENEFIT",
                "source_cue_ids": ["c1"],
                "target_cue_ids": ["c2"],
                "evidence_ids": ["e1", "e2"],
                "status": "SUPPORTED",
                "rationale_en": "The spoken benefit explicitly refers to the visible texture.",
                "provenance": "THEORY_OPERATIONALIZED",
            }
        ]

        report = audit_gold_bank(records, evidence, cues, relations)

        self.assertEqual(report["commerce_cue_coverage"], 1.0)
        self.assertEqual(report["commerce_cue_evidence_validity"], 1.0)
        self.assertEqual(report["commercial_relation_validity"], 1.0)
        self.assertEqual(report["relation_provenance_validity"], 1.0)
        self.assertEqual(report["capability_operator_coverage"], 1.0)
        self.assertEqual(report["causal_outcome_relation_count"], 0)
        self.assertEqual(report["canonical_chinese_field_count"], 0)
        self.assertEqual(report["audit_translation_stale_count"], 0)

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
