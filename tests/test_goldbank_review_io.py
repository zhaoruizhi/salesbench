from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.review_io import apply_human_reviews, validate_human_review_decisions  # noqa: E402


class GoldBankReviewIOTest(unittest.TestCase):
    def review_queue(self):
        return [
            {
                "review_item_id": "hr_v1_p1",
                "video_id": "v1",
                "proposal_id": "p1",
                "proposal": {
                    "proposal_id": "p1",
                    "video_id": "v1",
                    "task_type": "AE",
                    "task_subtype": "AUDIENCE_NEED_FIT",
                    "target": {"audience": "new parents"},
                    "proposed_gold": {"audience_need": "convenient breakfast"},
                    "evidence_ids": ["e1", "e2"],
                    "reasoning_edges": [],
                    "proposal_confidence": 0.8,
                    "capability": "USAGE_CONTEXT",
                    "reasoning_operator": "LOCALIZE_USAGE_CONTEXT",
                    "commerce_cue_ids": ["cue1"],
                    "commercial_relation_ids": [],
                    "question_intent": "Ask which usage context is represented by the content.",
                    "forbidden_inferences": ["Do not infer a real viewer profile."],
                },
            }
        ]

    def video_gold_records(self):
        return [
            {
                "video_id": "v1",
                "schema_version": "evidence-dataset-schema-v2",
                "evidence_unit_ids": ["e1", "e2"],
                "grounded_annotations": [],
                "coverage": {},
                "quality_summary": {},
                "observation_scope": {},
            }
        ]

    def test_validate_human_review_decisions_requires_revise_fields(self):
        errors = validate_human_review_decisions(
            [
                {
                    "review_item_id": "hr_v1_p1",
                    "proposal_id": "p1",
                    "decision": "REVISE",
                    "reviewer_id": "a1",
                    "reviewed_at": "2026-07-30T00:00:00Z",
                    "reason_code": "needs_fix",
                }
            ]
        )

        self.assertTrue(any("revised_value" in error for error in errors))

    def test_accept_gold_b_adds_human_accepted_item(self):
        reviewed = apply_human_reviews(
            self.video_gold_records(),
            self.review_queue(),
            [
                {
                    "review_item_id": "hr_v1_p1",
                    "proposal_id": "p1",
                    "decision": "ACCEPT_GOLD_B",
                    "reviewer_id": "a1",
                    "reviewed_at": "2026-07-30T00:00:00Z",
                    "reason_code": "ok",
                    "revised_value": None,
                    "revised_evidence_ids": [],
                }
            ],
        )

        item = reviewed[0]["grounded_annotations"][0]
        self.assertEqual(item["quality_status"], "INFERRED")
        self.assertEqual(item["review_status"], "human_accepted")
        self.assertEqual(item["capability"], "USAGE_CONTEXT")
        self.assertEqual(item["commerce_cue_ids"], ["cue1"])

    def test_unknown_review_item_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_human_reviews(
                self.video_gold_records(),
                self.review_queue(),
                [
                    {
                        "review_item_id": "missing",
                        "proposal_id": "p1",
                        "decision": "REJECT",
                        "reviewer_id": "a1",
                        "reviewed_at": "2026-07-30T00:00:00Z",
                        "reason_code": "bad",
                        "revised_value": None,
                        "revised_evidence_ids": [],
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
