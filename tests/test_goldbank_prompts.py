from __future__ import annotations

import unittest
import sys

sys.path.insert(0, "src")

from salesbench.goldbank.parsing import (  # noqa: E402
    ModelOutputError,
    parse_adjudication_response,
    parse_json_object,
    parse_proposal_response,
)
from salesbench.goldbank.prompts import (  # noqa: E402
    PROMPT_VERSION,
    build_adjudicator_prompt,
    build_challenger_prompt,
    build_evidence_extractor_prompt,
    build_proposer_prompt,
)


class GoldBankPromptTest(unittest.TestCase):
    def test_evidence_extractor_requests_units_not_questions(self):
        system, user_blocks = build_evidence_extractor_prompt("v1", {"C3_text_language": {"title": "hello"}})
        text = system + " " + str(user_blocks)

        self.assertEqual(PROMPT_VERSION, "evidence-prompt-v7")
        self.assertIn("EvidenceUnit", text)
        self.assertNotIn("generate questions", text.lower())
        self.assertIn("自然语言字段必须使用中文", system)
        self.assertIn("口播", system)
        self.assertIn("声称", system)
        self.assertIn("禁止把坐标", system)
        self.assertIn("start_s", system)
        self.assertIn("end_s", system)

    def test_all_proposers_must_reference_existing_evidence_ids(self):
        system, user = build_proposer_prompt(
            "operator",
            "v1",
            [{"evidence_id": "e1", "video_id": "v1", "subject": "claim"}],
        )

        self.assertIn("evidence_id", system + user)
        self.assertIn("e1", user)
        self.assertIn("abstentions", system + user)
        self.assertIn("至少两个不同的 evidence_ids", system)
        self.assertIn("proposal_confidence", system)
        self.assertIn("reasoning_edges", system)

    def test_proposer_contract_has_controlled_tasks_subtypes_and_exact_schema(self):
        consumer, _ = build_proposer_prompt("consumer", "v1", [])
        operator, _ = build_proposer_prompt("operator", "v1", [])
        strategist, _ = build_proposer_prompt("strategist", "v1", [])

        self.assertIn("AUDIENCE_NEED_FIT", consumer)
        self.assertIn("CLAIM_EVIDENCE_RELATION", operator)
        self.assertIn("HOOK_MECHANISM", strategist)
        self.assertIn('"task_type":"AE"', consumer)
        self.assertIn('"task_type":"CM"', operator)
        self.assertIn('"task_type":"SS"', strategist)
        self.assertNotIn('"task_type":"SS"', consumer)
        self.assertNotIn('"task_type":"SS"', operator)
        self.assertNotIn('"task_type":"AE"', operator)
        self.assertNotIn('"task_type":"CM"', strategist)
        self.assertNotIn('"task_type":"AE"', strategist)
        for prompt in (consumer, operator, strategist):
            self.assertIn('"target"', prompt)
            self.assertIn('"proposed_gold"', prompt)
            self.assertIn("不要输出 proposal_id", prompt)
            self.assertIn("自然语言", prompt)
        self.assertIn("audience_need", consumer)
        self.assertIn("usage_context", consumer)
        self.assertIn("decision_state", consumer)
        self.assertIn("content_motivation", consumer)
        self.assertIn("AE 禁止使用 claim", consumer)
        self.assertIn("至少两种不同模态", operator)
        self.assertIn("modality_pair", operator)
        self.assertIn("NOT_SHOWN", operator)
        self.assertIn("完整观察窗口", operator)

    def test_challenger_can_request_human_review_and_cannot_default_pass(self):
        system, user = build_challenger_prompt("v1", [], [])

        self.assertIn("HUMAN_REVIEW", system + user)
        self.assertIn("不得默认", system + user)
        self.assertIn("每个输入 proposal", system)
        self.assertIn("suggested_revision", system)
        for task_type in ("BP", "CM", "SS", "AE"):
            self.assertIn(task_type, system)
        self.assertIn("issues 中的自然语言必须使用中文", system)

    def test_adjudicator_returns_items_and_review_queue(self):
        system, user = build_adjudicator_prompt("v1", [], [], [])

        self.assertIn("accepted_groups", system + user)
        self.assertIn("human_review_queue", system + user)
        self.assertIn("source_proposal_ids", system)
        self.assertNotIn("grounded_annotations", system)
        self.assertNotIn('"gold_value"', system)
        self.assertIn("不得重写", system)

    def test_prompt_payload_never_contains_performance_data(self):
        _, user_blocks = build_evidence_extractor_prompt(
            "v1",
            {"performance_data": {"likes": 99}, "C3_text_language": {"title": "visible"}},
        )

        self.assertNotIn("likes", str(user_blocks))
        self.assertNotIn("performance_data", str(user_blocks))

    def test_parser_rejects_markdown_or_missing_top_level_key(self):
        with self.assertRaises(ModelOutputError):
            parse_json_object("```json\n{\"items\": []}\n```", "items")

        with self.assertRaises(ModelOutputError):
            parse_json_object("{\"items\": []}", "grounded_annotations")

    def test_parse_proposal_response_returns_items_and_abstentions(self):
        proposals, abstentions = parse_proposal_response(
            "{\"proposals\": [{\"proposal_id\": \"p1\"}], \"abstentions\": [{\"task_type\": \"AE\"}]}"
        )

        self.assertEqual(proposals[0]["proposal_id"], "p1")
        self.assertEqual(abstentions[0]["task_type"], "AE")

    def test_parse_adjudication_response_accepts_v7_decision_groups(self):
        groups, queue = parse_adjudication_response(
            '{"accepted_groups":[{"source_proposal_ids":["p1"],"reason":"证据充分"}],'
            '"human_review_queue":[]}'
        )

        self.assertEqual(groups[0]["source_proposal_ids"], ["p1"])
        self.assertTrue(groups[0]["_decision_only"])
        self.assertEqual(queue, [])


if __name__ == "__main__":
    unittest.main()
