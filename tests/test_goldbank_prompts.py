from __future__ import annotations

import unittest
import sys

sys.path.insert(0, "src")

from salesbench.goldbank.parsing import (  # noqa: E402
    ModelOutputError,
    parse_adjudication_response,
    parse_commerce_cue_response,
    parse_commercial_relation_response,
    parse_json_object,
    parse_proposal_response,
)
from salesbench.goldbank.prompts import (  # noqa: E402
    BP_COMPILER_CONTRACT,
    PROMPT_VERSION,
    build_adjudicator_prompt,
    build_challenger_prompt,
    build_commerce_cue_prompt,
    build_commercial_relation_prompt,
    build_evidence_extractor_prompt,
    build_proposer_prompt,
)


class GoldBankPromptTest(unittest.TestCase):
    def test_evidence_extractor_requests_units_not_questions(self):
        system, user_blocks = build_evidence_extractor_prompt("v1", {"C3_text_language": {"title": "hello"}})
        text = system + " " + str(user_blocks)

        self.assertEqual(PROMPT_VERSION, "evidence-prompt-v9")
        self.assertIn("EvidenceUnit", text)
        self.assertIn("Do not generate questions", system)
        self.assertIn("English", system)
        self.assertIn("subject='speaker', predicate='claims'", system)
        self.assertIn("verbatim", system)
        self.assertIn("start_s", system)
        self.assertIn("end_s", system)
        self.assertIn("content_en", system)
        self.assertIn("source_text_native", system)
        self.assertNotIn("EvidenceUnit.text_span", system)
        self.assertNotRegex(system, r"[\u4e00-\u9fff]")

    def test_v9_commerce_prompts_are_english_and_forbid_outcome_claims(self):
        evidence = [{
            "evidence_id": "e1",
            "video_id": "v1",
            "modality": "visual",
            "content_en": "The host wipes a marked surface.",
            "source_text_native": "",
        }]
        cue_system, cue_user = build_commerce_cue_prompt("v1", evidence)
        relation_system, relation_user = build_commercial_relation_prompt(
            "v1",
            evidence,
            [{
                "cue_id": "cue1",
                "cue_type": "PROCESS_DEMONSTRATION",
                "content_en": "The host wipes a marked surface.",
                "evidence_ids": ["e1"],
            }],
        )
        combined = cue_system + cue_user + relation_system + relation_user

        self.assertNotRegex(combined, r"[\u4e00-\u9fff]")
        self.assertIn("commerce_cues", cue_system)
        self.assertIn("commercial_relations", relation_system)
        self.assertIn(
            "Never claim that a viewer trusted, purchased, converted, or became less uncertain",
            combined,
        )
        self.assertIn("CLAIM_REPEATED_ACROSS_MODALITIES", relation_system)
        self.assertIn("existing cue IDs", relation_system)

    def test_bp_has_an_explicit_deterministic_compiler_contract(self):
        self.assertIn("BP", BP_COMPILER_CONTRACT)
        self.assertIn("deterministic", BP_COMPILER_CONTRACT.lower())
        self.assertIn("EvidenceUnit", BP_COMPILER_CONTRACT)
        self.assertNotRegex(BP_COMPILER_CONTRACT, r"[\u4e00-\u9fff]")

    def test_all_proposers_must_reference_existing_evidence_ids(self):
        system, user = build_proposer_prompt(
            "cm_proposer",
            "v1",
            [{"evidence_id": "e1", "video_id": "v1", "subject": "claim"}],
        )

        self.assertIn("evidence_id", system + user)
        self.assertIn("e1", user)
        self.assertIn("abstentions", system + user)
        self.assertIn("at least two distinct evidence_ids", system)
        self.assertIn("proposal_confidence", system)
        self.assertIn("reasoning_edges", system)

    def test_proposer_contract_has_controlled_tasks_subtypes_and_exact_schema(self):
        ae, _ = build_proposer_prompt("ae_proposer", "v1", [])
        cm, _ = build_proposer_prompt("cm_proposer", "v1", [])
        ss, _ = build_proposer_prompt("ss_proposer", "v1", [])

        self.assertIn("AUDIENCE_NEED_FIT", ae)
        self.assertIn("CLAIM_EVIDENCE_RELATION", cm)
        self.assertIn("HOOK_MECHANISM", ss)
        self.assertIn('"task_type":"AE"', ae)
        self.assertIn('"task_type":"CM"', cm)
        self.assertIn('"task_type":"SS"', ss)
        self.assertNotIn('"task_type":"SS"', ae)
        self.assertNotIn('"task_type":"SS"', cm)
        self.assertNotIn('"task_type":"AE"', cm)
        self.assertNotIn('"task_type":"CM"', ss)
        self.assertNotIn('"task_type":"AE"', ss)
        for prompt in (ae, cm, ss):
            self.assertIn('"target"', prompt)
            self.assertIn('"proposed_gold"', prompt)
            self.assertIn("Do not output proposal_id", prompt)
            self.assertIn("English", prompt)
            self.assertNotRegex(prompt, r"[\u4e00-\u9fff]")
        self.assertIn("audience_need", ae)
        self.assertIn("usage_context", ae)
        self.assertIn("decision_state", ae)
        self.assertIn("content_motivation", ae)
        self.assertIn("must not claim a real audience profile", ae)
        self.assertIn("two distinct modalities", cm)
        self.assertIn("modality_pair", cm)
        self.assertIn("NOT_SHOWN", cm)
        self.assertIn("complete-video observation window", cm)

    def test_unknown_or_legacy_proposer_name_is_rejected(self):
        for name in ("consumer", "operator", "strategist", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unknown task generator"):
                build_proposer_prompt(name, "v1", [])

    def test_challenger_can_request_human_review_and_cannot_default_pass(self):
        system, user = build_challenger_prompt("v1", [], [])

        self.assertIn("HUMAN_REVIEW", system + user)
        self.assertIn("must not default", system + user)
        self.assertIn("every input proposal", system)
        self.assertIn("suggested_revision", system)
        for task_type in ("BP", "CM", "SS", "AE"):
            self.assertIn(task_type, system)
        self.assertIn("issues and suggested_revision must use English", system)
        self.assertNotRegex(system, r"[\u4e00-\u9fff]")

    def test_adjudicator_returns_items_and_review_queue(self):
        system, user = build_adjudicator_prompt("v1", [], [], [])

        self.assertIn("accepted_groups", system + user)
        self.assertIn("human_review_queue", system + user)
        self.assertIn("source_proposal_ids", system)
        self.assertNotIn("grounded_annotations", system)
        self.assertNotIn('"gold_value"', system)
        self.assertIn("must not rewrite", system)
        self.assertNotRegex(system, r"[\u4e00-\u9fff]")

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

    def test_parse_commerce_graph_responses_returns_items_and_abstentions(self):
        cues, cue_abstentions = parse_commerce_cue_response(
            '{"commerce_cues":[{"cue_type":"PRICE"}],"abstentions":[]}'
        )
        relations, relation_abstentions = parse_commercial_relation_response(
            '{"commercial_relations":[{"relation_type":"OFFER_REQUIRES_CONDITION"}],'
            '"abstentions":[{"reason":"No condition cue."}]}'
        )

        self.assertEqual(cues[0]["cue_type"], "PRICE")
        self.assertEqual(cue_abstentions, [])
        self.assertEqual(relations[0]["relation_type"], "OFFER_REQUIRES_CONDITION")
        self.assertEqual(relation_abstentions[0]["reason"], "No condition cue.")

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
