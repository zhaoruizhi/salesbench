from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CueType,
    RelationProvenance,
    RelationType,
)
from salesbench.goldbank.normalizer import (  # noqa: E402
    normalize_commerce_cues,
    normalize_commercial_relations,
    normalize_evidence_units,
    normalize_proposals,
    normalize_task_subtype,
)
from salesbench.goldbank.schema import EvidenceModality, GoldTaskType  # noqa: E402
from salesbench.goldbank.validators import validate_evidence_unit  # noqa: E402


class EvidenceNormalizerTest(unittest.TestCase):
    def test_reasoning_operator_in_subtype_field_maps_back_to_capability(self):
        self.assertEqual(
            normalize_task_subtype(GoldTaskType.SS, "MAP_FEATURE_TO_BENEFIT"),
            "FEATURE_BENEFIT",
        )

    def test_normalizes_common_vlm_visual_aliases_and_frame_locator(self):
        unit = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "image",
                    "evidence_id": "frame_006.jpg",
                    "subject": "product",
                    "predicate": "has color",
                    "value": "red",
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
                    "text_span": "这个产品只要十几元",
                    "subject": "speaker",
                    "predicate": "claims a price",
                    "value": "a little over ten yuan",
                    "confidence": "medium",
                },
                {
                    "modality": "text",
                    "evidence_id": "asr_subtitles",
                    "text_span": "适合厨房收纳",
                    "subject": "speaker",
                    "predicate": "claims a use",
                    "value": "kitchen storage",
                    "confidence": 90,
                },
            ],
        )

        self.assertTrue(all(unit.modality == EvidenceModality.ASR for unit in units))
        self.assertEqual([unit.text_span for unit in units], ["这个产品只要十几元", "适合厨房收纳"])
        self.assertEqual([unit.confidence for unit in units], [0.75, 0.9])
        self.assertEqual(len({unit.evidence_id for unit in units}), 2)
        self.assertTrue(all(not validate_evidence_unit(unit) for unit in units))
        self.assertEqual(units[0].source_text_native, "这个产品只要十几元")
        self.assertEqual(units[0].content_en, "speaker claims a price a little over ten yuan")
        self.assertNotIn("text_span", units[0].to_dict())

    def test_normalizes_commerce_graph_with_local_ids_and_local_provenance(self):
        evidence_units = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "asr",
                    "text_span": "现在领券九块九",
                    "subject": "speaker",
                    "predicate": "states",
                    "value": "the price is 9.9 yuan after claiming a coupon",
                    "confidence": 0.95,
                },
                {
                    "modality": "ocr",
                    "evidence_id": "frame_002_ocr",
                    "text_span": "领券9.9元",
                    "subject": "on-screen text",
                    "predicate": "states",
                    "value": "9.9 yuan after coupon",
                    "confidence": 0.95,
                },
            ],
        )
        evidence = {unit.evidence_id: unit for unit in evidence_units}
        price, condition = normalize_commerce_cues(
            "v1",
            [
                {
                    "cue_type": "PRICE",
                    "content_en": "The offer price is 9.9 yuan.",
                    "source_text_native": "9.9元",
                    "evidence_ids": [evidence_units[0].evidence_id],
                    "attributes": {"amount": 9.9, "currency": "CNY"},
                    "directness": "DIRECT",
                    "theory_tags": ["offer_information"],
                    "confidence": 0.9,
                },
                {
                    "cue_type": "OFFER_CONDITION",
                    "content_en": "The price requires claiming a coupon.",
                    "source_text_native": "领券",
                    "evidence_ids": [evidence_units[1].evidence_id],
                    "attributes": {},
                    "directness": "DIRECT",
                    "theory_tags": ["offer_information"],
                    "confidence": 0.9,
                },
            ],
            evidence,
        )
        relation = normalize_commercial_relations(
            "v1",
            [
                {
                    "relation_type": "OFFER_REQUIRES_CONDITION",
                    "source_cue_ids": [price.cue_id],
                    "target_cue_ids": [condition.cue_id],
                    # The model may repeat only one endpoint's evidence. Local
                    # normalization must still preserve the complete graph edge.
                    "evidence_ids": [evidence_units[0].evidence_id],
                    "status": "SUPPORTED",
                    "rationale_en": "The 9.9-yuan offer is explicitly conditional on claiming a coupon.",
                    "provenance": "BENCHMARK_OPERATIONAL",
                    "directness": "DIRECT",
                    "confidence": 0.9,
                }
            ],
            {price.cue_id: price, condition.cue_id: condition},
            evidence,
        )[0]

        self.assertEqual(price.cue_type, CueType.PRICE)
        self.assertTrue(price.cue_id.startswith("v1_cue_price_"))
        self.assertEqual(relation.relation_type, RelationType.OFFER_REQUIRES_CONDITION)
        self.assertEqual(relation.provenance, RelationProvenance.THEORY_OPERATIONALIZED)
        self.assertEqual(relation.evidence_ids, tuple(unit.evidence_id for unit in evidence_units))
        self.assertTrue(relation.relation_id.startswith("v1_relation_offer_requires_condition_"))

    def test_proposal_operator_is_assigned_from_the_local_ontology(self):
        proposal = normalize_proposals(
            "v1",
            "ss_proposer",
            [
                {
                    "task_type": "SS",
                    "task_subtype": "FEATURE_BENEFIT",
                    "capability": "OUTCOME_DISPLAY",
                    "reasoning_operator": "controlled operator",
                    "target": {"specific_focus": "The braided jacket is presented as durable."},
                    "proposed_gold": {"answer": "The material feature is framed as durability."},
                    "evidence_ids": ["e1", "e2"],
                    "reasoning_edges": [],
                    "question_intent": "Ask how the braided jacket is connected to durability.",
                    "proposal_confidence": 0.9,
                }
            ],
        )[0]

        self.assertEqual(proposal.reasoning_operator, "MAP_FEATURE_TO_BENEFIT")
        self.assertEqual(proposal.capability, "FEATURE_BENEFIT")

    def test_proposal_graph_nodes_are_resolved_to_underlying_evidence(self):
        evidence_units = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "asr",
                    "text_span": "支持快充",
                    "subject": "speaker",
                    "predicate": "claims",
                    "value": "the cable supports fast charging",
                    "confidence": 0.95,
                },
                {
                    "modality": "visual",
                    "evidence_id": "frame_003.jpg",
                    "subject": "phone",
                    "predicate": "shows",
                    "value": "a charging indicator",
                    "confidence": 0.95,
                },
            ],
        )
        evidence = {unit.evidence_id: unit for unit in evidence_units}
        claim, demonstration = normalize_commerce_cues(
            "v1",
            [
                {
                    "cue_type": "FUNCTION_CLAIM",
                    "content_en": "The speaker claims that the cable supports fast charging.",
                    "source_text_native": "支持快充",
                    "evidence_ids": [evidence_units[0].evidence_id],
                    "attributes": {},
                    "directness": "DIRECT",
                    "theory_tags": ["product_claim"],
                    "confidence": 0.9,
                },
                {
                    "cue_type": "PROCESS_DEMONSTRATION",
                    "content_en": "A phone displays a charging indicator while connected to the cable.",
                    "source_text_native": "",
                    "evidence_ids": [evidence_units[1].evidence_id],
                    "attributes": {},
                    "directness": "DIRECT",
                    "theory_tags": ["product_demonstration"],
                    "confidence": 0.9,
                },
            ],
            evidence,
        )
        cues = {cue.cue_id: cue for cue in (claim, demonstration)}
        relation = normalize_commercial_relations(
            "v1",
            [
                {
                    "relation_type": "CLAIM_SUPPORTED_BY_DEMONSTRATION",
                    "source_cue_ids": [claim.cue_id],
                    "target_cue_ids": [demonstration.cue_id],
                    "evidence_ids": [evidence_units[0].evidence_id],
                    "status": "SUPPORTED",
                    "rationale_en": "The visible charging state addresses the spoken charging claim.",
                    "directness": "INFERRED",
                    "confidence": 0.9,
                }
            ],
            cues,
            evidence,
        )[0]

        proposal = normalize_proposals(
            "v1",
            "cm_proposer",
            [
                {
                    "task_type": "CM",
                    "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                    "target": {"specific_focus": "The spoken fast-charging claim."},
                    "proposed_gold": {"answer": "The claim is paired with a visible charging state."},
                    "evidence_ids": [evidence_units[0].evidence_id, demonstration.cue_id],
                    "commerce_cue_ids": [claim.cue_id, demonstration.cue_id],
                    "commercial_relation_ids": [relation.relation_id],
                    "reasoning_edges": [],
                    "question_intent": "Ask how the visible phone state relates to the spoken claim.",
                    "proposal_confidence": 0.9,
                }
            ],
            cues,
            {relation.relation_id: relation},
            set(evidence),
        )[0]

        self.assertEqual(set(proposal.evidence_ids), set(evidence))
        self.assertNotIn(demonstration.cue_id, proposal.evidence_ids)
        self.assertTrue(proposal.reasoning_edges)
        self.assertTrue(all(len(edge) == 3 and edge[0] in evidence for edge in proposal.reasoning_edges))

    def test_proposal_rejects_copied_schema_placeholders(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            normalize_proposals(
                "v1",
                "ae_proposer",
                [
                    {
                        "task_type": "AE",
                        "task_subtype": "CONTENT_IMPLIED_NEED",
                        "target": {"specific_focus": "English content-specific focus"},
                        "proposed_gold": {"answer": "A concrete answer."},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [["e1", "specific supported claim", "SUPPORTED"]],
                        "question_intent": "Ask a specific question about the cited product, offer, claim, or sequence.",
                        "proposal_confidence": 0.9,
                    }
                ],
            )

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

    def test_proposer_cannot_emit_another_tasks_subtype(self):
        with self.assertRaisesRegex(ValueError, "ae_proposer cannot emit"):
            normalize_proposals(
                "v1",
                "ae_proposer",
                [
                    {
                        "task_type": "CM",
                        "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                        "target": {"claim": "claim"},
                        "proposed_gold": {"relation": "SUPPORTED"},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )

        with self.assertRaisesRegex(ValueError, "ae_proposer cannot emit"):
            normalize_proposals(
                "v1",
                "ae_proposer",
                [
                    {
                        "task_type": "SS",
                        "task_subtype": "FEATURE_BENEFIT",
                        "target": {"segment": "opening"},
                        "proposed_gold": {"label": "value proposition"},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )

        with self.assertRaisesRegex(ValueError, "cm_proposer cannot emit"):
            normalize_proposals(
                "v1",
                "cm_proposer",
                [
                    {
                        "task_type": "SS",
                        "task_subtype": "PROBLEM_SOLUTION",
                        "target": {"segment": "opening"},
                        "proposed_gold": {"label": "question hook"},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )

    def test_placeholder_proposal_id_is_replaced_locally(self):
        proposal = normalize_proposals(
            "v1",
            "ss_proposer",
            [
                {
                    "proposal_id": "optional",
                    "task_type": "SS",
                    "task_subtype": "PROBLEM_SOLUTION",
                    "target": {"segment": "opening"},
                    "proposed_gold": {"label": "result-first hook"},
                    "evidence_ids": ["e1", "e2"],
                    "reasoning_edges": [],
                    "proposal_confidence": 0.9,
                }
            ],
        )[0]

        self.assertTrue(proposal.proposal_id.startswith("v1_ss_proposer_ss_000_"))

    def test_normalized_evidence_rejects_cjk_outside_verbatim_text_span(self):
        unit = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "asr",
                    "text_span": "这款产品采用编织材质",
                    "subject": "口播者",
                    "predicate": "claims",
                    "value": "braided material",
                    "confidence": 0.9,
                }
            ],
        )[0]

        issues = validate_evidence_unit(unit)

        self.assertTrue(any(issue.code == "NON_ENGLISH_NORMALIZED_TEXT" for issue in issues))

    def test_normalized_evidence_rejects_cjk_content_en_and_creator_handle(self):
        unit = normalize_evidence_units(
            "v1",
            [
                {
                    "modality": "ocr",
                    "evidence_id": "frame_003_ocr",
                    "frame_indices": [3],
                    "source_text_native": "抖音号: demo123",
                    "subject": "on-screen watermark",
                    "predicate": "shows",
                    "value": "creator handle demo123",
                    "content_en": "The watermark reads 抖音号: demo123.",
                    "confidence": 0.9,
                }
            ],
        )[0]

        codes = {issue.code for issue in validate_evidence_unit(unit)}

        self.assertIn("NON_ENGLISH_NORMALIZED_TEXT", codes)
        self.assertIn("CREATOR_METADATA_LEAK", codes)

    def test_normalized_proposal_rejects_cjk_natural_language(self):
        with self.assertRaisesRegex(ValueError, "English"):
            normalize_proposals(
                "v1",
                "ss_proposer",
                [
                    {
                        "task_type": "SS",
                    "task_subtype": "PROBLEM_SOLUTION",
                        "target": {"segment": "开场", "mechanism": "result first"},
                        "proposed_gold": {"label": "result-first", "answer": "The opening shows the result."},
                        "evidence_ids": ["e1", "e2"],
                        "reasoning_edges": [],
                        "proposal_confidence": 0.9,
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
