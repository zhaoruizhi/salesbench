from __future__ import annotations

import sys

sys.path.insert(0, "src")

from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CommerceCue,
    CommercialRelation,
    CueType,
    RelationProvenance,
    RelationType,
    make_cue_id,
    make_relation_id,
    parse_commerce_cue,
    parse_commercial_relation,
)
from salesbench.goldbank.schema import parse_evidence_unit  # noqa: E402


def test_commerce_contract_uses_english_semantics_and_stable_ids():
    cue_id = make_cue_id("v1", CueType.PRICE, ("e1",), "the displayed price is 9.9 yuan")
    cue = CommerceCue(
        cue_id=cue_id,
        video_id="v1",
        cue_type=CueType.PRICE,
        content_en="The displayed price is 9.9 yuan.",
        source_text_native="9.9元",
        evidence_ids=("e1",),
        attributes={"currency": "CNY", "amount": 9.9},
        directness="DIRECT",
        theory_tags=("offer_information",),
        extractor="fake",
        confidence=0.95,
    )

    assert parse_commerce_cue(cue.to_dict()) == cue
    assert make_cue_id("v1", CueType.PRICE, ("e1",), "the displayed price is 9.9 yuan") == cue_id
    assert "content_zh" not in cue.to_dict()


def test_relation_contract_round_trips_and_uses_stable_local_id():
    relation_id = make_relation_id(
        "v1",
        RelationType.OFFER_REQUIRES_CONDITION,
        ("cue_price",),
        ("cue_condition",),
    )
    relation = CommercialRelation(
        relation_id=relation_id,
        video_id="v1",
        relation_type=RelationType.OFFER_REQUIRES_CONDITION,
        source_cue_ids=("cue_price",),
        target_cue_ids=("cue_condition",),
        evidence_ids=("e1", "e2"),
        status="SUPPORTED",
        rationale_en="The lower price is explicitly conditional on claiming a coupon.",
        provenance=RelationProvenance.THEORY_OPERATIONALIZED,
        directness="DIRECT",
        extractor="fake",
        confidence=0.9,
    )

    assert parse_commercial_relation(relation.to_dict()) == relation
    assert relation.to_dict()["provenance"] == "THEORY_OPERATIONALIZED"
    assert "rationale_zh" not in relation.to_dict()


def test_v3_evidence_separates_english_semantics_from_native_source():
    unit = parse_evidence_unit(
        {
            "evidence_id": "e_asr",
            "video_id": "v1",
            "modality": "asr",
            "content_en": "The speaker claims that the product removes stains.",
            "source_text_native": "这个产品可以去除污渍",
            "subject": "speaker",
            "predicate": "claims",
            "value": "the product removes stains",
            "confidence": 0.9,
        }
    )

    serialized = unit.to_dict()
    assert unit.content_en.startswith("The speaker claims")
    assert unit.source_text_native == "这个产品可以去除污渍"
    assert unit.text_span == "这个产品可以去除污渍"
    assert "text_span" not in serialized
    assert "content_zh" not in serialized


def test_v2_evidence_reader_maps_text_span_without_writing_legacy_field():
    unit = parse_evidence_unit(
        {
            "evidence_id": "legacy_asr",
            "video_id": "v1",
            "modality": "asr",
            "text_span": "老版本原文",
            "subject": "speaker",
            "predicate": "states",
            "value": "a product attribute",
            "confidence": 0.8,
        }
    )

    assert unit.source_text_native == "老版本原文"
    assert unit.content_en == "speaker states a product attribute"
    assert "text_span" not in unit.to_dict()
