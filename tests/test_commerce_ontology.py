from __future__ import annotations

import sys

sys.path.insert(0, "src")

from salesbench.goldbank.commerce_ontology import RELATION_RULES  # noqa: E402
from salesbench.goldbank.commerce_schema import (  # noqa: E402
    CommerceCue,
    CommercialRelation,
    CueType,
    RelationProvenance,
    RelationType,
    make_cue_id,
    make_relation_id,
)
from salesbench.goldbank.schema import EvidenceModality, EvidenceUnit  # noqa: E402
from salesbench.goldbank.validators import (  # noqa: E402
    validate_commerce_cue,
    validate_commercial_relation,
)


def _evidence(evidence_id: str, modality: EvidenceModality = EvidenceModality.VISUAL) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        video_id="v1",
        modality=modality,
        start_s=0.0,
        end_s=1.0,
        frame_indices=(0,) if modality == EvidenceModality.VISUAL else (),
        text_span="这款产品只要九块九" if modality == EvidenceModality.ASR else "",
        subject="speaker" if modality == EvidenceModality.ASR else "product",
        predicate="states a price" if modality == EvidenceModality.ASR else "is displayed",
        value="9.9 yuan" if modality == EvidenceModality.ASR else "visible",
        attributes={},
        source_domains=("C2_audio_speech",) if modality == EvidenceModality.ASR else ("C6_raw_video",),
        extractor="test",
        confidence=0.95,
        timestamp_status="available",
        content_en="The speaker states a price of 9.9 yuan."
        if modality == EvidenceModality.ASR
        else "The product is visible.",
    )


def _cue(cue_type: CueType, cue_id: str, evidence_ids: tuple[str, ...], content: str) -> CommerceCue:
    return CommerceCue(
        cue_id=cue_id,
        video_id="v1",
        cue_type=cue_type,
        content_en=content,
        source_text_native="",
        evidence_ids=evidence_ids,
        attributes={},
        directness="DIRECT",
        theory_tags=("test",),
        extractor="test",
        confidence=0.9,
    )


def test_relation_catalog_excludes_consumer_outcomes():
    values = {item.value for item in RelationType}
    assert "INCREASES_TRUST" not in values
    assert "CAUSES_PURCHASE" not in values
    assert "IMPROVES_CONVERSION" not in values
    assert "CONTENT_ADDRESSES_FIT_UNCERTAINTY" in values
    assert set(RELATION_RULES) == set(RelationType)


def test_valid_offer_condition_relation_passes_deterministic_rules():
    price_evidence = _evidence("e_price", EvidenceModality.ASR)
    condition_evidence = _evidence("e_condition", EvidenceModality.ASR)
    price = _cue(CueType.PRICE, "cue_price", ("e_price",), "The offer price is 9.9 yuan.")
    condition = _cue(
        CueType.OFFER_CONDITION,
        "cue_condition",
        ("e_condition",),
        "The viewer must claim a coupon to receive the offer price.",
    )
    relation = CommercialRelation(
        relation_id=make_relation_id(
            "v1", RelationType.OFFER_REQUIRES_CONDITION, (price.cue_id,), (condition.cue_id,)
        ),
        video_id="v1",
        relation_type=RelationType.OFFER_REQUIRES_CONDITION,
        source_cue_ids=(price.cue_id,),
        target_cue_ids=(condition.cue_id,),
        evidence_ids=(price_evidence.evidence_id, condition_evidence.evidence_id),
        status="SUPPORTED",
        rationale_en="The stated price is explicitly conditional on claiming a coupon.",
        provenance=RelationProvenance.THEORY_OPERATIONALIZED,
        directness="DIRECT",
        extractor="test",
        confidence=0.9,
    )

    issues = validate_commercial_relation(
        relation,
        {price.cue_id: price, condition.cue_id: condition},
        {price_evidence.evidence_id: price_evidence, condition_evidence.evidence_id: condition_evidence},
    )

    assert not [issue for issue in issues if issue.severity == "ERROR"]


def test_relation_rejects_wrong_endpoint_type_and_consumer_outcome_language():
    first_evidence = _evidence("e1")
    second_evidence = _evidence("e2")
    product = _cue(CueType.PRODUCT_IDENTITY, "cue_product", ("e1",), "A cleaning spray is shown.")
    cta = _cue(CueType.CTA, "cue_cta", ("e2",), "The speaker asks viewers to buy now.")
    relation = CommercialRelation(
        relation_id=make_relation_id(
            "v1", RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION, (product.cue_id,), (cta.cue_id,)
        ),
        video_id="v1",
        relation_type=RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION,
        source_cue_ids=(product.cue_id,),
        target_cue_ids=(cta.cue_id,),
        evidence_ids=("e1", "e2"),
        status="SUPPORTED",
        rationale_en="This makes the viewer trust the host and purchase the product.",
        provenance=RelationProvenance.THEORY_OPERATIONALIZED,
        directness="INFERRED",
        extractor="test",
        confidence=0.9,
    )

    issues = validate_commercial_relation(
        relation,
        {product.cue_id: product, cta.cue_id: cta},
        {"e1": first_evidence, "e2": second_evidence},
    )
    codes = {issue.code for issue in issues}

    assert "INVALID_RELATION_SOURCE_TYPE" in codes
    assert "INVALID_RELATION_TARGET_TYPE" in codes
    assert "CONSUMER_OUTCOME_LEAK" in codes


def test_cue_rejects_non_english_canonical_semantics():
    evidence = _evidence("e1")
    cue = _cue(CueType.PRODUCT_IDENTITY, make_cue_id("v1", CueType.PRODUCT_IDENTITY, ("e1",), "产品"), ("e1",), "产品")

    issues = validate_commerce_cue(cue, {evidence.evidence_id: evidence})

    assert "NON_ENGLISH_CANONICAL_TEXT" in {issue.code for issue in issues}
