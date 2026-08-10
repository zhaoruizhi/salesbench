"""Theory provenance and deterministic endpoint rules for commerce graphs."""

from __future__ import annotations

from .commerce_schema import CueType, RelationProvenance, RelationType


CLAIM_CUES = {
    CueType.FUNCTION_CLAIM,
    CueType.EFFECT_CLAIM,
    CueType.PRICE_CLAIM,
    CueType.FIT_CLAIM,
    CueType.EXPERIENCE_REVIEW,
}

DEMONSTRATION_CUES = {
    CueType.PROCESS_DEMONSTRATION,
    CueType.OUTCOME_DISPLAY,
    CueType.BEFORE_AFTER,
    CueType.VICARIOUS_TRIAL,
}

OFFER_CUES = {
    CueType.PRICE,
    CueType.DISCOUNT,
    CueType.GIFT,
    CueType.OFFER_CONDITION,
    CueType.SERVICE_GUARANTEE,
}

CONCERN_CUES = {
    CueType.OBJECTION,
    CueType.FIT_CONSTRAINT,
    CueType.USAGE_DIFFICULTY,
    CueType.PRICE_CONCERN,
    CueType.RISK_CONCERN,
}

SOLUTION_CUES = {
    CueType.PRODUCT_IDENTITY,
    CueType.PRODUCT_ATTRIBUTE,
    CueType.PRODUCT_DESCRIPTION,
    CueType.BENEFIT,
    CueType.SERVICE_GUARANTEE,
    CueType.USAGE_SCENARIO,
    *DEMONSTRATION_CUES,
}


RELATION_RULES: dict[RelationType, dict[str, object]] = {
    RelationType.DESCRIPTION_REFERS_TO_PRODUCT: {
        "source": {CueType.PRODUCT_DESCRIPTION, CueType.PRODUCT_ATTRIBUTE, CueType.PRODUCT_VARIANT},
        "target": {CueType.PRODUCT_IDENTITY},
        "provenance": RelationProvenance.THEORY_DIRECT,
        "min_evidence": 1,
    },
    RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION: {
        "source": CLAIM_CUES,
        "target": DEMONSTRATION_CUES,
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CLAIM_REPEATED_ACROSS_MODALITIES: {
        "source": CLAIM_CUES,
        "target": CLAIM_CUES,
        "provenance": RelationProvenance.BENCHMARK_OPERATIONAL,
        "min_evidence": 2,
    },
    RelationType.CLAIM_PARTIALLY_SUPPORTED: {
        "source": CLAIM_CUES,
        "target": DEMONSTRATION_CUES,
        "provenance": RelationProvenance.BENCHMARK_OPERATIONAL,
        "min_evidence": 2,
    },
    RelationType.CLAIM_CONTRADICTED: {
        "source": CLAIM_CUES,
        "target": {
            CueType.PRODUCT_ATTRIBUTE,
            CueType.PRODUCT_VARIANT,
            CueType.QUANTITY,
            CueType.BUNDLE,
            *OFFER_CUES,
            *DEMONSTRATION_CUES,
        },
        "provenance": RelationProvenance.BENCHMARK_OPERATIONAL,
        "min_evidence": 2,
    },
    RelationType.CLAIM_TEMPORALLY_MISALIGNED: {
        "source": CLAIM_CUES,
        "target": DEMONSTRATION_CUES,
        "provenance": RelationProvenance.BENCHMARK_OPERATIONAL,
        "min_evidence": 2,
    },
    RelationType.FEATURE_FRAMED_AS_BENEFIT: {
        "source": {CueType.PRODUCT_ATTRIBUTE, CueType.PRODUCT_DESCRIPTION},
        "target": {CueType.BENEFIT},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 1,
    },
    RelationType.DEMONSTRATION_SHOWS_STATE_CHANGE: {
        "source": {CueType.PROCESS_DEMONSTRATION, CueType.BEFORE_AFTER},
        "target": {CueType.OUTCOME_DISPLAY, CueType.BEFORE_AFTER},
        "provenance": RelationProvenance.THEORY_DIRECT,
        "min_evidence": 2,
    },
    RelationType.PROBLEM_ADDRESSED_BY_SOLUTION: {
        "source": {CueType.PAIN_POINT, CueType.NEED},
        "target": SOLUTION_CUES,
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.OBJECTION_RESPONDED_BY_CUE: {
        "source": CONCERN_CUES,
        "target": SOLUTION_CUES | OFFER_CUES | {CueType.CREDIBILITY_SIGNAL, CueType.COMPARISON_ANCHOR},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CONTENT_ADDRESSES_FIT_UNCERTAINTY: {
        "source": {CueType.FIT_CONSTRAINT, CueType.OBJECTION, CueType.FIT_CLAIM},
        "target": {
            CueType.PRODUCT_ATTRIBUTE,
            CueType.PRODUCT_VARIANT,
            CueType.VICARIOUS_TRIAL,
            CueType.USAGE_SCENARIO,
            CueType.PROCESS_DEMONSTRATION,
        },
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CONTENT_ADDRESSES_USAGE_UNCERTAINTY: {
        "source": {CueType.USAGE_DIFFICULTY, CueType.OBJECTION, CueType.NEED},
        "target": {CueType.PROCESS_DEMONSTRATION, CueType.USAGE_SCENARIO, CueType.PRODUCT_DESCRIPTION},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CONTENT_ADDRESSES_PRICE_UNCERTAINTY: {
        "source": {CueType.PRICE_CONCERN, CueType.OBJECTION, CueType.NEED},
        "target": {CueType.PRICE, CueType.DISCOUNT, CueType.BUNDLE, CueType.QUANTITY, CueType.OFFER_CONDITION},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.OFFER_REQUIRES_CONDITION: {
        "source": {CueType.PRICE, CueType.DISCOUNT, CueType.GIFT},
        "target": {CueType.OFFER_CONDITION},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CONTENT_PRECEDES_CTA: {
        "source": {
            CueType.PAIN_POINT,
            CueType.NEED,
            CueType.BENEFIT,
            CueType.PRODUCT_DESCRIPTION,
            CueType.COMPARISON_ANCHOR,
            CueType.CREDIBILITY_SIGNAL,
            *OFFER_CUES,
            *DEMONSTRATION_CUES,
        },
        "target": {CueType.CTA},
        "provenance": RelationProvenance.BENCHMARK_OPERATIONAL,
        "min_evidence": 2,
    },
}


def relation_rule(relation_type: RelationType | str) -> dict[str, object]:
    relation = (
        relation_type
        if isinstance(relation_type, RelationType)
        else RelationType(str(relation_type).strip().upper())
    )
    return RELATION_RULES[relation]
