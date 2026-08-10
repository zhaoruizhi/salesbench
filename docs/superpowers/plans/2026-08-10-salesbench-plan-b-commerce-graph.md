# SalesBench Plan B Commerce Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade SalesBench from generic frame/ASR/OCR QA generation to an evidence-first benchmark of product-and-offer grounding, claim–evidence verification, sales-logic reconstruction, and need–objection–offer alignment for presenter-led e-commerce short videos.

**Architecture:** Preserve raw `EvidenceUnit` as the factual layer, add theory-grounded `CommerceCue` nodes and auditable `CommercialRelation` edges, then let four task planners create semantic `QuestionSpec` records. A separate English question realizer produces natural surface questions; local validation, human review, and a frozen task-aware LLM-as-Judge remain the final quality gates.

**Tech Stack:** Python 3.13, dataclasses and enums, JSON/JSONL, OpenAI-compatible multimodal/text APIs, pytest, local HTML/CSS/JavaScript audit workbench, ffmpeg/ffprobe.

## Global Constraints

- Public tasks remain exactly `BP`, `CM`, `SS`, and `AE`; compatibility IDs do not change.
- Public names become Product & Offer Grounding, Claim–Evidence Verification, Persuasion & Sales Logic, and Need–Objection–Offer Alignment.
- Public questions, Gold answers, generator prompts, and Judge prompts use English only.
- Chinese ASR/OCR remains verbatim in `content_native`; `content_en` is an auditable English normalization.
- `EvidenceUnit` contains directly localizable frame, OCR, or ASR facts only; marketing interpretation must not be written back into raw evidence.
- `CommerceCue` may summarize an observable commercial presentation cue but must cite one or more valid `EvidenceUnit` IDs from the same video.
- `CommercialRelation` may connect validated cues but must cite the supporting EvidenceUnits and declare whether it is direct, bounded inference, or needs review.
- Never emit causal outcome relations such as `INCREASES_TRUST`, `CAUSES_PURCHASE`, `IMPROVES_CONVERSION`, or `REDUCES_INTERACTION_RISK`.
- Interaction counts, follower counts, titles, creator metadata, and private analysis fields never enter Evidence, QA, model payloads, Gold, or Judge payloads.
- Do not force every video to generate all four tasks; balance is enforced across the cohort.
- The primary evaluation remains a frozen five-level LLM-as-Judge; no separate deterministic or key-fact scoring engine is required for v9.
- Smoke and formal outputs are physically separate and never overwrite v6/v7/v8 artifacts.
- New versions are `evidence-dataset-schema-v3`, `evidence-prompt-v9`, `evidence-first-pipeline-v8`, `evidence-qa-compiler-v5`, and `judge-prompt-v4`.
- Every implementation task ends in focused tests and a Git commit.

---

## 1. Literature provenance and what the relation labels mean

The commercial ontology is a benchmark operationalization informed by literature; it is not copied verbatim from one paper. Every cue and relation must record one of three provenance classes:

- `THEORY_DIRECT`: the source literature explicitly studies the content construct, such as product description, product demonstration, vicarious trial, process presentation, outcome presentation, or product uncertainty.
- `THEORY_OPERATIONALIZED`: SalesBench converts a theory construct into a video-annotation edge, such as linking a demonstrated feature to a stated benefit.
- `BENCHMARK_OPERATIONAL`: the edge is required to test multimodal grounding or temporal reasoning, such as claim repetition versus independent visual support.

### 1.1 Primary domain sources

1. Guo et al., *Analyzing and Predicting Consumer Response to Short Videos in E-Commerce*, ACM TMIS 2024, identifies product description, product demonstration, pleasure, and aesthetics in 23,001 Taobao e-commerce short videos: <https://doi.org/10.1145/3690393>.
2. *Process Reveal or Product Display?*, Journal of Retailing and Consumer Services 2026, separates process-oriented and outcome-oriented short-video product presentation: <https://doi.org/10.1016/j.jretconser.2026.104827>.
3. *The Effects of Mini-detail Short Videos on Consumer Purchase Intention on Taobao*, Entertainment Computing 2024, motivates product details, functions, specific usage, application scenarios, multimedia presentation, and virtual experience: <https://doi.org/10.1016/j.entcom.2024.100745>.
4. Lu and Chen, *Live Streaming Commerce and Consumers' Purchase Intention: An Uncertainty Reduction Perspective*, Information & Management 2021, supplies vicarious product trial, product-fit uncertainty, and product/social signal distinctions: <https://doi.org/10.1016/j.im.2021.103509>.
5. *What Reduces Product Uncertainty in Live Streaming E-Commerce?*, Journal of Retailing and Consumer Services 2023, motivates anchor–product and content–product signal consistency: <https://doi.org/10.1016/j.jretconser.2023.103441>.
6. *What Drives Taobao Live Streaming Commerce?*, Heliyon 2022, supplies source credibility, presenter–product congruence, and parasocial constructs; SalesBench uses only their observable in-video cues: <https://doi.org/10.1016/j.heliyon.2022.e09676>.
7. *Let TikTokers Talk About Products*, Journal of Interactive Advertising 2026, distinguishes experiential verbal review from visual product demonstration in short-form influencer marketing: <https://doi.org/10.1080/15252019.2026.2642022>.
8. E-VAds supplies e-commerce-video tasks for basic perception, cross-modal detection, marketing logic, consumer insight, and evidence-grounded QA: <https://arxiv.org/html/2602.08355>.

### 1.2 Corrected cue and relation boundary

The earlier draft mixed presentation events and actual graph edges. V9 corrects that boundary:

| Earlier label | V9 representation | Why |
| --- | --- | --- |
| `DESCRIBES_PRODUCT` | `PRODUCT_DESCRIPTION` cue | A description is an observable content unit before it is linked to a product entity. |
| `DEMONSTRATES_CLAIM` | `CLAIM_SUPPORTED_BY_DEMONSTRATION` relation | The relation connects a claim cue to a separate demonstration cue. |
| `REPEATS_CLAIM` | `CLAIM_REPEATED_ACROSS_MODALITIES` relation | Repetition is not independent proof and must remain distinguishable from support. |
| `PARTIALLY_SUPPORTS` | `CLAIM_PARTIALLY_SUPPORTED` relation | Only part of a composite claim is shown. |
| `CONTRADICTS` | `CLAIM_CONTRADICTED` relation | A cited cue conflicts with a cited claim. |
| `FRAMES_FEATURE_AS_BENEFIT` | `FEATURE_FRAMED_AS_BENEFIT` relation | Connects a concrete feature to the benefit asserted in speech/OCR. |
| `PRESENTS_PROBLEM` | `PAIN_POINT` cue | The problem itself is a node. |
| `ADDRESSES_PROBLEM` | `PROBLEM_ADDRESSED_BY_SOLUTION` relation | Connects a problem cue to a solution/demo cue. |
| `RAISES_OBJECTION` | `OBJECTION` cue | The objection itself is a node. |
| `RESPONDS_TO_OBJECTION` | `OBJECTION_RESPONDED_BY_CUE` relation | Connects the objection to the response, guarantee, comparison, or demo. |
| `REDUCES_FIT_UNCERTAINTY` | `CONTENT_ADDRESSES_FIT_UNCERTAINTY` relation | The video can address uncertainty; it cannot prove the viewer's uncertainty was reduced. |
| `REDUCES_USAGE_UNCERTAINTY` | `CONTENT_ADDRESSES_USAGE_UNCERTAINTY` relation | Avoids a causal consumer-outcome claim. |
| `REDUCES_PRICE_UNCERTAINTY` | `CONTENT_ADDRESSES_PRICE_UNCERTAINTY` relation | Avoids a causal consumer-outcome claim. |
| `ESTABLISHES_OFFER_CONDITION` | `OFFER_REQUIRES_CONDITION` relation | Connects an offer to the quantity, coupon, time, membership, or action condition. |
| `BUILDS_CREDIBILITY_SIGNAL` | `CREDIBILITY_SIGNAL` cue | The video presents a signal; actual credibility is not observable. |
| `PRECEDES_ACTION_PROMPT` | `CONTENT_PRECEDES_CTA` relation | A benchmark-operational temporal edge, not a marketing-outcome claim. |

### 1.3 Final relation catalog

| Relation | Meaning and acceptance rule | Provenance | Main task |
| --- | --- | --- | --- |
| `DESCRIPTION_REFERS_TO_PRODUCT` | Description/attribute cue explicitly refers to the grounded product or variant. | `THEORY_DIRECT` | BP |
| `CLAIM_SUPPORTED_BY_DEMONSTRATION` | A distinct visual demonstration shows the material part of a spoken/OCR claim. Repeating the claim in another modality is insufficient. | `THEORY_OPERATIONALIZED` | CM |
| `CLAIM_REPEATED_ACROSS_MODALITIES` | Speech and OCR/visual text repeat semantically equivalent promotional wording without independent demonstration. | `BENCHMARK_OPERATIONAL` | CM |
| `CLAIM_PARTIALLY_SUPPORTED` | Evidence supports a separable subset of a composite claim and leaves another subset unshown. | `BENCHMARK_OPERATIONAL` | CM |
| `CLAIM_CONTRADICTED` | A localized cue is incompatible with the claim under the same product, condition, and time window. | `BENCHMARK_OPERATIONAL` | CM |
| `CLAIM_TEMPORALLY_MISALIGNED` | Claim and proposed evidence refer to different temporal stages or product states. | `BENCHMARK_OPERATIONAL` | CM |
| `FEATURE_FRAMED_AS_BENEFIT` | Speech/OCR explicitly connects a visible or described feature to a buyer-facing benefit. | `THEORY_OPERATIONALIZED` | SS |
| `DEMONSTRATION_SHOWS_STATE_CHANGE` | Before/process/after cues establish an observable state change. | `THEORY_DIRECT` | BP/SS |
| `PROBLEM_ADDRESSED_BY_SOLUTION` | The content presents a problem and subsequently presents the product, feature, or demo as its response. | `THEORY_OPERATIONALIZED` | SS |
| `OBJECTION_RESPONDED_BY_CUE` | An explicit concern is followed by a relevant explanation, demo, comparison, guarantee, or service cue. | `THEORY_OPERATIONALIZED` | SS/AE |
| `CONTENT_ADDRESSES_FIT_UNCERTAINTY` | Try-on, dimensions, physical comparison, or stated constraints clarify whether the product fits a represented use/person/object. | `THEORY_OPERATIONALIZED` | AE |
| `CONTENT_ADDRESSES_USAGE_UNCERTAINTY` | Instructions or process demonstration clarify how, where, or how difficult the product is to use. | `THEORY_OPERATIONALIZED` | AE |
| `CONTENT_ADDRESSES_PRICE_UNCERTAINTY` | Price composition, bundle quantity, discount comparison, or conditions clarify what the buyer receives and pays. | `THEORY_OPERATIONALIZED` | BP/AE |
| `OFFER_REQUIRES_CONDITION` | A price, gift, or discount depends on a localized and explicit condition. | `THEORY_OPERATIONALIZED` | BP/AE |
| `CONTENT_PRECEDES_CTA` | A cited hook, problem, demo, offer, or guarantee occurs before a cited purchase/action prompt. | `BENCHMARK_OPERATIONAL` | SS |

The schema must reject any relation that silently changes “content addresses X” into “consumer becomes less uncertain,” “consumer trusts the host,” or “consumer purchases.”

---

## 2. CommerceCue catalog

`CommerceCue` is a grounded semantic node, not a consumer-effect label.

| Family | Cue types | Observable boundary |
| --- | --- | --- |
| Product | `PRODUCT_IDENTITY`, `PRODUCT_ATTRIBUTE`, `PRODUCT_VARIANT`, `QUANTITY`, `BUNDLE` | Product, variant, attribute, count, or bundle visible/stated in supplied evidence. |
| Offer | `PRICE`, `DISCOUNT`, `GIFT`, `OFFER_CONDITION`, `SERVICE_GUARANTEE` | Exact offer and its explicit conditions; no inferred market value. |
| Presentation | `PRODUCT_DESCRIPTION`, `PROCESS_DEMONSTRATION`, `OUTCOME_DISPLAY`, `BEFORE_AFTER`, `VICARIOUS_TRIAL`, `USAGE_SCENARIO` | What the presenter says/shows and whether process or outcome is visible. |
| Claim | `FUNCTION_CLAIM`, `EFFECT_CLAIM`, `PRICE_CLAIM`, `FIT_CLAIM`, `EXPERIENCE_REVIEW` | A statement remains a claim until separately demonstrated. |
| Need and barrier | `PAIN_POINT`, `NEED`, `OBJECTION`, `FIT_CONSTRAINT`, `USAGE_DIFFICULTY`, `PRICE_CONCERN`, `RISK_CONCERN` | Explicitly represented problem, question, limitation, or concern. |
| Sales signal | `BENEFIT`, `COMPARISON_ANCHOR`, `CREDIBILITY_SIGNAL`, `SOCIAL_PROOF`, `SCARCITY`, `URGENCY`, `CTA` | Observable presentation cue only; no claim that it changes behavior. |

Every cue stores `content_en`, optional `content_native`, `evidence_refs`, `attributes`, `directness`, `confidence`, and `theory_tags`.

---

## 3. How the four tasks consume evidence, cues, and relations

The four task layers do not duplicate evidence extraction. Their inputs form a hierarchy:

```text
BP: EvidenceUnit -> CommerceCue, with optional single relation
CM: two or more EvidenceUnits -> two or more CommerceCues -> one claim relation
SS: multiple CommerceCues -> one relation or an ordered relation path
AE: represented need/constraint/objection cues -> bounded alignment relation path
```

### 3.1 BP — Product & Offer Grounding

| Sub-capability | What it asks | Required source |
| --- | --- | --- |
| `PRODUCT_IDENTITY` | What product is being presented? | Product cue plus visual/ASR/OCR grounding. |
| `ATTRIBUTE_AND_VARIANT` | Which material, size, flavor, model, color, or variant is specified? | Attribute/variant cue linked to the product. |
| `QUANTITY_AND_BUNDLE` | How many units or what bundle is included? | Quantity/bundle cue; reconcile OCR and speech when both exist. |
| `PRICE_AND_DISCOUNT` | What price or explicit discount is presented? | Price/discount cue; do not infer savings without an explicit comparator. |
| `OFFER_CONDITION` | What condition must hold for the offer? | Offer cue plus `OFFER_REQUIRES_CONDITION`. |
| `USAGE_STEP` | What observable step does the presenter perform? | Process-demonstration cue with localized frames. |
| `DEMONSTRATED_STATE_CHANGE` | What visible before/after change occurs? | `DEMONSTRATION_SHOWS_STATE_CHANGE`. |
| `USAGE_SCENARIO` | In what represented scenario is the product used? | Usage-scenario cue; no real-audience inference. |

BP is primarily cue-level. It should no longer compile every raw EvidenceUnit into a generic fact question.

### 3.2 CM — Claim–Evidence Verification

| Sub-capability | What it tests | Required relation |
| --- | --- | --- |
| `SPEECH_VISUAL_COREFERENCE` | Whether speech and frames refer to the same product, part, or action. | Grounded cue references under one observation window. |
| `OCR_SPEECH_OFFER_ALIGNMENT` | Whether spoken and displayed offer information agree. | Offer cues from two modalities. |
| `CLAIM_DEMONSTRATION_STATUS` | Whether a claim is demonstrated, repeated, partially supported, contradicted, or unshown in the available window. | One controlled claim relation. |
| `REPETITION_VS_INDEPENDENT_EVIDENCE` | Whether a second modality adds proof or merely repeats promotional language. | `CLAIM_SUPPORTED_BY_DEMONSTRATION` or `CLAIM_REPEATED_ACROSS_MODALITIES`. |
| `PARTIAL_SUPPORT` | Which part of a composite claim is supported and which is not. | `CLAIM_PARTIALLY_SUPPORTED`. |
| `CONTRADICTION` | What localized evidence conflicts with the claim. | `CLAIM_CONTRADICTED`. |
| `TEMPORAL_MISALIGNMENT` | Whether claim and evidence concern different stages/states. | `CLAIM_TEMPORALLY_MISALIGNED`. |
| `NOT_DEMONSTRATED` | What is stated but not demonstrated in a complete observation window. | Complete-window metadata plus absence rule; sampled-frame absence alone is insufficient. |

CM is relation-level and requires a genuine information gap across modalities.

### 3.3 SS — Persuasion & Sales Logic

| Sub-capability | What it reconstructs | Cue/relation path |
| --- | --- | --- |
| `PROBLEM_SOLUTION` | How the content moves from a represented problem to a product response. | Pain-point cue -> `PROBLEM_ADDRESSED_BY_SOLUTION`. |
| `FEATURE_BENEFIT` | How a feature is explicitly presented as useful to the buyer. | Attribute cue -> `FEATURE_FRAMED_AS_BENEFIT` -> benefit cue. |
| `PROCESS_DEMONSTRATION` | Why showing the process matters to the sales explanation. | Process cue plus steps and result; no effectiveness prediction. |
| `OUTCOME_DISPLAY` | What result is foregrounded and how it relates to the offer/claim. | Outcome cue plus claim or sequence relation. |
| `BEFORE_AFTER_COMPARISON` | What changed and what comparison the video asks the viewer to make. | Before/after cues -> state-change relation. |
| `VICARIOUS_TRIAL` | What the presenter tries on/uses/tastes on behalf of the viewer. | Vicarious-trial cue plus fit/use evidence. |
| `PRICE_VALUE_FRAMING` | How quantity, feature, or service is used to contextualize price. | Price cue plus bundle/benefit/service cue. |
| `REFERENCE_PRICE_ANCHORING` | Which explicit comparator is presented before/with the offer. | Comparison-anchor cue plus price cue. |
| `CREDIBILITY_SIGNAL` | What in-video expertise, first-person use, guarantee, or limitation signal is presented. | Credibility-signal cue; never “the host is trustworthy.” |
| `SOCIAL_PROOF` | What reviews, counts, testimonials, or quoted users are shown inside the video. | Social-proof cue; platform snapshot interactions remain private and forbidden. |
| `LIMITATION_DISCLOSURE` | What limitation or applicable boundary the presenter acknowledges. | Objection/constraint cue plus response. |
| `OBJECTION_HANDLING` | How an explicit buyer concern is addressed. | `OBJECTION_RESPONDED_BY_CUE`. |
| `SCARCITY_AND_URGENCY` | What time, stock, or price-window cue is presented. | Scarcity/urgency cue plus offer. |
| `CTA_SEQUENCE` | What content precedes the call to action. | One or more `CONTENT_PRECEDES_CTA` edges. |

SS is relation/path-level. Questions must refer to the concrete claim, demonstration, comparison, or sequence rather than ask for an abstract “mechanism.”

### 3.4 AE — Need–Objection–Offer Alignment

| Sub-capability | What it infers within bounds | Cue/relation path |
| --- | --- | --- |
| `CONTENT_IMPLIED_NEED` | What need is represented by the demonstrated problem/use, without asserting a real audience profile. | Need/pain-point cue plus product response. |
| `USAGE_CONTEXT` | Where or when the content presents the product being used. | Usage-scenario cue. |
| `FIT_CONSTRAINT` | What size, body, object, compatibility, or environment constraint is made visible/stated. | Fit-constraint cue. |
| `QUALITY_UNCERTAINTY` | What represented quality concern the demo/comparison attempts to address. | Objection cue plus response relation. |
| `USAGE_UNCERTAINTY` | What “how to use/how difficult” question the process clarifies. | `CONTENT_ADDRESSES_USAGE_UNCERTAINTY`. |
| `PRICE_UNCERTAINTY` | What price, bundle, or condition ambiguity is clarified. | `CONTENT_ADDRESSES_PRICE_UNCERTAINTY`. |
| `SERVICE_OR_RISK_CONCERN` | What guarantee, return, safety, or service concern is addressed in-video. | Risk concern plus guarantee/response cue. |
| `DECISION_BARRIER` | What explicit obstacle to considering the offer is represented. | Objection/constraint plus response. |
| `OFFER_NEED_ALIGNMENT` | How the stated offer conditions or bundle correspond to the represented use/need. | Need cue -> offer/condition relation path. |

AE is bounded interpretation. It must not infer actual viewer demographics, purchase intention, conversion, or popularity.

---

## 4. Target artifact flow

```text
frames + ASR + OCR
  -> evidence_units.jsonl
  -> commerce_cues.jsonl
  -> commercial_relations.jsonl
  -> video_evidence_dataset.jsonl
  -> Evidence/Cue/Relation human review
  -> video_evidence_dataset_reviewed.jsonl
  -> qa_specs.jsonl
  -> qa_realizations.jsonl
  -> QA human review
  -> vqa_gold_private.jsonl + vqa_public.jsonl
  -> predictions.jsonl
  -> LLM-as-Judge details + four-task macro-average
```

Formal delivery contains:

- EvidenceDataset v3: atomic evidence, cues, relations, reviewed grounded annotations.
- Prompt manifest: exact extractor, cue builder, relation builder, task planner, realizer, Challenger, Adjudicator, and Judge prompts.
- QA: public English questions and private Gold/evidence references.
- Evaluation: predictions, per-item Judge decisions, error tags, per-task metrics, macro-average, and human calibration summary.

---

## 5. File structure

### Create

- `src/salesbench/goldbank/commerce_schema.py`: cue/relation enums, dataclasses, parsers, stable IDs.
- `src/salesbench/goldbank/commerce_ontology.py`: cue catalog, relation catalog, provenance, and allowed endpoint types.
- `src/salesbench/vqa/specs.py`: `QuestionSpec` and `QuestionRealization` contracts.
- `src/salesbench/vqa/prompts.py`: English question-realizer prompt.
- `src/salesbench/vqa/realizer.py`: resumable LLM realization runner and response parser.
- `tests/test_commerce_schema.py`: round-trip and stable-ID tests.
- `tests/test_commerce_ontology.py`: relation endpoint and causal-label firewall tests.
- `tests/test_vqa_specs.py`: semantic question contract tests.
- `tests/test_vqa_realizer.py`: natural English generation and failure tests.
- `configs/evidence_smoke_v9_5videos.json`: v9 smoke cohort fingerprint.
- `configs/evidence_pilot_v9_64videos.json`: v9 formal pilot fingerprint.

### Modify

- `src/salesbench/goldbank/schema.py`: v3 record references and explicit task semantics.
- `src/salesbench/goldbank/ontology.py`: replace generic v8 subtypes with the four Plan B capability sets.
- `src/salesbench/goldbank/prompts.py`: add cue/relation prompts and v9 task-planner contracts.
- `src/salesbench/goldbank/parsing.py`: parse cue and relation model responses.
- `src/salesbench/goldbank/normalizer.py`: normalize cue/relation IDs and English/native fields.
- `src/salesbench/goldbank/validators.py`: local cue/relation and task-capability rules.
- `src/salesbench/goldbank/pipeline.py`: run atomic evidence -> cue -> relation -> four task planners.
- `src/salesbench/goldbank/runner.py`: persist new JSONL outputs and v8 pipeline fingerprints.
- `src/salesbench/goldbank/review_io.py`: apply human decisions to cues, relations, and annotations.
- `src/salesbench/goldbank/audit.py`: coverage and provenance metrics.
- `src/salesbench/vqa/question_programs.py`: remove fixed public templates; retain only legacy compatibility helpers.
- `src/salesbench/vqa/compiler.py`: consume reviewed question realizations and enforce diversity/English/private-field gates.
- `src/salesbench/vqa_evaluate/context.py`: provide reviewed evidence/cue/relation summaries to Judge.
- `src/salesbench/vqa_evaluate/prompts.py`: task-aware v4 Judge rubric and error tags.
- `src/salesbench/vqa_evaluate/judge.py`: parse error tags while preserving five-level main score.
- `src/salesbench/cli.py`: add `realize-qa` and v9 input/output options.
- `tools/audit_workbench/build.py`: render cue cards, relation paths, QuestionSpec, natural QA, and Judge error tags.
- `tools/audit_workbench/evidence_assets.py`: attach lazy-loaded frames to cue and relation evidence.
- Existing `tests/test_goldbank_*.py`, `tests/test_vqa_question_programs.py`, `tests/test_vqa_evaluate.py`, `tests/test_audit_workbench.py`, and `tests/test_evidence_vqa_e2e.py`: migrate fixtures and assertions to v3/v9.
- `docs/Pilot_64_Video_Execution_Guide.md`: replace v8 commands with the reviewed v9 sequence after implementation.
- `docs/data/EvidenceDataset_Data_Card_v3.md`: document the v3 schema; keep the v2 data card unchanged as the legacy contract.

---

### Task 1: Add CommerceCue and CommercialRelation contracts

**Files:**
- Create: `src/salesbench/goldbank/commerce_schema.py`
- Create: `tests/test_commerce_schema.py`
- Modify: `src/salesbench/goldbank/schema.py`

**Interfaces:**
- Consumes: `stable_digest`, `QualityStatus`, and normalized `EvidenceUnit` IDs.
- Produces: `CommerceCue`, `CommercialRelation`, `parse_commerce_cue`, `parse_commercial_relation`, `make_cue_id`, and `make_relation_id`.

- [ ] **Step 1: Write failing round-trip and stable-ID tests**

```python
from salesbench.goldbank.commerce_schema import (
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


def test_commerce_contract_round_trip_and_ids_are_stable():
    cue_id = make_cue_id("v1", CueType.PRICE, ("e1",), "price is 9.9 yuan")
    cue = CommerceCue(
        cue_id=cue_id,
        video_id="v1",
        cue_type=CueType.PRICE,
        content_en="The displayed price is 9.9 yuan.",
        content_native="9.9元",
        evidence_ids=("e1",),
        attributes={"currency": "CNY", "amount": 9.9},
        directness="DIRECT",
        theory_tags=("offer_information",),
        extractor="fake",
        confidence=0.95,
    )
    assert parse_commerce_cue(cue.to_dict()) == cue

    relation_id = make_relation_id(
        "v1", RelationType.OFFER_REQUIRES_CONDITION, (cue_id,), ("condition-cue",)
    )
    relation = CommercialRelation(
        relation_id=relation_id,
        video_id="v1",
        relation_type=RelationType.OFFER_REQUIRES_CONDITION,
        source_cue_ids=(cue_id,),
        target_cue_ids=("condition-cue",),
        evidence_ids=("e1", "e2"),
        rationale_en="The displayed price is explicitly limited to a two-item purchase.",
        provenance=RelationProvenance.THEORY_OPERATIONALIZED,
        directness="INFERRED",
        confidence=0.9,
    )
    assert parse_commercial_relation(relation.to_dict()) == relation
```

- [ ] **Step 2: Run the test and verify missing-module failure**

Run: `pytest tests/test_commerce_schema.py -q`

Expected: FAIL with `ModuleNotFoundError: salesbench.goldbank.commerce_schema`.

- [ ] **Step 3: Implement enums, dataclasses, parsers, and stable IDs**

```python
class RelationProvenance(str, Enum):
    THEORY_DIRECT = "THEORY_DIRECT"
    THEORY_OPERATIONALIZED = "THEORY_OPERATIONALIZED"
    BENCHMARK_OPERATIONAL = "BENCHMARK_OPERATIONAL"


@dataclass(frozen=True)
class CommerceCue:
    cue_id: str
    video_id: str
    cue_type: CueType
    content_en: str
    content_native: str
    evidence_ids: tuple[str, ...]
    attributes: dict[str, object]
    directness: str
    theory_tags: tuple[str, ...]
    extractor: str
    confidence: float


@dataclass(frozen=True)
class CommercialRelation:
    relation_id: str
    video_id: str
    relation_type: RelationType
    source_cue_ids: tuple[str, ...]
    target_cue_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rationale_en: str
    provenance: RelationProvenance
    directness: str
    confidence: float
```

Define every cue and relation token exactly as listed in Sections 1.3 and 2. Re-export the new contracts from `goldbank/schema.py` and set `SCHEMA_VERSION = "evidence-dataset-schema-v3"`.

- [ ] **Step 4: Run contract tests**

Run: `pytest tests/test_commerce_schema.py tests/test_goldbank_schema.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/goldbank/commerce_schema.py src/salesbench/goldbank/schema.py tests/test_commerce_schema.py tests/test_goldbank_schema.py
git commit -m "feat: add commerce cue and relation contracts"
```

### Task 2: Encode ontology provenance and deterministic validation

**Files:**
- Create: `src/salesbench/goldbank/commerce_ontology.py`
- Create: `tests/test_commerce_ontology.py`
- Modify: `src/salesbench/goldbank/validators.py`

**Interfaces:**
- Consumes: cue/relation contracts from Task 1.
- Produces: `validate_commerce_cue(cue, evidence_by_id)` and `validate_commercial_relation(relation, cues_by_id, evidence_by_id)`.

- [ ] **Step 1: Write failing causal-firewall and endpoint tests**

```python
def test_relation_catalog_rejects_causal_outcomes_and_invalid_endpoints():
    assert "INCREASES_TRUST" not in {item.value for item in RelationType}
    assert "CAUSES_PURCHASE" not in {item.value for item in RelationType}

    issues = validate_commercial_relation(
        relation_fixture(
            relation_type="FEATURE_FRAMED_AS_BENEFIT",
            source_cue_ids=("price-cue",),
            target_cue_ids=("cta-cue",),
        ),
        cues_by_id={
            "price-cue": cue_fixture("PRICE"),
            "cta-cue": cue_fixture("CTA"),
        },
        evidence_by_id={"e1": evidence_fixture()},
    )
    assert "INVALID_RELATION_ENDPOINT" in {issue.code for issue in issues}
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_commerce_ontology.py -q`

Expected: FAIL because the catalog and validators do not exist.

- [ ] **Step 3: Implement relation metadata and local rules**

`commerce_ontology.py` must expose `RELATION_RULES` with exact allowed endpoints and provenance. For example:

```python
RELATION_RULES = {
    RelationType.FEATURE_FRAMED_AS_BENEFIT: {
        "source": {CueType.PRODUCT_ATTRIBUTE, CueType.PRODUCT_DESCRIPTION},
        "target": {CueType.BENEFIT},
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
    RelationType.CLAIM_SUPPORTED_BY_DEMONSTRATION: {
        "source": {CueType.FUNCTION_CLAIM, CueType.EFFECT_CLAIM, CueType.FIT_CLAIM},
        "target": {
            CueType.PROCESS_DEMONSTRATION,
            CueType.OUTCOME_DISPLAY,
            CueType.BEFORE_AFTER,
            CueType.VICARIOUS_TRIAL,
        },
        "provenance": RelationProvenance.THEORY_OPERATIONALIZED,
        "min_evidence": 2,
    },
}
```

Validators must reject missing/same-video-invalid IDs, empty English content, private fields, invalid endpoints, confidence outside `[0, 1]`, causal outcome wording, and unsupported “NOT_SHOWN” decisions without a complete observation scope.

- [ ] **Step 4: Run validators**

Run: `pytest tests/test_commerce_ontology.py tests/test_goldbank_validators.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/goldbank/commerce_ontology.py src/salesbench/goldbank/validators.py tests/test_commerce_ontology.py tests/test_goldbank_validators.py
git commit -m "feat: validate commerce graph semantics"
```

### Task 3: Add v9 cue and relation extraction prompts

**Files:**
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/parsing.py`
- Modify: `src/salesbench/goldbank/normalizer.py`
- Modify: `tests/test_goldbank_prompts.py`
- Modify: `tests/test_goldbank_normalizer.py`

**Interfaces:**
- Consumes: validated `EvidenceUnit` dictionaries and v3 ontology.
- Produces: `build_commerce_cue_prompt`, `build_commercial_relation_prompt`, normalized cue/relation lists.

- [ ] **Step 1: Write prompt contract tests**

```python
def test_v9_commerce_prompts_separate_observation_from_consumer_outcomes():
    cue_system, cue_user = build_commerce_cue_prompt("v1", [evidence_fixture()])
    relation_system, relation_user = build_commercial_relation_prompt(
        "v1", [evidence_fixture()], [cue_fixture()]
    )
    combined = cue_system + cue_user + relation_system + relation_user
    assert PROMPT_VERSION == "evidence-prompt-v9"
    assert "content_native" in combined
    assert "content_en" in combined
    assert "CLAIM_REPEATED_ACROSS_MODALITIES" in combined
    assert "CONTENT_ADDRESSES_USAGE_UNCERTAINTY" in combined
    assert "Never claim that a viewer trusted, purchased, converted, or became less uncertain" in combined
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_goldbank_prompts.py::test_v9_commerce_prompts_separate_observation_from_consumer_outcomes -q`

Expected: FAIL because the v9 builders do not exist.

- [ ] **Step 3: Implement strict JSON prompt contracts**

The cue prompt top-level output is exactly:

```json
{
  "commerce_cues": [
    {
      "cue_type": "PROCESS_DEMONSTRATION",
      "content_en": "The host applies the cleaner and wipes the surface.",
      "content_native": "",
      "evidence_ids": ["existing_visual_id"],
      "attributes": {"presentation_stage": "process"},
      "directness": "DIRECT",
      "theory_tags": ["product_demonstration"],
      "confidence": 0.92
    }
  ],
  "abstentions": []
}
```

The relation prompt top-level output is exactly:

```json
{
  "commercial_relations": [
    {
      "relation_type": "CLAIM_SUPPORTED_BY_DEMONSTRATION",
      "source_cue_ids": ["existing_claim_cue"],
      "target_cue_ids": ["existing_demo_cue"],
      "evidence_ids": ["existing_asr_id", "existing_visual_id"],
      "rationale_en": "The spoken cleaning claim is supported by the visible before-and-after change.",
      "provenance": "THEORY_OPERATIONALIZED",
      "directness": "INFERRED",
      "confidence": 0.88
    }
  ],
  "abstentions": []
}
```

Local code generates canonical IDs and overwrites model-provided provenance with the ontology catalog.

- [ ] **Step 4: Run prompt, parser, and normalizer tests**

Run: `pytest tests/test_goldbank_prompts.py tests/test_goldbank_normalizer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/goldbank/prompts.py src/salesbench/goldbank/parsing.py src/salesbench/goldbank/normalizer.py tests/test_goldbank_prompts.py tests/test_goldbank_normalizer.py
git commit -m "feat: add v9 commerce graph prompts"
```

### Task 4: Persist the commerce graph in the EvidenceDataset pipeline

**Files:**
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `src/salesbench/goldbank/review_io.py`
- Modify: `tests/test_goldbank_pipeline.py`
- Modify: `tests/test_goldbank_runner.py`
- Modify: `tests/test_goldbank_review_io.py`

**Interfaces:**
- Consumes: normalized EvidenceUnits, cues, relations, and validators.
- Produces: `GoldBankResult.commerce_cues`, `GoldBankResult.commercial_relations`, `commerce_cues.jsonl`, and `commercial_relations.jsonl`.

- [ ] **Step 1: Write a failing stage-order test**

```python
def test_pipeline_builds_cues_and_relations_before_task_planners():
    result = pipeline_with_v9_responses().run_video(bundle_fixture(), frames_b64=["frame"])
    assert result.status == "ok"
    assert len(result.evidence_units) == 2
    assert len(result.commerce_cues) == 2
    assert len(result.commercial_relations) == 1
    stages = [trace["stage"] for trace in result.agent_traces]
    assert stages.index("evidence_extraction") < stages.index("commerce_cue_extraction")
    assert stages.index("commerce_cue_extraction") < stages.index("commercial_relation_building")
    assert stages.index("commercial_relation_building") < stages.index("task_proposal")
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_goldbank_pipeline.py::test_pipeline_builds_cues_and_relations_before_task_planners -q`

Expected: FAIL because `GoldBankResult` lacks the new fields.

- [ ] **Step 3: Implement the new stage order and output files**

Use this exact order in `run_video`:

```python
evidence_units = self._extract_evidence(bundle, frames_b64, traces)
commerce_cues = self._extract_commerce_cues(bundle.video_id, evidence_units, traces)
commercial_relations = self._build_commercial_relations(
    bundle.video_id, evidence_units, commerce_cues, traces
)
proposals = self._build_task_proposals(
    bundle.video_id, evidence_units, commerce_cues, commercial_relations, traces
)
```

`runner.py` must save part-level and aggregate cue/relation rows, include their counts in `generation_meta.json`, and set `PIPELINE_VERSION = "evidence-first-pipeline-v8"`. Resume fingerprints include schema, prompt, pipeline, video hash, frame strategy, vision model, and text model.

- [ ] **Step 4: Run pipeline, runner, and review tests**

Run: `pytest tests/test_goldbank_pipeline.py tests/test_goldbank_runner.py tests/test_goldbank_review_io.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/goldbank/pipeline.py src/salesbench/goldbank/runner.py src/salesbench/goldbank/review_io.py tests/test_goldbank_pipeline.py tests/test_goldbank_runner.py tests/test_goldbank_review_io.py
git commit -m "feat: persist v3 commerce graph outputs"
```

### Task 5: Replace generic subtypes with Plan B capabilities

**Files:**
- Modify: `src/salesbench/goldbank/ontology.py`
- Modify: `src/salesbench/goldbank/schema.py`
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `tests/test_goldbank_schema.py`
- Modify: `tests/test_goldbank_prompts.py`
- Modify: `tests/test_goldbank_pipeline.py`

**Interfaces:**
- Consumes: CommerceGraph from Task 4.
- Produces: four task planners that emit `capability`, `reasoning_operator`, cue/relation refs, bounded Gold, and question intent.

- [ ] **Step 1: Write task-boundary tests**

```python
def test_plan_b_capabilities_map_to_graph_levels():
    assert capability_level("BP", "OFFER_CONDITION") == "RELATION_OPTIONAL"
    assert capability_level("CM", "CLAIM_DEMONSTRATION_STATUS") == "RELATION_REQUIRED"
    assert capability_level("SS", "PROBLEM_SOLUTION") == "RELATION_PATH_REQUIRED"
    assert capability_level("AE", "FIT_CONSTRAINT") == "CUE_OR_RELATION"
    assert "AUDIENCE_NEED_FIT" not in allowed_subtypes("AE")
    assert "TRUST_MECHANISM" not in allowed_subtypes("SS")
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_goldbank_schema.py tests/test_goldbank_prompts.py -q`

Expected: FAIL against the v8 subtype catalog.

- [ ] **Step 3: Implement the capability catalogs and proposal contract**

Add these explicit fields to `GoldProposal` and `GoldItem`:

```python
capability: str
reasoning_operator: str
commerce_cue_ids: tuple[str, ...]
commercial_relation_ids: tuple[str, ...]
question_intent: str
forbidden_inferences: tuple[str, ...]
```

The BP planner remains deterministic but consumes Product/Offer/Presentation cues rather than compiling every raw evidence row. CM, SS, and AE proposers consume the same graph with task-specific allowed capabilities from Sections 3.2–3.4. Every proposer may abstain; no per-video all-task requirement is allowed.

- [ ] **Step 4: Run schema, prompt, and pipeline tests**

Run: `pytest tests/test_goldbank_schema.py tests/test_goldbank_prompts.py tests/test_goldbank_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/goldbank/ontology.py src/salesbench/goldbank/schema.py src/salesbench/goldbank/prompts.py src/salesbench/goldbank/pipeline.py tests/test_goldbank_schema.py tests/test_goldbank_prompts.py tests/test_goldbank_pipeline.py
git commit -m "feat: align four tasks with commerce graph capabilities"
```

### Task 6: Add QuestionSpec planning and natural English realization

**Files:**
- Create: `src/salesbench/vqa/specs.py`
- Create: `src/salesbench/vqa/prompts.py`
- Create: `src/salesbench/vqa/realizer.py`
- Create: `tests/test_vqa_specs.py`
- Create: `tests/test_vqa_realizer.py`
- Modify: `src/salesbench/cli.py`

**Interfaces:**
- Consumes: reviewed v3 GroundedAnnotations and evidence/cue/relation lookup tables.
- Produces: `qa_specs.jsonl` and resumable `qa_realizations.jsonl`.

- [ ] **Step 1: Write failing QuestionSpec and realizer tests**

```python
def test_realizer_uses_specific_referents_without_annotation_jargon():
    spec = QuestionSpec(
        spec_id="qs_v1_cm_001",
        video_id="v1",
        task_type="CM",
        capability="CLAIM_DEMONSTRATION_STATUS",
        reasoning_operator="DISTINGUISH_STATED_FROM_SHOWN",
        question_intent="Distinguish the universal stain-removal claim from the demonstrated result.",
        gold_answer="The video shows one stain becoming lighter, but the universal claim is only stated.",
        evidence_refs=("e1", "e2"),
        commerce_cue_refs=("c1", "c2"),
        commercial_relation_refs=("r1",),
        forbidden_inferences=("The product removes every stain.",),
        confidence=0.9,
    )
    realization = realize_with_fake_client(spec, "What is shown after the stain-removal claim, and what remains only stated?")
    assert realization.question.startswith("What is shown")
    assert "mechanism" not in realization.question.lower()
    assert "support the answer" not in realization.question.lower()
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_vqa_specs.py tests/test_vqa_realizer.py -q`

Expected: FAIL because QuestionSpec and realizer modules do not exist.

- [ ] **Step 3: Implement contracts and the English realizer prompt**

The model returns exactly:

```json
{
  "spec_id": "qs_v1_cm_001",
  "question": "What is shown after the stain-removal claim, and what remains only stated?"
}
```

The prompt forbids generic openings such as `What mechanism`, `What strategy`, `What audience`, and the suffix `Support the answer with evidence`. It requires concrete product/action/offer referents, one reasoning operator, no answer leakage, and no task labels. Local code validates English, length, ID equality, and absence of forbidden phrases.

Add CLI:

```text
python salesbench.py realize-qa \
  --evidence-dir <reviewed-evidence-dir> \
  --output-dir <qa-dir> \
  --text-model gpt-4o \
  --max-workers 2
```

- [ ] **Step 4: Run spec, realizer, and CLI tests**

Run: `pytest tests/test_vqa_specs.py tests/test_vqa_realizer.py tests/test_goldbank_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/vqa/specs.py src/salesbench/vqa/prompts.py src/salesbench/vqa/realizer.py src/salesbench/cli.py tests/test_vqa_specs.py tests/test_vqa_realizer.py tests/test_goldbank_cli.py
git commit -m "feat: realize evidence-specific English questions"
```

### Task 7: Compile reviewed realizations and enforce dataset diversity

**Files:**
- Modify: `src/salesbench/vqa/question_programs.py`
- Modify: `src/salesbench/vqa/compiler.py`
- Modify: `src/salesbench/cli.py`
- Modify: `tests/test_vqa_question_programs.py`
- Modify: `tests/test_goldbank_qa_compiler.py`
- Modify: `tests/test_goldbank_cli.py`

**Interfaces:**
- Consumes: reviewed GroundedAnnotations, `qa_specs.jsonl`, and `qa_realizations.jsonl`.
- Produces: private/public VQA JSONL, diversity report, and rejection reasons.

- [ ] **Step 1: Write failing no-template and no-forced-task tests**

```python
def test_compiler_uses_realized_questions_and_allows_per_video_abstention():
    records, validation = compile_qa_records(
        gold_items=[reviewed_cm_item(), reviewed_ss_item()],
        realizations={
            "qs_cm": realization("qs_cm", "What does the demonstration show beyond the spoken claim?"),
            "qs_ss": realization("qs_ss", "How does the video connect the cleaning problem to the product demo?"),
        },
        policy=CompilePolicy(max_questions_per_video=6, max_per_task=3),
    )
    assert {row["task_type"] for row in records} == {"CM", "SS"}
    assert all("Support the answer" not in row["question"] for row in records)
    assert not any(row.get("reason") == "missing_bp_for_video" for row in validation)
```

- [ ] **Step 2: Run and verify signature failure**

Run: `pytest tests/test_goldbank_qa_compiler.py -q`

Expected: FAIL because the compiler does not accept realizations.

- [ ] **Step 3: Implement compiler v5 and diversity metrics**

Set `COMPILER_VERSION = "evidence-qa-compiler-v5"`. Compile only realized specs whose source annotations are reviewed/eligible. Emit `qa_diversity.json` with:

```json
{
  "exact_duplicate_rate": 0.0,
  "largest_normalized_stem_cluster_rate": 0.03125,
  "within_video_semantic_duplicate_count": 0,
  "cross_task_same_evidence_count": 4,
  "cross_task_non_marginal_count": 0
}
```

Reject exact duplicates and within-video semantic duplicates. Flag cross-task evidence reuse unless capability, reasoning operator, and normalized answer differ. Require all four tasks only at cohort level, not per video.

Extend `compile-vqa` with the required `--realizations` argument and fail before writing outputs when the reviewed realization file is missing or contains an unknown `spec_id`.

- [ ] **Step 4: Run compiler tests**

Run: `pytest tests/test_vqa_question_programs.py tests/test_goldbank_qa_compiler.py tests/test_goldbank_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/vqa/question_programs.py src/salesbench/vqa/compiler.py src/salesbench/cli.py tests/test_vqa_question_programs.py tests/test_goldbank_qa_compiler.py tests/test_goldbank_cli.py
git commit -m "feat: compile diverse reviewed QA realizations"
```

### Task 8: Extend the lazy-loading audit workbench to commerce graphs and QuestionSpecs

**Files:**
- Modify: `tools/audit_workbench/build.py`
- Modify: `tools/audit_workbench/evidence_assets.py`
- Modify: `tools/audit_workbench/review_queue.py`
- Modify: `tests/test_audit_workbench.py`
- Modify: `tests/test_audit_review_queue.py`

**Interfaces:**
- Consumes: EvidenceUnits, CommerceCues, CommercialRelations, annotations, QuestionSpecs, realizations, QA, and Judge details.
- Produces: a self-contained HTML shell plus lazy frame thumbnails under `outputs/audit/assets/frames/`.

- [ ] **Step 1: Write failing graph-visibility tests**

```python
def test_workbench_shows_cue_relation_question_spec_and_linked_frames():
    html = render_workbench(workbench_v9_fixture(), collect_prompt_snapshot())
    assert "Commerce Cue" in html
    assert "CLAIM_SUPPORTED_BY_DEMONSTRATION" in html
    assert "DISTINGUISH_STATED_FROM_SHOWN" in html
    assert "content_native" in html
    assert "content_en" in html
    assert 'loading="lazy"' in html
    assert "INCREASES_TRUST" not in html
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_audit_workbench.py tests/test_audit_review_queue.py -q`

Expected: FAIL because v8 workbench data does not contain cues, relations, or specs.

- [ ] **Step 3: Implement readable graph cards and review exports**

Evidence cards show native/English text and exact frames. Cue cards show cue type, content, directness, theory tags, evidence contents, and frames. Relation cards show source cue -> relation -> target cue, provenance, rationale, supporting evidence, and frames. QA cards show capability, reasoning operator, QuestionSpec intent, final question, Gold, forbidden inferences, relations, evidence contents, and frames.

Review exports must use stable IDs and support `accept`, `revise`, `reject`, and `defer`, with editable replacement fields stored only in the exported JSON until `apply-evidence-reviews` is run.

- [ ] **Step 4: Run workbench tests**

Run: `pytest tests/test_audit_workbench.py tests/test_audit_review_queue.py -q`

Expected: PASS and generated test HTML contains no private interaction fields.

- [ ] **Step 5: Commit**

```bash
git add tools/audit_workbench/build.py tools/audit_workbench/evidence_assets.py tools/audit_workbench/review_queue.py tests/test_audit_workbench.py tests/test_audit_review_queue.py
git commit -m "feat: audit commerce graphs and question specs"
```

### Task 9: Keep LLM-as-Judge as the primary evaluator and add diagnostic error tags

**Files:**
- Modify: `src/salesbench/vqa_evaluate/context.py`
- Modify: `src/salesbench/vqa_evaluate/prompts.py`
- Modify: `src/salesbench/vqa_evaluate/judge.py`
- Modify: `src/salesbench/vqa_evaluate/runner.py`
- Modify: `tests/test_vqa_evaluate.py`

**Interfaces:**
- Consumes: question, task/capability, reference answer, reviewed evidence/cue/relation summary, and model answer.
- Produces: one five-level score plus diagnostic `error_tags`; macro-average remains the primary leaderboard metric.

- [ ] **Step 1: Write failing v4 Judge contract tests**

```python
def test_judge_v4_returns_main_score_and_error_tags():
    parsed = parse_judge_response(
        '{"score":0.5,"correctness":0.5,"grounding":0.5,"completeness":0.5,'
        '"error_tags":["CLAIM_EVIDENCE_CONFUSION"],'
        '"reason":"The answer treats repeated text as visual proof.",'
        '"evidence_alignment":"The frames do not independently demonstrate the claim."}'
    )
    assert JUDGE_PROMPT_VERSION == "judge-prompt-v4"
    assert parsed["score"] == 0.5
    assert parsed["error_tags"] == ["CLAIM_EVIDENCE_CONFUSION"]
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_vqa_evaluate.py -q`

Expected: FAIL because v3 does not parse error tags.

- [ ] **Step 3: Implement task-aware rubric without new scoring engines**

Keep the existing allowed scores `{0, 0.25, 0.5, 0.75, 1}` and main macro-average. Add controlled tags:

```python
JUDGE_ERROR_TAGS = {
    "FACTUAL_ERROR",
    "UNSUPPORTED_INFERENCE",
    "MISSING_KEY_INFORMATION",
    "CLAIM_EVIDENCE_CONFUSION",
    "OFFER_CONDITION_MISSING",
    "TEMPORAL_ERROR",
    "TASK_MISUNDERSTANDING",
    "UNANSWERED",
}
```

The Judge receives reviewed evidence summaries and graph relations but no interaction/private metadata. Missing model answers remain 0. Do not add deterministic answer matching or an independent key-fact scorer in v9.

- [ ] **Step 4: Run evaluation tests**

Run: `pytest tests/test_vqa_evaluate.py tests/test_evidence_vqa_e2e.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/vqa_evaluate/context.py src/salesbench/vqa_evaluate/prompts.py src/salesbench/vqa_evaluate/judge.py src/salesbench/vqa_evaluate/runner.py tests/test_vqa_evaluate.py tests/test_evidence_vqa_e2e.py
git commit -m "feat: add commerce-aware Judge diagnostics"
```

### Task 10: Wire v9 CLI, configs, metadata, and end-to-end tests

**Files:**
- Modify: `src/salesbench/cli.py`
- Create: `configs/evidence_smoke_v9_5videos.json`
- Create: `configs/evidence_pilot_v9_64videos.json`
- Modify: `tests/test_goldbank_cli.py`
- Modify: `tests/test_evidence_vqa_e2e.py`
- Modify: `tests/test_benchmark_convergence.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: executable v9 smoke/formal commands and one complete mocked integration path.

- [ ] **Step 1: Write a failing full-flow test**

```python
def test_v9_context_to_judge_flow_contains_commerce_graph_without_private_fields(tmp_path):
    outputs = run_v9_mock_flow(tmp_path)
    assert outputs["counts"]["commerce_cues"] > 0
    assert outputs["counts"]["commercial_relations"] > 0
    assert set(outputs["task_counts"]) == {"BP", "CM", "SS", "AE"}
    payload_text = json.dumps(outputs["model_and_judge_payloads"], ensure_ascii=False)
    assert "likes" not in payload_text
    assert "followers" not in payload_text
    assert "private_analysis_metadata" not in payload_text
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_evidence_vqa_e2e.py tests/test_benchmark_convergence.py -q`

Expected: FAIL because v8 fixtures do not include the commerce graph.

- [ ] **Step 3: Wire CLI and versioned cohort fingerprints**

Both v9 configs set:

```json
{
  "prompt_version": "evidence-prompt-v9",
  "schema_version": "evidence-dataset-schema-v3",
  "pipeline_version": "evidence-first-pipeline-v8",
  "frame_strategy": "hook_plus_uniform",
  "frames_per_video": 16,
  "min_confidence": 0.7
}
```

The smoke config retains the existing five v8 smoke video IDs; the pilot config retains the existing 64 v8 pilot IDs so changes measure ontology/prompt effects rather than cohort changes.

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/test_goldbank_cli.py tests/test_evidence_vqa_e2e.py tests/test_benchmark_convergence.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/cli.py configs/evidence_smoke_v9_5videos.json configs/evidence_pilot_v9_64videos.json tests/test_goldbank_cli.py tests/test_evidence_vqa_e2e.py tests/test_benchmark_convergence.py
git commit -m "feat: wire the v9 SalesBench pilot flow"
```

### Task 11: Update audit metrics, data card, and execution guide

**Files:**
- Modify: `src/salesbench/goldbank/audit.py`
- Modify: `docs/Pilot_64_Video_Execution_Guide.md`
- Create: `docs/data/EvidenceDataset_Data_Card_v3.md`
- Modify: `tests/test_goldbank_audit.py`

**Interfaces:**
- Consumes: final v3 artifacts.
- Produces: provenance/graph/capability coverage metrics and exact human execution instructions.

- [ ] **Step 1: Write failing audit metric tests**

```python
def test_audit_reports_graph_and_capability_coverage():
    report = audit_gold_bank(v3_records(), v3_evidence(), v3_cues(), v3_relations())
    assert report["commerce_cue_coverage"] == 1.0
    assert report["commercial_relation_validity"] == 1.0
    assert report["causal_outcome_relation_count"] == 0
    assert report["capability_counts"]["CLAIM_DEMONSTRATION_STATUS"] == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_goldbank_audit.py -q`

Expected: FAIL because audit currently accepts only records and evidence.

- [ ] **Step 3: Implement metrics and document exact v9 commands**

Add metrics for cue evidence validity, relation endpoint validity, provenance distribution, causal label leakage, capability counts, operator counts, cross-task evidence reuse, and per-video abstention. Update the data card with the v2/v3 non-upgrade boundary and update the execution guide with the commands in Section 6 below.

- [ ] **Step 4: Run documentation-linked tests**

Run: `pytest tests/test_goldbank_audit.py tests/test_goldbank_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/salesbench/goldbank/audit.py docs/Pilot_64_Video_Execution_Guide.md docs/data/EvidenceDataset_Data_Card_v3.md tests/test_goldbank_audit.py
git commit -m "docs: document and audit the v9 commerce benchmark"
```

### Task 12: Run full verification before any API smoke

**Files:**
- Verify only; commit fixes only if verification exposes defects.

**Interfaces:**
- Consumes: Tasks 1–11.
- Produces: a clean, tested implementation ready for API use.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_commerce_schema.py tests/test_commerce_ontology.py \
  tests/test_goldbank_prompts.py tests/test_goldbank_pipeline.py \
  tests/test_vqa_specs.py tests/test_vqa_realizer.py \
  tests/test_goldbank_qa_compiler.py tests/test_audit_workbench.py \
  tests/test_vqa_evaluate.py tests/test_evidence_vqa_e2e.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete suite and static checks**

Run:

```bash
pytest -q
python -m compileall -q src tools
git diff --check
```

Expected: all tests pass, compileall exits 0, and `git diff --check` prints nothing.

- [ ] **Step 3: Inspect version constants**

Run:

```bash
rg -n "evidence-dataset-schema-v3|evidence-prompt-v9|evidence-first-pipeline-v8|evidence-qa-compiler-v5|judge-prompt-v4" src configs docs
```

Expected: every active v9 path reports the intended version; v6/v7/v8 remain only in explicit legacy/migration text.

- [ ] **Step 4: Route any discovered defect back to its owning task**

If verification fails, add a focused regression test in the task that owns the failing component, implement the minimal fix there, rerun that task's tests, and commit only the exact files named by that task. If verification passes without file changes, do not create an empty or aggregate verification commit.

---

## 6. Post-implementation execution runbook

### 6.1 Load the ignored relay credentials without printing them

```bash
set -a
source .env
set +a
```

### 6.2 Run the five-video v9 smoke

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_smoke_v9_5videos.json \
  --output-dir outputs/evidence/v9_smoke5_gpt4o_yunwu \
  --vision-base-url "$OPENAI_BASE_URL" \
  --vision-model gpt-4o \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 1
```

Smoke acceptance:

- 5/5 videos complete without schema-stage failure.
- Atomic Evidence, CommerceCue, and CommercialRelation files are non-empty.
- No causal outcome relation is present.
- Claim repetition is not misclassified as visual support.
- At least one BP, CM, SS, and AE capability exists across the five-video set; missing a task for one individual video is acceptable.
- Linked frames and native/English text are readable in the audit workbench.

### 6.3 Build the smoke audit workbench

Create a v9 delivery manifest by copying `configs/pilot64_gpt4o_v6_delivery.json` and changing only the formal/smoke source paths to the v9 directories. Then run:

```bash
python -m tools.audit_workbench.build \
  --manifest configs/pilot64_gpt4o_v9_delivery.json \
  --output outputs/audit/SalesBench_Prompt_Audit_Workbench_v9.html \
  --skip-organize
```

Audit every cue, relation, QuestionSpec, and QA in the five-video smoke. Revise the ontology or prompt before any 64-video call if relation endpoints, causal boundaries, translations, or question naturalness are unstable.

### 6.4 Run the formal 64-video EvidenceDataset

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_pilot_v9_64videos.json \
  --output-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --vision-base-url "$OPENAI_BASE_URL" \
  --vision-model gpt-4o \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 2
```

### 6.5 Human-review and freeze EvidenceDataset v3

Review order:

1. Atomic evidence transcription/localization.
2. CommerceCue type, English normalization, native text, and evidence refs.
3. CommercialRelation endpoints, provenance, rationale, and causal boundary.
4. GroundedAnnotation capability, operator, Gold, and forbidden inferences.

Apply exported decisions:

```bash
python salesbench.py apply-evidence-reviews \
  --evidence-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --decisions outputs/reviews/v9_pilot64_evidence_decisions.json \
  --output outputs/evidence/v9_pilot64_gpt4o_yunwu/video_evidence_dataset_reviewed.jsonl
```

### 6.6 Realize and audit natural English QA

```bash
python salesbench.py realize-qa \
  --evidence-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu \
  --text-base-url "$OPENAI_BASE_URL" \
  --text-model gpt-4o \
  --max-workers 2
```

Human-review every pilot question. Exact duplicates must be zero, within-video semantic duplicates must be zero, and generic `What mechanism/strategy/audience` questions must be rejected.

### 6.7 Compile reviewed QA

```bash
python salesbench.py compile-vqa \
  --evidence-dir outputs/evidence/v9_pilot64_gpt4o_yunwu \
  --dataset-file video_evidence_dataset_reviewed.jsonl \
  --realizations outputs/vqa/v9_pilot64_gpt4o_yunwu/qa_realizations_reviewed.jsonl \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu \
  --max-questions-per-video 6 \
  --max-per-task 3
```

Target volume is approximately 192–384 QA, determined by valid evidence rather than a fixed per-video quota.

### 6.8 Run one model and LLM-as-Judge

```bash
python salesbench.py run-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --vqa outputs/vqa/v9_pilot64_gpt4o_yunwu/vqa_public.jsonl \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu/run_gpt4o \
  --base-url "$OPENAI_BASE_URL" \
  --model gpt-4o \
  --max-workers 2

python salesbench.py evaluate-vqa-benchmark \
  --config configs/benchmark_v1.json \
  --gold outputs/vqa/v9_pilot64_gpt4o_yunwu/vqa_gold_private.jsonl \
  --predictions outputs/vqa/v9_pilot64_gpt4o_yunwu/run_gpt4o/predictions.jsonl \
  --output-dir outputs/vqa/v9_pilot64_gpt4o_yunwu/evaluation_gpt4o \
  --base-url "$OPENAI_BASE_URL" \
  --judge-model gpt-4o \
  --max-workers 2
```

The primary leaderboard value is the macro-average of BP, CM, SS, and AE Judge scores. Micro-average, capability slices, reasoning-operator slices, and error tags are diagnostics.

---

## 7. Final acceptance criteria

- `pytest -q`, `compileall`, and `git diff --check` pass.
- Public payloads contain no private interaction/profile/title fields.
- Referenced evidence, cues, and relations are same-video and resolvable.
- No causal consumer-outcome relation exists.
- All retained pilot Evidence/Cue/Relation/QA records have a human decision.
- Every QA maps to exactly one primary capability and reasoning operator.
- Exact duplicate questions: 0 in the 64-video pilot.
- Within-video semantic duplicates: 0.
- Largest normalized question-stem cluster: at most 5%.
- CM: 100% genuine cross-modal information gap.
- SS/AE: no real viewer profile, actual trust, purchase, interaction, or conversion conclusion.
- All four tasks are non-empty across the cohort, without forcing all tasks per video.
- Missing model answers score 0.
- Judge model/version/prompt are frozen and recorded.
- A human-scored calibration subset covers all four tasks and all five Judge score levels when available.
- Formal and smoke directories remain physically separate.
