# SalesBench Run Integrity and Final QA Browser Design

## Status

Approved in chat on 2026-09-07. This design extends `2026-08-30-salesbench-quality-gates-v10-design.md`. It does not change the four public tasks or the Evidence-First data model. It closes run-integrity gaps, treats missing ASR timestamps as an expected capability boundary, strengthens content selection, and separates production auditing from read-only final QA browsing.

## Goal

Produce one internally consistent SalesBench candidate release in which Evidence, QuestionSpecs, realized QA, compiled QA, translations, evaluation, and audit HTML are traceable to one immutable run. Obvious low-value or semantically invalid items must be rejected during production. Human reviewers handle only genuine semantic ambiguity and stratified quality-control samples. The HTML workbench must also expose a separate, read-only Chinese view of every release-eligible final QA item.

## Confirmed current-state findings

The current path name contains `v10`, but component versions are independent:

```text
benchmark experiment       evidence-smoke-v10
Evidence prompt            evidence-prompt-v10.2
Evidence pipeline          evidence-first-pipeline-v10.3
Evidence schema            evidence-dataset-schema-v4
generated QA compiler      evidence-qa-compiler-v7
current source compiler    evidence-qa-compiler-v8
question realizer          question-realizer-prompt-v2
QA quality prompt          qa-quality-prompt-v1
```

The QA run recorded 83 EvidenceUnits. The Evidence directory was subsequently regenerated in place and now contains 89 EvidenceUnits. The QA artifacts therefore reference an earlier Evidence snapshot under the same path. Of 29 compiled QA rows, only two preserve a complete reference closure against the current Evidence directory.

This is not a v9-versus-v10 mix-up. It is an immutable-run failure inside the v10 experiment: a stable path was reused for different content, and downstream stages did not verify content fingerprints.

The input ASR has no usable timestamps. `timestamp_status=unavailable` is an expected source limitation, not an Evidence defect and not human-review work. It limits which downstream capabilities may be generated.

## Version and run identity model

Three identities are kept separate:

```text
benchmark_release  externally meaningful candidate or frozen release
run_id             one immutable execution of the complete pipeline
component_versions independent implementations and prompt/schema versions
```

Every run writes a manifest with this minimum contract:

```json
{
  "benchmark_release": "salesbench-v10-candidate.2",
  "run_id": "v10c2-smoke5-qwen-deepseek-20260907-001",
  "release_status": "CANDIDATE",
  "components": {
    "evidence_schema": "evidence-dataset-schema-v4",
    "evidence_prompt": "evidence-prompt-v10.3",
    "evidence_pipeline": "evidence-first-pipeline-v10.4",
    "quality_prompt": "quality-gate-prompt-v3",
    "question_realizer": "question-realizer-prompt-v3",
    "qa_quality_prompt": "qa-quality-prompt-v2",
    "qa_compiler": "evidence-qa-compiler-v9",
    "audit_workbench": "audit-workbench-v11"
  },
  "artifacts": {},
  "fingerprints": {}
}
```

Component numbers do not have to equal the benchmark release number. The workbench must present them as component versions and must never describe `compiler-v9` as the overall benchmark version.

## Immutable run layout

New executions use an immutable run root:

```text
outputs/runs/<run_id>/
  run_manifest.json
  evidence/
  qa/
  evaluation/
  translations/
  audit/
```

An existing run root may be resumed only when its configuration and source fingerprints match. A different Evidence configuration, prompt, model, input video hash, frame-selection policy, or source content produces a new run ID. A stage may atomically replace its own incomplete files while resuming the same fingerprint, but it may not overwrite a completed artifact with a different fingerprint.

Legacy output paths remain readable but are candidate-only and cannot be promoted to a frozen release.

## Artifact fingerprint contract

Each stage records a canonical SHA-256 over the semantic inputs it consumes. Absolute paths, output timestamps, API latency, and cost are excluded from content fingerprints.

### Evidence fingerprint

The Evidence fingerprint includes:

- ordered video fingerprints;
- frame-selection configuration and sampled-frame hashes;
- ASR source hash;
- Evidence schema, prompt, pipeline, and quality-prompt versions;
- generation and verifier model identifiers;
- minimum-confidence and strict-verification settings;
- canonical digests of accepted EvidenceUnits, CommerceCues, CommercialRelations, and GroundedAnnotations.

### QA realization fingerprint

The QA realization fingerprint includes:

- Evidence fingerprint;
- complete QuestionSpec;
- realizer and semantic-verifier models;
- realizer and QA-quality prompt versions;
- local QA policy version.

Resume parts are stored at:

```text
qa/.parts/<evidence_fingerprint>/<spec_id>.json
```

Parts under another Evidence fingerprint are ignored. The current run never merges orphaned part files into `qa_realizations.jsonl`.

### Compile fingerprint

The compile fingerprint includes:

- Evidence fingerprint;
- QA realization fingerprint;
- compiler version;
- quota, diversity, task-balance, and ranking policies;
- canonical digest of the selected QA records.

The compiler fails before writing public or private QA if any upstream fingerprint is missing or mismatched.

## Referential closure gate

Every final QA item must resolve all references against the same run:

```text
source_annotation_ids       -> GroundedAnnotation
evidence_refs               -> EvidenceUnit
commerce_cue_ids            -> CommerceCue
commercial_relation_ids     -> CommercialRelation
```

The compiler validates existence, video ownership, and closure before selection. A missing reference is a release-blocking compile error rather than an item-level skip. The compile summary reports counts for all four reference classes and must show 100% closure.

The audit builder independently repeats this check as defense in depth. If the HTML input is stale, it displays a blocking state and does not label any QA as final.

## ASR timestamp capability boundary

Unavailable source timing is represented as:

```json
{
  "start_s": null,
  "end_s": null,
  "timestamp_status": "unavailable"
}
```

`0.0/0.0` is reserved for a genuinely localized zero-length event at the beginning of a video and is not used as a missing-value sentinel.

An ASR EvidenceUnit with `timestamp_status=unavailable` remains eligible for non-temporal tasks, including product identity, price, quantity, seller claims, usage contexts, claim-versus-visual comparison, and other relations that do not depend on order.

Before relation construction, the pipeline derives a capability mask:

```text
asr_temporal_order=false
cross_modal_semantics=true
spoken_claim_extraction=true
```

When `asr_temporal_order=false`, the pipeline does not propose or compile relations and capabilities whose truth requires ASR ordering, including `CONTENT_PRECEDES_CTA`, ASR before/after comparisons, and `CTA_SEQUENCE`. This is recorded as a normal abstention with `TEMPORAL_CAPABILITY_UNAVAILABLE`. It is not an error, risk badge, or human-review item.

The workbench displays a neutral source-capability notice. The existing missing-time risk applies only when a producer incorrectly claims temporal localization or emits a temporal relation despite unavailable timing.

## Evidence and CommerceCue semantic policy

### Product identity

`PRODUCT_IDENTITY` must identify a product category, brand, model, variant, or explicit product label. A sentence that only says a person holds, points to, lifts, rotates, or shows an object is not product identity.

### Product handling versus demonstration

Visible actions are classified as:

```text
BACKGROUND_HANDLING    holding, pointing, generic page flipping; not QA-eligible
PRODUCT_INSPECTION     exposing a package, connector, cross-section, or printed content
FUNCTIONAL_OPERATION   applying, connecting, assembling, scanning, measuring, or using
OUTCOME_DEMONSTRATION  a supported observable state change or result
```

Only functional operation and outcome demonstration map directly to `USAGE_STEP` or demonstration-oriented commercial reasoning. Product inspection may support product identity or attribute QA. Background handling remains Evidence but does not become a QA-generating CommerceCue.

### Assertion scope

Evidence and derived objects preserve one controlled scope:

```text
OBSERVED_FACT
SPOKEN_CLAIM
CONDITIONAL
INSTRUCTION
HYPOTHETICAL
PROMOTIONAL_PROMISE
```

Derived Cue, Relation, Annotation, and Gold text may narrow wording but may not promote a conditional or seller claim into an observed or verified fact.

### Gold constraints

- one answer center per Gold;
- no Evidence, Cue, Relation, Annotation, task, schema, or internal IDs in natural-language fields;
- BP maximum 20 English words unless a structured enumeration is necessary;
- CM and AE maximum 45 English words;
- SS maximum 50 English words;
- seller claims remain explicitly attributed;
- AE selects one bounded need, context, constraint, objection, or offer alignment instead of bundling all video claims.

## QA semantic quality policy

The QA quality verifier returns these dimensions:

```text
answerable_from_evidence
gold_supported
unique_answer
task_aligned
content_specific
commerce_relevant
natural_question
non_trivial
commercially_diagnostic
claim_scope_preserved
intended_modality_required
reference_closed
```

PASS requires all applicable dimensions. BP is allowed to be simpler than CM, SS, and AE, but generic background actions are rejected. The compiled benchmark targets 20–30% simple BP calibration questions; remaining BP items prioritize identity, offer terms, OCR, measurable attributes, functional operation, or observable state change.

Question realization and semantic verification use different model roles. In the Qwen/DeepSeek configuration, DeepSeek realizes English questions and Qwen performs semantic quality verification. Deterministic local rules remain authoritative for reference closure, language, ID leakage, length, temporal capability, and task contracts.

## Candidate ranking and compilation

The compiler no longer orders eligible items by `gold_id`. It computes an auditable selection score from normalized sub-scores:

```text
semantic quality
commercial relevance
Evidence confidence
modality diversity
relation depth
state-change or functional-operation value
question novelty
- triviality penalty
- redundancy penalty
- answer-length penalty
- claim-scope risk
```

Tie-breaking uses a stable ID only after quality scores are equal. Each video contributes at most one item per capability unless a frozen policy explicitly allows more. Selection first satisfies task coverage, then quality and diversity quotas. The compile report stores every candidate's score, selected/skipped state, and reason.

## Human review routing

Human review is reserved for two materially plausible semantic interpretations that remain unresolved after local validation and verifier review.

These cases never enter human review:

- unavailable ASR timing;
- missing references;
- schema or parse failure;
- low confidence;
- internal-ID leakage;
- background handling;
- missing graph path;
- explicit semantic rejection;
- duplicate or quota skip.

The five-video smoke run presents all compiled candidate QA for calibration. The 64-video run uses the human ambiguity queue plus a stratified 10–20% accepted sample after smoke precision is established.

## Workbench information architecture

The HTML workbench contains six top-level views:

1. Run Overview;
2. Current Prompts;
3. Evidence Audit;
4. QA Audit;
5. Final QA Browser;
6. Evaluation and Judge.

### Audit views

Evidence Audit and QA Audit show production decisions: human ambiguity, accepted monitoring sample, automatic rejection, pipeline diagnostics, and abstentions. Only human ambiguity and accepted monitoring samples expose audit decision controls.

### Final QA Browser

The Final QA Browser is a separate read-only product view. It consumes `vqa_gold_private.jsonl` plus `vqa_gold_private_zh.jsonl` from the same verified run and displays one item per page with:

- QA ID, video ID, task, capability, and reasoning operator;
- Chinese question and Gold answer as the primary reading view;
- Chinese Evidence, CommerceCue, and CommercialRelation summaries;
- lazy-loaded linked frame thumbnails;
- collapsible English canonical Question, Gold, and provenance;
- run ID, release status, component versions, and reference-closure status;
- task, video, capability, modality, and free-text filters;
- previous, next, and direct item navigation.

The browser has no Accept, Revise, Reject, or Pending controls. Chinese content is audit-only and never replaces the canonical English benchmark. Before Evidence and QA are frozen, the view is labeled `CANDIDATE`; only a reviewed immutable release may be labeled `FROZEN`.

If fingerprints or references are invalid, the view shows a release-blocking explanation and no final-item count. Historical stale QA may remain available only under an explicitly labeled legacy diagnostics view.

## Regeneration sequence

After implementation, validation uses this order:

```text
1. create a new immutable smoke run ID
2. generate Evidence with Qwen visual and DeepSeek text roles
3. validate and freeze the smoke Evidence candidate snapshot
4. derive current QuestionSpecs
5. realize English QA with DeepSeek
6. verify QA semantics with Qwen
7. compile QA with quality ranking and full reference closure
8. generate Chinese audit sidecars
9. build the six-view HTML workbench
10. review all five-video final candidate QA
11. run model evaluation only after the QA candidate passes review
12. apply the same contract to a new 64-video run ID
```

No existing v10 Evidence or QA directory is overwritten. Current stale QA remains diagnostic-only.

## Verification and acceptance

Automated tests must prove:

- a changed Evidence fingerprint invalidates QA resume and compile;
- orphan parts from another fingerprint are ignored;
- all four QA reference classes close within one video and one run;
- unavailable ASR timing is stored as null, is not a risk, and does not enter human review;
- temporal relations and QA are absent when temporal capability is unavailable;
- non-temporal ASR-derived items still compile;
- action-only product identity and background handling are not QA-eligible;
- conditional and seller-claim scope survives Evidence-to-Gold compilation;
- internal IDs and oversized or multi-focus Gold answers are rejected;
- QA semantic verification includes the new quality dimensions;
- compiler ranking selects a meaningful candidate over a trivial candidate;
- final QA browsing is distinct from QA auditing and contains no decision buttons;
- stale runs block the Final QA Browser;
- public JSONL remains English-only and contains no Gold, private metadata, interaction data, or internal evidence graph;
- Chinese sidecars and HTML remain audit-only;
- smoke and formal runs remain physically separated.

Release acceptance requires 100% reference closure, zero stale parts selected, zero temporal QA under unavailable ASR timing, zero internal-ID leakage, all four tasks non-empty, explainable selection scores, and a successful full test suite, compile check, diff check, and five-video smoke run.

## Git and delivery policy

Implementation is divided into independently testable commits:

1. immutable fingerprints and reference closure;
2. ASR capability boundary;
3. Evidence, Cue, and Gold semantic filters;
4. QA quality dimensions and compiler ranking;
5. Final QA Browser and stale-run blocking;
6. regenerated smoke artifacts and audit report, with generated `outputs/` excluded from Git.

