# SalesBench v10 Quality Gates Design

## Status

Approved in chat on 2026-08-30. This design replaces the v9 behavior that mixed parse failures, low-confidence candidates, abstentions, automatic rejections, repairable candidates, and genuinely ambiguous content in one `human_review_queue.jsonl`.

## Goal

Build a fail-closed quality pipeline in which obvious bad Evidence, CommerceCue, CommercialRelation, GroundedAnnotation, and QA candidates are repaired once or rejected during production. Human reviewers see only semantic ambiguity and a small stratified quality-control sample.

## Current failure modes

The v9 64-video run emitted 440 human-review rows:

- 195 proposal graph validation failures;
- 131 CommercialRelation validation failures;
- 30 low-confidence CommerceCues;
- 29 Challenger `REVISE` results;
- 26 abstentions;
- 15 invalid CommerceCues;
- 8 invalid final annotations;
- 3 semantic duplicates;
- 3 relation parse failures.

At least 411 of these records are deterministic rejection or diagnostic events rather than content decisions. The remaining 29 must receive one automatic repair attempt before any human escalation.

The content false-negative problem has separate causes:

- Evidence confidence aliases are not normalized consistently;
- Evidence is accepted without applying `min_confidence`;
- lifecycle status and grounding type are conflated;
- relation validation checks endpoint types but not proposition entailment;
- the Challenger is text-only and cannot reinspect cited frames;
- every non-PASS Challenger result is escalated to humans;
- QuestionSpec creation ignores lifecycle readiness;
- the Question Realizer validates only surface form;
- diversity is reported after compilation instead of gating selection;
- audit rendering tries to reconstruct candidate payloads that production discarded.

## Quality decision model

Every generated object receives exactly one disposition:

```text
ACCEPT              eligible for the next production stage
REPAIR              one automatic repair attempt is allowed
REJECT              terminal automatic rejection
HUMAN_REVIEW        genuine semantic ambiguity only
```

Every decision is recorded in `quality_decisions.jsonl`. The pipeline writes distinct outputs:

```text
accepted candidates in their canonical stage files
repaired_candidates.jsonl
rejected_candidates.jsonl
pipeline_diagnostics.jsonl
human_review_queue.jsonl
quality_decisions.jsonl
```

`human_review_queue.jsonl` must never contain parse failures, schema failures, low-confidence candidates, duplicates, generator abstentions, or explicit model rejections.

## Status separation

Grounding and publication lifecycle are independent:

```text
GroundingType:
  DIRECT_OBSERVATION
  SPEAKER_CLAIM
  BOUNDED_INFERENCE

LifecycleStatus:
  AUTO_ACCEPTED_CANDIDATE
  NEEDS_HUMAN_REVIEW
  HUMAN_ACCEPTED
  REJECTED
```

The legacy `quality_status` remains readable during migration, but it no longer authorizes compilation. Candidate compilation must be explicitly enabled. Formal compilation accepts `HUMAN_ACCEPTED`, or auto-accepted rows only under an explicit calibrated release policy.

## Evidence gate

Evidence normalization is per record so one malformed item cannot invalidate the whole batch. `confidence` and historical `numeric_confidence` are accepted as aliases; a missing or malformed confidence is a schema error rather than silent `0.0`.

EvidenceUnit v4 adds:

```text
assertion_type: OBSERVED | SPOKEN_CLAIM | OCR_TEXT
temporal_scope: FRAME | SHORT_CLIP | FULL_VIDEO | LONG_TERM_CLAIM | UNSPECIFIED
```

Hard validation requires:

- visual Evidence has valid frame indices;
- ASR/OCR Evidence has a native text span and temporal localization;
- Evidence meets the configured minimum confidence;
- normalized semantic fields are English;
- private metadata is absent;
- the declared assertion type matches its modality.

Strict mode uses a batched frame-aware verifier to recheck visual Evidence. Unsupported Evidence is rejected; ambiguous Evidence alone may enter human review.

## CommerceCue gate

Cue-to-modality contracts are enforced locally. Spoken effect and durability claims remain claim cues and cannot become direct product attributes. Demonstration and outcome cues require visual Evidence. Price and quantity cues must preserve the numeric tokens present in ASR/OCR Evidence.

Invalid or low-confidence cues are rejected, not escalated. Repairable schema output gets one repair attempt.

## CommercialRelation gate

Relation validation enforces:

- endpoint ontology;
- exact relation/status compatibility;
- endpoint Evidence union;
- modality requirements;
- proposition compatibility;
- claim scope versus observation scope;
- no consumer-outcome, long-term, causal, or subjective overclaim.

Short visual demonstrations cannot independently support long-term, subjective, causal, durability, or psychological claims. OCR/speech alignment requires matching subject, attribute, and value rather than the mere presence of two modalities.

Strict mode sends each video's accepted relation candidates and their cited frames to a batched VLM relation verifier. `PASS` continues, `REJECT` is terminal, and `AMBIGUOUS` enters human review.

## Proposal and annotation gate

The proposer prompt receives an explicit capability contract, including required relation types. A generator must abstain when the required graph path is absent.

Challenger outcomes route as follows:

```text
PASS          continue
REVISE        automatic repair, then full revalidation
REJECT        automatic rejection
HUMAN_REVIEW  human queue
```

Repairs may not invent Evidence, Cue, or Relation IDs. Repair is limited to one attempt. Exhausted or invalid repair is rejected.

Accepted annotations use `AUTO_ACCEPTED_CANDIDATE`; production code must never write the ambiguous label `verified`.

## QA gate

QuestionSpec creation requires an accepted lifecycle state and a passed item-validity decision. Question surface realization is separate from semantic validation.

The local QA gate checks:

- English and punctuation;
- question length;
- answer leakage and premise leakage;
- task-specific answer schema;
- evidence-reference minimums;
- duplicate and template-cluster limits;
- question/target/gold consistency that can be decided structurally.

Strict mode adds an item-validity verifier over Question, Gold, Evidence, Cue, and Relation context. It returns `PASS`, `REJECT`, or `HUMAN_REVIEW` with answerability, gold support, answer uniqueness, task alignment, and domain-specificity checks.

CM uses controlled labels and stores rationale separately. BP uses short fact or numeric answers where applicable. SS and AE remain open-answer tasks.

## Audit workbench

The audit workbench consumes already separated files and exposes five paginated views:

1. Human Review: actionable ambiguous items only;
2. Auto-Accepted Sample: deterministic 5–10% per-task monitoring sample;
3. Auto-Rejected: read-only rejection trace;
4. Pipeline Diagnostics: parse/API/schema events;
5. Abstentions/Resample: coverage decisions.

Only Human Review has Accept/Edit/Reject controls. Every row stores its full candidate snapshot, verifier snapshot, Evidence refs, Cue refs, Relation refs, and linked lazy-loaded frames.

## Release policy

The operational target is:

```text
human ambiguity queue <= 5% of generated candidates
pipeline fails calibration if ambiguity queue > 10%
accepted sample fraction = 10% per task for the 64-video pilot
accepted sample human precision >= 95%
```

The percentages do not force acceptance. If the ambiguity rate is too high, the run is blocked and the pipeline must be improved instead of shifting work to humans.

## Versioning and compatibility

v10 introduces new schema, prompt, pipeline, compiler, and audit versions. v9 artifacts remain readable as legacy candidate data but cannot be silently promoted to v10 formal Gold. Resume fingerprints include the quality policy, verifier model, verifier prompt version, and frame-selection configuration.

## Verification

Required tests cover:

- confidence alias normalization and missing confidence rejection;
- per-unit Evidence normalization;
- Evidence confidence filtering;
- Cue modality contracts;
- relation/status and temporal-scope contracts;
- queue routing and one-repair limit;
- lifecycle compilation policy;
- QuestionSpec lifecycle filtering;
- QA answer schema and diversity gating;
- separated runner outputs;
- audit views consuming the separated outputs;
- full Evidence-to-QA integration;
- privacy firewall for model, Judge, and public JSONL payloads.

The implementation is complete only when the full test suite, compile checks, diff checks, and a 5-video strict-mode smoke run pass.
