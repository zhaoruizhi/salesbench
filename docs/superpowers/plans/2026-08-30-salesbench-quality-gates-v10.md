# SalesBench v10 Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed v9 review backlog with fail-closed production quality gates that repair once, reject obvious errors automatically, and reserve humans for semantic ambiguity and stratified quality monitoring.

**Architecture:** Add a shared quality-decision contract, separate lifecycle state from grounding state, and route every pipeline item to accept, repair, reject, or human review. Add task-aware local validators first, then optional strict frame-aware semantic verification, and make QA compilation depend on passed validity decisions rather than the legacy `verified` label.

**Tech Stack:** Python 3.11+, dataclasses/enums, JSONL artifacts, existing OpenAI-compatible VLM client, pytest, standalone HTML/CSS/JavaScript audit workbench.

**Spec:** `docs/superpowers/specs/2026-08-30-salesbench-quality-gates-v10-design.md`

## Global Constraints

- Canonical Evidence, Prompt, QA, Gold, model outputs, and Judge outputs remain English; Chinese is audit-only.
- Public and Judge payloads must never contain interaction metrics, followers, creator metadata, or private analysis fields.
- Strict semantic verification must reuse cited frames and preserve resume fingerprints.
- Every repair is limited to one attempt and may not invent IDs.
- Parse errors, low confidence, duplicates, abstentions, and explicit rejections never enter the human content queue.
- Current v9 outputs remain untouched and are not overwritten.
- Every source/config change is committed to Git.

---

### Task 1: Quality decisions, lifecycle states, and output routing

**Files:**
- Create: `src/salesbench/goldbank/quality_gate.py`
- Modify: `src/salesbench/goldbank/schema.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `src/salesbench/goldbank/review_io.py`
- Test: `tests/test_goldbank_quality_gate.py`
- Test: `tests/test_goldbank_pipeline.py`
- Test: `tests/test_goldbank_runner.py`

**Interfaces:**
- Produces `QualityDisposition`, `LifecycleStatus`, `QualityDecision`, and `route_quality_decision(...)`.
- Extends `GoldBankResult` with `quality_decisions`, `repaired_candidates`, `rejected_candidates`, and `pipeline_diagnostics`.
- Writes five separate quality artifacts while retaining canonical accepted stage files.

- [ ] Write failing tests that low-confidence, validation-failed, duplicate, abstention, and REJECT records do not enter `human_review_queue`.
- [ ] Run the focused tests and verify failures describe the existing mixed routing.
- [ ] Implement decision enums, decision serialization, and deterministic routing.
- [ ] Change pipeline records to `auto_accepted_candidate` and retain `human_accepted` for actual decisions.
- [ ] Persist full `candidate_snapshot` and verifier information for actionable human rows.
- [ ] Extend runner merge/resume payloads and generation metadata for separated artifacts.
- [ ] Run focused tests, then the full suite.
- [ ] Commit with `refactor: separate quality decisions from human review`.

### Task 2: Evidence normalization and hard semantic contracts

**Files:**
- Modify: `src/salesbench/goldbank/schema.py`
- Modify: `src/salesbench/goldbank/normalizer.py`
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/validators.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Test: `tests/test_goldbank_normalizer.py`
- Test: `tests/test_goldbank_prompts.py`
- Test: `tests/test_goldbank_validators.py`
- Test: `tests/test_goldbank_pipeline.py`

**Interfaces:**
- Adds `EvidenceAssertionType` and `EvidenceTemporalScope` to EvidenceUnit serialization.
- Produces `normalize_evidence_unit(video_id, raw, ordinal)` for per-record failure isolation.
- Makes `confidence` and `numeric_confidence` aliases and rejects missing confidence.

- [ ] Write failing tests for confidence aliases, missing confidence, per-unit normalization, and Evidence confidence filtering.
- [ ] Run focused tests and confirm the existing silent-zero and batch-failure behavior.
- [ ] Add v4 Evidence fields with backward-compatible parsers.
- [ ] Update all Evidence prompts to exact JSON field names and enums.
- [ ] Add confidence, timestamp, assertion/modality, Cue/modality, numeric-token, relation/status, and claim-scope validators.
- [ ] Apply Evidence `min_confidence` before accepted Evidence enters downstream stages.
- [ ] Run focused tests and the full suite.
- [ ] Commit with `fix: enforce evidence and commerce quality contracts`.

### Task 3: Automatic repair and strict frame-aware semantic verification

**Files:**
- Create: `src/salesbench/goldbank/semantic_verifier.py`
- Create: `src/salesbench/goldbank/quality_prompts.py`
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/ontology.py`
- Modify: `src/salesbench/goldbank/commerce_schema.py`
- Modify: `src/salesbench/goldbank/commerce_ontology.py`
- Modify: `src/salesbench/goldbank/runner.py`
- Create: `tests/test_goldbank_semantic_verifier.py`
- Modify: `tests/test_goldbank_pipeline.py`
- Modify: `tests/test_commerce_ontology.py`

**Interfaces:**
- Produces batched `verify_relations(...)` and verifier-result parsing.
- Adds strict-mode config with verifier model/prompt fingerprints.
- Adds a one-attempt proposal repair stage for Challenger `REVISE`.

- [ ] Write failing tests for unsupported long-term demonstration, ambiguous verifier routing, explicit rejection, and the repair-attempt limit.
- [ ] Run focused tests and verify the strict-mode APIs do not yet exist.
- [ ] Add frame-index-to-image lookup and cited-frame selection.
- [ ] Add relation and proposal-repair prompts with strict ID-preservation contracts.
- [ ] Implement batched verifier parsing and quality-decision integration.
- [ ] Add missing cross-modal relation types and explicit capability-to-relation contracts.
- [ ] Route PASS, REVISE, REJECT, and HUMAN_REVIEW without polluting the human queue.
- [ ] Add verifier configuration to fingerprints and generation metadata.
- [ ] Run focused tests and the full suite.
- [ ] Commit with `feat: verify and repair semantic graph candidates`.

### Task 4: QuestionSpec, QA validity, answer schemas, and diversity gates

**Files:**
- Create: `src/salesbench/vqa/item_validator.py`
- Modify: `src/salesbench/vqa/specs.py`
- Modify: `src/salesbench/vqa/realizer.py`
- Modify: `src/salesbench/vqa/prompts.py`
- Modify: `src/salesbench/vqa/compiler.py`
- Modify: `src/salesbench/vqa/goldbank_loader.py`
- Modify: `src/salesbench/cli.py`
- Test: `tests/test_vqa_item_validator.py`
- Modify: `tests/test_vqa_specs.py`
- Modify: `tests/test_vqa_realizer.py`
- Modify: `tests/test_goldbank_qa_compiler.py`

**Interfaces:**
- Produces `validate_question_spec(...)` and `validate_qa_candidate(...)`.
- Adds `CompilePolicy.allow_auto_candidates`, task-specific `answer_type`, and active diversity selection.
- Formal compilation defaults to human-accepted rows only; candidate compilation is explicit.

- [ ] Write failing tests for lifecycle filtering, CM enums, BP answer types, speculative AE ambiguity, target/gold mismatch, template clusters, and formal compiler policy.
- [ ] Run focused tests and confirm current permissive behavior.
- [ ] Add local spec and QA item decisions before and after surface realization.
- [ ] Expand surface validation for length, generic stems, premise leakage, and normalized answer leakage.
- [ ] Assign task-specific answer schemas and normalize CM labels.
- [ ] Turn diversity reports into selection gates with per-video subtype quotas.
- [ ] Add explicit CLI flag for candidate compilation and keep formal default fail-closed.
- [ ] Run focused tests and the full suite.
- [ ] Commit with `feat: gate qa validity and candidate compilation`.

### Task 5: Actionable paginated audit workbench

**Files:**
- Modify: `tools/audit_workbench/review_queue.py`
- Modify: `tools/audit_workbench/build.py`
- Modify: `tests/test_audit_review_queue.py`
- Modify: `tests/test_audit_workbench.py`

**Interfaces:**
- Consumes separated quality artifacts directly.
- Exposes Human Review, Accepted Sample, Auto-Rejected, Diagnostics, and Abstentions as separate paginated views.
- Exports decisions only for actionable human rows.

- [ ] Write failing tests that diagnostic/rejected/abstention records have no review controls and full candidate snapshots remain readable.
- [ ] Run focused tests and confirm current inference-based workbench behavior.
- [ ] Load the separated artifacts and remove post-hoc queue reconstruction as the primary path.
- [ ] Render the five paginated views with lazy frames and readable Cue/Relation/Evidence content.
- [ ] Keep legacy v9 adapters read-only for historical audit files.
- [ ] Run focused tests and the full suite.
- [ ] Commit with `feat: separate actionable audit quality views`.

### Task 6: v10 configuration, integration verification, and smoke readiness

**Files:**
- Create: `configs/evidence_smoke_v10_5videos.json`
- Create: `configs/evidence_pilot_v10_64videos.json`
- Create: `configs/pilot64_gpt4o_v10_delivery.json`
- Modify: `tests/test_evidence_vqa_e2e.py`
- Modify: `tests/test_benchmark_convergence.py`
- Modify: `README.md`

**Interfaces:**
- Enables strict verification for v10 while leaving v9 immutable.
- Freezes schema/prompt/pipeline/compiler/audit versions and quality-policy fingerprints.

- [ ] Write failing integration tests for Evidence to quality decisions to QA to Judge input.
- [ ] Run focused integration tests and confirm missing v10 contracts.
- [ ] Add strict v10 configurations and delivery manifest.
- [ ] Verify public/model/Judge payloads contain no private fields.
- [ ] Verify human ambiguity queue routing and accepted-sample selection.
- [ ] Run `pytest -q`, compileall, `git diff --check`, and secret/private-field scans.
- [ ] Run the existing 5-video artifacts through non-network local compilation checks; run a new API smoke only when explicitly authorized in the execution turn.
- [ ] Commit with `feat: add v10 quality-gated benchmark pipeline`.
