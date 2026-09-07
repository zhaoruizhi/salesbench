# SalesBench Run Integrity and QA Quality Implementation Plan

> **REQUIRED SUB-SKILL:** Use `superpowers:executing-plans` to execute this plan task by task and `superpowers:test-driven-development` for every behavior change.

**Goal:** Produce a fresh five-video SalesBench candidate whose Evidence, QuestionSpecs, realized QA, compiled QA, Chinese audit translations, and HTML are bound to one immutable run; automatically reject structurally invalid or low-value items; reserve human review for genuine ambiguity and a small monitoring sample; and expose all accepted Chinese QA in a separate read-only paginated browser.

**Architecture:** Add a shared run-integrity module that creates canonical content fingerprints and validates upstream/downstream contracts. Evidence generation records source capabilities and a content fingerprint. QA realization consumes that fingerprint, namespaces its resume cache, applies deterministic and Qwen semantic gates, and records a realization fingerprint. Compilation fails closed on stale inputs or unresolved references and ranks valid candidates by explicit quality signals. The audit workbench repeats integrity checks and renders production queues separately from a Chinese-first final-product browser.

**Tech Stack:** Python 3.13, dataclasses/enums, SHA-256 canonical JSON fingerprints, JSON/JSONL artifacts, pytest, existing OpenAI-compatible `VLMClient`, DeepSeek text generation, Qwen multimodal extraction and QA semantic verification, static HTML/CSS/JavaScript audit workbench.

**Spec:** `docs/superpowers/specs/2026-09-07-salesbench-run-integrity-final-qa-design.md`

**Global Constraints:** Read API credentials only from the existing local `.env`; never print or commit credentials. Keep canonical Evidence, Gold, QA, prompts, model answers, and Judge output in English. Chinese is audit-only. Do not modify or commit `reporting/`. Do not commit generated `outputs/`. Treat unavailable ASR timing as a neutral source capability boundary, never as a human-review defect. Run all implementation work on the existing `codex/prompt-v8` feature branch because the user explicitly requested continuous commits in the current project.

---

## Task 1: Add canonical artifact fingerprints and reference-closure primitives

**Files:**

- Create: `src/salesbench/run_integrity.py`
- Create: `tests/test_run_integrity.py`
- Modify: `src/salesbench/__init__.py`

### Step 1: Write failing unit tests

Add tests that define:

- canonical hashes are independent of dictionary key order and volatile fields such as timestamps, latency, cost, and absolute output paths;
- an Evidence fingerprint changes when any accepted EvidenceUnit, CommerceCue, CommercialRelation, GroundedAnnotation, source hash, model, prompt, schema, pipeline, or policy input changes;
- `validate_reference_closure` checks annotation, Evidence, Cue, and Relation existence plus video ownership;
- unresolved or cross-video references raise `ReferenceClosureError` with stable error codes and counts;
- `read_required_fingerprint` rejects a missing or empty upstream fingerprint.

Run:

```bash
pytest -q tests/test_run_integrity.py
```

Expected RED: import failure because `salesbench.run_integrity` does not exist.

### Step 2: Implement the minimal shared contract

Implement:

```python
VOLATILE_FINGERPRINT_KEYS: frozenset[str]
class ReferenceClosureError(ValueError)
def canonical_sha256(value: object, *, exclude_keys: Iterable[str] = ()) -> str
def compute_evidence_fingerprint(...inputs...) -> str
def compute_qa_realization_fingerprint(...inputs...) -> str
def compute_compile_fingerprint(...inputs...) -> str
def read_required_fingerprint(meta_path: Path, field: str) -> str
def validate_reference_closure(dataset_rows, evidence_rows, cue_rows, relation_rows) -> dict[str, int]
```

Return full 64-character SHA-256 values for artifact identity. Keep existing short `stable_digest` IDs unchanged.

### Step 3: Verify GREEN and regressions

```bash
pytest -q tests/test_run_integrity.py tests/test_evidence_vqa_e2e.py
```

### Step 4: Commit

```bash
git add src/salesbench/run_integrity.py src/salesbench/__init__.py tests/test_run_integrity.py
git commit -m "feat: add artifact integrity contracts"
```

---

## Task 2: Bind Evidence output to one immutable run

**Files:**

- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `src/salesbench/goldbank/schema.py`
- Modify: `src/salesbench/cli.py`
- Modify: `tests/test_goldbank_runner.py`
- Modify: `tests/test_goldbank_cli.py`
- Modify: `tests/test_goldbank_schema.py`

### Step 1: Write failing runner and CLI tests

Add tests asserting:

- Evidence `generation_meta.json` contains `benchmark_release`, `run_id`, `release_status=CANDIDATE`, `component_versions`, `source_fingerprint`, and `evidence_fingerprint`;
- the Evidence fingerprint covers final accepted graph files, not merely configuration;
- a completed Evidence directory with a different run ID or source/config fingerprint fails rather than being overwritten;
- resume remains allowed when run ID and fingerprints match;
- `build-evidence-dataset` accepts `--run-id` and `--benchmark-release` and passes them to the runner.

Run:

```bash
pytest -q tests/test_goldbank_runner.py tests/test_goldbank_cli.py tests/test_goldbank_schema.py
```

Expected RED: missing CLI options and missing fingerprint fields.

### Step 2: Implement run metadata and immutable completion checks

- Bump Evidence pipeline to `evidence-first-pipeline-v10.4` and prompt/schema versions only where their contracts change.
- Extend `build_gold_bank_dataset`/`run_gold_bank_records` with `run_id` and `benchmark_release`.
- Write `run_manifest.json` at the run root when `--run-id` is supplied and Evidence output resides under `outputs/runs/<run_id>/evidence`.
- Compute `source_fingerprint` before API work and `evidence_fingerprint` after `_merge_outputs` writes canonical accepted artifacts.
- Reject reuse of a completed stage when its stored run ID or source fingerprint differs.
- Keep per-video resume parts valid only for the matching pipeline/source fingerprint.

### Step 3: Verify GREEN

```bash
pytest -q tests/test_goldbank_runner.py tests/test_goldbank_cli.py tests/test_goldbank_schema.py tests/test_evidence_vqa_e2e.py
```

### Step 4: Commit

```bash
git add src/salesbench/goldbank/runner.py src/salesbench/goldbank/schema.py src/salesbench/cli.py tests/test_goldbank_runner.py tests/test_goldbank_cli.py tests/test_goldbank_schema.py
git commit -m "feat: make evidence runs immutable"
```

---

## Task 3: Normalize unavailable ASR time as a capability boundary

**Files:**

- Modify: `src/salesbench/goldbank/normalizer.py`
- Modify: `src/salesbench/goldbank/validators.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/commerce_ontology.py`
- Modify: `tools/audit_workbench/build.py`
- Modify: `tests/test_goldbank_normalizer.py`
- Modify: `tests/test_goldbank_validators.py`
- Modify: `tests/test_goldbank_pipeline.py`
- Modify: `tests/test_audit_workbench.py`

### Step 1: Write failing timestamp/capability tests

Cover these behaviors:

- ASR `0/0` with `timestamp_status=unavailable` normalizes to `start_s=null`, `end_s=null`;
- unavailable ASR is valid for non-temporal claims and receives no `MISSING_TEMPORAL_LOCALIZATION` issue;
- OCR still requires a frame/time locator when the producer claims it is available;
- a capability mask reports `spoken_claim_extraction=true`, `cross_modal_semantics=true`, and `asr_temporal_order=false`;
- temporal-only relation requests are emitted as `TEMPORAL_CAPABILITY_UNAVAILABLE` abstentions/diagnostics and never enter `human_review_queue.jsonl`;
- the HTML presents a neutral source-capability notice rather than a red missing-time risk.

Run:

```bash
pytest -q tests/test_goldbank_normalizer.py tests/test_goldbank_validators.py tests/test_goldbank_pipeline.py tests/test_audit_workbench.py
```

Expected RED: ASR is currently flagged as missing temporal localization and temporal relations can reach review.

### Step 2: Implement capability-aware normalization and routing

- Canonicalize unavailable timestamps to `None`.
- Add a pure `derive_source_capabilities(evidence)` helper.
- Partition relation types/capabilities into temporal-required and non-temporal sets.
- Skip temporal proposal/adjudication inputs when timing is unavailable.
- Record abstention counts and capability flags in Evidence metadata without creating human work.
- Update workbench risk routing so only contradictory producer claims about timing are errors.

### Step 3: Verify GREEN

```bash
pytest -q tests/test_goldbank_normalizer.py tests/test_goldbank_validators.py tests/test_goldbank_pipeline.py tests/test_audit_workbench.py
```

### Step 4: Commit

```bash
git add src/salesbench/goldbank/normalizer.py src/salesbench/goldbank/validators.py src/salesbench/goldbank/pipeline.py src/salesbench/goldbank/commerce_ontology.py tools/audit_workbench/build.py tests/test_goldbank_normalizer.py tests/test_goldbank_validators.py tests/test_goldbank_pipeline.py tests/test_audit_workbench.py
git commit -m "fix: treat missing asr timing as capability boundary"
```

---

## Task 4: Enforce Evidence, CommerceCue, and Gold semantic quality

**Files:**

- Modify: `src/salesbench/goldbank/schema.py`
- Modify: `src/salesbench/goldbank/commerce_schema.py`
- Modify: `src/salesbench/goldbank/commerce_ontology.py`
- Modify: `src/salesbench/goldbank/normalizer.py`
- Modify: `src/salesbench/goldbank/validators.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/quality_prompts.py`
- Modify: `tests/test_commerce_schema.py`
- Modify: `tests/test_commerce_ontology.py`
- Modify: `tests/test_goldbank_prompts.py`
- Modify: `tests/test_goldbank_validators.py`
- Modify: `tests/test_goldbank_pipeline.py`
- Modify: `tests/test_goldbank_quality_gate.py`

### Step 1: Write failing semantic-policy tests

Test that production rejects or downgrades:

- `PRODUCT_IDENTITY` that merely says a presenter holds, points to, shows, rotates, or flips an unspecified object;
- generic page flipping/holding as `PROCESS_DEMONSTRATION` or `USAGE_STEP`;
- a spoken claim rewritten as an observed fact;
- a conditional, instruction, hypothetical, or promotional promise whose scope is promoted in Cue, Relation, or Gold;
- Gold answer text containing internal IDs;
- multi-focus Gold answers;
- answers over BP 20, CM/AE 45, or SS 50 English words;
- background handling that produces a BP/SS/AE proposal.

Also test valid product identity, product inspection, functional operation, and observable outcome examples.

Run:

```bash
pytest -q tests/test_commerce_schema.py tests/test_commerce_ontology.py tests/test_goldbank_prompts.py tests/test_goldbank_validators.py tests/test_goldbank_pipeline.py tests/test_goldbank_quality_gate.py
```

Expected RED: action-only identity and overlong/multi-focus Gold currently survive.

### Step 2: Extend controlled semantics

- Add `AssertionScope`: `OBSERVED_FACT`, `SPOKEN_CLAIM`, `CONDITIONAL`, `INSTRUCTION`, `HYPOTHETICAL`, `PROMOTIONAL_PROMISE`.
- Add `ActionRole`: `BACKGROUND_HANDLING`, `PRODUCT_INSPECTION`, `FUNCTIONAL_OPERATION`, `OUTCOME_DEMONSTRATION`.
- Preserve backward-compatible defaults while parsing old candidate artifacts.
- Bump prompt to `evidence-prompt-v10.3` and quality prompt to `quality-gate-prompt-v3`.
- Require extractors to state scope/role explicitly and abstain when identity or commercial relevance is absent.
- Add deterministic scope-monotonicity, product-identity, action-role, internal-ID, answer-length, and single-focus checks before Challenger/Adjudicator output can become Gold.
- Route deterministic failures to `auto_rejected` or `pipeline_diagnostics`; only genuinely ambiguous supported readings become human review.

### Step 3: Verify GREEN

```bash
pytest -q tests/test_commerce_schema.py tests/test_commerce_ontology.py tests/test_goldbank_prompts.py tests/test_goldbank_validators.py tests/test_goldbank_pipeline.py tests/test_goldbank_quality_gate.py
```

### Step 4: Commit

```bash
git add src/salesbench/goldbank/schema.py src/salesbench/goldbank/commerce_schema.py src/salesbench/goldbank/commerce_ontology.py src/salesbench/goldbank/normalizer.py src/salesbench/goldbank/validators.py src/salesbench/goldbank/pipeline.py src/salesbench/goldbank/prompts.py src/salesbench/goldbank/quality_prompts.py tests/test_commerce_schema.py tests/test_commerce_ontology.py tests/test_goldbank_prompts.py tests/test_goldbank_validators.py tests/test_goldbank_pipeline.py tests/test_goldbank_quality_gate.py
git commit -m "feat: enforce commercial evidence semantics"
```

---

## Task 5: Bind QA realization to Evidence and separate DeepSeek/Qwen roles

**Files:**

- Modify: `src/salesbench/vqa/realizer.py`
- Modify: `src/salesbench/vqa/prompts.py`
- Modify: `src/salesbench/vqa/semantic_verifier.py`
- Modify: `src/salesbench/vqa/item_validator.py`
- Modify: `src/salesbench/cli.py`
- Modify: `tests/test_vqa_realizer.py`
- Modify: `tests/test_vqa_semantic_verifier.py`
- Modify: `tests/test_vqa_item_validator.py`
- Modify: `tests/test_goldbank_cli.py`

### Step 1: Write failing QA-gate and provider-routing tests

Define:

- `run_qa_realizer` refuses Evidence without `evidence_fingerprint`;
- resume parts live in `.parts/<evidence_fingerprint>/<spec_id>.json` and orphan parts are ignored;
- `qa_realizer_meta.json` records the exact Evidence fingerprint and a full `qa_realization_fingerprint`;
- the verifier response includes all twelve quality booleans from the design;
- PASS is converted to REJECT if any applicable dimension is false;
- generic/simple background-action questions, scope promotion, modality shortcuts, and commercially non-diagnostic CM/SS/AE questions are rejected;
- `realize-qa --strict-semantic-verification` constructs a DeepSeek realizer client from `DEEPSEEK_*` and an independent Qwen verifier client from `QWEN_*`, and passes both to `run_qa_realizer`.

Run:

```bash
pytest -q tests/test_vqa_realizer.py tests/test_vqa_semantic_verifier.py tests/test_vqa_item_validator.py tests/test_goldbank_cli.py
```

Expected RED: Evidence fingerprint is not consumed, parts are unnamespaced, five dimensions are missing, and the verifier reuses DeepSeek.

### Step 2: Implement prompt v3 and QA quality v2

- Bump `QUESTION_REALIZER_PROMPT_VERSION` to `question-realizer-prompt-v3`.
- Bump `QA_QUALITY_PROMPT_VERSION` to `qa-quality-prompt-v2`.
- Add `content_specific`, `commerce_relevant`, `natural_question`, `non_trivial`, `commercially_diagnostic`, `claim_scope_preserved`, `intended_modality_required`, and `reference_closed`, retaining the existing core fields under a single strict schema.
- Add deterministic local checks for background-action stems, internal IDs, answer length, obvious tautology, scope markers, and task-required graph/modality references.
- Pass an independent Qwen `semantic_verifier_client` through CLI flags/env: `--verifier-api-key`, `--verifier-base-url`, `--verifier-model`, with `QWEN_*` fallbacks.
- Store rejection reason codes and quality dimensions for later ranking and audit.

### Step 3: Verify GREEN

```bash
pytest -q tests/test_vqa_realizer.py tests/test_vqa_semantic_verifier.py tests/test_vqa_item_validator.py tests/test_goldbank_cli.py
```

### Step 4: Commit

```bash
git add src/salesbench/vqa/realizer.py src/salesbench/vqa/prompts.py src/salesbench/vqa/semantic_verifier.py src/salesbench/vqa/item_validator.py src/salesbench/cli.py tests/test_vqa_realizer.py tests/test_vqa_semantic_verifier.py tests/test_vqa_item_validator.py tests/test_goldbank_cli.py
git commit -m "feat: bind qa realization to verified evidence"
```

---

## Task 6: Fail closed at compile time and rank by QA quality

**Files:**

- Create: `src/salesbench/vqa/ranking.py`
- Modify: `src/salesbench/vqa/compiler.py`
- Modify: `src/salesbench/vqa/specs.py`
- Modify: `tests/test_goldbank_qa_compiler.py`
- Modify: `tests/test_vqa_specs.py`
- Modify: `tests/test_evidence_vqa_e2e.py`

### Step 1: Write failing compiler tests

Add tests that require:

- compile aborts before writing outputs when Evidence and realization fingerprints differ;
- compile aborts on any missing/cross-video annotation, Evidence, Cue, or Relation reference;
- the closure report shows 100% for all reference classes;
- quality score outranks `gold_id` hash under per-video/task quotas;
- one capability does not crowd out all other useful capabilities for a video;
- rejected, human-review, pipeline-diagnostic, temporal-unavailable, or failed semantic-verifier rows never compile;
- `qa_selection.jsonl` records selected and skipped candidates with component scores/reasons;
- compile metadata contains Evidence, realization, and compile fingerprints.

Run:

```bash
pytest -q tests/test_goldbank_qa_compiler.py tests/test_vqa_specs.py tests/test_evidence_vqa_e2e.py
```

Expected RED: compiler only checks partial semantic pass state and sorts by task then hash.

### Step 2: Implement deterministic ranking and fail-closed compilation

Implement `score_qa_candidate` with explicit components:

- evidence directness and confidence;
- valid multimodal/commerce relation strength;
- QA semantic quality dimensions;
- content specificity and commercial diagnostic value;
- penalties for simple background actions, repeated capability, verbosity, and near-duplicate phrasing.

Then:

- bump compiler to `evidence-qa-compiler-v9`;
- validate fingerprints and full reference closure before creating output files;
- sort by descending score, then stable ID as tie-breaker;
- diversify by task and capability while respecting quotas;
- emit `qa_selection.jsonl`, `reference_closure.json`, and fingerprint-rich `generation_meta.json`.

### Step 3: Verify GREEN

```bash
pytest -q tests/test_goldbank_qa_compiler.py tests/test_vqa_specs.py tests/test_evidence_vqa_e2e.py
```

### Step 4: Commit

```bash
git add src/salesbench/vqa/ranking.py src/salesbench/vqa/compiler.py src/salesbench/vqa/specs.py tests/test_goldbank_qa_compiler.py tests/test_vqa_specs.py tests/test_evidence_vqa_e2e.py
git commit -m "feat: compile qa by integrity and quality"
```

---

## Task 7: Add a Chinese-first paginated final QA browser

**Files:**

- Modify: `tools/audit_workbench/build.py`
- Modify: `tools/audit_workbench/review_queue.py`
- Modify: `src/salesbench/audit_translation.py`
- Modify: `tests/test_audit_workbench.py`
- Modify: `tests/test_audit_review_queue.py`
- Modify: `tests/test_audit_translation.py`

### Step 1: Write failing workbench tests

Require:

- six distinct tabs: Run Overview, Prompt, Evidence Audit, QA Audit, Final QA Browser, Evaluation/Judge;
- Final QA Browser reads only compiled accepted `vqa_gold_private.jsonl`, never rejected/spec/parts rows;
- it displays Chinese question and answer first, then translated Evidence/Cue/Relation content and lazy-loaded linked frames;
- canonical English is available in an expandable block;
- exactly one final QA is visible per page with previous/next, index, video, task, capability, and quality filters;
- the page has no accept/reject audit buttons;
- stale fingerprints or incomplete closure show a blocking banner and suppress the “final” label;
- unavailable ASR time appears as a neutral notice;
- private interaction/follower/title metadata never enters the HTML payload.

Run:

```bash
pytest -q tests/test_audit_workbench.py tests/test_audit_review_queue.py tests/test_audit_translation.py
```

Expected RED: accepted QA is currently mixed into audit-oriented data and lacks final-product integrity state.

### Step 2: Implement workbench v11

- Build a dedicated `final_qa` payload from compiled QA plus translation sidecar and graph lookups.
- Repeat reference-closure and fingerprint checks when loading a group.
- Add paginated client-side rendering with `loading="lazy"` thumbnails and no eager full-frame embedding.
- Keep Evidence/QA audit queues paginated and actionable; keep final products read-only.
- Extend translation jobs only for canonical fields needed by final QA; never translate identifiers or private metadata.

### Step 3: Verify GREEN

```bash
pytest -q tests/test_audit_workbench.py tests/test_audit_review_queue.py tests/test_audit_translation.py
```

### Step 4: Commit

```bash
git add tools/audit_workbench/build.py tools/audit_workbench/review_queue.py src/salesbench/audit_translation.py tests/test_audit_workbench.py tests/test_audit_review_queue.py tests/test_audit_translation.py
git commit -m "feat: add final qa product browser"
```

---

## Task 8: Add the fresh v10 candidate run configuration and operating guide

**Files:**

- Create: `configs/evidence_smoke_v10_candidate2_5videos.json`
- Create: `configs/v10_candidate2_smoke_delivery.json`
- Modify: `docs/Pilot_v10_Quality_Gate_Execution_Guide.md`
- Modify: `tests/test_benchmark_convergence.py`
- Modify: `tests/test_goldbank_cli.py`

### Step 1: Write failing config/document-contract tests

Test that:

- the candidate config declares the immutable `run_id`, benchmark release, five exact cohort video IDs, strict Evidence verification, strict QA verification, and review sampling rates;
- delivery sources all resolve under the same `outputs/runs/<run_id>/` root;
- no API key value is present in tracked config or guide;
- the guide names Qwen extraction/verification and DeepSeek realization, and explains why unavailable ASR timing disables temporal tasks.

Run:

```bash
pytest -q tests/test_benchmark_convergence.py tests/test_goldbank_cli.py
```

Expected RED: candidate2 configs and documented immutable flow do not exist.

### Step 2: Add exact configs and commands

Use run ID:

```text
v10c2-smoke5-qwen-deepseek-20260907-001
```

All generated paths must be children of:

```text
outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/
```

The guide must load `.env` without echoing values and use these existing variable names:

```text
QWEN_API_KEY, QWEN_BASE_URL, QWEN_VISION_MODEL
DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
```

### Step 3: Verify GREEN and scan secrets

```bash
pytest -q tests/test_benchmark_convergence.py tests/test_goldbank_cli.py
git grep -nE 'sk-[A-Za-z0-9._-]{12,}' -- ':!.env'
```

The secret scan must return no newly introduced credentials.

### Step 4: Commit

```bash
git add configs/evidence_smoke_v10_candidate2_5videos.json configs/v10_candidate2_smoke_delivery.json docs/Pilot_v10_Quality_Gate_Execution_Guide.md tests/test_benchmark_convergence.py tests/test_goldbank_cli.py
git commit -m "docs: define immutable v10 candidate smoke run"
```

---

## Task 9: Run full verification before spending API calls

**Files:** No source changes expected.

### Step 1: Run the full local suite

```bash
pytest -q
python -m compileall -q src tools tests
git diff --check
git status --short
```

Expected:

- all tests pass;
- compileall and diff check pass;
- only generated ignored outputs and user-owned `reporting/` remain outside commits.

### Step 2: Stop on any failure

Do not invoke remote APIs until the local suite is green. Fix failures using a new RED/GREEN cycle and commit the correction.

---

## Task 10: Generate and audit one fresh five-video candidate

**Files:** Generated under ignored `outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/`; do not commit.

### Step 1: Load current local provider configuration safely

Use a shell that sources `.env` without printing it. Verify only that required variable names are non-empty and that model/base URL host selections are correct; never print credential values.

### Step 2: Generate Evidence with Qwen vision and DeepSeek text reasoning

```bash
python salesbench.py build-evidence-dataset \
  --config configs/benchmark_v1.json \
  --cohort-config configs/evidence_smoke_v10_candidate2_5videos.json \
  --output-dir outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/evidence \
  --run-id v10c2-smoke5-qwen-deepseek-20260907-001 \
  --benchmark-release salesbench-v10-candidate.2 \
  --max-workers 2
```

Inspect counts, graph validity, automatic rejections, genuine ambiguity, source capability mask, private-field leakage, answer length, internal-ID leakage, and action-role distribution before continuing.

### Step 3: Realize English QA with DeepSeek and verify with Qwen

```bash
python salesbench.py realize-qa \
  --evidence-dir outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/evidence \
  --dataset-file video_evidence_dataset.jsonl \
  --output-dir outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/qa/realizations \
  --allow-auto-candidates \
  --strict-semantic-verification \
  --max-workers 2
```

Confirm metadata names DeepSeek as realizer and Qwen as verifier and that all parts are beneath the Evidence fingerprint namespace.

### Step 4: Compile only fingerprint-matched, reference-closed QA

```bash
python salesbench.py compile-vqa \
  --evidence-dir outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/evidence \
  --dataset-file video_evidence_dataset.jsonl \
  --realizations outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/qa/realizations/qa_realizations.jsonl \
  --output-dir outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/qa/compiled \
  --allow-auto-candidates \
  --allow-missing-tasks
```

Require `reference_closure.json` to report 100% for annotations, Evidence, Cues, and Relations. Review `qa_selection.jsonl` and all compiled rows. Do not continue if integrity is blocked.

### Step 5: Generate audit-only Chinese translations

```bash
python salesbench.py build-audit-translations \
  --manifest configs/v10_candidate2_smoke_delivery.json \
  --output outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/translations/audit_translations.jsonl \
  --model "$DEEPSEEK_MODEL" \
  --base-url "$DEEPSEEK_BASE_URL" \
  --batch-size 20
```

Preserve strict controlled-token and number checks. Missing translations remain visibly missing; do not weaken validation.

### Step 6: Build the lazy-loading audit HTML

```bash
python -m tools.audit_workbench.build \
  --manifest configs/v10_candidate2_smoke_delivery.json \
  --translations outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/translations/audit_translations.jsonl \
  --output outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/audit/SalesBench_v10c2_Smoke_Audit.html \
  --fragment outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/audit/SalesBench_v10c2_Smoke_Audit_fragment.html \
  --group smoke \
  --skip-organize
```

### Step 7: Inspect the generated product before evaluation

Verify:

- no stale/incomplete banner;
- five processed videos and exact artifact fingerprints;
- no public/private metadata leakage;
- no temporal ASR review items;
- Evidence/QA audit queues contain only genuine ambiguities plus deterministic monitoring samples;
- Final QA Browser contains every compiled row, Chinese first, one item per page, readable cited graph content, and lazy-linked frames;
- none of the four previously identified unnatural/simple question patterns survives unless its commercial/diagnostic value is explicit.

Start a local server only after the checks pass:

```bash
python -m http.server 8765 --directory outputs/runs/v10c2-smoke5-qwen-deepseek-20260907-001/audit
```

Deliver the local URL for user review. Do not run the 64-video cohort until the user accepts this smoke candidate.

---

## Task 11: Final source verification and handoff

**Files:** No generated files committed.

### Step 1: Verify repository state

```bash
pytest -q
python -m compileall -q src tools tests
git diff --check
git status --short
git log --oneline -12
```

### Step 2: Report exact candidate status

Report:

- commit IDs for each implementation slice;
- test count and verification commands;
- Evidence/Cue/Relation/Annotation and QA counts by task;
- auto-rejected, pipeline-diagnostic, genuine human-review, and monitoring-sample counts separately;
- fingerprint/reference-closure status;
- translation coverage;
- local HTML URL and generated artifact paths;
- any abstentions or failures without relabeling them as success;
- the explicit gate that the 64-video run waits for human acceptance of the five-video candidate.
