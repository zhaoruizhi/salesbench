# SalesBench Prompt v8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an English-only Prompt v8 runtime with explicit BP/CM/SS/AE generation paths and a canonical, readable review-queue contract that remains compatible with historical v6/v7 artifacts.

**Architecture:** Keep BP as a deterministic evidence-to-candidate compiler and route CM, SS, and AE through one task-specific proposer each. Normalize public EvidenceDataset/QA/Judge natural language to English while retaining verbatim source-language OCR/ASR only in provenance fields. Emit one canonical review item shape at pipeline time and use a compatibility adapter to normalize older nested proposal and abstention rows for the HTML audit workbench.

**Tech Stack:** Python 3.13, dataclasses and JSON/JSONL, pytest, static HTML/CSS/JavaScript audit workbench, Git.

## Global Constraints

- Public EvidenceDataset normalized fields, QA, Prompt v8 instructions, and Judge outputs use English.
- Verbatim OCR/ASR evidence remains in its original language in `text_span`; it is never silently translated.
- Public model/Judge payloads never include private interaction metrics, fan counts, titles, or private analysis metadata.
- BP is generated deterministically from validated EvidenceUnits; CM, SS, and AE each have one dedicated LLM proposer.
- Challenger, Adjudicator, and local validators remain shared independent quality gates.
- Historical v6/v7 outputs are not rewritten or relabeled as v8.
- Every production behavior change starts with a failing test and ends with a focused Git commit.

---

### Task 1: Prompt v8 and four task generation paths

**Files:**
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/normalizer.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `tests/test_goldbank_prompts.py`
- Modify: `tests/test_goldbank_normalizer.py`
- Modify: `tests/test_goldbank_pipeline.py`

**Interfaces:**
- Consumes: validated `EvidenceUnit` dictionaries and task-specific model JSON.
- Produces: `BP_COMPILER_CONTRACT`, Prompt v8 messages, canonical generator names `bp_compiler`, `cm_proposer`, `ss_proposer`, `ae_proposer`, and English normalized candidates.

- [x] **Step 1: Write failing prompt, routing, and language-contract tests**

  Add tests asserting Prompt v8 contains English task contracts, the three LLM calls use task-specific generator names, BP remains local, and normalized non-verbatim fields reject Chinese output while `text_span` accepts original Chinese.

- [x] **Step 2: Run focused tests and confirm contract failures**

  Run: `pytest -q tests/test_goldbank_prompts.py tests/test_goldbank_normalizer.py tests/test_goldbank_pipeline.py`

  Expected: failures identify the v7 prompt version, legacy perspective names, and missing English-language validation.

- [x] **Step 3: Implement Prompt v8 and generator routing**

  Rewrite all stage instructions and examples in English; expose the deterministic BP compiler contract; replace consumer/operator/strategist with CM/SS/AE proposers; assign BP candidates to `bp_compiler`; bump prompt and pipeline fingerprints.

- [x] **Step 4: Run focused and full tests**

  Run: `pytest -q tests/test_goldbank_prompts.py tests/test_goldbank_normalizer.py tests/test_goldbank_pipeline.py`

  Run: `pytest -q`

- [x] **Step 5: Commit**

  Commit message: `feat: add English prompt v8 task generators`

### Task 2: English QA compiler and Judge contract

**Files:**
- Modify: `src/salesbench/vqa/question_programs.py`
- Modify: `src/salesbench/vqa/compiler.py`
- Modify: `src/salesbench/vqa_evaluate/prompts.py`
- Modify: `tests/test_vqa_question_programs.py`
- Modify: `tests/test_vqa_compiler.py`
- Modify: `tests/test_vqa_evaluate.py`

**Interfaces:**
- Consumes: accepted English `GroundedAnnotation` records.
- Produces: English BP/CM/SS/AE questions and reference answers, English task-specific Judge payloads, and a bumped QA compiler/Judge prompt fingerprint.

- [x] **Step 1: Write failing English-output and Judge tests**

  Add literal fixtures asserting all four compiled tasks produce English questions/answers, reject CJK in normalized public QA fields, preserve original-language evidence separately, and request English-only Judge explanations.

- [x] **Step 2: Run focused tests and confirm failures**

  Run: `pytest -q tests/test_vqa_question_programs.py tests/test_vqa_compiler.py tests/test_vqa_evaluate.py`

- [x] **Step 3: Translate templates and enforce public-language boundaries**

  Replace Chinese templates/default labels with English, add compiler validation at the public dataset boundary, rewrite the Judge prompt/rubrics in English, and bump compiler/Judge prompt versions.

- [x] **Step 4: Run focused and full tests**

  Run the focused command from Step 2, then `pytest -q`.

- [x] **Step 5: Commit**

  Commit message: `feat: normalize vqa and judge output to English`

### Task 3: Canonical review queue and historical adapter

**Files:**
- Create: `tools/audit_workbench/review_queue.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `tools/audit_workbench/build.py`
- Modify: `tools/audit_workbench/assets/app.js`
- Modify: `tools/audit_workbench/assets/styles.css`
- Create: `tests/test_audit_review_queue.py`
- Modify: `tests/test_audit_workbench.py`

**Interfaces:**
- Consumes: canonical v8 queue rows and legacy rows containing `source_proposal_ids`, nested proposer records, nested GoldItems, or abstentions.
- Produces: `normalize_review_queue_row(row, proposal_index)` with `review_item_id`, `stage`, `item_type`, `task_type`, `task_subtype`, `reason_code`, `reason`, `target`, `candidate_gold`, `evidence_refs`, `source_proposal_ids`, and `issues`.

- [x] **Step 1: Write failing adapter and UI-data tests**

  Cover canonical rows, source-ID lookup, nested proposal aliases, nested GoldItem aliases, abstentions, unresolved references, and task recovery without generic `UNKNOWN` when the source contains a task.

- [x] **Step 2: Run focused tests and confirm failures**

  Run: `pytest -q tests/test_audit_review_queue.py tests/test_audit_workbench.py tests/test_goldbank_pipeline.py`

- [x] **Step 3: Emit and normalize the canonical queue shape**

  Add pipeline queue builders for candidates, abstentions, adjudication conflicts, and stage failures. Add the historical adapter and make the workbench consume only its normalized output.

- [x] **Step 4: Render type-specific readable audit cards**

  Candidate/conflict cards show target, candidate Gold, readable evidence content, and issues; abstentions show “No candidate generated” plus the exact reason; failures show stage/reason and available representative frames; unresolved references are explicitly labeled.

- [x] **Step 5: Run focused and full tests**

  Run the focused command from Step 2, then `pytest -q`.

- [x] **Step 6: Commit**

  Commit message: `feat: normalize review queue audit records`

### Task 4: v8 configs, documentation, generated workbench, and release verification

**Files:**
- Create: `configs/evidence_smoke_v8_5videos.json`
- Create: `configs/evidence_pilot_v8_64videos.json`
- Modify: `docs/Pilot_64_Video_Execution_Guide.md`
- Modify: `docs/data/EvidenceDataset_Data_Card_v2.md`
- Modify: `tools/audit_workbench/build.py`
- Generate (ignored): `outputs/audit/SalesBench_Prompt_Audit_Workbench_v8.html`

**Interfaces:**
- Consumes: v8 source contracts and existing v6/v7 audit artifacts.
- Produces: separate smoke/formal configs, an explicit v8 execution guide, and a self-contained audit workbench that labels source-result versions honestly.

- [x] **Step 1: Add versioned smoke/formal configs and update execution documentation**

  Preserve v7 files as historical records. Document that v8 code does not create v8 EvidenceDataset/QA/evaluation results until the API pipeline is rerun.

- [x] **Step 2: Build the v8 workbench**

  Run the workbench builder against the currently available historical artifacts and output `outputs/audit/SalesBench_Prompt_Audit_Workbench_v8.html` with clear runtime/result-version labels.

- [x] **Step 3: Verify generated data and HTML behavior**

  Check that no private fields enter public payloads, prompt tabs show BP/CM/SS/AE paths, queue rows have readable content, and lazy frame loading remains intact.

- [x] **Step 4: Run release verification**

  Run: `pytest -q`

  Run: `python -m compileall -q src tools`

  Run: `git diff --check`

- [x] **Step 5: Commit and push**

  Commit message: `docs: publish prompt v8 pilot workflow`

  Push branch: `git push -u origin codex/prompt-v8`
