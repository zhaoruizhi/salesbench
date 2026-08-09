# SalesBench Visual Evidence Audit and Prompt v7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Evidence, QA, and Judge audit record human-readable with lazily loaded linked frames, and introduce a schema-tight Chinese-natural-language Prompt v7 without relabeling existing v6 outputs.

**Architecture:** Split the audit builder into evidence enrichment, thumbnail materialization, and HTML rendering responsibilities. Frame associations are derived deterministically from EvidenceUnit refs and frame-cache manifests; generated thumbnails live beside the HTML and are loaded lazily. Prompt v7 changes the generation contracts while the workbench carries separate runtime-v6 and current-v7 version labels.

**Tech Stack:** Python 3.13 standard library, macOS `sips` with source-copy fallback, semantic HTML/CSS, vanilla JavaScript, pytest.

## Global Constraints

- Never embed complete videos or interaction metadata in the audit page.
- Never expose `.env`, API keys, relay URLs, or private interaction fields.
- Existing 64-video artifacts remain labeled `evidence-prompt-v6`.
- Current code Prompt becomes `evidence-prompt-v7`; this does not imply the 64 videos were rerun.
- Natural-language outputs are Chinese; schema keys and controlled enums remain English.
- Full audit HTML references relative lazy-loaded thumbnails; conversation preview remains below 1 MB.
- Every production behavior change follows a failing-test-first cycle.

---

### Task 1: Enrich audit records with EvidenceUnit content and frame descriptors

**Files:**
- Create: `tools/audit_workbench/evidence_assets.py`
- Modify: `tools/audit_workbench/build.py`
- Test: `tests/test_audit_workbench.py`

**Interfaces:**
- Produces `load_frame_manifests(frame_cache_root: Path, video_ids: set[str]) -> dict[str, dict[int, dict[str, Any]]]`.
- Produces `enrich_evidence_refs(video_id: str, evidence_refs: list[str], evidence_by_id: dict[str, dict[str, Any]], frame_manifests: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]`.
- Each enriched item contains the EvidenceUnit semantic fields, localization note, and frame descriptors.

- [ ] Write failing tests for direct visual/OCR frames, missing-time ASR representative frames, and missing refs.
- [ ] Run `pytest -q tests/test_audit_workbench.py -k 'evidence_content or representative_frames'` and confirm assertion failures.
- [ ] Implement manifest loading, direct frame association, nearest/representative selection, and readable localization notes.
- [ ] Join enriched evidence into queue, annotation-risk, QA, and Judge records.
- [ ] Run the targeted tests and commit the independently working enrichment layer.

### Task 2: Materialize unique thumbnails and lazy-load them in HTML

**Files:**
- Modify: `tools/audit_workbench/evidence_assets.py`
- Modify: `tools/audit_workbench/build.py`
- Test: `tests/test_audit_workbench.py`

**Interfaces:**
- Produces `materialize_thumbnails(data: dict[str, Any], repo_root: Path, asset_root: Path, html_parent: Path) -> dict[str, int]`.
- Rewrites each frame descriptor with a POSIX relative `thumbnail_src`.
- Uses unique `(video_id, frame_index)` keys and never copies video files.

- [ ] Write failing tests proving deduplication, relative paths, missing-image status, and `loading="lazy"` HTML output.
- [ ] Run targeted tests and confirm failure because thumbnail materialization does not exist.
- [ ] Implement `sips -Z 360 -s format jpeg -s formatOptions 60` with `shutil.copy2` fallback.
- [ ] Render responsive evidence cards, frame captions, missing-image placeholder, and click-to-open lightbox.
- [ ] Run targeted tests and commit the thumbnail/lazy-loading layer.

### Task 3: Make QA and Judge cards readable without ID lookup

**Files:**
- Modify: `tools/audit_workbench/build.py`
- Test: `tests/test_audit_workbench.py`

**Interfaces:**
- QA rows expose `evidence_items` and source annotation summary.
- Judge rows reuse QA evidence via `vqa_id` and expose the same `evidence_items`.

- [ ] Write failing tests asserting QA and Judge HTML contains semantic Evidence text and frame captions rather than only refs.
- [ ] Confirm the tests fail against the current renderer.
- [ ] Add reusable JavaScript renderers for evidence semantics and frames.
- [ ] Verify filters and local review decisions still operate after expanded cards.
- [ ] Run targeted tests and commit the readable QA/Judge audit UI.

### Task 4: Introduce evidence-prompt-v7

**Files:**
- Modify: `src/salesbench/goldbank/prompts.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `tests/test_goldbank_prompts.py`
- Modify: `tests/test_goldbank_pipeline.py`

**Interfaces:**
- `PROMPT_VERSION = "evidence-prompt-v7"`.
- Proposer roles become Consumer→AE, Operator→CM, Strategist→SS.
- Adjudicator returns decisions/groups rather than rewriting annotations; pipeline reconstructs accepted annotations from source proposals.

- [ ] Write failing Prompt contract tests for Chinese natural-language fields, exact subtype schemas, role separation, CM modality diversity, canonical local IDs, and accepted-group Adjudicator output.
- [ ] Run tests and confirm v6 violates the new contracts.
- [ ] Rewrite Prompt builders in Chinese while preserving English schema keys/enums.
- [ ] Update parsing/pipeline only where required for the reduced Adjudicator response, keeping local validation authoritative.
- [ ] Run Goldbank Prompt and pipeline tests and commit Prompt v7.

### Task 5: Draft the task-specific Chinese Judge v7 contract

**Files:**
- Modify: `src/salesbench/vqa_evaluate/prompts.py`
- Modify: `tests/test_vqa_evaluate.py`

**Interfaces:**
- Judge JSON keeps backward-compatible `score` and adds `correctness`, `grounding`, and `completeness` controlled scores.
- `reason` and `evidence_alignment` must be Chinese.

- [ ] Write failing tests for task-specific Chinese rubrics and Chinese output requirements.
- [ ] Confirm current English generic Judge prompt fails.
- [ ] Implement the Chinese rubric with separate BP/CM/SS/AE criteria and backward-compatible final score.
- [ ] Run evaluator tests and commit Judge Prompt v7.

### Task 6: Rebuild and verify the audit deliverables

**Files:**
- Generate: `outputs/audit/SalesBench_Prompt_Audit_Workbench_v7.html`
- Generate: `outputs/audit/assets/frames/`
- Generate: thread visualization fragment and its `assets/frames/`

**Interfaces:**
- Full workbench contains all records and all required thumbnails.
- Preview contains prioritized records, matching thumbnails, and stays below 1 MB excluding external assets.

- [ ] Run the builder using `configs/pilot64_gpt4o_v6_delivery.json`, with explicit v6 runtime metadata and v7 current Prompt snapshot.
- [ ] Verify no video files appear under audit assets and every referenced thumbnail exists.
- [ ] Run `pytest -q`, `python -m compileall -q src tools`, `git diff --check`, and privacy scans.
- [ ] Inspect Evidence, QA, Judge and Prompt tabs at 736px and 360px; test lazy image loading, lightbox, filters and local decisions.
- [ ] Commit source/config/test/doc changes, keep generated private audit artifacts ignored, and push `main`.
