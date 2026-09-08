# Qwen 16-Frame URL Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably run the complete 16-frame SalesBench visual Evidence stage through official Qwen without embedding all JPEG bytes in the inference request.

**Architecture:** A focused DashScope temporary-OSS uploader converts cached local frames into expiring `oss://` references. The existing runner resolves references before the pipeline, the API client injects the required resolver header, and the pipeline distinguishes network failure from valid-but-bad model output. Resume reuses successful language Evidence and preserves the best fingerprint-matched part.

**Tech Stack:** Python 3.13 standard library, OpenAI-compatible SDK, pytest, existing SalesBench JSON/JSONL utilities.

**Spec:** `docs/superpowers/specs/2026-09-08-qwen-16-frame-url-transport-design.md`

## Global Constraints

- Preserve all 16 `hook_plus_uniform` frames; never silently fall back to a smaller frame count.
- Use official DashScope temporary OSS only for smoke/development; document stable object storage as a prerequisite for formal release runs.
- Never persist API keys, upload-policy credentials, signatures, or signed HTTP URLs.
- Keep old run `v10c2-smoke5-qwen-deepseek-20260907-004` immutable and create `v10c2-smoke5-qwen37-deepseek-20260908-005`.
- Use test-first development and commit each independently verified change.

---

### Task 1: Retryable DashScope frame uploader

**Files:**
- Create: `src/salesbench/vlm/dashscope_oss.py`
- Create: `tests/test_dashscope_oss.py`

**Interfaces:**
- Produces: `DashScopeTemporaryOSSUploader(api_key, cache_root, retry_max, retry_backoff_s)`.
- Produces: `upload_frames(frame_paths: list[str], model: str) -> list[str]` returning ordered `oss://` references.
- Produces: `classify_transport_error(exc: Exception) -> str` returning `retryable`, `non_retryable`, or `unknown`.

- [ ] **Step 1: Write failing tests** for ordered 16-frame upload, digest/model cache reuse, expired-cache refresh, retryable SSL/5xx behavior, and immediate non-retryable 4xx failure.
- [ ] **Step 2: Run `pytest -q tests/test_dashscope_oss.py`** and confirm the missing module/API failures.
- [ ] **Step 3: Implement the uploader** with injectable HTTP and sleep boundaries, multipart encoding, atomic cache writes, and redacted exceptions.
- [ ] **Step 4: Run `pytest -q tests/test_dashscope_oss.py`** and confirm all uploader tests pass.
- [ ] **Step 5: Commit** `feat: add retryable dashscope frame transport`.

### Task 2: Provider headers and pipeline transport semantics

**Files:**
- Modify: `src/salesbench/vlm/api_client.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `tests/test_vlm_api_client.py`
- Modify: `tests/test_goldbank_pipeline.py`

**Interfaces:**
- Extends: `VLMClient(..., default_headers: dict[str, str] | None = None)`.
- Extends: `GoldBankPipeline.run_video(..., frame_urls: list[str] | None = None, resume_result: GoldBankResult | None = None)`.

- [ ] **Step 1: Write failing tests** proving the DashScope resolver header reaches SDK and fallback HTTP calls, all 16 URL blocks keep their frame labels, a failed visual call does not invoke repair, and a successful malformed/invalid visual response does invoke one repair.
- [ ] **Step 2: Run the focused tests** and confirm each new assertion fails for the intended missing behavior.
- [ ] **Step 3: Implement the minimum client and pipeline changes** while retaining Base64 compatibility for non-Qwen providers.
- [ ] **Step 4: Run `pytest -q tests/test_vlm_api_client.py tests/test_goldbank_pipeline.py`**.
- [ ] **Step 5: Commit** `fix: separate qwen transport from visual repair`.

### Task 3: Monotonic stage-aware resume

**Files:**
- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `src/salesbench/goldbank/pipeline.py`
- Modify: `tests/test_goldbank_runner.py`
- Modify: `tests/test_goldbank_pipeline.py`

**Interfaces:**
- Produces: `_load_retry_seeds(...) -> dict[str, GoldBankResult]` for fingerprint-matched nonterminal parts.
- Produces: `_prefer_result(previous, candidate) -> GoldBankResult` using status, Gold record, successful modalities, evidence count, and failed-trace count.
- Extends: the pipeline reuses only successful ASR Evidence from a matching retry seed; all downstream derived data is rebuilt.

- [ ] **Step 1: Write failing tests** proving successful ASR extraction is skipped on a visual retry and a worse result cannot overwrite a better partial part.
- [ ] **Step 2: Run the focused tests** and verify failures name the absent retry-seed and monotonic-write behaviors.
- [ ] **Step 3: Implement retry-seed loading, safe ASR reuse, and deterministic result preference**.
- [ ] **Step 4: Run `pytest -q tests/test_goldbank_runner.py tests/test_goldbank_pipeline.py`**.
- [ ] **Step 5: Commit** `fix: preserve successful stages across evidence resume`.

### Task 4: Fail-closed preflight and immutable Qwen 3.7 run config

**Files:**
- Modify: `src/salesbench/goldbank/runner.py`
- Modify: `src/salesbench/cli.py`
- Modify: `tests/test_goldbank_runner.py`
- Modify: `tests/test_goldbank_cli.py`
- Create: `configs/evidence_smoke_v10_candidate2_qwen37_5videos.json`
- Create: `configs/v10_candidate2_qwen37_smoke_delivery.json`
- Modify: `.env.example`
- Modify: `docs/Pilot_v10_Quality_Gate_Execution_Guide.md`

**Interfaces:**
- Cohort keys: `vision_transport`, `vision_preflight`, `vision_upload_retry_max`, and `vision_upload_retry_backoff_s`.
- Runner behavior: resolve exactly `frames_per_video` references and abort before merge if the first full-frame preflight cannot complete.

- [ ] **Step 1: Write failing tests** for exact frame-count enforcement, preflight abort, run fingerprint transport binding, and CLI option/config propagation.
- [ ] **Step 2: Run focused runner/CLI tests** and confirm the new contracts fail.
- [ ] **Step 3: Wire the uploader into dataset construction**, add private preflight diagnostics, and create the `-005` configs.
- [ ] **Step 4: Update operator documentation** with temporary-versus-formal storage boundaries and exact commands.
- [ ] **Step 5: Run focused tests** and commit `feat: gate qwen evidence runs on 16-frame preflight`.

### Task 5: Verification and five-video smoke

**Files:**
- Generated only: `outputs/runs/v10c2-smoke5-qwen37-deepseek-20260908-005/`

**Interfaces:**
- Consumes: the official Qwen and DeepSeek settings already stored in ignored `.env`.
- Produces: a new immutable smoke Evidence result, followed by QA/audit only if Evidence passes its gate.

- [ ] **Step 1: Run `pytest -q`**, `python -m compileall -q src tools tests`, `git diff --check`, and a tracked-file secret scan.
- [ ] **Step 2: Run the 16-frame health gate and Evidence command** for the `-005` run with `--max-workers 1`.
- [ ] **Step 3: If Evidence succeeds, run realization, compilation, model answers, Judge, audit translation, and paginated audit HTML using the delivery manifest.**
- [ ] **Step 4: Inspect counts, traces, public-field leakage, linked-frame rendering, and run fingerprints; if preflight fails, stop without producing mixed or reduced-frame QA.**
- [ ] **Step 5: Commit any final tracked documentation/config corrections** and report generated outputs separately from committed code.

