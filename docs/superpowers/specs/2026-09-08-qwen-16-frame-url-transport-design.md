# Qwen 16-Frame URL Transport Design

## Objective

Keep the benchmark's complete 16-frame sampling policy while removing large
Base64 image bodies from Qwen inference requests. Smoke runs upload sampled
frames to DashScope's account-scoped temporary OSS, then submit all 16
`oss://` URLs with explicit frame indices and timestamps. Formal runs may use
the same transport only for validation; a release run must use stable,
operator-managed object URLs because temporary objects expire after 48 hours.

## Confirmed failure boundary

Qwen visual models support multi-image and video inputs. Earlier SalesBench
runs also completed with 16 Base64 frames. The September 8 failures occurred
before token usage and crossed several Qwen model aliases, with SSL EOF,
broken-pipe, remote-protocol, and timeout errors. This is therefore treated as
a request/upload transport failure, not as evidence that Qwen cannot process
video and not as malformed model output.

## Architecture

1. `DashScopeTemporaryOSSUploader` obtains an upload policy, uploads local JPEG
   frames, returns account/model-bound `oss://` URLs, and caches only the URL,
   file digest, model, and expiry under ignored `outputs/cache/`.
2. The Gold runner selects a frame transport from the cohort config. Base64
   remains supported for provider compatibility; the Qwen smoke config selects
   `dashscope_temporary_oss`.
3. The pipeline receives provider-ready frame references plus the existing
   frame metadata. It always retains all 16 logical frame positions and emits
   label/image pairs so irregular `hook_plus_uniform` timestamps remain
   observable to the model.
4. Qwen calls using `oss://` references send
   `X-DashScope-OssResourceResolve: enable`. Temporary upload credentials are
   never written to artifacts or logs.
5. Transport/API failure ends the visual extraction stage and is routed to
   pipeline diagnostics. Visual repair is attempted only when the call
   succeeded but parsing or evidence validation produced no acceptable visual
   unit.
6. Resume loads a fingerprint-matched partial result as a retry seed, reuses a
   previously successful language-evidence stage, and recomputes downstream
   commercial graph and QA candidates after visual recovery. A deterministic
   quality comparison prevents a worse retry from overwriting a better part.
7. A preflight uploads and submits the full sampled frame set before the cohort
   run. Failure aborts before the immutable dataset is merged. Its report is a
   private run diagnostic, not part of the public benchmark.

## Failure policy

- Retry connection reset, SSL EOF, broken pipe, timeout, HTTP 408/409/429, and
  HTTP 5xx with bounded exponential backoff.
- Fail immediately for authentication, permission, invalid model, malformed
  request, and other non-retryable HTTP 4xx responses.
- Refresh the temporary upload policy after a retryable policy/upload failure.
- Model-output parse/validation repair is separate from transport retry.
- Do not reduce the frame count as an automatic fallback. A failed 16-frame
  preflight blocks the run so a different visual input contract cannot silently
  enter the same candidate release.

## Run identity

The first new smoke identity is
`v10c2-smoke5-qwen37-deepseek-20260908-005`. Its fingerprints include the
visual model, frame transport, frame count, upload cache version, and pipeline
version. The existing `-004` directory remains an immutable diagnostic record.

## Acceptance criteria

- Unit tests prove retry classification, temporary upload caching, header
  injection, 16-reference payload preservation, transport/repair separation,
  partial-stage reuse, monotonic part writes, and preflight fail-closed behavior.
- The full test suite, compile check, diff check, and secret scan pass.
- The `-005` smoke run either completes five videos with 16 URL-backed frames
  each or stops at the preflight with a transport diagnostic. It must never
  continue with fewer frames or mix `-004` artifacts.
- Generated public Evidence/QA JSONL contains no API keys, signed upload
  policies, temporary URLs, or private interaction metadata.

