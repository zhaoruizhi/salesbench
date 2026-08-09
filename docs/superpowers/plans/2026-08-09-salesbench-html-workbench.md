# SalesBench Interactive Algorithm Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a code-grounded, Chinese interactive HTML workbench that first exposes the current SalesBench algorithm for section-by-section review, then evolves into a paper-style Method view and standalone final deliverable after the user freezes the algorithm.

**Architecture:** Maintain one host-compatible HTML fragment in the thread-scoped visualization directory. It embeds a versioned `WORKBENCH` content model for modules M1–M8, renders algorithm and paper views from the same facts, and provides a feedback channel through `window.openai.sendFollowUpMessage` with a standalone JSON-export fallback. After the algorithm is frozen, render the fragment into one standalone HTML file and add a visually verified ImageGen framework figure.

**Tech Stack:** Semantic HTML, theme-aware CSS, vanilla JavaScript, inline SVG, Codex `window.openai` host bridge, browser `localStorage`, bundled visualization renderer, built-in ImageGen, Python 3.13 `unittest` for content-contract validation.

## Global Constraints

- The user-facing review and final deliverables must be HTML, not Markdown.
- The internal Markdown spec and plan are audit artifacts only and are not final deliverables.
- The initial version is algorithm baseline `v0.1`; paper prose must not conceal or override implementation facts.
- Stable module IDs are `M1` through `M8`: overview, dataset, evidence, tasks, multi-agent, deterministic QA, evaluation, reproducibility/boundaries.
- Only sampled frames and ASR/subtitles are public observations; C1, C3, C4, and C5 remain internal and do not enter the Evidence Extractor, Proposers, tested VLM, or Judge.
- Evidence modalities admitted by validation are exactly `visual`, `ocr`, and `asr`.
- The multi-agent system generates and reviews grounded annotations; final questions are rendered by deterministic templates.
- Interaction counts and creator/account metadata never enter Evidence, public QA, tested-model input, Judge input, or leaderboard metrics.
- The fragment must contain no `<!doctype>`, `<html>`, `<head>`, or `<body>` tags and must remain below 1 MB.
- The fragment must work at 736px and 360px without overlap, clipping, fixed viewport layouts, or internal horizontal scrolling.
- All controls must be native keyboard-accessible controls; presentation colors must use host theme variables.
- Do not generate the final ImageGen figure or standalone deliverable before the user explicitly says the algorithm is frozen.

---

### Task 1: Add a deterministic workbench content validator

**Files:**
- Create: `tools/method_workbench/__init__.py`
- Create: `tools/method_workbench/validate_workbench.py`
- Create: `tests/test_method_workbench_validator.py`

**Interfaces:**
- Consumes: an absolute path to a host-compatible HTML fragment.
- Produces: `validate_fragment(path: Path) -> list[str]`; an empty list means the fragment satisfies the structural and algorithm-content contract.

- [ ] **Step 1: Write failing unit tests for required modules, forbidden document wrappers, and feedback hooks**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.method_workbench.validate_workbench import validate_fragment


class WorkbenchValidatorTest(unittest.TestCase):
    def test_minimal_fragment_reports_missing_algorithm_contract(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "workbench.html"
            path.write_text('<section id="salesbench-method-workbench"></section>', encoding="utf-8")
            errors = validate_fragment(path)
        self.assertIn("missing module M1", errors)
        self.assertIn("missing module M8", errors)
        self.assertIn("missing host feedback bridge", errors)
        self.assertIn("missing standalone feedback export", errors)

    def test_document_wrapper_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "workbench.html"
            path.write_text('<!doctype html><html><body></body></html>', encoding="utf-8")
            errors = validate_fragment(path)
        self.assertIn("fragment contains document wrapper", errors)
```

- [ ] **Step 2: Run the tests and verify they fail because the validator does not exist**

Run: `python -m unittest tests.test_method_workbench_validator -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.method_workbench'`.

- [ ] **Step 3: Implement the validator with exact contract checks**

```python
from __future__ import annotations

from pathlib import Path


REQUIRED_MODULES = tuple(f"M{index}" for index in range(1, 9))
REQUIRED_FACTS = (
    "1,200",
    "hook_plus_uniform",
    "EvidenceUnit",
    "GroundedAnnotation",
    "BP",
    "CM",
    "SS",
    "AE",
    "0.70",
    "每视频最多 8 题",
    "每任务最多 2 题",
    "MacroRA",
    "judge_failed_count",
)


def validate_fragment(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    errors: list[str] = []
    if any(token in lower for token in ("<!doctype", "<html", "<head", "<body")):
        errors.append("fragment contains document wrapper")
    if path.stat().st_size >= 1_000_000:
        errors.append("fragment exceeds 1 MB")
    if 'id="salesbench-method-workbench"' not in text:
        errors.append("missing unique workbench root")
    for module_id in REQUIRED_MODULES:
        if f'id="module-{module_id.lower()}"' not in text and f'"id":"{module_id}"' not in text.replace(" ", ""):
            errors.append(f"missing module {module_id}")
    for fact in REQUIRED_FACTS:
        if fact not in text:
            errors.append(f"missing fact: {fact}")
    if "window.openai.sendFollowUpMessage" not in text:
        errors.append("missing host feedback bridge")
    if "exportFeedback" not in text:
        errors.append("missing standalone feedback export")
    if "localStorage" not in text:
        errors.append("missing local feedback persistence")
    return errors


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    if not args.fragment.exists():
        print(f"fragment not found: {args.fragment}")
        return 1
    errors = validate_fragment(args.fragment)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("workbench contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run validator tests and the existing test suite**

Run: `python -m unittest tests.test_method_workbench_validator -v`

Expected: PASS, 2 tests.

Run: `pytest -q`

Expected: all existing SalesBench tests pass.

- [ ] **Step 5: Commit the validation harness**

```bash
git add tools/method_workbench/__init__.py tools/method_workbench/validate_workbench.py tests/test_method_workbench_validator.py
git commit -m "test: validate SalesBench method workbench"
```

---

### Task 2: Build algorithm baseline v0.1 as a host-compatible HTML fragment

**Files:**
- Create: `/Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html`

**Interfaces:**
- Consumes: code facts from `src/salesbench/`, fixed cohort configuration, current data profile, and the approved design spec.
- Produces: `WORKBENCH` JavaScript object; `renderNavigation()`, `renderModule(moduleId)`, and `setView(viewName)` functions; an immediately useful first render focused on M1.

- [ ] **Step 1: Run the validator against the absent fragment and confirm the expected failure**

Run:

```bash
python tools/method_workbench/validate_workbench.py /Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html
```

Expected: non-zero exit with `file not found`.

- [ ] **Step 2: Create the fragment root, versioned content model, and shared layout**

The fragment must begin with a unique root and keep all custom selectors rooted beneath it:

```html
<section id="salesbench-method-workbench" aria-labelledby="workbench-title">
  <header class="workbench-header">
    <div>
      <div class="text-small text-muted">算法基线 v0.1 · 代码事实优先</div>
      <h1 id="workbench-title">SalesBench 算法审阅工作台</h1>
    </div>
    <div class="viz-controls" aria-label="视图切换">
      <button type="button" class="btn btn-primary" data-view="algorithm" aria-pressed="true">算法视图</button>
      <button type="button" class="btn" data-view="paper" aria-pressed="false">论文视图</button>
    </div>
  </header>
  <div class="workbench-layout">
    <nav id="workbench-navigation" aria-label="算法模块"></nav>
    <main id="workbench-content"></main>
  </div>
</section>
```

Embed exactly eight module records in the same file. Each record must contain `id`, `title`, `summary`, `inputs`, `process`, `outputs`, `equations`, `codeEvidence`, `boundaries`, and `paperText`. The stable titles are:

```javascript
const WORKBENCH = {
  version: "v0.1",
  activeModule: "M1",
  activeView: "algorithm",
  changelog: [],
  modules: [
    { id: "M1", title: "总体算法与数据流" },
    { id: "M2", title: "数据集构建与 Cohort" },
    { id: "M3", title: "多模态观测与证据构建" },
    { id: "M4", title: "BP / CM / SS / AE 任务本体" },
    { id: "M5", title: "Evidence-First 多智能体状态机" },
    { id: "M6", title: "确定性 QA 编译" },
    { id: "M7", title: "被测模型与评估聚合" },
    { id: "M8", title: "质量、复现与实现边界" }
  ]
};
```

- [ ] **Step 3: Populate all M1–M8 algorithm facts from the code**

The implementation must include these exact invariants in the relevant module:

```javascript
const requiredFacts = {
  M2: ["1,200 条记录", "1,196 条视频资产", "1,158 条商品图像", "20 个 anchor", "seed = 42"],
  M3: ["目标 16 帧", "3 个 Hook 帧", "13 个均匀帧", "visual / OCR / ASR", "EvidenceUnit"],
  M4: ["BP 至少 1 个证据", "CM / SS / AE 至少 2 个不同证据", "DIRECT", "INFERRED"],
  M5: ["Local BP", "Consumer", "Operator", "Strategist", "Challenger", "Adjudicator", "γ = 0.70"],
  M6: ["每视频最多 8 题", "每任务最多 2 题", "BP → CM → SS → AE", "GroundedAnnotation"],
  M7: ["0 / 0.25 / 0.5 / 0.75 / 1", "StrictAcc", "RelaxedAcc", "MacroRA", "judge_failed_count"],
  M8: ["C1–C6 不是公开输入", "creator split 尚未接入主 CLI", "互动分析不进入排行榜"]
};
```

M5 must explicitly state that agents propose grounded annotations and deterministic code renders final QA. M7 must state that missing model answers receive local score 0, while Judge API/parse failures are reported but excluded from accuracy denominators by the current aggregator.

- [ ] **Step 4: Render module navigation, algorithm view, paper view, and expandable code evidence**

```javascript
function renderModule(moduleId) {
  const module = WORKBENCH.modules.find(item => item.id === moduleId);
  const content = document.getElementById("workbench-content");
  content.innerHTML = WORKBENCH.activeView === "algorithm"
    ? renderAlgorithmModule(module)
    : renderPaperModule(module);
}

function setView(viewName) {
  WORKBENCH.activeView = viewName;
  document.querySelectorAll("[data-view]").forEach(button => {
    const active = button.dataset.view === viewName;
    button.setAttribute("aria-pressed", String(active));
    button.classList.toggle("btn-primary", active);
  });
  renderModule(WORKBENCH.activeModule);
}
```

Use native `<details>` for code evidence and implementation boundaries. Code evidence must cite concrete paths such as `src/salesbench/goldbank/pipeline.py`, `src/salesbench/vqa/compiler.py`, and `src/salesbench/vqa_evaluate/metrics.py` without inventing line numbers.

- [ ] **Step 5: Validate the complete content contract**

Run:

```bash
python tools/method_workbench/validate_workbench.py /Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html
```

Expected: exit 0 and `workbench contract: PASS`.

---

### Task 3: Add the interactive algorithm graph and feedback round-trip

**Files:**
- Modify: `/Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html`

**Interfaces:**
- Consumes: `WORKBENCH.activeModule`, `WORKBENCH.version`, and M1–M8 module records.
- Produces: a clickable `renderFlow()` overview; `submitFeedback(moduleId, text) -> Promise<void>`; `saveFeedback(moduleId, text) -> void`; `exportFeedback() -> void`.

- [ ] **Step 1: Add a pre-implementation contract check for missing feedback behavior**

Run the validator before adding the functions.

Expected: FAIL with `missing host feedback bridge`, `missing standalone feedback export`, or `missing local feedback persistence`.

- [ ] **Step 2: Implement a compact clickable flow graph in M1**

Use an accessible inline SVG whose nodes map to modules and whose arrows show the real data path:

```javascript
const FLOW_NODES = [
  ["M2", "Dataset"],
  ["M3", "EvidenceUnit"],
  ["M5", "Grounded Annotation"],
  ["M6", "Deterministic QA"],
  ["M7", "Judge Metrics"]
];
```

Node activation must update `WORKBENCH.activeModule`, navigation selection, and the current module content. Add a separate red dashed firewall annotation reading `Title / Metadata / Creator / Engagement: excluded` rather than drawing those fields into the main path.

- [ ] **Step 3: Implement local feedback persistence and host submission**

```javascript
function feedbackKey(moduleId) {
  return `salesbench-workbench:${WORKBENCH.version}:${moduleId}`;
}

function saveFeedback(moduleId, text) {
  localStorage.setItem(feedbackKey(moduleId), text);
}

async function submitFeedback(moduleId, text) {
  saveFeedback(moduleId, text);
  const prompt = [
    `SalesBench 算法工作台 ${WORKBENCH.version} 修改意见`,
    `模块：${moduleId}`,
    text,
    "请重新核对项目代码，说明受影响的算法模块，并更新工作台版本。"
  ].join("\n");
  if (window.openai?.sendFollowUpMessage) {
    await window.openai.sendFollowUpMessage({ prompt, title: `发送 ${moduleId} 修改意见` });
    return;
  }
  exportFeedback();
}
```

Each module view must provide one labeled `<textarea class="form-control">`, a secondary save button, and one primary send button. Restore saved text when switching modules.

After defining `bindFeedbackForm(module)`, update `renderModule(moduleId)` to call it immediately after replacing `workbench-content`, ensuring the textarea and buttons exist before event listeners are attached.

- [ ] **Step 4: Implement standalone JSON export**

```javascript
function exportFeedback() {
  const feedback = Object.fromEntries(
    WORKBENCH.modules.map(module => [module.id, localStorage.getItem(feedbackKey(module.id)) || ""])
  );
  const blob = new Blob([JSON.stringify({ version: WORKBENCH.version, feedback }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `salesbench-feedback-${WORKBENCH.version}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
```

- [ ] **Step 5: Re-run the content validator**

Expected: `workbench contract: PASS`.

---

### Task 4: Render and visually verify baseline v0.1

**Files:**
- Read: `/Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html`
- Generate for QA only: `/private/tmp/salesbench-algorithm-workbench-standalone.html`

**Interfaces:**
- Consumes: completed host fragment.
- Produces: evidence that content, interaction, accessibility, and responsive behavior work at 736px and 360px.

- [ ] **Step 1: Render a standalone QA wrapper with the bundled renderer**

Run:

```bash
/Users/zhaoruizhi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /Users/zhaoruizhi/.codex/plugins/cache/openai-bundled/visualize/1.0.20/skills/visualize/scripts/render.py \
  /Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html \
  /private/tmp/salesbench-algorithm-workbench-standalone.html
```

Expected: standalone HTML is created without fragment parsing errors.

- [ ] **Step 2: Use the in-app browser to inspect 736px behavior**

Verify all of the following manually:

- M1 is visible on first render;
- selecting M2–M8 changes the content and selected state;
- algorithm/paper switching changes only the content mode, not the active module;
- code evidence and implementation boundaries expand with native `<details>`;
- feedback persists when navigating away and back;
- flow nodes navigate to the corresponding module;
- no text, controls, or diagram labels overlap or clip.

- [ ] **Step 3: Inspect 360px behavior and keyboard access**

Verify navigation wraps or stacks, content remains readable, all controls are reachable in native tab order, textarea labels remain associated, and no internal horizontal scroll is introduced.

- [ ] **Step 4: Fix every visual or runtime defect and repeat both widths**

Apply one targeted correction per defect, then re-run the validator and both visual inspections. Do not accept clipped labels, dead controls, undefined identifiers, or missing module content.

- [ ] **Step 5: Present baseline v0.1 inline for user review**

Return the visualization reference using the absolute fragment path and no Markdown file link:

```text
visualize{"path":"/Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html","mode":"wide","title":"SalesBench 算法审阅工作台 v0.1"}
```

---

### Task 5: Process review rounds without losing code traceability

**Files:**
- Modify each round: `/Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html`

**Interfaces:**
- Consumes: structured feedback containing workbench version, module ID, and requested change.
- Produces: next `v0.x` version with updated algorithm facts, paper mapping, and changelog.

- [ ] **Step 1: Resolve every feedback item to authoritative code evidence**

For each request, inspect the named module's source files. Record one of: `accepted-code-aligned`, `accepted-user-design-change`, or `not-code-aligned`. Do not silently rewrite an implemented fact merely to make the prose sound cleaner.

- [ ] **Step 2: Update the shared fact record before updating paper prose**

Change `inputs`, `process`, `outputs`, `equations`, `codeEvidence`, or `boundaries` first; regenerate that module's paper view from the new fact record second.

- [ ] **Step 3: Append a concrete changelog entry and increment the version**

```javascript
WORKBENCH.version = "v0.2";
WORKBENCH.changelog.unshift({
  version: "v0.2",
  module: "M5",
  change: "Clarified that only PASS proposals enter adjudication.",
  evidence: "src/salesbench/goldbank/pipeline.py"
});
```

- [ ] **Step 4: Re-run contract and responsive checks before each presentation**

Run the validator, render the QA wrapper, and test the changed module plus M1 at 736px and 360px. Include the visualization reference in the same response.

---

### Task 6: Freeze the algorithm, generate the paper figure, and export the final standalone HTML

**Files:**
- Create after explicit user freeze: `deliverables/figures/salesbench_method_framework.png`
- Create after explicit user freeze: `deliverables/SalesBench_Algorithm_Method_Workbench.html`
- Modify after explicit user freeze: `/Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html`

**Interfaces:**
- Consumes: user-approved frozen workbench version and its M1–M8 content model.
- Produces: visually verified framework PNG and standalone HTML with no Codex-only dependency.

- [ ] **Step 1: Confirm the user explicitly said the algorithm is frozen**

Do not infer freeze from silence or from approval of a single module. Record the exact frozen workbench version in the final HTML.

- [ ] **Step 2: Generate the framework figure with built-in ImageGen**

Use this normalized prompt. At execution time, append the exact current `WORKBENCH.version` value to the `Text (verbatim)` line as a label in the form `Version: v0.x`; no other label may be changed:

```text
Use case: infographic-diagram
Asset type: ACL/EMNLP paper method framework figure
Primary request: visualize the frozen SalesBench Evidence-First benchmark pipeline from data and observation through grounded annotation, deterministic QA compilation, tested VLM, and evidence-aware judging
Style/medium: clean flat vector-like academic infographic, white background, precise arrows, no decorative people
Composition/framing: 16:9 landscape, three stages from left to right
Text (verbatim): "Data & Observation", "16 Frames + ASR", "EvidenceUnit", "Local BP", "Consumer", "Operator", "Strategist", "Challenger", "Adjudicator", "Local Validation", "Human Review", "GroundedAnnotation", "Deterministic QA Compiler", "Public QA", "Private Gold", "Tested VLM", "Evidence-aware Judge", "MacroRA", "Private Firewall", plus the exact runtime version label
Constraints: blue-gray main flow; orange quality gates; red dashed private firewall; title, metadata, creator, and engagement remain outside the main flow; no extra modules; no watermark; version label shown exactly once
```

- [ ] **Step 3: Inspect the generated image before using it**

Check every required label, arrow direction, firewall placement, and module count. If any label is corrupted or an arrow changes the algorithm, make one targeted regeneration. Copy the accepted PNG into `deliverables/figures/`; never leave the project-referenced final only under the generated-images directory.

- [ ] **Step 4: Replace the Codex-only feedback action with standalone export behavior**

In the standalone rendering path, `submitFeedback()` must always save locally and call `exportFeedback()`; it must not reference `window.openai`. Preserve localStorage and JSON export.

- [ ] **Step 5: Render the frozen fragment to the final standalone HTML**

Run the bundled renderer with destination `deliverables/SalesBench_Algorithm_Method_Workbench.html`, then embed or reference the project-local framework PNG using a portable relative path.

- [ ] **Step 6: Run final completion checks**

Run:

```bash
python tools/method_workbench/validate_workbench.py /Users/zhaoruizhi/.codex/visualizations/2026/08/09/019fe4fa-1ee4-7871-a08f-9d9bb6897b1b/salesbench-algorithm-workbench.html
pytest -q
python -m compileall -q src tools
git diff --check
```

Open the final standalone HTML at 736px and 360px and verify all M1–M8 content, paper view, formula rendering, changelog, image, local feedback save, and JSON export.

- [ ] **Step 7: Commit only the frozen project deliverables and validation code**

```bash
git add deliverables/SalesBench_Algorithm_Method_Workbench.html deliverables/figures/salesbench_method_framework.png tools/method_workbench/validate_workbench.py tests/test_method_workbench_validator.py
git commit -m "docs: deliver interactive SalesBench method workbench"
```
