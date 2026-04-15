# CSAM Publication Blocker Technical Gap Register

Date: 2026-04-15
Audience: Engineering leads, benchmark owners, QA owners, publication leads
Status: Current-state blocker register (supersedes prior stale gap writeup)

## 1. Purpose

This document lists only the technical and process gaps that currently block publication-safe claims.

A gap is considered a publication blocker if it can invalidate one or more of:
1. fairness of benchmark comparisons,
2. reproducibility and statistical stability,
3. correctness of core metrics,
4. integrity of the final claims narrative.

This version intentionally excludes already-fixed historical gaps unless they have regressed.

## 2. Current Blocker Summary

| ID | Severity | Blocker | Why It Blocks Publication |
|---|---|---|---|
| PB-01 | Critical | Execution-plan artifact is empty | No canonical implementation tracker or closure record exists in the intended plan file. |
| PB-02 | Critical | LoCoMo multimodel protocol is scope-limited | Current evidence is based on one conversation and 10 QA, too narrow for broad claims. |
| PB-03 | Critical | Baseline comparison is not apples-to-apples | Baseline uses different model stack and weaker retrieval budget than CSAM benchmark path. |
| PB-04 | Critical | Seed controls are missing in benchmark CLIs | Reproducibility cannot be guaranteed across LoCoMo, MuSiQue, and HotPotQA runs. |
| PB-05 | High | Variance artifacts are missing | No persisted multi-seed outputs for ablation/sweep; significance language remains weak. |
| PB-06 | High | Tests are smoke-style, not assertion-enforced | Passing tests do not reliably fail on regressions. |
| PB-07 | High | Coverage gate is documented but not supported | Coverage command is prescribed but dependency is missing in project declarations. |
| PB-08 | High | Ablation overall metric can be inflated | Overall F1 excludes zero-valued categories, which can overstate performance. |
| PB-09 | High | Configuration/formula drift across docs/instructions | Conflicting formulas and defaults can create inconsistent implementation/reporting. |
| PB-10 | High | Existing gap document is stale and contradictory | It marks resolved fixes as open critical issues, causing incorrect execution priorities. |

## 3. Detailed Blocker Cards

## PB-01: Execution-plan artifact is empty

Severity: Critical

Evidence:
- `csam_project/docs/publication_engineering_execution_plan.md` (currently empty file content).

Why this blocks publication:
- There is no authoritative, versioned execution tracker for what was fixed, validated, and signed off.
- Without this artifact, reviewer questions on closure and traceability cannot be answered rigorously.

Required fix:
1. Restore plan content in `csam_project/docs/publication_engineering_execution_plan.md`.
2. Add owners, deadlines, and closure criteria per blocker.
3. Add immutable run/artifact references for each closed item.

Done criteria:
- File is populated and reviewed.
- Each blocker in this document is mapped to a plan task and status.

---

## PB-02: LoCoMo multimodel protocol is scope-limited

Severity: Critical

Evidence:
- Single conversation hardcoded in [csam_project/benchmarks/benchmark_multimodel.py](csam_project/benchmarks/benchmark_multimodel.py#L149).
- QA subset truncation in [csam_project/benchmarks/benchmark_multimodel.py](csam_project/benchmarks/benchmark_multimodel.py#L250).
- CLI default of 10 questions in [csam_project/benchmarks/benchmark_multimodel.py](csam_project/benchmarks/benchmark_multimodel.py#L437).
- Output confirms 10-question scope in [csam_project/benchmarks/results_multimodel_summary.json](csam_project/benchmarks/results_multimodel_summary.json#L4).

Why this blocks publication:
- Broad LoCoMo statements cannot be defended when evidence is generated from one conversation and 10 QA.
- Scope under-sampling increases variance and selection bias risk.

Required fix:
1. Add conversation-scope controls (`--conversation-index`, `--max-conversations`, full-scope mode).
2. Add question-scope controls (`--questions-per-conversation`, default all).
3. Persist per-conversation and aggregate metrics with scope metadata.

Done criteria:
- Canonical multimodel artifact includes multi-conversation coverage.
- Protocol metadata explicitly records scope and sampling choices.

---

## PB-03: Baseline comparison is not apples-to-apples

Severity: Critical

Evidence:
- Baseline retrieval budget `k=5` in [csam_project/benchmarks/benchmark_baseline_rag.py](csam_project/benchmarks/benchmark_baseline_rag.py#L47).
- Baseline uses local Ollama 3B in [csam_project/benchmarks/benchmark_baseline_rag.py](csam_project/benchmarks/benchmark_baseline_rag.py#L109).
- Baseline uses first conversation only in [csam_project/benchmarks/benchmark_baseline_rag.py](csam_project/benchmarks/benchmark_baseline_rag.py#L100).
- CSAM LoCoMo path uses `k=20` in [csam_project/benchmarks/benchmark_multimodel.py](csam_project/benchmarks/benchmark_multimodel.py#L267) and top-10 context in [csam_project/benchmarks/benchmark_multimodel.py](csam_project/benchmarks/benchmark_multimodel.py#L271).

Why this blocks publication:
- CSAM-vs-baseline comparisons are confounded by model/provider/retrieval-budget differences.
- Architecture advantage is not isolated from protocol and model effects.

Required fix:
1. Add hosted baseline benchmark with matched model/provider settings.
2. Match retrieval budget and context assembly to CSAM benchmark protocol.
3. Produce paired summary artifacts with explicit protocol fingerprint.

Done criteria:
- Baseline and CSAM use identical benchmark protocol except architecture.
- Comparison table contains parity metadata fields.

---

## PB-04: Seed controls missing in benchmark CLIs

Severity: Critical

Evidence:
- Multimodel parser exposes questions but no seed in [csam_project/benchmarks/benchmark_multimodel.py](csam_project/benchmarks/benchmark_multimodel.py#L437).
- MuSiQue parser exposes questions but no seed in [csam_project/benchmarks/benchmark_musique.py](csam_project/benchmarks/benchmark_musique.py#L387).
- HotPotQA parser exposes questions but no seed in [csam_project/benchmarks/benchmark_hotpotqa.py](csam_project/benchmarks/benchmark_hotpotqa.py#L388).

Why this blocks publication:
- Runs are not reproducible or directly comparable across reruns.
- Reported differences can be sampling/noise artifacts without deterministic controls.

Required fix:
1. Add `--seed` to all benchmark CLIs.
2. Pass seed through dataset sampling, shuffling, and generation calls where supported.
3. Persist seed in every output artifact.

Done criteria:
- Each benchmark JSON contains a seed field.
- Reruns with same seed produce reproducible scope and near-identical metrics.

---

## PB-05: Variance artifacts are missing

Severity: High

Evidence:
- `csam_project/evaluation` currently has no persisted `variance_ablation_results.json` or `variance_sweep_results.json` outputs (directory listing check).
- Runner exists in `csam_project/evaluation/variance_runner.py`, but artifacts are absent.

Why this blocks publication:
- No mean/std confidence framing for key claims.
- Significance statements cannot be defended robustly.

Required fix:
1. Run `variance_runner.py` for ablation and sweep modes.
2. Commit output artifacts for both modes.
3. Add run manifest with commands, timestamps, model IDs, and seeds.

Done criteria:
- Variance JSON artifacts present and versioned.
- Paper claims cite multi-seed stability statistics.

---

## PB-06: Tests are smoke-style, not assertion-enforced

Severity: High

Evidence:
- No `assert` statements found in current test files (workspace search).
- Print-driven flow in [csam_project/tests/test_working_memory.py](csam_project/tests/test_working_memory.py#L20).
- Branch checks without hard assertion in [csam_project/tests/test_npc_l1_integration.py](csam_project/tests/test_npc_l1_integration.py#L71) and [csam_project/tests/test_npc_l1_integration.py](csam_project/tests/test_npc_l1_integration.py#L94).
- Similar pattern in [csam_project/tests/test_metadata_filtering.py](csam_project/tests/test_metadata_filtering.py#L146).

Why this blocks publication:
- The suite can pass without enforcing key correctness conditions.
- Regressions can slip through while tests still report success.

Required fix:
1. Convert tests to assertion-first style.
2. Add focused tests for current fragile paths (store_interaction flag, rebuild_index behavior, metadata filter scope).
3. Keep logs optional, not as pass/fail mechanism.

Done criteria:
- Core tests fail when expected behavior is broken.
- CI/local gates represent real correctness constraints.

---

## PB-07: Coverage gate documented but not supported

Severity: High

Evidence:
- Coverage command is documented in [\.claude/CLAUDE.md](.claude/CLAUDE.md#L80).
- `pyproject.toml` declares pytest but not pytest-cov in [pyproject.toml](pyproject.toml#L16).

Why this blocks publication:
- Team process expects coverage evidence, but project dependencies do not enable it reliably.
- Quality gate is not executable in clean environments.

Required fix:
1. Add `pytest-cov` to project dependencies.
2. Verify documented command executes in clean venv.
3. Set and enforce minimum coverage for core modules.

Done criteria:
- Coverage command runs without argument errors.
- Coverage report is generated and attached to release evidence.

---

## PB-08: Ablation overall F1 can inflate reported performance

Severity: High

Evidence:
- Overall score uses positive categories only in [csam_project/evaluation/run_ablation.py](csam_project/evaluation/run_ablation.py#L456):
  - `overall_f1 = np.mean([s for s in f1_scores.values() if s > 0])`

Why this blocks publication:
- Zero-performing categories can be excluded from overall aggregate, potentially overstating performance.
- Cross-script comparability is weakened when aggregate definitions differ.

Required fix:
1. Define one canonical aggregate policy (include all categories, or weighted by question counts).
2. Apply the same aggregate across all evaluation scripts.
3. Document the metric definition in benchmark docs.

Done criteria:
- Aggregate policy is centralized and reused.
- Reported overall metrics are directly comparable.

---

## PB-09: Configuration/formula drift across project instructions

Severity: High

Evidence:
- Formula in [\.github/copilot-instructions.md](.github/copilot-instructions.md#L25): `0.2*R + 0.2*(1-I) + 0.3*C + 0.3*D`.
- Formula in [\.claude/CLAUDE.md](.claude/CLAUDE.md#L24): equal 0.25 weighting.
- Runtime default in code is equal weighting in [csam_project/csam_core/forgetting_engine.py](csam_project/csam_core/forgetting_engine.py#L188).

Why this blocks publication:
- Different formula statements create ambiguity about what was actually evaluated.
- Reproducibility and methods clarity are compromised.

Required fix:
1. Define canonical default formula and threshold in one authoritative place.
2. Mark alternative weight sets as experimental configurations.
3. Harmonize instructions/docs with runtime code defaults.

Done criteria:
- Docs, instructions, and code agree on defaults.
- Benchmark artifacts explicitly state if non-default weights are used.

---

## PB-10: Existing technical gap document is stale and contradictory

Severity: High

Evidence:
- Current implementation shows these previously reported items are already fixed:
  - consolidation threshold default is 0.3 in [csam_project/csam_core/forgetting_engine.py](csam_project/csam_core/forgetting_engine.py#L192),
  - Counter-based F1 is active in [csam_project/evaluation/run_ablation.py](csam_project/evaluation/run_ablation.py#L456),
  - metadata filter logic is active in [csam_project/csam_core/memory_repository.py](csam_project/csam_core/memory_repository.py#L228) and [csam_project/csam_core/memory_repository.py](csam_project/csam_core/memory_repository.py#L244).
- The intended execution-plan anchor file is currently empty: `csam_project/docs/publication_engineering_execution_plan.md`.

Why this blocks publication:
- Engineering priorities become misaligned with actual current blockers.
- Publication team can cite obsolete problem statements.

Required fix:
1. Replace stale gap writeup with this current blocker register.
2. Separate historical-fixed issues from active blockers.
3. Link blockers directly to execution tasks.

Done criteria:
- Only active blockers remain in the canonical gaps doc.
- Historical items are marked fixed or moved to archive notes.

## 4. Blocker-to-Workstream Mapping

| Blocker | Execution Workstream |
|---|---|
| PB-01 | Plan governance restore (documentation control) |
| PB-02 | P0-A canonical LoCoMo protocol |
| PB-03 | P0-B apples-to-apples baseline |
| PB-04 | P0-C seed and reproducibility controls |
| PB-05 | P0-C variance evidence artifacts |
| PB-06 | P1-B test hardening |
| PB-07 | P1-B coverage gate enablement |
| PB-08 | P1-A metric consistency |
| PB-09 | P1-D documentation/formula reconciliation |
| PB-10 | P1-D technical gap doc reconciliation |

## 5. Minimum Closure Sequence (Publication-safe)

Recommended closure order:
1. PB-01 (restore non-empty execution plan artifact).
2. PB-02 + PB-03 (fairness of primary evaluation evidence).
3. PB-04 + PB-05 (reproducibility and variance).
4. PB-08 (metric aggregation correctness).
5. PB-06 + PB-07 (quality gates and regression protection).
6. PB-09 + PB-10 (documentation and methods consistency).

## 6. Immediate Next Actions (48-hour window)

1. Repopulate `publication_engineering_execution_plan.md` and map owner/status fields.
2. Patch benchmark scripts to add seed and expanded scope controls.
3. Run and persist variance artifacts.
4. Replace smoke tests with assert-based checks and enable coverage dependency.
5. Sync formula/default statements across all instruction and docs files.

## 7. Definition Of Done For This Gap Register

This document can be marked complete when:
- Every PB-* item is either Closed with artifact evidence or explicitly deferred with risk acceptance.
- The execution plan file contains mapped tasks and closure evidence.
- Publication draft claim lines reference only closed or accepted-risk items.
