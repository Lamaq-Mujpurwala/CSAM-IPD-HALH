# 10-Day Publication Sprint Plan: CSAM (April 15–25, 2026)

Generated: 2026-04-15
Last updated: 2026-04-15 (post-audit)
Status: ACTIVE — submission deadline April 25, 2026
Full audit doc: `csam_project/docs/codebase_audit_and_improvements.md`

---

## Newly Found Bugs (from Codebase Audit — April 15)

These were found in the deep audit and must be folded into the sprint:

| ID | File | Issue | Severity | Day |
|----|------|-------|----------|-----|
| BUG-01 | `run_ablation.py:456` | F1 excludes zeros (same as PB-08) | CRITICAL | 4 |
| BUG-02 | `llm_hosted.py:40` | `CEREBAS_API_KEY` typo — Cerebras broken | HIGH | 1 |
| BUG-03 | `retrieval.py:122` | async `retrieve()` calls sync method with `await` — broken async path | HIGH | 5 |
| BUG-04 | `llm_hosted.py` rate limit | Retry loop has no max — can hang forever | MEDIUM | 1 |
| BUG-05 | No test file | No test verifies consolidation-recall invariant (core claim) | CRITICAL | 4 |
| BUG-06 | `benchmark_e2e.py` | E2E ablation may not trigger consolidation before forgetting | HIGH | 4 |
| BUG-07 | `memory_repository.py` | Soft-delete leaves stale HNSW mappings | MEDIUM | 5 |
| BUG-08 | `memory_repository.py` | Metadata filter silently returns <k results | MEDIUM | 5 |
| BUG-10 | `working_memory.py` | Typo: "Recent Coversation:" | LOW | 5 |

---

## Must-Close vs Accept-Risk

**Must-Close (no exceptions):**
- PB-02: LoCoMo scope (1 conv / 10 QA indefensible)
- PB-03: Apples-to-apples baseline (confounded comparison invalidates main claim)
- PB-04: Seed controls (reproducibility is table stakes)
- PB-08 / BUG-01: Ablation F1 inflation (`if s > 0` filter in `run_ablation.py:456`)
- PB-09: Formula drift (copilot-instructions vs code mismatch)
- PB-01: Execution plan restored
- **BUG-05: Consolidation-recall invariant test** (core claim is unverified without it)
- **BUG-06: Verify E2E ablation triggers consolidation** (results may be meaningless otherwise)

**Accept-Risk with explicit note:**
- PB-05: Variance — attempt 3–5 seeds; if rate-limited, cite infrastructure and defer
- PB-06: Tests — convert 3 smoke files + add new assertion tests; 60% coverage floor
- PB-07: pytest-cov — 30-minute fix, no excuse to defer
- PB-10: Gap doc — already current, just archive stale items
- BUG-03: Remove broken async path — low risk (not used), 30-minute cleanup
- BUG-07/08: Soft-delete and metadata filter — fix for demo quality, not paper blocking

---

## Day 1 — April 15 (Today): Foundation & Governance + Quick Bugfixes

**Files:** `pyproject.toml`, `.github/copilot-instructions.md`, `benchmark_musique.py`, `benchmark_hotpotqa.py`, `docs/publication_engineering_execution_plan.md`, `llm_hosted.py`

**Tasks:**

0. **Fix Cerebras typo** (BUG-02 — 5 min) in `csam_project/csam_core/services/llm_hosted.py:40`:
   - Change `"env_key": "CEREBAS_API_KEY"` → `"env_key": "CEREBRAS_API_KEY"`

0b. **Add rate limit max retries** (BUG-04 — 30 min) in `llm_hosted.py` rate limit handling:
   - Add `_retry_count: int = 0` parameter to `generate()`, MAX_RETRIES = 5
   - Raise `RuntimeError` after 5 failed retries instead of looping forever

1. **Restore execution plan** (`csam_project/docs/publication_engineering_execution_plan.md`)
   - Populate with this sprint plan task table, mapping each PB-* to owner, deadline, done criteria
   - Add artifact references for already-completed items

2. **Fix formula drift** (PB-09)
   - `.github/copilot-instructions.md` line 25: change `0.2*R + 0.2*(1-I) + 0.3*C + 0.3*D` → `0.25*R + 0.25*(1-I) + 0.25*C + 0.25*D`
   - Add comment noting 0.2/0.2/0.3/0.3 is tracked as experimental in `grid_search_results.json`
   - `.claude/CLAUDE.md` already has 0.25 — no change needed

3. **Add `pytest-cov`** (PB-07)
   - In `pyproject.toml`: add `"pytest-cov>=4.0.0"` alongside existing pytest

4. **Add `--seed` to `benchmark_musique.py`** (PB-04)
   - `parser.add_argument("--seed", type=int, default=42)`
   - After parse: `random.seed(args.seed); np.random.seed(args.seed)`
   - Append `"seed": args.seed` to every output JSON

5. **Add `--seed` to `benchmark_hotpotqa.py`** (PB-04)
   - Same pattern as musique

**Day 1 Checklist:**
- [ ] `llm_hosted.py:40` Cerebras typo fixed (BUG-02)
- [ ] Rate limit retry has max 5 retries (BUG-04)
- [ ] `publication_engineering_execution_plan.md` populated
- [ ] `.github/copilot-instructions.md` formula corrected to 0.25 equal
- [ ] `pyproject.toml` has `pytest-cov>=4.0.0`
- [ ] `benchmark_musique.py` has `--seed` arg + written to output JSON
- [ ] `benchmark_hotpotqa.py` has `--seed` arg + written to output JSON
- [ ] `pytest csam_project/tests/ -v` passes

---

## Day 2 — April 16: LoCoMo Protocol Expansion (PB-02, PB-04)

**File:** `csam_project/benchmarks/benchmark_multimodel.py`

**Tasks:**

6. **Add conversation-scope CLI args** to `main()` at line ~430:
   - `--conversation-index INT` — single index override (default None = all)
   - `--max-conversations INT` — cap on conversations (default None = all)
   - `--questions-per-conv INT` — QA per conversation (default None = all)
   - `--seed INT` — random seed (default 42)
   - Change `--questions` default from 10 → None

7. **Refactor `run_single_benchmark`** — accept `conv_data` as parameter, remove `data[0]` hardcode at line 149

8. **Add outer conversation loop** — new `run_multi_conversation_benchmark()`:
   - Load dataset, slice by `max_conversations`
   - Fresh NPC instance per conversation
   - Collect per-conversation results, aggregate macro/micro F1 + std
   - Bootstrap CI (95%) across question-level F1 (1000 samples, 2.5th–97.5th pct)

9. **Add protocol metadata block** to every output JSON:
   ```json
   "protocol": {
     "dataset": "locomo10.json",
     "num_conversations": N,
     "questions_per_conv": M_or_all,
     "retrieval_k": 20,
     "context_top_k": 10,
     "seed": 42,
     "timestamp": "...",
     "model_id": "...",
     "provider": "..."
   }
   ```

10. **Validation run:** `python -m csam_project.benchmarks.benchmark_multimodel --provider groq --model llama-3.1-8b-instant --max-conversations 3 --seed 42`

**Day 2 Checklist:**
- [ ] All 4 new CLI args present
- [ ] `data[0]` hardcode removed; outer loop iterates conversations
- [ ] Per-conversation + aggregate metrics in output JSON
- [ ] Bootstrap CI (95%) in output
- [ ] Protocol fingerprint block in output
- [ ] Validation run with 3 conversations completes

---

## Day 3 — April 17: Apples-to-Apples Baseline (PB-03)

**New files:** `csam_project/benchmarks/benchmark_baseline_rag_hosted.py`, `csam_project/benchmarks/compare_csam_vs_baseline.py`

**Tasks:**

11. **Create `benchmark_baseline_rag_hosted.py`** — class `HostedBaselineRAGAgent`:
    - L2 HNSW only (no L1, no L3, no consolidation, no forgetting)
    - Same embedding service (all-MiniLM-L6-v2)
    - Same hosted LLM provider/model (Groq)
    - Same retrieval: `k=20`, top-10 context
    - Same ingest format: `[date] Speaker: content`
    - Same QA prompt as CSAM benchmark
    - CLI args mirror `benchmark_multimodel.py` exactly
    - Protocol fingerprint block with identical field names

12. **Create `compare_csam_vs_baseline.py`**:
    - Takes two JSON paths (CSAM result, baseline result)
    - Verifies protocol fingerprints match (same model, provider, seed, num_conversations, retrieval_k)
    - Produces: per-conversation F1 delta, mean delta, bootstrap 95% CI
    - Warns if any protocol fields differ
    - Output: `results_comparison_csam_vs_baseline.json`

13. **Run baseline:** `python -m csam_project.benchmarks.benchmark_baseline_rag_hosted --provider groq --model llama-3.1-8b-instant --max-conversations 3 --seed 42`

14. **Run compare script**, verify CSAM delta is positive on LoCoMo

**Day 3 Checklist:**
- [ ] `benchmark_baseline_rag_hosted.py` created; passes linting
- [ ] Architecture is flat RAG only (no L1/L3/consolidation/forgetting)
- [ ] Protocol fingerprint identical structure to CSAM output
- [ ] `compare_csam_vs_baseline.py` validates protocol parity
- [ ] Baseline run completes, output saved
- [ ] Comparison shows positive CSAM delta

---

## Day 4 — April 18: Metric Consistency + Ablation Fix + Variance Start (PB-08, PB-05)

**Files:** `csam_project/benchmarks/metrics.py` (new), `csam_project/evaluation/run_ablation.py` (line 456), `csam_project/tests/test_metrics.py` (new)

**Tasks:**

15. **Create `csam_project/benchmarks/metrics.py`** (canonical metrics module):
    - `normalize_text(text: str) -> str`
    - `token_f1(prediction: str, ground_truth: str) -> float` — Counter intersection F1
    - `exact_match(prediction: str, ground_truth: str) -> bool`
    - `aggregate_f1(scores: list[float], include_zeros: bool = True) -> float` — `include_zeros=True` is default

16. **Fix `run_ablation.py` line 456** (PB-08):
    - Change: `overall_f1 = np.mean([s for s in f1_scores.values() if s > 0])`
    - To: `overall_f1 = float(np.mean(list(f1_scores.values())))`
    - Record before/after numbers for paper methods section

17. **Create `csam_project/tests/test_metrics.py`**:
    - `test_token_f1_exact_match`: identical strings → 1.0
    - `test_token_f1_no_overlap`: different tokens → 0.0
    - `test_token_f1_partial`: known partial case
    - `test_aggregate_f1_includes_zeros`: `[0.5, 0.0, 0.5]` → 1/3, not 0.5
    - `test_normalize_text_strips_punctuation`

18. **Start variance runs** (background):
    ```
    python -m csam_project.evaluation.variance_runner --mode ablation --seeds 42 43 44 45 46
    python -m csam_project.evaluation.variance_runner --mode sweep --seeds 42 43 44
    ```

**Additional Day 4 tasks (from audit + publication review):**

19b. **Verify BUG-06** — check `benchmark_e2e.py`: does the test explicitly call `npc.consolidate()` or equivalent between ingestion and the forgetting phase? If not, `ConsolidationAwareForgetting` is running with `C(m)=0` for all memories (degenerate — everything is protected), making the ablation meaningless.
- Read `benchmark_e2e.py` consolidation trigger logic
- Add explicit `system.consolidate()` call after ingestion if missing
- Re-run ablation with fixed trigger; record new numbers

20b. **Create `test_consolidation_recall_invariant.py`** (BUG-05, TEST-01 — 2 hours):
- Ingest memory M
- Force consolidation → L3 node N created
- Forget M from L2
- Assert querying M's content still returns L3 content
- This is the single most important test for paper credibility

21b. **Create `test_forgetting_strategy_ranking.py`** (TEST-03 — 1 hour):
- Assert unconsolidated memories get score 0 (protected)
- Assert consolidated memories (C(m) > threshold) get non-zero score
- Assert equal-weight formula produces correct ordering

22b. **[PB-11 — CRITICAL] Add CA-no-gate as 5th ablation strategy** (~30 min):
- In `run_ablation.py` `strategies` list (currently 4 entries), add:
  ```python
  ("CA-Formula-Only (no gate)", ConsolidationAwareForgetting(
      alpha=0.25, beta=0.25, gamma=0.25, delta=0.25,
      consolidation_threshold=0.0   # gate disabled — formula runs but never protects
  )),
  ```
- This is the single most important addition for a defensible paper.
- Expected: CA-full > CA-no-gate. If not, the gate claim must be revised before submission.
- Include this 5th strategy in all variance runner seeds.

23b. **[PB-12 — HIGH] Add False Forgetting Rate logging** (~2 hours):
- In `forgetting_engine.py`: expose `last_forgotten_ids: list[str]` property, cleared each cycle.
- In `run_ablation.py` evaluation loop: after each forgetting event, check if any forgotten memory ID is a known supporting-fact memory (the ablation dataset tracks which memories were ingested per QA).
- Accumulate `false_forgetting_events` / `total_forgetting_events` per strategy.
- Add `"false_forgetting_rate": float` to `EvaluationResult` and output JSON.
- Expected: CA has lowest false forgetting rate across all strategies.

24b. **[PB-13 — MEDIUM] Add `--no-l3` flag to `benchmark_musique.py`** (~1 hour):
- When `--no-l3` is set, skip L3 KG retrieval and use L2 context only.
- Run MuSiQue with and without L3 for `llama-3.1-8b-instant` (1 model is enough).
- Report F1 broken down by hop count for both variants.
- If L3 helps on 3-hop/4-hop and not 1-hop, cite this in Architecture section.
- If L3 doesn't help, note honestly in Limitations — do not overclaim.

**Day 4 Checklist:**
- [ ] `metrics.py` created with all 4 functions
- [ ] `run_ablation.py:456` fixed (no more zero exclusion)
- [ ] `test_metrics.py` — all 5+ tests pass
- [ ] **BUG-06 verified/fixed** — E2E ablation explicitly triggers consolidation
- [ ] **`test_consolidation_recall_invariant.py` created and passes** (core claim verified)
- [ ] **`test_forgetting_strategy_ranking.py` created and passes**
- [ ] **[PB-11] CA-no-gate 5th strategy added to `run_ablation.py`**
- [ ] **[PB-12] False Forgetting Rate field in `EvaluationResult` and output JSON**
- [ ] **[PB-13] `--no-l3` flag added to `benchmark_musique.py`; hop-breakdown run**
- [ ] Variance ablation run started (5 strategies × 5 seeds)
- [ ] Variance sweep run started
- [ ] Before/after ablation F1 numbers recorded
- [ ] `consolidation_status` added to ablation output JSON

---

## Day 5 — April 19: Test Hardening + Full Artifact Collection (PB-06)

**Files:** `test_working_memory.py`, `test_npc_l1_integration.py`, `test_metadata_filtering.py`

**Tasks:**

19. **Rewrite `test_working_memory.py`** — assertion-first style, zero `print()`:
    - `test_lru_eviction_at_capacity`
    - `test_player_isolation`
    - `test_get_fact_returns_correct_value`
    - `test_empty_cache_returns_empty_list`

20. **Rewrite `test_npc_l1_integration.py`**:
    - `assert hasattr(npc, 'working_memory')`
    - `assert len(recent_items) > 0` after adding messages
    - `assert len(recent_items) <= 20` after adding 25+ items

21. **Rewrite `test_metadata_filtering.py`**:
    - After Bob+Alice memories with `player_name="Bob"` filter, assert zero Alice items in results

22. **Run full test suite with coverage:**
    ```
    pytest csam_project/tests/ -v --cov=csam_project/csam_core --cov-report=term-missing
    ```
    Record coverage % for paper methods section.

23. **Run full LoCoMo multimodel** (all 3 models, 3 conversations, seed 42):
    - `llama-3.1-8b-instant`, `llama-4-scout-17b-16e-instruct`, `llama-3.3-70b-versatile`
    - Output: `results_multimodel_locomo_canonical_{model}.json`

24. **Run matched baseline** (same 3 models, same scope):
    - Output: `results_baseline_rag_hosted_{model}.json`

**Additional Day 5 tasks (from audit):**

24b. **Fix BUG-03** — remove broken async `retrieve()` in `retrieval.py`:
- Delete or comment out the `async def retrieve()` method (lines 97–184)
- Rename `retrieve_sync()` to `retrieve()` for clean API
- Update any callers (check for `await retriever.retrieve(`)

25b. **Fix BUG-07** — soft-delete mapping cleanup in `memory_repository.py`:
- In `delete()` and `delete_batch()`, also remove entries from `_id_to_index` and `_index_to_id`
- Add `stale_ratio` property; trigger `rebuild_index()` when ratio > 0.20

26b. **Fix BUG-10** — typo in `working_memory.py`: `"Recent Coversation:"` → `"Recent Conversation:"`
- Also fix in any test that checks for this string

27b. **Create `test_l1_contamination.py`** (TEST-02):
- Verify `clear_all()` empties L1
- Verify ingestion turns don't pollute QA context after clear

**Day 5 Checklist:**
- [ ] All 3 original test files: assertion-based, no `print()`
- [ ] **`test_l1_contamination.py` created and passes** (TEST-02)
- [ ] **BUG-03 fixed** — async `retrieve()` removed; `retrieve_sync()` renamed
- [ ] **BUG-07 fixed** — soft-delete cleans up `_id_to_index` mappings
- [ ] **BUG-10 fixed** — typo corrected in `working_memory.py`
- [ ] `pytest csam_project/tests/ -v` — all pass
- [ ] Coverage % recorded
- [ ] Full LoCoMo canonical runs for all 3 models complete
- [ ] Baseline runs for all 3 models complete

---

## Day 6 — April 20: Paper Writing (Architecture, Background) + Process Artifacts

**Paper sections (lead author):**
- Abstract (draft with placeholders — use `[TABLE-LOCOMO]` until Day 7 numbers confirmed)
- Introduction — motivation, gap, CSAM's 3 claims
- Background / Related Work — RAG, episodic memory for agents, KG integration, forgetting mechanisms
- Architecture — L1/L2/L3, forgetting formula (0.25 equal), consolidation gate (C(m) < 0.3 → F=0)

**Engineering track:**
25. **Process variance outputs** — verify `variance_ablation_results.json` and `variance_sweep_results.json`

26. **Create `csam_project/docs/claim_to_artifact_matrix.md`**:
    - Each row: claim text | artifact file | PB-* blocker closed | status

27. **Run compare script** for all 3 model pairs

**Day 6 Checklist:**
- [ ] Abstract drafted (with placeholders)
- [ ] Introduction drafted
- [ ] Background / Related Work drafted
- [ ] Architecture section drafted (correct formula, correct gate condition)
- [ ] `claim_to_artifact_matrix.md` created
- [ ] Comparison artifacts for all 3 models generated

---

## Day 7 — April 21: Finalize Benchmark Data + Paper Experiments Section

**Tasks:**

28. **Run MuSiQue with `--seed 42`** for 8B and 70B:
    - `python -m csam_project.benchmarks.benchmark_musique --provider groq --model llama-3.1-8b-instant --seed 42`

29. **Run HotPotQA with `--seed 42`** for 8B and 70B

30. **Verify variance outputs** — if incomplete, restart with 3 seeds minimum

31. **Create `csam_project/evaluation/run_manifest_publication.json`**:
    - Every artifact: file path, run command, timestamp, seed, model, question/conversation count

**Paper sections:**
- Datasets — LoCoMo (N conversations, QA count), MuSiQue (50 Q, 2–4 hop), HotPotQA (50 Q, 2-hop)
- Baselines — hosted RAG (same model/k), ablation strategies
- Results: LoCoMo — fill `[TABLE-LOCOMO]` from Day 5 runs; architecture-dominates-model-size
- Results: Multi-hop — MuSiQue + HotPotQA tables; model-size-adds-gains
- Results: Ablation — corrected (PB-08 fixed) numbers

**Day 7 Checklist:**
- [ ] MuSiQue `--seed 42` runs complete
- [ ] HotPotQA `--seed 42` runs complete
- [ ] `run_manifest_publication.json` created
- [ ] Experiments section drafted with confirmed numbers
- [ ] All tables cross-referenced in `claim_to_artifact_matrix.md`

---

## Day 8 — April 22: Paper Analysis, Discussion, Ablation, Limitations

**Paper sections:**
- Ablation Analysis — variance mean ± std from `variance_ablation_results.json`; consolidation-aware forgetting is dominant strategy
- Threshold Sensitivity — `variance_sweep_results.json`; θ=0.3 sits in stable region (0.2–0.4)
- Discussion — architecture vs model size tradeoff; when CSAM helps most; failure modes
- Limitations — single-language, offline consolidation, HNSW index rebuild frequency

**Engineering:**
32. **Finalize `claim_to_artifact_matrix.md`** with all paper claim rows

33. **Commit all code + artifacts:**
    ```
    git add csam_project/benchmarks/ csam_project/evaluation/ csam_project/tests/ csam_project/docs/ .github/ pyproject.toml
    git commit -m "fix: close PB-01 through PB-10 for publication readiness"
    ```

**Day 8 Checklist:**
- [ ] Ablation analysis section drafted (mean ± std)
- [ ] Threshold sensitivity section drafted
- [ ] Discussion drafted
- [ ] Limitations section drafted
- [ ] `claim_to_artifact_matrix.md` finalized
- [ ] All code committed; tests pass

---

## Day 9 — April 23: Full Draft Assembly + Figures + Internal Review

**Paper assembly:**
- Write: Conclusion, Future Work, Appendix (reproducibility notes citing `run_manifest_publication.json`)
- Polish: Introduction, Abstract (finalize with confirmed numbers)
- Generate figures (see DIAG-01 through DIAG-06 in `codebase_audit_and_improvements.md`):
  1. **Architecture diagram** — add capacity labels: `L1: LRU cap=20`, `L2: HNSW ≤200`, `L3: NetworkX`, gate annotation `C(m) ≥ 0.3`
  2. **Forgetting decision tree** (NEW) — visual flowchart: C(m) < θ? → Protected / → Score → Top-N? → Delete
  3. **Consolidation flow** — add memory-to-node mapping with coverage values shown
  4. **LoCoMo F1 bar chart** — CSAM vs baseline (same model, same k), architecture-dominates
  5. **Ablation bar chart** — with ±std error bars from variance runner; corrected (zero-inclusive) F1
  6. **Threshold sweep curve** — ±std shading; annotate θ=0.3 as "stable region" midpoint
  7. **Memory growth chart** (existing) — add horizontal cap line at 200, vertical forgetting event ticks

**Internal review checklist:**
- [ ] Every number traces to a closed PB-* and a `run_manifest_publication.json` entry
- [ ] Formula in paper body = `forgetting_engine.py` (0.25 equal)
- [ ] No single-seed claim without CI or caveat
- [ ] Baseline section explicitly states model/provider/k parity
- [ ] No `if s > 0` filtered F1 cited anywhere
- [ ] Paper formatted to target venue style

---

## Day 10 — April 24: Final Review + Submission

- Proofread: abstract, introduction, conclusion (most-read sections)
- Cross-check all numbers against artifact JSONs one final time
- Verify references: all citations present, DOIs valid
- PDF: confirm no broken references, figures render correctly
- Submit to venue portal
- `git tag -a v1.0-submission -m "Paper submission April 24 2026"`

---

## April 25 — Buffer Day (Hard Deadline)

Reserved for: portal issues, PDF format rejection, co-author revision feedback.
**No new implementation permitted on this day.**

---

## Parallel Tracks Summary

| Day | Track A (Engineering) | Track B (Paper Writing) |
|-----|----------------------|------------------------|
| 1 | Governance, formula, pyproject, seed patches | — |
| 2 | LoCoMo multi-conv refactor | — |
| 3 | Baseline hosted benchmark | — |
| 4 | Metrics module, ablation fix, variance runs | — |
| 5 | Test hardening, full artifact collection | — |
| 6 | Process variance, comparison artifacts, claim matrix | Abstract, Intro, Background, Architecture |
| 7 | MuSiQue/HotPotQA reruns, run manifest | Experiments section |
| 8 | Docs commit, final verification | Analysis, Discussion, Ablation, Limitations |
| 9 | Internal review checklist | Full draft, figures, venue formatting |
| 10 | — | Proofread, submit |

---

## Critical Path

```
Day 1: pyproject fix → formula fix → seed patches (musique, hotpotqa)
         |
Day 2: LoCoMo multi-conv (removes 1-conv hardcode, adds protocol block)
         |
Day 3: Hosted baseline (benchmark_baseline_rag_hosted.py + compare script)
         |
Day 4: Metrics module → ablation fix (run_ablation.py:456) | Variance runs START
         |                                                    |
Day 5: Test hardening | Full artifact runs (all models, all benchmarks)
         |                                                    |
Day 6: Claim matrix | Compare artifacts        | Paper: Arch/Intro/Background
         |                                      |
Day 7: Seed-controlled reruns | run_manifest   | Paper: Experiments (confirmed numbers)
         |
Day 8: Paper: Analysis + Ablation + Discussion | Docs committed
         |
Day 9: Full draft + 4 figures + internal review + venue formatting
         |
Day 10: Proofread → Submit → git tag v1.0-submission
```

---

## Key Risk Mitigations

**Risk 1: Groq rate limits block variance runs (PB-05)**
→ Run ablation mode first (may support local LLM). If blocked, reduce to 3 seeds and note in paper.

**Risk 2: LoCoMo dataset has only 1 conversation**
→ Check `len(data)` in `locomo10.json`. If N=1, reframe as "full question set" (remove 10 QA cap).
→ `--questions-per-conv None` removes the cap and is still a valid evidence expansion.

**Risk 3: Corrected F1 (PB-08) lowers ablation numbers**
→ Record old vs new. If CSAM CA strategy is still best-in-class among forgetting strategies, claim holds.

**Risk 4: Hosted baseline outperforms CSAM on some metrics**
→ Report honestly. Scope the baseline comparison to LoCoMo. Multi-hop tasks are model-size-dominated by design — that is already the stated finding.

---

## Final Submission Checklist

### Engineering
- [ ] PB-01: Execution plan populated
- [ ] PB-02: LoCoMo results ≥3 conversations; protocol metadata in output
- [ ] PB-03: Baseline same model/provider/k=20/top-10; comparison artifact present
- [ ] PB-04: `--seed` in all 3 benchmark CLIs; seed in every output JSON
- [ ] PB-05: Variance JSONs present (≥3 seeds) or risk explicitly accepted in paper
- [ ] PB-06: 3 test files assertion-based; suite passes
- [ ] PB-07: `pytest-cov>=4.0.0` in `pyproject.toml`
- [ ] PB-08: `run_ablation.py:456` no longer filters zeros; `metrics.py` exists
- [ ] PB-09: All 3 places (copilot-instructions, CLAUDE.md, forgetting_engine.py) say 0.25
- [ ] PB-10: Gap doc is current-state; no stale items

### Traceability
- [ ] `run_manifest_publication.json` — every artifact with command, seed, model, timestamp
- [ ] `claim_to_artifact_matrix.md` — every paper claim mapped to JSON file
- [ ] `git tag v1.0-submission` applied

### Paper Content
- [ ] Formula in paper = 0.25 equal weights (matches code)
- [ ] Baseline comparison states protocol parity explicitly
- [ ] Ablation table uses corrected all-category F1
- [ ] Variance / stability reported (mean ± std or CI)
- [ ] Limitations section present
- [ ] All figures render in final PDF

### Audit Bug Closures (from codebase_audit_and_improvements.md)
- [ ] BUG-01: `run_ablation.py:456` zeros included (same as PB-08)
- [ ] BUG-02: Cerebras typo fixed in `llm_hosted.py:40`
- [ ] BUG-03: Broken async `retrieve()` removed from `retrieval.py`
- [ ] BUG-04: Rate limit retry capped at 5 max retries
- [ ] BUG-05: `test_consolidation_recall_invariant.py` exists and passes
- [ ] BUG-06: E2E ablation verified to trigger consolidation before forgetting
- [ ] BUG-07: Soft-delete cleans up `_id_to_index` mappings
- [ ] BUG-10: "Coversation" typo fixed

---

## Post-Submission: TUI Demo Plan

After paper is submitted, the primary engineering effort shifts to the Textual TUI demo.

**Framework:** `textual>=0.47.0` — Python-native TUI (terminal UI) built on Rich.
**Why Textual:** Runs in any terminal, no browser needed, looks professional for conference demos, recordable with `asciinema`.

**Full spec:** See Section 6 of `csam_project/docs/codebase_audit_and_improvements.md`

### Layout Overview
```
╔═══════════════════════════════════════════════════════════════╗
║  CSAM — Cognitive Sparse Access Memory Demo          v1.0     ║
╠══════════════════╦════════════════════════════════════════════╣
║  MEMORY LAYERS   ║  CONVERSATION                              ║
║  L1 ▓▓▓▓▓░░░░   ║  [14:23] You: Tell me about Alice          ║
║  12/20 items     ║  [14:23] Aric: Alice visited the library.. ║
║  L2 ▓▓▓▓▓▓▓░░░  ║                                            ║
║  143/200         ║  > _  (input)                              ║
║  L3 28 nodes     ║                                            ║
║     41 edges     ║  TABS: [Chat] [Memory] [Graph] [Stats]     ║
╠══════════════════╬════════════════════════════════════════════╣
║  Last retrieval  ║  F1:Skip  F2:Consolidate  F3:Switch NPC   ║
║  L2:12  L3:3     ║  F4:Memory Inspector  F9:Save Transcript   ║
╚══════════════════╩════════════════════════════════════════════╝
```

### File Structure
```
csam_project/simulation/
  tui_demo.py                    — Main Textual app
  tui_widgets/
    memory_sidebar.py            — Live L1/L2/L3 stat panels
    conversation_panel.py        — Chat history
    memory_inspector.py          — Browsable memory table
    graph_view.py                — ASCII knowledge graph
    stats_panel.py               — Session statistics
  demo_scripts/
    demo_script_publication.json — Scripted demo for paper appendix
```

### CLI Interface
```bash
python -m csam_project.simulation.tui_demo \
    --npc aric \
    --seed 42 \
    --replay demo_script_publication.json \
    --record output_transcript.json
```

### Add to pyproject.toml
```
"textual>=0.47.0",
```

### TUI Development Schedule (Post-Apr 25)
| Week | Work |
|------|------|
| Week 1 | Textual app skeleton + memory sidebar (TUI-01, TUI-02) |
| Week 2 | Memory Inspector + Graph tab (TUI-03, TUI-04) |
| Week 3 | Stats tab + Keybindings (TUI-05, TUI-06) |
| Week 4 | Replay/record mode + polish (TUI-07) |

---

## HANDOVER — Session 2 (Apr 15, 2026)

**Last updated:** 2026-04-15 (end of session 2)

### What Was Completed This Session

| Item | File(s) Changed | Status |
|------|----------------|--------|
| BUG-02: Cerebras API key typo fixed | `csam_core/services/llm_hosted.py` | DONE |
| BUG-04: Rate limit retry now capped at MAX_RETRIES=5 | `csam_core/services/llm_hosted.py` | DONE |
| `llm_hosted.py` full rewrite | `csam_core/services/llm_hosted.py` | DONE |
| 4 new providers added (Fireworks, Together, NVIDIA NIM, OpenRouter) | `csam_core/services/llm_hosted.py` | DONE |
| `PUBLICATION_MODELS` list defined (4 canonical models) | `csam_core/services/llm_hosted.py` | DONE |
| `FALLBACK_PROVIDERS` dict with rate-limit fallback routing | `csam_core/services/llm_hosted.py` | DONE |
| PB-09: Formula drift fixed | `.github/copilot-instructions.md:25` | DONE |
| PB-07: `pytest-cov>=4.0.0` added | `pyproject.toml` | DONE |
| PB-04: `--seed` added to `benchmark_musique.py` | `benchmarks/benchmark_musique.py` | DONE |
| PB-04: `--seed` added to `benchmark_hotpotqa.py` | `benchmarks/benchmark_hotpotqa.py` | DONE |
| Seed-based random shuffle of questions (reproducible samples) | both benchmark files | DONE |
| `checkpoint.py` created — atomic per-question save/resume | `benchmarks/checkpoint.py` | DONE |
| Checkpointing wired into `benchmark_musique.py` | `benchmarks/benchmark_musique.py` | DONE |
| Checkpointing wired into `benchmark_hotpotqa.py` | `benchmarks/benchmark_hotpotqa.py` | DONE |
| Output filenames now include seed (`_s42.json`) | both benchmark files | DONE |

### Still Pending from Day 1–2

| Item | File | Notes |
|------|------|-------|
| Restore execution plan | `docs/publication_engineering_execution_plan.md` | Low priority — sprint plan covers it |
| `--seed` for `benchmark_multimodel.py` (PB-04) | `benchmarks/benchmark_multimodel.py` | Day 2 task |
| Checkpointing for `benchmark_multimodel.py` | `benchmarks/benchmark_multimodel.py` | Day 2 task |
| LoCoMo multi-conversation loop (PB-02) | `benchmarks/benchmark_multimodel.py` | Day 2 — CRITICAL |
| Baseline RAG benchmark (PB-03) | `benchmarks/benchmark_baseline_rag_hosted.py` (new) | Day 3 — CRITICAL |
| F1 zero-exclusion fix (BUG-01) | `evaluation/run_ablation.py:456` | Day 4 — CRITICAL |
| Consolidation-recall invariant test (BUG-05) | `tests/test_consolidation_recall_invariant.py` (new) | Day 4 — CRITICAL |
| BUG-03: Remove broken async `retrieve()` | `csam_core/retrieval.py:122` | Day 5 |
| BUG-07: Soft-delete stale HNSW mappings | `csam_core/memory_repository.py` | Day 5 |
| BUG-10: Typo in working memory | `csam_core/working_memory.py` | Day 5 |

### When to Start Notebooks & Cloud Testing

**Prerequisites before notebooks are useful:**
1. PB-02 (LoCoMo multi-conv) must be fixed in `benchmark_multimodel.py` — Day 2
2. PB-03 (baseline RAG) must exist — Day 3
3. BUG-01 (F1 zeros fix) must be in `run_ablation.py` — Day 4

**Earliest notebook start: Day 3 (April 17)** — hotpotqa and musique can run on Colab/Kaggle as soon as PB-04 seed work is done (it is now). LoCoMo/multimodel needs Day 2 fix first.

**Do we need to re-run all tests from scratch?**

- **HotPotQA & MuSiQue**: YES — old result files have no `seed` field and used non-shuffled ordering, so they are not reproducible baselines. Re-run with `--seed 42 --questions 100` to get canonical results. Existing JSON files (`results_hotpotqa_*.json`, `results_musique_*.json`) should be treated as preliminary.
- **LoCoMo / multimodel**: YES — PB-02 fix changes the scope fundamentally (1 conv → full dataset). All old LoCoMo numbers are invalidated.
- **Ablation**: YES — BUG-01 (F1 zero exclusion) inflated prior results. Re-run after Day 4 fix.
- **Baseline RAG**: N/A — does not exist yet, first run = canonical.

**Notebook plan (Day 3+):**
1. `colab_hotpotqa.ipynb` — mounts Drive, installs deps, runs `benchmark_hotpotqa.py --all --seed 42 --questions 100 --checkpoint-dir /gdrive/...`
2. `colab_musique.ipynb` — same pattern for MuSiQue
3. `colab_multimodel.ipynb` — LoCoMo runs (after Day 2 fix)
4. `colab_baseline.ipynb` — baseline RAG comparison (after Day 3 fix)

All notebooks should: (a) load `.env` from Google Drive, (b) use `--checkpoint-dir` pointing to Drive so progress survives session resets, (c) save final JSON to Drive.

---

## PUBLICATION DEFENSIBILITY VERDICT (Apr 15)

### Final Assessment

| Tier | Verdict |
|------|---------|
| Workshop / national conference (CIS, ICCIS, AIC) | **Submission-ready after Day 5** if PB-02/03/11 are closed |
| Mid-tier (CoLLAs, ACL findings) | **Submission-ready after Day 7** — needs canonical numbers + CA-gate ablation |
| Top-tier (EMNLP main) | Needs 1 more cycle — variance, FFR metric, and L3 isolation all expected |

### Why These 3 New Gaps Are Sprint-Critical (Not Separate)

**They must be fixed before the canonical runs start.** If PB-11/12/13 are left until later:
- You will run the ablation on Day 4 with only 4 strategies → realize Day 8 you're missing the gate comparison → re-run everything from scratch, burning 2 days you don't have.
- The False Forgetting Rate cannot be computed retroactively without re-running with logging enabled.
- The L3 hop-breakdown needs one extra run on MuSiQue — cheapest to do Day 4 while cloud is already running.

**Each costs ≤2h of code. The cost of deferring is a full re-run.**

### What Makes This Paper Defensible

The paper will be defensible when it can answer all of the following without hesitation:
1. *Why is your baseline fair?* → Same model, same k, same embedding, same prompt (PB-03 ✓)
2. *Is the gate the contribution or just the formula?* → CA-full beats CA-no-gate (PB-11 ✓)
3. *How do you know you're not randomly forgetting?* → False Forgetting Rate (PB-12 ✓)
4. *Does L3 actually help?* → MuSiQue F1-by-hop with/without L3 (PB-13 ✓)
5. *Are your numbers stable?* → ±std across 5 seeds (PB-05 ✓)
6. *Are your metrics fair?* → Zero-inclusive F1, canonical metrics.py (PB-08 ✓)

### Next Agent Starting Point

**Day 2 — `benchmark_multimodel.py`** (PB-02 is the critical path blocker):
1. Add `--seed`, `--max-conversations` args; wire in `BenchmarkCheckpoint` (`benchmarks/checkpoint.py` exists)
2. Fix PB-02: remove `data[0]` hardcode at line ~149; add outer conversation loop
3. Add bootstrap 95% CI and protocol fingerprint block to output JSON

**Day 3 — `benchmark_baseline_rag_hosted.py`** (new file, PB-03):
- Flat L2-only agent, identical model/k/embedding/prompt as CSAM benchmark
- `compare_csam_vs_baseline.py` validates protocol parity before computing delta

**Day 4 — Ablation fixes (PB-08, PB-11, PB-12, PB-13 all in same session):**
- Fix F1 zero-exclusion bug (`run_ablation.py:456`)
- Add 5th strategy CA-no-gate (PB-11) — 30 min, HIGHEST PRIORITY
- Add False Forgetting Rate logging (PB-12) — 2h
- Add `--no-l3` to MuSiQue (PB-13) — 1h
- BUG-05: `test_consolidation_recall_invariant.py`
- BUG-06: verify consolidation fires before forgetting in E2E
- Start variance runs (5 strategies × 5 seeds) overnight

**All Day 4 engineering must be done before starting any cloud notebook runs for ablation.**
