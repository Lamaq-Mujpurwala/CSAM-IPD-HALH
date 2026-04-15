# CSAM Codebase Audit & Improvement Plan

Generated: 2026-04-15
Scope: Full audit of all core modules, benchmarks, evaluation scripts, tests, and demo layer.
Purpose: Publication readiness + demo improvements (TUI direction).

---

## Section 1: Confirmed Bugs (Fix Before Any Submission)

These are verified bugs that either produce wrong results, break functionality, or undermine the research claims.

---

### BUG-01 — Ablation F1 Excludes Zero Categories (CRITICAL — KNOWN PB-08)

**File:** `csam_project/evaluation/run_ablation.py:456`
**Code:** `overall_f1 = np.mean([s for s in f1_scores.values() if s > 0])`
**Problem:** If any QA category scores exactly 0.0 (e.g., multi-hop type fails completely), that category is silently excluded from the aggregate. This inflates the reported overall F1 for strategies that fail hard on some category types.
**Impact:** Results for all 4 strategies in the ablation are potentially inflated by an unknown amount. The *relative ranking* may still be valid, but absolute numbers cannot be cited in the paper.
**Fix:**
```python
overall_f1 = float(np.mean(list(f1_scores.values())))
```
**Priority:** Fix in Day 4 of sprint. Record before/after numbers.

---

### BUG-02 — Cerebras Provider Completely Non-Functional (HIGH)

**File:** `csam_project/csam_core/services/llm_hosted.py:40`
**Code:** `"env_key": "CEREBAS_API_KEY"` — missing letter 'R'
**Problem:** The environment variable lookup is `os.environ.get("CEREBAS_API_KEY")` (no 'R'). The real key is stored as `CEREBRAS_API_KEY`. Result: Cerebras provider always returns None for the API key and fails silently or raises auth errors.
**Note in file:** Comment says "typo preserved from .env" — if the .env file also has the typo, this accidentally works. But this is not reliable and should be fixed properly.
**Fix:** Change to `"env_key": "CEREBRAS_API_KEY"` and update .env if needed.
**Priority:** Fix immediately (5-minute fix, no publication impact since only Groq is used for benchmarks).

---

### BUG-03 — Async `retrieve()` Calls Sync Method with `await` (HIGH)

**File:** `csam_project/csam_core/retrieval.py:122`
**Code:** `l2_results = await self.memory_repo.retrieve(...)` — `MemoryRepository.retrieve()` is a synchronous method.
**Problem:** Any code calling `HybridRetriever.retrieve()` (the async path) would hang or raise `TypeError: object NoneType can't be used in 'await' expression`.
**Current state:** All benchmarks use `retrieve_sync()` which works correctly. The async path is dead and broken.
**Fix options:**
1. Remove the async `retrieve()` method entirely and keep only `retrieve_sync()`.
2. Convert `MemoryRepository.retrieve()` to async.
**Recommended:** Option 1 — remove broken async method. The `retrieve_sync()` path is correct and that's all that's used.
**Priority:** Fix before paper — broken API will confuse reviewers reading the code.

---

### BUG-04 — Rate Limit Retry Has No Maximum (MEDIUM)

**File:** `csam_project/csam_core/services/llm_hosted.py` (rate limit handling block)
**Problem:** On HTTP 429, the code reads `retry_after` from response header then recursively calls `generate()` again. If the API keeps returning 429 (e.g., quota exhausted for the day), this loop never terminates.
**Impact:** Benchmark runs can hang indefinitely with no user feedback.
**Fix:**
```python
def generate(self, ..., _retry_count: int = 0) -> str:
    MAX_RETRIES = 5
    ...
    if response.status_code == 429 and _retry_count < MAX_RETRIES:
        retry_after = int(response.headers.get("Retry-After", 30))
        time.sleep(retry_after)
        return self.generate(..., _retry_count=_retry_count + 1)
    elif response.status_code == 429:
        raise RuntimeError(f"Rate limit exceeded after {MAX_RETRIES} retries")
```
**Priority:** Fix before multi-seed variance runs (which are most likely to hit rate limits).

---

### BUG-05 — Consolidation Never Verified to Preserve Recall After Forgetting (CRITICAL — Core Claim)

**Files:** No test file covers this. `csam_project/tests/` has no test for the core invariant.
**Problem:** The entire paper claims that "consolidation-aware forgetting preserves recall because L3 absorbs the semantic content before deletion." But there is no test that:
1. Ingests memory M
2. Consolidates M into L3 node N
3. Forgets M from L2
4. Queries for M's content
5. Asserts N (or an L3-equivalent response) is returned

Without this test, the core claim is an assertion, not a verified property.
**Fix:** Add `csam_project/tests/test_consolidation_recall_invariant.py` (see Section 3).
**Priority:** CRITICAL — must exist before submission. Even a single end-to-end scenario test suffices.

---

### BUG-06 — E2E Ablation May Never Trigger Actual Consolidation (HIGH)

**File:** `csam_project/benchmarks/benchmark_e2e.py`
**Problem:** The ablation compares forgetting strategies (NoForgetting, LRU, Importance, ConsolidationAware). But `ConsolidationAwareForgetting` relies on `consolidation_tracker.get_coverage(m.id)` returning non-zero values. If consolidation is never explicitly triggered during the test run, `C(m) = 0.0` for ALL memories, making the ConsolidationAware strategy degenerate to a score of 0.0 for everything (all memories protected).
**Result:** In that degenerate state, CSAM forgets nothing (same as NoForgetting) — not because it's superior, but because the threshold gate fires on everything.
**Fix:** Verify that the E2E benchmark explicitly calls `npc.consolidate()` or the consolidation pipeline between ingestion and the forgetting phase. Add a log assertion: after consolidation, assert `len(tracker.get_consolidated_memories()) > 0`.
**Priority:** HIGH — if this is happening, the ablation results that show CA as "best" may be misleading.

---

### BUG-07 — Soft-Delete Leaves Stale Index Mappings (MEDIUM)

**File:** `csam_project/csam_core/memory_repository.py` (delete/delete_batch methods)
**Problem:** HNSW does not support removal. When a memory is "deleted," it is removed from `self._memories` dict but the HNSW index still contains its vector. The `_id_to_index` and `_index_to_id` mappings are never cleaned up on delete. Only `rebuild_index()` clears them.
**Impact:** Long-running sessions accumulate ghost entries. After 100 forgetting cycles without rebuilds, `len(_id_to_index)` >> `len(_memories)`. Retrieval performance degrades.
**Fix:** On delete, immediately clean up the mappings:
```python
def _cleanup_deleted_mapping(self, memory_id: str) -> None:
    if memory_id in self._id_to_index:
        idx = self._id_to_index.pop(memory_id)
        self._index_to_id.pop(idx, None)
```
And trigger `rebuild_index()` when stale ratio exceeds 20%:
```python
@property
def stale_ratio(self) -> float:
    total = len(self._id_to_index)
    active = len(self._memories)
    return (total - active) / max(total, 1)
```
**Priority:** MEDIUM — affects long-running demos and multi-NPC scaling benchmarks.

---

### BUG-08 — Metadata Filter Silently Returns Fewer Than k Results (MEDIUM)

**File:** `csam_project/csam_core/memory_repository.py` (around line 228)
**Problem:** When `metadata_filter` is active, the code fetches `k * 3` candidates from HNSW then post-filters. If all 3k candidates fail the filter (e.g., very selective player filter in a large shared memory pool), the caller receives 0 results with no warning.
**Impact:** In multiplayer scenarios with metadata filtering, one player can receive empty context silently — degrading QA quality for that player without any error signal.
**Fix:**
```python
if len(results) < k and metadata_filter:
    logger.warning(
        "metadata_filter returned only %d/%d results — "
        "consider increasing k or relaxing filter",
        len(results), k
    )
```
**Priority:** MEDIUM — important for correct multiplayer behavior in demo and scaling benchmark.

---

### BUG-09 — `rerank_simple()` Has Misleading Signature (LOW)

**File:** `csam_project/csam_core/mmr.py:121-142`
**Problem:** `rerank_simple()` passes `embeddings[0]` (first candidate's embedding) as `query_embedding` to `rerank()`. However, `rerank()` never actually uses `query_embedding` in its implementation — relevance is already precomputed. This is a dead parameter with a misleading value.
**Impact:** No functional bug, but confusing for anyone reading the code or trying to add query-aware MMR in the future.
**Fix:**
```python
def rerank_simple(self, items, embeddings, relevance_scores, k, query_embedding=None):
    candidates = list(zip(items, embeddings, relevance_scores))
    dummy_query = query_embedding if query_embedding is not None else np.zeros_like(embeddings[0])
    return self.rerank(candidates, dummy_query, k)
```
And add a comment in `rerank()` noting query_embedding is currently unused.

---

### BUG-10 — `working_memory.py` Typo in Output String (LOW)

**File:** `csam_project/csam_core/working_memory.py` (context output method)
**Problem:** `"Recent Coversation:"` — missing 't' in "Conversation".
**Impact:** Visible in demo output and NPC chat; looks unprofessional. The integration test `test_npc_l1_integration.py` may check for this string and would fail if fixed without also fixing the test.
**Fix:** Change to `"Recent Conversation:"` and update any test that checks for the old string.

---

### BUG-11 — ConsolidationTracker Coverage Cache is Monotonically Increasing (MEDIUM)

**File:** `csam_project/csam_core/consolidation_tracker.py` (around line 103)
**Problem:** Coverage cache uses `max(current, new_coverage)` — once a memory's coverage is set high, it can never decrease. If the L3 graph is later pruned or the consolidation is corrected, the cached coverage remains stale.
**Impact:** In experiments with iterative consolidation runs (e.g., ablation with different configurations), stale coverage scores can cause incorrect forgetting decisions.
**Fix:** Allow coverage to update on re-consolidation: replace with most recent value, or add a `force_update` parameter.

---

## Section 2: Implementation Quality Improvements

These are not bugs per se but significantly affect reproducibility, paper credibility, and demo quality.

---

### IMP-01 — Centralize All Metric Calculations (Required for Publication)

**Problem:** `token_f1()` is reimplemented independently in:
- `benchmark_multimodel.py`
- `benchmark_musique.py`
- `benchmark_hotpotqa.py`
- `benchmark_baseline_rag.py`
- `run_ablation.py`

Each implementation has slight differences in normalization (punctuation handling, whitespace, lowercase). This means the same prediction can score differently depending on which script runs it.

**Fix:** Create `csam_project/benchmarks/metrics.py`:
```python
import re
from collections import Counter
from typing import Optional

def normalize_text(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles, strip punctuation."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def token_f1(prediction: str, ground_truth: str) -> float:
    """Counter-based token overlap F1 (SQuAD standard)."""
    pred_tokens = Counter(normalize_text(prediction).split())
    gold_tokens = Counter(normalize_text(ground_truth).split())
    common = pred_tokens & gold_tokens
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / sum(pred_tokens.values())
    recall = num_common / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)

def aggregate_f1(scores: list[float], include_zeros: bool = True) -> float:
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))  # always include zeros
```
**Priority:** Day 4 of sprint. Replace all duplicated implementations.

---

### IMP-02 — Add Seed Controls Everywhere (Required for Reproducibility)

**Problem:** `benchmark_multimodel.py`, `benchmark_musique.py`, `benchmark_hotpotqa.py` have no `--seed` parameter. Any run that shuffles data (question sampling, conversation selection) produces non-reproducible results.

**Fix:** Add to every benchmark CLI parser:
```python
parser.add_argument("--seed", type=int, default=42,
    help="Random seed for reproducibility (default: 42)")
```
And immediately after `args = parser.parse_args()`:
```python
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
```
And write to every output JSON: `"seed": args.seed`.

**Priority:** Day 1 (musique, hotpotqa) and Day 2 (multimodel).

---

### IMP-03 — Add Protocol Fingerprint to All Result JSONs (Required for Reproducibility)

**Problem:** Current result JSONs don't record: dataset file used, scope (how many questions/conversations), retrieval settings (k=20, top_context=10), model and provider, timestamp.

**Fix:** Add to every benchmark output:
```json
"protocol": {
    "dataset": "locomo10.json",
    "num_conversations": 3,
    "questions_per_conv": "all",
    "retrieval_k": 20,
    "context_top_k": 10,
    "mmr_lambda": 0.0,
    "seed": 42,
    "model_id": "llama-3.1-8b-instant",
    "provider": "groq",
    "csam_version": "v1.0",
    "timestamp": "2026-04-17T14:32:00Z"
}
```
**Priority:** Day 2 (multimodel) and Day 3 (baseline).

---

### IMP-04 — Add `consolidation_status` to Output JSON (Research Transparency)

**Problem:** No benchmark output records whether L3 consolidation was actually triggered. For ConsolidationAwareForgetting to be meaningful, consolidation must have run. Without logging this, it's impossible to know if the core mechanism was active.

**Fix:** Add to every ablation/benchmark result:
```json
"consolidation_status": {
    "triggered": true,
    "num_l3_nodes": 42,
    "consolidated_memories": 38,
    "avg_coverage": 0.67
}
```
**Priority:** Day 4 of sprint, alongside ablation fix.

---

### IMP-05 — Forgetting Engine: Separate Score Calculation from Protection Logic

**Problem:** The protection gate (`if C < threshold: score = 0.0`) mixes two concerns — scoring and filtering. This makes it harder to study the scores of protected memories (you lose the actual score value).

**Fix:** Return scores and protection status separately:
```python
def compute_forget_scores(self, ...) -> tuple[dict[str, float], dict[str, bool]]:
    # Returns (scores, is_protected)
    # Protected memories get their real score, but is_protected=True
```
This lets you log "memory X had score 0.73 but was protected because C=0.21 < 0.3" — valuable for debugging and paper discussion.
**Priority:** Nice-to-have before submission; critical for understanding behavior during ablation.

---

### IMP-06 — Add Bootstrap Confidence Intervals to All Benchmark Outputs

**Problem:** Current results are single-point estimates. No confidence interval or variance information.

**Fix:** After collecting per-question F1 scores:
```python
import numpy as np

def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(scores, size=len(scores))) for _ in range(n_bootstrap)]
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)
```
Add to all benchmark JSON output:
```json
"f1_mean": 0.365,
"f1_ci95_low": 0.298,
"f1_ci95_high": 0.432,
"n_questions": 47
```
**Priority:** Day 2 (multimodel) and update all benchmarks by Day 7.

---

## Section 3: Baseline Tests Required for Publication

These are the tests that MUST exist and pass before submitting to any venue.

---

### TEST-01 — Consolidation-Recall Invariant Test (CRITICAL)

**File to create:** `csam_project/tests/test_consolidation_recall_invariant.py`

This test verifies the core claim: forgetting a consolidated memory does NOT degrade recall because L3 preserves the semantic content.

```python
def test_forgotten_memory_still_retrievable_via_l3():
    """
    Core invariant: if memory M is consolidated to L3 node N,
    then after M is forgotten from L2, querying for M's content
    should still return a result containing N's content.
    """
    # 1. Setup
    npc = NPC(personality=TEST_PERSONALITY, max_memories=10)
    npc.add_memory("Alice visited the library on Tuesday to borrow a book on alchemy.",
                   importance=0.9)
    
    # 2. Force consolidation
    npc.consolidate(force=True)
    
    # 3. Verify L3 has something
    l3_nodes = npc.knowledge_graph.get_all_nodes()
    assert len(l3_nodes) > 0, "L3 should have at least one node after consolidation"
    
    # 4. Get coverage of the memory
    mem_ids = list(npc.memory_repo._memories.keys())
    coverage = npc.consolidation_tracker.get_coverage(mem_ids[0])
    assert coverage > 0.0, "Memory should have non-zero coverage after consolidation"
    
    # 5. Forget the memory
    npc.memory_repo.delete(mem_ids[0])
    assert len(npc.memory_repo._memories) == 0, "L2 should be empty"
    
    # 6. Query for the forgotten content
    query_embedding = npc.embedding_service.encode("Alice alchemy library")
    l3_results = npc.knowledge_graph.query_by_embedding(query_embedding, k=3)
    
    # 7. L3 should still have relevant content
    assert len(l3_results) > 0, "L3 should return results for forgotten memory's content"
    top_node_text = l3_results[0][0].content
    assert any(word in top_node_text.lower() for word in ["alice", "library", "alchemy"]), \
        f"L3 node should contain memory content. Got: {top_node_text}"
```

---

### TEST-02 — Working Memory L1 Clear Before QA (HIGH)

**File to create:** `csam_project/tests/test_l1_contamination.py`

Verifies that L1 contamination from ingestion doesn't pollute QA context.

```python
def test_l1_clear_prevents_context_contamination():
    """L1 should be clearable; clearing removes ingestion pollution."""
    npc = NPC(personality=TEST_PERSONALITY, max_memories=50)
    # Simulate ingestion
    for i in range(25):
        npc.add_memory(f"Turn {i}: random filler text about weather")
    # L1 has last 20 filler turns
    assert len(npc.working_memory.get_recent("Player", k=20)) > 0
    
    # Clear L1 (as done before QA phase)
    npc.working_memory.clear_all()
    
    # L1 should be empty
    assert len(npc.working_memory.get_recent("Player", k=20)) == 0, \
        "L1 should be empty after clear_all()"
```

---

### TEST-03 — Forgetting Strategy Ranking Test (HIGH)

**File to create:** `csam_project/tests/test_forgetting_strategy_ranking.py`

Verifies that ConsolidationAware correctly ranks consolidated memories ABOVE unconsolidated ones for forgetting.

```python
def test_ca_forgetting_protects_unconsolidated_memory():
    """Unconsolidated memories should have forget score = 0."""
    strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
    
    old_memory = Memory(id="old", text="Old fact", importance=0.1,
                        timestamp=datetime(2023, 1, 1), last_accessed=datetime(2023, 1, 1))
    new_memory = Memory(id="new", text="New fact", importance=0.9,
                        timestamp=datetime.now(), last_accessed=datetime.now())
    
    # No consolidation tracker — all C(m) = 0
    scores = strategy.compute_forget_scores([old_memory, new_memory])
    
    # Both should have score 0 (protected by threshold)
    assert scores["old"] == 0.0, "Old unconsolidated memory should be protected"
    assert scores["new"] == 0.0, "New unconsolidated memory should be protected"

def test_ca_forgetting_scores_consolidated_memory():
    """After consolidation, a memory should get a real forget score."""
    strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
    tracker = ConsolidationTracker()
    
    old_memory = Memory(id="old", text="Old consolidated fact", importance=0.1,
                        timestamp=datetime(2023, 1, 1), last_accessed=datetime(2023, 1, 1))
    
    # Simulate consolidation
    tracker.record_consolidation("old", "l3_node_1", coverage=0.8)
    
    scores = strategy.compute_forget_scores([old_memory], consolidation_tracker=tracker)
    
    # Should have a real (non-zero) score now
    assert scores["old"] > 0.0, "Consolidated memory should have non-zero forget score"
```

---

### TEST-04 — Metric Consistency Test (HIGH)

**File to create:** `csam_project/tests/test_metrics.py`

```python
from csam_project.benchmarks.metrics import token_f1, exact_match, normalize_text, aggregate_f1

def test_token_f1_identical_strings():
    assert token_f1("the quick brown fox", "the quick brown fox") == 1.0

def test_token_f1_no_overlap():
    assert token_f1("apple orange banana", "car truck train") == 0.0

def test_token_f1_partial():
    # "fox jumps" vs "quick fox" → 1 common token "fox", precision=0.5, recall=0.5, F1=0.5
    assert abs(token_f1("quick fox", "fox jumps") - 0.5) < 0.01

def test_aggregate_f1_includes_zeros():
    scores = [0.5, 0.0, 0.5]
    result = aggregate_f1(scores, include_zeros=True)
    assert abs(result - (1.0/3.0)) < 0.01, "Zero-inclusive aggregate should be 1/3"

def test_normalize_strips_articles():
    assert normalize_text("The quick brown fox") == "quick brown fox"

def test_normalize_strips_punctuation():
    assert normalize_text("Hello, world!") == "hello world"
```

---

### TEST-05 — HNSW Index Rebuild Consistency (MEDIUM)

**File to add to:** `csam_project/tests/test_memory_repository.py`

```python
def test_rebuild_index_preserves_retrieval():
    """After adding and deleting memories, rebuild should restore retrieval."""
    repo = MemoryRepository(dim=384)
    emb = EmbeddingService()
    
    # Add 10 memories
    for i in range(10):
        text = f"Memory about topic {i}"
        embedding = emb.encode(text)
        repo.add(Memory(id=str(i), text=text, embedding=embedding, importance=0.5))
    
    # Delete 5
    for i in range(5):
        repo.delete(str(i))
    
    # Rebuild
    repo.rebuild_index()
    
    # Retrieval should work and return only live memories
    query = emb.encode("Memory about topic 7")
    results = repo.retrieve(query, k=3)
    result_ids = [m.id for m, _ in results]
    
    assert all(rid in ["5", "6", "7", "8", "9"] for rid in result_ids), \
        "Rebuilt index should not return deleted memories"
```

---

### TEST-06 — Metadata Filter Scope Test (MEDIUM)

**File to rewrite:** `csam_project/tests/test_metadata_filtering.py`

```python
def test_player_metadata_filter_isolates_memories():
    """Bob's filter should never return Alice's memories."""
    repo = MemoryRepository(dim=384)
    emb = EmbeddingService()
    
    # Add Bob's and Alice's memories
    bob_emb = emb.encode("Bob went to the market")
    alice_emb = emb.encode("Alice visited the library")
    
    repo.add(Memory(id="bob1", text="Bob went to the market",
                    embedding=bob_emb, metadata={"player_name": "Bob"}))
    repo.add(Memory(id="alice1", text="Alice visited the library",
                    embedding=alice_emb, metadata={"player_name": "Alice"}))
    
    # Query with Bob's filter
    results = repo.retrieve(bob_emb, k=5, metadata_filter={"player_name": "Bob"})
    result_ids = [m.id for m, _ in results]
    
    assert "alice1" not in result_ids, "Bob's filter must exclude Alice's memories"
    assert "bob1" in result_ids, "Bob's filter must return Bob's own memories"
```

---

## Section 4: Optional Tests to Strengthen the Paper

These are not required for submission but would make the paper significantly stronger and more defensible under review.

---

### OPT-01 — Scaling Test: Memory Pressure with Forgetting

**Purpose:** Demonstrate that bounded forgetting strategies maintain constant memory footprint while NoForgetting grows linearly.

```python
def test_memory_growth_under_bounded_forgetting():
    """Bounded strategies should plateau; no-forgetting grows linearly."""
    MAX_MEM = 50
    npc_ca = NPC(..., max_memories=MAX_MEM, strategy='consolidation_aware')
    npc_nf = NPC(..., max_memories=99999, strategy='none')  # unbounded
    
    for i in range(200):
        npc_ca.add_memory(f"Event {i}")
        npc_nf.add_memory(f"Event {i}")
    
    # CA should be bounded
    assert len(npc_ca.memory_repo) <= MAX_MEM * 1.1  # allow 10% slack
    
    # No-forgetting should grow
    assert len(npc_nf.memory_repo) > MAX_MEM * 2
```
**Paper value:** Validates the "memory growth dynamics" figure directly.

---

### OPT-02 — Forgetting Strategy Comparison Under Long Conversation

**Purpose:** Show that CA forgetting yields better recall than LRU after 500+ interaction turns.

```python
def test_ca_vs_lru_recall_after_500_turns():
    """CA should outperform LRU on recall of early-turn facts after long conversation."""
    # Setup 2 NPCs, one CA one LRU
    # Ingest 500 turns with 10 critical facts seeded in first 100 turns
    # After 500 turns, test recall of the 10 critical facts
    # Assert CA recall > LRU recall on critical facts
```
**Paper value:** This is the *qualitative* story that motivates CSAM. If this test passes, it is a compelling demo result.

---

### OPT-03 — Threshold Sensitivity Test

**Purpose:** Verify that F1 performance is stable in the θ=[0.2, 0.4] range and unstable at extremes.

```python
@pytest.mark.parametrize("threshold", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0])
def test_threshold_effect_on_protection_rate(threshold):
    """At threshold=0: nothing protected. At threshold=1.0: everything protected."""
    strategy = ConsolidationAwareForgetting(consolidation_threshold=threshold)
    # Use a set of memories with known coverage values [0.1, 0.2, ..., 1.0]
    # Assert protection rate matches expected value for each threshold
```
**Paper value:** Directly supports the threshold sensitivity figure.

---

### OPT-04 — Ablation Completeness Test

**Purpose:** Verify that all 4 components (R, I, C, D) contribute independently to the forgetting score.

```python
def test_all_four_components_contribute():
    """Removing any one component should change the ranking of some memory pair."""
    configs = [
        ConsolidationAwareForgetting(alpha=1, beta=0, gamma=0, delta=0),  # R only
        ConsolidationAwareForgetting(alpha=0, beta=1, gamma=0, delta=0),  # I only
        ConsolidationAwareForgetting(alpha=0, beta=0, gamma=1, delta=0),  # C only
        ConsolidationAwareForgetting(alpha=0, beta=0, gamma=0, delta=1),  # D only
    ]
    # Assert each config produces different rankings
```
**Paper value:** Shows each component of the formula is non-redundant.

---

### OPT-05 — Multi-Seed Variance Baseline

**Purpose:** Run each forgetting strategy 5 times with different seeds, report mean ± std.

**Implementation:** Already in `variance_runner.py` — just needs to be run and artifacts committed.

**Paper value:** Statistical stability claim. Without this, "CA outperforms LRU" could be a single-seed artifact.

---

### OPT-06 — Cross-Model Architecture-Dominance Smoke Test

**Purpose:** On a small subset (5 questions), verify that the pattern "8B with CSAM ≈ 70B with CSAM" holds under the current protocol.

```python
def test_architecture_dominates_model_size_on_locomo(locomo_sample):
    """8B model F1 should be within 0.10 of 70B model F1 on LoCoMo."""
    f1_8b = run_locomo_qa(model="llama-3.1-8b-instant", questions=locomo_sample)
    f1_70b = run_locomo_qa(model="llama-3.3-70b-versatile", questions=locomo_sample)
    assert abs(f1_8b - f1_70b) < 0.15, \
        f"Architecture should dominate: 8B ({f1_8b:.3f}) vs 70B ({f1_70b:.3f})"
```
**Paper value:** Makes the central thesis testable and falsifiable — which is scientifically stronger.

---

## Section 5: Diagram Improvements

The current diagrams in `csam_project/diagrams/` need the following additions and revisions.

---

### DIAG-01 — Architecture Diagram: Add Data Types and Size Labels

**Current:** Shows L1/L2/L3 boxes and arrows.
**Problem:** Doesn't show: LRU capacity (20 items), HNSW dim (384), max memories (200), consolidation threshold (0.3).
**Improvement:**
- Label L1: `LRU Cache (cap=20, O(1) access)`
- Label L2: `HNSW Index (384-dim, ≤200 entries)`
- Label L3: `NetworkX Graph (semantic nodes + relations)`
- Label forgetting gate: `C(m) ≥ θ=0.3`
- Add a "forgetting pressure" annotation showing when forgetting triggers

---

### DIAG-02 — Add Forgetting Decision Tree Diagram (NEW)

**Purpose:** Visually show the forgetting decision process:
```
Memory m in L2
    │
    ▼
C(m) < θ (0.3)?
    │YES → PROTECTED (score=0) → stays in L2
    │NO
    ▼
ForgetScore(m) = 0.25·R + 0.25·(1-I) + 0.25·C + 0.25·D
    │
    ▼
Is m in Top-N most forgettable?
    │YES → DELETE from L2
    │NO → stays in L2
```
**Paper value:** Readers immediately understand the gating mechanism without reading the math.

---

### DIAG-03 — Consolidation Flow Diagram: Show Memory-to-Node Mapping

**Current:** Shows arrows from L2 to L3.
**Improvement:** Show:
- A batch of 5-10 L2 memories → LLM consolidation call → 1-3 L3 nodes
- Arrow from L3 node back to coverage tracker
- Coverage tracker signal feeding into the forgetting gate
- Show coverage values: "C(m₁)=0.8, C(m₂)=0.6, C(m₃)=0.3"

---

### DIAG-04 — Add Timeline Diagram: Write Path vs Read Path (NEW)

**Purpose:** Show the temporal separation of operations:
```
Time ──────────────────────────────────────────────────────▶
│ Conversation turn 1-20 │ Consolidation │ Forgetting │ QA │
│ L1 fill + L2 write     │ L2→L3         │ L2 prune   │ L2+L3 retrieve │
```
This explains why CSAM can maintain response latency while still doing expensive consolidation offline.

---

### DIAG-05 — Memory Growth Chart: Add Y-Axis Cap Line

**Current:** Shows growth curves for bounded vs unbounded strategies.
**Improvement:** Add a horizontal dashed line at `max_memories=200` labeled "Configured memory cap." Add annotations showing when forgetting events fire (vertical tick marks). This makes the "plateau effect" of bounded strategies visually obvious.

---

### DIAG-06 — Ablation Bar Chart: Add Error Bars and Corrected Aggregate

**Current:** Bar chart without error bars.
**Improvement:**
- Add error bars from variance runner (mean ± std over 3-5 seeds)
- Use corrected (zero-inclusive) F1 values after BUG-01 fix
- Add a horizontal baseline for "random" performance
- Add p-value annotation between CA and best baseline (if significant)

---

### DIAG-07 — Add Memory Layer Activity Heatmap (NEW — Optional)

**Purpose:** Show which memory layer (L1/L2/L3) contributes to the final context across different question types.

Columns: question types (temporal, entity, event, multi-hop)
Rows: memory source (L1 working, L2 episodic, L3 semantic)
Cell value: % of correct answers where that layer contributed context

**Paper value:** Demonstrates that L3 is essential for semantic questions while L1 helps temporal ones — validates the architecture partitioning decision.

---

## Section 6: Demo TUI Improvements

### Direction: Textual-Based TUI

Replace `csam_project/simulation/demo_cli.py` with a full **Textual** TUI (`csam_project/simulation/tui_demo.py`).

**Framework:** `textual` (https://textual.textualize.io/) — Python-native TUI framework built on Rich. Add to `pyproject.toml` dependencies: `"textual>=0.47.0"`.

---

### TUI-01 — Layout Design

```
╔═══════════════════════════════════════════════════════════════╗
║  CSAM — Cognitive Sparse Access Memory Demo          v1.0     ║
╠══════════════════╦════════════════════════════════════════════╣
║  MEMORY LAYERS   ║  CONVERSATION                              ║
║  ┌─────────────┐ ║  ┌────────────────────────────────────┐   ║
║  │ L1 Working  │ ║  │ [14:23] You: Tell me about Alice   │   ║
║  │ 12/20 items │ ║  │ [14:23] Aric: Alice visited the    │   ║
║  │ ▓▓▓▓▓▓░░░░ │ ║  │  library last Tuesday...            │   ║
║  └─────────────┘ ║  │                                    │   ║
║  ┌─────────────┐ ║  │ [14:24] You: What did she study?   │   ║
║  │ L2 Episodic │ ║  │ [14:24] Aric: She was reading...   │   ║
║  │ 143/200     │ ║  │                                    │   ║
║  │ ▓▓▓▓▓▓▓░░░ │ ║  └────────────────────────────────────┘   ║
║  └─────────────┘ ║                                            ║
║  ┌─────────────┐ ║  ACTIVE NPC: Aric the Blacksmith           ║
║  │ L3 Graph    │ ║  ┌──────────────────────────────────────┐ ║
║  │ 28 nodes    │ ║  │ > _                                  │ ║
║  │ 41 edges    │ ║  └──────────────────────────────────────┘ ║
║  └─────────────┘ ║                                            ║
╠══════════════════╬════════════════════════════════════════════╣
║  LAST RETRIEVAL  ║  TABS: [Chat] [Memory] [Graph] [Stats]     ║
║  L2: 12 results  ║  F1: Skip N turns  F2: Force consolidate   ║
║  L3: 3 nodes     ║  F3: Switch NPC    F4: Memory inspector     ║
║  Top-k: 5        ║  F5: Run ablation  F9: Save transcript      ║
╚══════════════════╩════════════════════════════════════════════╝
```

---

### TUI-02 — Memory Layer Sidebar (Live Updates)

**Widget:** Custom Textual `Widget` that polls NPC stats every 500ms.

Shows for each layer:
- L1: Items count, fill bar (e.g., `▓▓▓▓▓▓░░░░ 12/20`)
- L2: Memory count, fill bar, stale HNSW ratio
- L3: Node count, edge count, avg coverage of L2 memories
- Color coding: L1=cyan, L2=green, L3=yellow
- Alerts: Red when L2 > 180/200 (forgetting pressure)

---

### TUI-03 — Memory Inspector Tab

A dedicated tab showing the current memory contents, browsable and filterable:

```
[Memory Inspector]
Filter: [ alice           ] [Type: ALL ▼] [Source: L2 ▼]

ID          Score  Layer  Text (truncated)                  C(m)  Age
──────────────────────────────────────────────────────────────────────
mem_0042    0.87   L2     [2023-05-08] Alice visited the l  0.73  2d
mem_0041    0.82   L2     [2023-05-07] Player mentioned Al  0.61  3d
node_012    0.78   L3     Entity: Alice (person, frequente  N/A   1d
mem_0038    0.71   L2     [2023-05-05] Alice and Bob went   0.45  5d
```

**Interactive:** Arrow keys to browse, Enter to expand full text, D to delete, C to force-consolidate.

---

### TUI-04 — Graph Visualization Tab (ASCII)

A simple ASCII representation of the L3 knowledge graph:

```
[Knowledge Graph — L3]
                    ┌──────────┐
                    │  Alice   │
                    │ (person) │
                    └────┬─────┘
              ┌──────────┤──────────┐
              ▼          ▼          ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Library  │ │  Bob     │ │ Alchemy  │
       │(location)│ │ (person) │ │ (topic)  │
       └──────────┘ └──────────┘ └──────────┘

Nodes: 28  Edges: 41  Coverage: 67% of L2 memories
```

**Note:** For large graphs, show only the neighborhood of the most recently active node.

---

### TUI-05 — Stats / Ablation Tab

Live performance stats panel:

```
[Session Statistics]
Duration: 00:14:32          Conversations: 47
Tokens used: 12,847         Avg latency: 823ms

Memory Events:
  Ingested:      47
  Consolidated:  12 (3 L3 nodes created)
  Forgotten:      8 (2 protected by gate)
  L2→L3 ratio:  25%

Retrieval Quality (this session):
  Avg L2 results: 14.2/20
  Avg L3 hits:     2.1/5
  Avg context chars: 1,847

Forgetting Strategy: Consolidation-Aware
  θ threshold:     0.3
  α (recency):    0.25
  β (importance): 0.25
  γ (coverage):   0.25
  δ (redundancy): 0.25
```

---

### TUI-06 — Keybindings

| Key | Action |
|-----|--------|
| `F1` | Skip N turns (prompts for N) |
| `F2` | Force consolidation cycle |
| `F3` | Switch active NPC (cycle through available) |
| `F4` | Toggle Memory Inspector tab |
| `F5` | Run quick ablation (3 questions, current NPC) |
| `F9` | Save session transcript to file |
| `Ctrl+R` | Force HNSW index rebuild |
| `Ctrl+C` | Quit |
| `Tab` | Cycle through tabs |
| `/` | Focus filter bar in Memory Inspector |

---

### TUI-07 — Seed and Deterministic Mode

Add CLI args for the TUI:
```bash
python -m csam_project.simulation.tui_demo \
    --npc aric \
    --seed 42 \
    --replay demo_script_publication.json \
    --record output_transcript.json
```

`--replay` feeds a scripted interaction sequence (questions + expected answers) for demo/presentation mode. `--record` saves the full session for paper appendix use.

---

### TUI-08 — Implementation Stack

**Dependencies to add to `pyproject.toml`:**
```toml
"textual>=0.47.0",
"rich>=13.0.0",  # usually a textual dependency
```

**File structure:**
```
csam_project/simulation/
  tui_demo.py                    — Main Textual app entry point
  tui_widgets/
    __init__.py
    memory_sidebar.py            — L1/L2/L3 live stat panels
    conversation_panel.py        — Chat history with formatting
    memory_inspector.py          — Browsable memory table
    graph_view.py                — ASCII knowledge graph
    stats_panel.py               — Session statistics
  demo_scripts/
    demo_script_publication.json — Scripted demo for paper appendix
```

---

## Section 7: Implementation Priority Matrix

Combining all findings into a single prioritized list:

| Priority | Item | Category | Day to Fix | Impact |
|----------|------|----------|-----------|--------|
| P0 | BUG-01: Ablation F1 excludes zeros | Bug | Day 4 | Results invalid |
| P0 | BUG-05: No consolidation-recall test | Test | Day 4 | Core claim unverified |
| P0 | BUG-06: E2E ablation skips consolidation | Bug | Day 4 | Core claim possibly invalid |
| P0 | IMP-01: Centralize metrics | Code | Day 4 | Cross-script inconsistency |
| P1 | BUG-03: Async path broken | Bug | Day 5 | API confusion |
| P1 | BUG-02: Cerebras typo | Bug | Day 1 | Provider broken |
| P1 | BUG-04: Rate limit loop | Bug | Before variance runs | Hangs |
| P1 | TEST-01-06: All publication tests | Test | Day 4-5 | Paper credibility |
| P1 | IMP-02: Seed controls | Code | Day 1-2 | Reproducibility |
| P1 | IMP-03: Protocol fingerprint | Code | Day 2 | Reproducibility |
| P1 | IMP-06: Bootstrap CI | Code | Day 2 | Statistical validity |
| P2 | BUG-07: Soft-delete mappings | Bug | Day 5 | Long-running stability |
| P2 | BUG-08: Metadata filter silent loss | Bug | Day 5 | Multiplayer accuracy |
| P2 | DIAG-01-07: Diagram improvements | Diagrams | Day 9 | Paper presentation |
| P2 | OPT-01-06: Optional tests | Test | Day 6-7 | Paper strength |
| P3 | TUI-01-08: Textual TUI | Demo | Post-submission | Demo quality |
| P3 | BUG-10: Typo in working_memory | Bug | Day 5 | Presentation polish |
| P3 | BUG-09: MMR misleading signature | Bug | Day 5 | Code clarity |

---

## Section 8: Updated Day 4 Tasks (Critical Additions)

The original sprint plan's Day 4 must now include:

**Original Day 4 tasks:**
- Create `metrics.py` (IMP-01)
- Fix `run_ablation.py:456` (BUG-01)
- Create `test_metrics.py` (TEST-04)
- Start variance runs

**Additional Day 4 tasks:**
- Fix BUG-02 (Cerebras typo in `llm_hosted.py:40`) — 5 minutes
- Fix BUG-04 (rate limit loop) — 30 minutes
- Verify BUG-06 (check if E2E ablation triggers consolidation) — 1 hour
- Create `test_consolidation_recall_invariant.py` (TEST-01) — 2 hours
- Create `test_forgetting_strategy_ranking.py` (TEST-03) — 1 hour

**Day 5 additions:**
- Fix BUG-03 (remove broken async `retrieve()`)
- Fix BUG-07 (soft-delete mapping cleanup)
- Create `test_l1_contamination.py` (TEST-02)
- Create `test_metadata_filtering.py` (TEST-06 rewrite)

---

## Section 9: TUI Development Timeline (Post-Submission)

If submission is made by April 25, the TUI can be the main post-submission engineering effort:

| Week | TUI Work |
|------|---------|
| Week 1 post-submit | Setup Textual app skeleton + layout (TUI-01, TUI-02) |
| Week 2 | Memory Inspector tab + Graph tab (TUI-03, TUI-04) |
| Week 3 | Stats tab + Keybindings (TUI-05, TUI-06) |
| Week 4 | Seed/replay/record mode (TUI-07) + Polish |

**Why Textual:** Zero browser dependency, runs in any terminal, looks professional for conference demos, easily recorded as terminal cast (using `asciinema`). The live L1/L2/L3 fill bars updating in real-time during conversation is a compelling visual demonstration of the architecture working.
