# CSAM Metrics Reference — Timing & Memory Optimization

Your research is based on **memory efficiency** and **speed optimization** within CSAM's 3-tier architecture. Here's exactly what we're measuring across all 6 notebooks to validate those claims.

---

## The Three Core Claims We're Testing

| Claim | How We Measure | Where | Expected Result |
|---|---|---|---|
| **Speed**: CSAM is fast enough for real-time QA | Latency per question (`latency_ms`) | NB-1, NB-2, NB-4, NB-6 | <500ms per Q |
| **Memory**: CSAM uses less memory than unbounded baselines | Memory footprint, eviction strategy | NB-4 (ablation) | CA-Ours uses similar memory as LRU but with better recall |
| **Quality**: CSAM maintains accuracy despite memory constraints | F1 + semantic similarity + exact match | All notebooks | CSAM F1 ≥ Baseline F1 at same memory |

---

## Timing Metrics — Detailed Breakdown

### 1. Per-Question Latency (`latency_ms`)

**What it measures:** Time from user query to answer returned (end-to-end).

**Includes:**
- Embedding the user query (all-MiniLM-L6-v2)
- HNSW retrieval (L2 vector search)
- LLM API call + response parsing
- (Optional) Knowledge graph traversal (L3)

**Where collected:**
- NB-1: `results_locomo_csam_*.json` → `per_conversation[i].per_question[j].latency_ms`
- NB-2: `results_hotpotqa_*.json` → `per_question[j].latency_ms`
- NB-4: `ablation_*.json` → `avg_latency_ms` per strategy
- NB-6: `results_*.json` → `avg_latency_ms` per Q count

**Example JSON:**
```json
{
  "per_question": [
    {
      "question": "What is the capital of France?",
      "latency_ms": 287.4,
      "f1": 0.95,
      "semantic_sim": 0.88
    },
    {
      "question": "Who was the first president of the USA?",
      "latency_ms": 312.5,
      "f1": 1.0,
      "semantic_sim": 0.92
    }
  ],
  "avg_latency_ms": 299.95
}
```

### 2. Average Latency (`avg_latency_ms`)

**What it measures:** Mean latency across all questions in a run.

**Interpretation:**
- < 200ms: Very fast (small models, simple retrieval)
- 200–400ms: Normal (typical LLM API roundtrip)
- 400–800ms: Slower (big models, multi-hop retrieval)
- > 1000ms: Bottleneck (network issue or LLM timeout)

**Where collected:** Top-level key in every benchmark output JSON.

**Example:**
```json
{
  "model": "llama-3.1-8b-instant",
  "avg_latency_ms": 312.4,
  "num_questions": 100
}
```

### 3. Throughput (Derived Metric)

**Calculation:**
```
Throughput = num_questions / sum(latencies_ms) * 1000
           = questions per second
           = 100 / (312.4 * 100) * 1000 = 3.2 questions/sec
```

**Example interpretation:**
- 2 Q/sec: Fine for batch processing, questionable for real-time
- 5 Q/sec: Good for real-time scenarios (200ms per Q)
- 10+ Q/sec: Excellent (< 100ms per Q)

---

## Memory Metrics — Detailed Breakdown

### 1. Memory Count (`memory_count`)

**What it measures:** Total number of memory entries stored in L2 HNSW at end of benchmark.

**Why it matters:** Tests the **memory cap** enforcement. Should never exceed 200 (hard limit) or should be consistent across strategies.

**Where collected:**
- NB-4 (ablation): `ablation_*.json` → `results[strategy].memory_count`
- NB-5 (scaling): `scaling_*.json` → `memory_count` at various NPC scales

**Example JSON:**
```json
{
  "results": [
    {
      "strategy": "No-Forgetting",
      "memory_count": 450,  ← UNBOUNDED (memory leaks)
      "overall_f1": 0.75
    },
    {
      "strategy": "LRU",
      "memory_count": 200,  ← CAPPED (evicts oldest)
      "overall_f1": 0.72
    },
    {
      "strategy": "Consolidation-Aware (Ours)",
      "memory_count": 200,  ← CAPPED (evicts unconsolidated)
      "overall_f1": 0.74   ← Better F1 than LRU at same memory!
    }
  ]
}
```

### 2. Estimated Memory Footprint (`estimated_memory_mb`)

**What it measures:** Total RAM used by L2 HNSW index + metadata + L3 knowledge graph.

**Calculation (rough):**
```
Memory (MB) = num_memories * (384 floats * 4 bytes + 200 bytes metadata) / 1e6
            = 200 * (384*4 + 200) / 1e6
            ≈ 0.33 MB per 200 memories
```

**Where collected:**
- NB-4 (ablation): `ablation_*.json` → `results[strategy].memory_bytes_mb`
- NB-5 (scaling): `scaling_*.json` → `estimated_memory_mb` at various NPC counts

**Example JSON:**
```json
{
  "strategy": "Consolidation-Aware (Ours)",
  "memory_count": 200,
  "memory_bytes_mb": 0.33,     ← Fits in memory
  "memory_bytes": 331200,       ← Raw bytes
  "avg_latency_ms": 42.5        ← Retrieval latency (should scale O(log n))
}
```

### 3. Memory Efficiency Ratio (Derived Metric)

**Calculation:**
```
Memory_Efficiency = F1 / (memory_count * estimated_memory_mb)
                  = Quality per unit memory
```

**Higher is better.** Example:
```
Strategy A: F1=0.75, memory=200, mem_mb=0.33 → Efficiency = 0.75 / (200*0.33) = 0.0114
Strategy B: F1=0.72, memory=200, mem_mb=0.33 → Efficiency = 0.72 / (200*0.33) = 0.0109

Strategy A is 4.6% more efficient (same memory, better F1).
```

---

## Quality Metrics (Kept for Completeness)

### 1. Token-Level F1 (`avg_f1`)

**What it measures:** Bag-of-words token overlap between predicted and ground-truth answers (SQuAD standard).

**Range:** 0.0 (completely wrong) to 1.0 (perfect match)

**Why:** Comparable to other QA systems (HotPotQA benchmarks report F1).

**Example:**
```
Prediction: "The capital of France is Paris, France."
Ground truth: "Paris"

Tokens: {capital, france, is, paris}
Overlap: {paris} = 1 token
Precision = 1/4 = 0.25
Recall = 1/1 = 1.0
F1 = 2 * (0.25 * 1.0) / (0.25 + 1.0) = 0.4
```

### 2. Semantic Similarity (`avg_semantic_sim`)

**What it measures:** Cosine distance between predicted and ground-truth embeddings (using all-MiniLM-L6-v2).

**Range:** 0.0 (opposite meaning) to 1.0 (identical)

**Why:** Captures paraphrases and synonyms that token F1 misses.

**Example:**
```
Prediction: "The capital of France is Paris"
Ground truth: "Paris"

Embedding similarity = cos(embed(prediction), embed(ground_truth))
                    = 0.85 (high similarity despite different wording)

Token F1 = 0.4 (token overlap is low)
Semantic F1 = 0.85 (meaning is clear)
```

### 3. Exact Match (`avg_em`)

**What it measures:** Fraction of predictions that match ground truth exactly (0 or 1 per question).

**Range:** 0.0 to 1.0

**Typical values:** 0.3–0.7 for multi-hop (harder), 0.6–0.9 for single-hop (easier)

---

## Specialization: What Each Notebook Optimizes For

### NB-1 (LoCoMo): Speed Under Load

**Primary metrics:**
- `avg_latency_ms` — can CSAM handle long conversations fast?
- `micro_f1` — does memory pressure degrade quality?

**Expected finding:**
- CSAM ~300–400ms per question (similar to baseline)
- But CSAM F1 slightly better (memory consolidation helps multi-turn)

### NB-2 (HotPotQA): Speed Across Model Sizes

**Primary metrics:**
- `avg_latency_ms` per model
- `avg_f1` per model
- Ratio: `F1 / latency` (quality per ms)

**Expected finding:**
```
8B:   latency ~300ms, F1=0.60 → 0.20 quality/ms
70B:  latency ~800ms, F1=0.72 → 0.09 quality/ms

Bigger model = slower but higher quality.
At what point is speed trade-off worth it?
```

### NB-3 (MuSiQue): Memory Efficiency via L3

**Primary metrics:**
- `avg_f1` WITH L3 vs WITHOUT L3 (delta = L3 contribution)
- `avg_latency_ms` WITH L3 vs WITHOUT L3 (L3 overhead?)

**Expected finding:**
```
Without L3: F1=0.42, latency=250ms
With L3:    F1=0.54, latency=280ms (only 30ms slower!)

→ L3 knowledge graph gives +12 F1 points for only +30ms latency!
   This is the **memory-optimization trade-off we designed for**.
```

### NB-4 (Ablation): Memory vs Accuracy Trade-Off

**Primary metrics:**
- `false_forgetting_rate` (FFR) — memory eviction strategy quality
- `memory_count` — how many memories stored?
- `avg_latency_ms` — retrieval latency at this memory count
- `overall_f1` — accuracy

**Expected finding:**
```
Strategy                     Mem   FFR   Latency   F1
─────────────────────────────────────────────────────
No-Forgetting (unbounded)   450   0%    580ms    0.78 (memory leak!)
LRU                         200   8%    320ms    0.72 (evicts good stuff)
CA-Formula-Only             200  15%    310ms    0.71 (gate disabled = bad)
CA-Ours (Full)              200   0%    315ms    0.74 ← SWEET SPOT
─────────────────────────────────────────────────────

CA-Ours:
✓ Bounds memory (200 entries)
✓ Zero false forgetting (never evicts unconsolidated)
✓ Only 3ms slower than LRU (negligible overhead)
✓ Better F1 than LRU (consolidation helps)
```

### NB-5 (Scaling): Memory System Throughput

**Primary metrics:**
- `avg_latency_ms` vs `num_npcs` (HNSW should scale O(log n))
- `estimated_memory_mb` vs `num_npcs` (linear growth expected)

**Expected finding:**
```
NPCs   Memories   Latency   Memory
─────────────────────────
10     100        8.2ms     0.033 MB
25     250        9.5ms     0.083 MB
50     500        11.1ms    0.166 MB ← log(50) ≈ 5.64
100    1000       12.7ms    0.331 MB ← log(100) ≈ 6.64

Latency grows ~2.5ms per 10× memory (log-linear).
Ideal for scaling to millions of memories!
```

### NB-6 (Question Scaling): Robustness Across Dataset Sizes

**Primary metrics:**
- `avg_f1` at 50Q, 100Q, 200Q, 500Q (stability test)
- `avg_latency_ms` (should NOT increase with Q count)

**Expected finding:**
```
Q Count   F1     Latency   Semantic
───────────────────────────────────
50        0.58   305ms     0.62
100       0.60   308ms     0.63
200       0.59   310ms     0.63
500       0.60   312ms     0.63

F1 is stable ±2 points (not cherry-picked from easy questions).
Latency per-Q is constant (Q count doesn't slow retrieval).
```

---

## Summary Table: What You're Measuring Across All 6 Notebooks

| Metric | NB-1 | NB-2 | NB-3 | NB-4 | NB-5 | NB-6 | Unit | What It Proves |
|---|---|---|---|---|---|---|---|---|
| **latency_ms** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ms | Speed is acceptable |
| **memory_count** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | entries | Memory cap enforced |
| **estimated_memory_mb** | - | - | - | ✅ | ✅ | - | MB | Footprint bounded |
| **false_forgetting_rate** | - | - | - | ✅ | - | - | % | Consolidation works |
| **avg_f1** | ✅ | ✅ | ✅ | ✅ | - | ✅ | ratio | Accuracy maintained |
| **avg_semantic_sim** | ✅ | ✅ | ✅ | ✅ | - | ✅ | ratio | Meaning preserved |
| **L3_contribution** | - | - | ✅ | - | - | - | F1 Δ | KG helps multi-hop |
| **throughput** | Derived | Derived | Derived | Derived | Derived | Derived | Q/sec | Scalable? |

---

## How to Validate Timing/Memory Results

After all notebooks complete, you'll have ~20 JSON files. To verify timing and memory optimization:

### 1. Latency Verification

```bash
# For each JSON file, extract avg_latency_ms:
grep "avg_latency_ms" results/**/*.json

# Expected ranges:
# 8B model:  200–400ms per question
# Scout 17B: 150–350ms per question
# 70B model: 400–800ms per question
# 120B model: 500–1000ms per question

# If ANY model is >2000ms, something is wrong (network/API timeout).
```

### 2. Memory Verification (NB-4)

```bash
# Extract memory_count from ablation results:
grep "memory_count" results/nb4_ablation/*.json

# Expected:
# No-Forgetting:        > 200 (unbounded, memory leak)
# LRU / Importance / CA-Formula:  = 200 (capped)
# CA-Ours:              = 200 (capped)
# All FFR > 0 except CA-Ours:     = 0
```

### 3. Quality vs Memory Trade-Off (NB-4)

```json
// Compare these two entries:

// LRU (baseline memory strategy):
{
  "strategy": "LRU",
  "memory_count": 200,
  "avg_latency_ms": 320,
  "overall_f1": 0.72,
  "false_forgetting_rate": 0.08
}

// CA-Ours (our strategy):
{
  "strategy": "Consolidation-Aware (Ours)",
  "memory_count": 200,
  "avg_latency_ms": 315,    ← Only 5ms slower!
  "overall_f1": 0.74,       ← Better F1!
  "false_forgetting_rate": 0.0  ← Perfect (never evicts unconsolidated)
}

// Conclusion:
// CA-Ours uses THE SAME memory footprint as LRU
// but achieves BETTER accuracy WITHOUT evicting "premature" memories.
// This proves the consolidation-aware design works!
```

---

## What to Report to the Team

After all 6 notebooks complete, summarize:

```markdown
# Timing & Memory Results Summary

## Speed Metrics
- 8B model average latency: XXX ms ✓
- Bigger models (70B, 120B) latency: XXX–XXX ms ✓
- Throughput at 100Q: X.X questions/second ✓

## Memory Optimization
- Memory cap enforced at 200 entries ✓
- CSAM (CA-Ours) FFR = 0% (never evicts unconsolidated) ✓
- Baseline (LRU) FFR = X% (evicts some good stuff) ✓
- CSAM F1 = 0.XX vs LRU F1 = 0.XX (+X% improvement) ✓

## L3 Knowledge Graph Contribution (NB-3)
- Without L3: F1 = 0.XX, latency = XXX ms
- With L3:    F1 = 0.XX (+X points!), latency = XXX ms (+Z ms overhead)
- Conclusion: KG provides strong multi-hop signal with minimal latency cost ✓

## Memory Scaling (NB-5)
- HNSW retrieval scales O(log n) ✓
- At 1000 memories: latency = XX ms (acceptable) ✓
- Memory footprint at 100 NPCs: X.X MB (fits in mobile) ✓
```

---

**You're measuring EXACTLY what your research claims: memory efficiency + speed optimization. All the metrics are already being collected. Just run the notebooks and validate the results match expectations.**
