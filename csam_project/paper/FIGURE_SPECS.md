# CSAM Paper — Figure Specifications
*For designer / chart creator. Each figure has an exact data spec and layout description.*

---

## Figure 1 — CSAM Three-Tier Architecture Diagram
**File:** `diagrams/csam_architecture.pdf`  
**Type:** System architecture diagram (vector, no data)  
**Size:** Full column width (NeurIPS: ~\linewidth)

### What to draw
Three boxes left-to-right connected by arrows:

**Box 1 — L1: Working Memory**
- Label: "L1 — Working Memory"
- Sub-label: "LRU Cache · 20 items · <1 ms"
- Color: light blue

**Box 2 — L2: Episodic Memory**
- Label: "L2 — Episodic Memory"
- Sub-label: "HNSW Index · 384-dim · O(log N)"
- Color: light orange
- Show: cylinder database icon

**Box 3 — L3: Semantic Memory**
- Label: "L3 — Knowledge Graph"
- Sub-label: "NetworkX Graph · Entities + Relations · Unbounded"
- Color: light green
- Show: small graph with 3-4 nodes and edges

**Arrows (query flow, top):**
- User Query → L1 (labeled "context lookup")
- L1 → L2 (labeled "k=20 HNSW search")
- L2 → L3 (labeled "graph traversal")
- L3 → box labeled "LLM Context Assembly" (on right)

**Arrows (consolidation loop, bottom):**
- L2 → L3 with curved arrow (labeled "Consolidation Pipeline")
- Small box below L2-L3 bridge: "Consolidation Tracker"
  - Arrow from tracker to L2: "C(m) ← coverage"

**Eviction arrow:**
- From L2, downward: "Forgetting Engine" → trash icon
- Small label: "CA-Ours: FFR = 0"

---

## Figure 2 — Ablation Study: F1 and FFR per Strategy
**File:** `diagrams/ablation_results.pdf`  
**Type:** Two-panel bar chart  
**Size:** Full column width

### Panel (a) — Mean F1
**X-axis:** Strategy (5 bars):
1. No-Forgetting
2. LRU
3. Importance
4. CA-Formula-Only
5. CA-Ours

**Y-axis:** Mean F1 (0.40 to 0.55)

**Values:**
| Strategy | Mean F1 | Std |
|---|---|---|
| No-Forgetting | 0.4847 | ~0.042 |
| LRU | 0.4854 | ~0.056 |
| Importance | 0.4582 | ~0.082 |
| CA-Formula-Only | 0.5016 | ~0.030 |
| CA-Ours | **0.5095** | ~0.034 |

**Error bars:** ±1 std  
**Highlight:** CA-Ours bar in red/bold. Add value label "0.5095" above.  
**Note:** "84.8% memory reduction vs. No-Forgetting"

### Panel (b) — False Forgetting Rate (FFR)
**X-axis:** Same 5 strategies  
**Y-axis:** FFR (0.0 to 0.25)

**Values:**
| Strategy | FFR |
|---|---|
| No-Forgetting | 0.000 |
| LRU | 0.128 |
| Importance | 0.197 |
| CA-Formula-Only | 0.068 |
| CA-Ours | **0.000** |

**Add:** Horizontal dashed red line at y=0 labeled "Target: FFR = 0"  
**Highlight:** CA-Ours bar in green. No-Forgetting also 0 but unbounded.  
**Add text:** "Only CA-Ours achieves FFR=0 with bounded memory"

---

## Figure 3 — HotPotQA Results
**File:** `diagrams/hotpotqa_results.pdf`  
**Type:** Three-panel figure  
**Size:** Full column width

### Panel (a) — F1 by Model
**X-axis:** Models (4 bars): 8B, Scout-17B, 70B, GPT-OSS-120B  
**Y-axis:** F1 (0.55 to 0.80)

**Values:**
| Model | F1 | Std |
|---|---|---|
| 8B | 0.672 | ±0.016 |
| Scout-17B | 0.698 | n/a |
| 70B | **0.742** | n/a |
| GPT-OSS-120B | 0.641 | n/a |

**Error bar on 8B only** (3 seeds: 0.691, 0.664, 0.662)  
**Highlight:** 70B bar as best.

### Panel (b) — Bridge vs. Comparison F1
**X-axis:** Models (4 groups of 2 bars each)  
**Y-axis:** F1 (0.45 to 0.85)  
**Two bars per model:** Bridge (dark) and Comparison (light)

| Model | Bridge F1 | Comparison F1 |
|---|---|---|
| 8B | 0.702 | 0.650 |
| Scout-17B | 0.749 | 0.509 |
| 70B | 0.777 | 0.609 |
| GPT-OSS-120B | 0.612 | **0.750** |

**Note:** GPT-OSS strongest on comparison but weakest on bridge.

### Panel (c) — Latency vs. F1 (Efficiency Frontier)
**X-axis:** Avg latency (ms): 1994, 2399, 3735, 5403  
**Y-axis:** F1: 0.641, 0.691, 0.698, 0.742  
**Points:** One per model, labeled  
**Add:** Pareto frontier curve. Circle Scout-17B as Pareto-optimal.

---

## Figure 4 — LoCoMo: CSAM vs. Flat-RAG Baseline
**File:** `diagrams/locomo_results.pdf`  
**Type:** Grouped bar chart  
**Size:** Single column

**X-axis:** Models (4 groups): 8B, Scout-17B, 70B, GPT-OSS-120B  
**Y-axis:** Macro F1 (0.15 to 0.40)

**Two bars per model:** CSAM (solid blue) and Flat-RAG Baseline (hatched gray)

| Model | CSAM | Baseline |
|---|---|---|
| 8B | 0.3419 | 0.3413 |
| Scout-17B | 0.3384 | 0.3378 |
| 70B | 0.3528 | 0.3521 |
| GPT-OSS-120B | 0.2269 | 0.2262 |

**Error bars:** 95% CI from Micro F1 data  
**Add note/annotation:** "Δ ≈ +0.0006 (not significant)" with arrow pointing to 8B pair  
**Add note:** "GPT-OSS-120B: likely instruction format mismatch" with arrow  

---

## Figure 5 — Multi-Agent Scaling
**File:** `diagrams/scaling_results.pdf`  
**Type:** Two-panel line chart  
**Size:** Full column width

### Panel (a) — Avg Query Latency vs. NPC Count
**X-axis:** NPC count (log scale: 1, 5, 10, 25, 50, 100)  
**Y-axis:** Avg latency (ms) (range: 11.5 to 14.0)

| NPCs | Avg Lat (ms) |
|---|---|
| 1 | 12.5 |
| 5 | 13.0 |
| 10 | 12.9 |
| 25 | 12.9 |
| 50 | 12.8 |
| 100 | 12.4 |

**Add:** Shaded band ±0.3 ms representing std across 4 runs  
**Add:** Horizontal dashed line at y=12.8 (reference "baseline latency")  
**Add:** Label "O(log N) — flat scaling" in plot  

### Panel (b) — Memory Footprint vs. NPC Count
**X-axis:** NPC count (linear: 1, 5, 10, 25, 50, 100)  
**Y-axis:** Memory (MB) (0 to 18)

| NPCs | Memory (MB) |
|---|---|
| 1 | 0.17 |
| 5 | 0.83 |
| 10 | 1.66 |
| 25 | 4.14 |
| 50 | 8.28 |
| 100 | 16.56 |

**Note:** Perfectly linear — 0.165 MB per 100 memories  
**Add:** Linear fit line r² = 1.00  
**Add label:** "Recall = 100% at all scales" as annotation inside plot

---

## Figure 6 — Memory Compression Diagram (Optional / for presentation)
**File:** `diagrams/memory_compression.pdf`  
**Type:** Visual comparison  
**Size:** Single column

Left side: "No-Forgetting" stack of 500 memory blocks  
Right side: "CA-Ours" stack of 76 memory blocks + KG icon  
Arrow between: "84.8% reduction"  
Below: "CA-Ours F1 = 0.5095 > No-Forgetting F1 = 0.4847"  
Bottom note: "868 KB → 132 KB"

---

## Chart Style Guidelines (All Figures)

- **Font:** Computer Modern or Times New Roman (matches LaTeX body)
- **Font size:** 9–10pt axis labels, 8pt tick labels
- **Colors:** Use colorblind-safe palette (matplotlib tab10 or seaborn colorblind)
  - CA-Ours: `#d62728` (red — highlight)
  - LRU: `#1f77b4` (blue)
  - Importance: `#ff7f0e` (orange)
  - CA-Formula-Only: `#2ca02c` (green)
  - No-Forgetting: `#7f7f7f` (gray)
  - CSAM: `#1f77b4` (blue)
  - Baseline: `#aec7e8` (light blue / hatched)
- **Export:** PDF (vector) for LaTeX inclusion
- **DPI:** 300+ if PNG fallback needed
- **No background grid lines** (only horizontal reference lines where noted)
