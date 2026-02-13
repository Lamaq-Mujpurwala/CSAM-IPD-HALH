# CSAM - Cognitive Sparse Access Memory

A 3-tier hierarchical memory architecture for AI agents with **consolidation-aware forgetting**.

**Hardware:** Intel i7-13HX, 16GB RAM, NVIDIA RTX 4060 8GB  
**Models:** Llama 3.2 3B (local), Llama 3.1 8B / Llama 4 Scout 17B / Llama 3.3 70B (Groq API)  
**Embeddings:** all-MiniLM-L6-v2 (384-dim)

---

## Research Contribution

Our novel contribution is **consolidation-aware forgetting**:

```
ForgetScore(m) = 0.2*R(m) + 0.2*(1-I(m)) + 0.3*C(m) + 0.3*D(m)
```

Where:
- `R(m)` = Recency decay (how old is the memory)
- `I(m)` = Importance score
- `C(m)` = **Consolidation coverage** (how much of m is backed by L3 knowledge graph)
- `D(m)` = **L3 redundancy** (semantic similarity to any L3 node)

**Key insight:** If a memory's content has been absorbed into the knowledge graph,
the original memory is redundant and can be safely forgotten -- bounded memory with no accuracy loss.

---

## Architecture

| Layer | Role | Implementation | Latency |
|-------|------|---------------|---------|
| **L1** | Working Memory | LRU Cache (last 20 items) | <1ms |
| **L2** | Long-Term Memory | HNSW Vector Index (384-dim) | ~5ms |
| **L3** | Knowledge Graph | NetworkX entity-relationship graph | ~2ms |

**Memory is capped at 200 entries.** Forgetting engine selects which memories to evict using the 4-factor formula above. Consolidation pipeline (L2 -> L3) extracts entity-relationship triples from accumulated memories.

---

## Project Structure

```
csam_project/
  csam_core/                        # Core memory architecture
    memory_repository.py            # L2 HNSW vector store
    working_memory.py               # L1 LRU cache
    knowledge_graph.py              # L3 NetworkX graph
    forgetting_engine.py            # Novel forgetting strategies
    consolidation.py                # L2 -> L3 consolidation pipeline
    consolidation_tracker.py        # Tracks L2-L3 mappings
    retrieval.py                    # Hybrid L2+L3 retriever with MMR
    mmr.py                          # Maximal Marginal Relevance
    services/
      embedding.py                  # all-MiniLM-L6-v2 wrapper
      llm.py                        # Ollama (local) LLM service
      llm_hosted.py                 # Groq API (hosted) LLM service

  simulation/                       # Interactive demo
    npc.py                          # NPC class with full CSAM memory
    demo_cli.py                     # CLI demo (talk/skip/remember/recall)
    multi_agent_orchestrator.py     # Multi-NPC orchestration

  benchmarks/                       # All benchmark scripts
    benchmark_e2e.py                # End-to-end recall (custom)
    benchmark_locomo.py             # LoCoMo conversational QA
    benchmark_multimodel.py         # LoCoMo multi-model (8B/17B/70B/120B)
    benchmark_musique.py            # MuSiQue multi-hop QA
    benchmark_hotpotqa.py           # HotPotQA multi-hop QA
    benchmark_baseline_rag.py       # Baseline RAG comparison
    benchmark_scaling.py            # Multi-NPC scalability
    generate_charts.py              # Chart generation (18 charts)
    generate_diagrams.py            # Architecture diagrams

  evaluation/                       # Ablation studies
    npc_locomo.py                   # LoCoMo evaluation adapter
    run_ablation.py                 # Forgetting strategy comparison

  tests/                            # Unit tests
    test_working_memory.py          # L1 tests
    test_npc_l1_integration.py      # L1-NPC integration tests
    test_metadata_filtering.py      # Metadata filtering tests

  docs/                             # Documentation
    comprehensive_results_report.md # Full 9-experiment report
    benchmark_documentation.md      # Benchmark methodology
  
  diagrams/                         # 18 generated charts + 3 architecture diagrams
  research/                         # Research paper source
```

---

## Benchmark Results

### 9 experiments across 3 published datasets, 4 LLM sizes

| Benchmark | Dataset | Best F1 | Best Model | Key Finding |
|-----------|---------|---------|------------|-------------|
| LoCoMo QA | LoCoMo [Maharana 2024] | **0.365** | 17B | Architecture dominates; 8B=0.324, 70B=0.329 |
| MuSiQue | MuSiQue [Trivedi 2022] | **0.535** | 70B | 2-hop works (F1=0.62), 4-hop fails (F1=0.0) |
| HotPotQA | HotPotQA [Yang 2018] | **0.729** | 70B | Competitive with published BM25 baselines (~0.45) |
| E2E Recall | Custom | **80%** | 3B | 5 facts recalled through 200+ memories |
| CSAM vs RAG | LoCoMo | **2.7x** | 3B | 0.136 vs 0.051 on same hardware/model |
| Ablation | Custom | **Same** | 3B | Forgetting = No-forgetting accuracy; 62% less memory |
| Scalability | Custom | **<7ms** | 3B | 5 NPCs: +4% latency, sub-linear scaling |

**Core claim:** Architecture fixes gave a **16x accuracy gain** (0.02 -> 0.33). Scaling models from 8B to 70B gave **+0.005**. The architecture provided 3,200% of the improvement; model scaling provided 0.15%.

---

## Setup

```bash
# Clone and setup
git clone https://github.com/Lamaq-Mujpurwala/CSAM-IPD-HALH.git
cd "CSAM-IPD-HALH"

# Create virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r csam_project/requirements.txt

# Install Ollama for local LLM (optional -- hosted API also works)
# Download from https://ollama.com/download
ollama pull llama3.2:3b
```

For hosted benchmarks, create `.env`:
```
GROQ_API_KEY=your_key_here
```

---

## Running

### Interactive Demo
```bash
cd csam_project
python simulation/demo_cli.py           # With Ollama running
python simulation/demo_cli.py --no-llm  # Without LLM (retrieval-only mode)
```

Demo commands: `talk`, `skip`, `remember`, `recall`, `stats`, `memories`, `quit`

### Unit Tests
```bash
cd csam_project
python -m pytest tests/ -v    # All 3 test suites, ~5 seconds
```

### Benchmarks
```bash
cd csam_project

# E2E recall (local, ~30 min with Ollama)
python benchmarks/benchmark_e2e.py --strategy consolidation

# LoCoMo multi-model (needs GROQ_API_KEY, ~20 min)
python benchmarks/benchmark_multimodel.py --all

# MuSiQue multi-hop (needs GROQ_API_KEY, ~30 min)
python benchmarks/benchmark_musique.py --all --questions 50

# HotPotQA multi-hop (needs GROQ_API_KEY, ~30 min)
python benchmarks/benchmark_hotpotqa.py --all --questions 50

# Generate all charts
python benchmarks/generate_charts.py
python benchmarks/generate_diagrams.py
```

### Scalability Benchmark (no LLM needed)
```bash
python benchmarks/benchmark_scaling.py   # ~3 min, tests 1-10 NPCs
```

---

## Key Files

| File | Description |
|------|-------------|
| `csam_core/forgetting_engine.py` | Novel forgetting strategies (5 implementations) |
| `csam_core/consolidation.py` | L2->L3 knowledge extraction pipeline |
| `csam_core/memory_repository.py` | HNSW-backed vector store |
| `simulation/npc.py` | NPC with full 3-tier memory |
| `simulation/demo_cli.py` | Interactive CLI for judges |
| `docs/comprehensive_results_report.md` | Full results with analysis |
| `PANEL_PRESENTATION_PLAN.md` | Presentation strategy |

---

## Total Cost

All 9 experiments: **$0**. Free-tier Groq API (~460K tokens) + local Ollama.

## License

Research use only.
