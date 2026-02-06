# CSAM Project: Enhanced Memory Architecture

This project implements a state-of-the-art **Consolidated Semantic Associative Memory (CSAM)** system for autonomous agents. It replaces a legacy synchronous prototype with a high-performance, asynchronous, 3-layer memory architecture capable of sustained, multi-turn conversations.

---

## 🚀 Key Features

### 1. 3-Layer Memory Hierarchy
We moved beyond simple vector storage to a biologically-inspired hierarchy:
- **L1 Working Memory**: Ultra-fast, transient cache (last ~20 items) injected directly into the system prompt.
- **L2 Episodic Memory**: Persistent vector storage (FAISS) for long-term history, featuring **Consolidation-Aware Forgetting**.
- **L3 Semantic Memory**: A Knowledge Graph that stores distilled facts and relationships.

### 2. High-Performance Async Pipeline
- **Async/Await**: The entire pipeline (embedding → retrieval → generation) is non-blocking.
- **GPU Acceleration**: Utilizes `AsyncGPUEmbeddingService` to batch requests dynamically for CUDA devices.

### 3. Advanced Retrieval
- **Hybrid Retrieval**: Combines vector similarity (L2) with graph traversal (L3).
- **MMR Re-ranking**: Uses *Maximal Marginal Relevance* to ensure response diversity.
- **Context Filtering**: Scopes retrieval to specific conversation partners to prevent hallucinations.

---

## 🛠️ How to Run the Benchmark

The `sustained_conversation.py` script is the primary tool for validating system stability. It simulates continuous conversations between multiple pairs of NPCs.

### Basic Command
```bash
python csam_project/benchmarks/sustained_conversation.py
```
*Runs with default settings: 3 pairs, 50 turns each.*

### Configuration Options

| Option | Default | Description |
| :--- | :---: | :--- |
| `--pairs N` | `3` | Number of NPC pairs to run **currently**. usage: scaling this tests system throughput. |
| `--turns N` | `50` | Number of exchanges per pair. usage: scaling this tests long-term stability and forgetting mechanisms. |
| `--gpu` | `False` | Enable GPU acceleration for embeddings. **Highly recommended** for performance. |
| `--quiet` | `False` | Suppress verbose log output. |

### usage Examples

**1. Quick Smoke Test** (Verify code works)
```bash
python csam_project/benchmarks/sustained_conversation.py --pairs 1 --turns 5
```

**2. Standard Performance Test** (Use GPU)
```bash
python csam_project/benchmarks/sustained_conversation.py --pairs 3 --turns 50 --gpu
```

**3. Stress Test** (High concurrency & duration)
```bash
# Simulates 50 agents (25 pairs) talking for 100 turns
python csam_project/benchmarks/sustained_conversation.py --pairs 25 --turns 100 --gpu --quiet
```

### 📊 Reading the Results
The benchmark is "self-documenting" via its output plots. After a run, check the `benchmark_results/sustained/` folder for a PNG image showing:
- **Memory Growth**: Are memories accumulating correctly?
- **Forgetting Events**: Did the L2 `ConsolidationAwareForgetting` trigger when the limit was reached?
- **Latency**: Is retrieval speed stable (O(log N)) or degrading?

---

## ⚡ Troubleshooting

**"FAISS-GPU not available"**
- **Cause**: The `faiss-gpu` package is Linux-only. On Windows, the system automatically falls back to `faiss-cpu`.
- **Solution**: This is normal on Windows. The `--gpu` flag will still accelerate *embeddings* (PyTorch), even if the vector index is on CPU.

**"Unclosed client session"**
- **Cause**: Minor cleanup warning from the async HTTP client.
- **Impact**: Harmless. Can be ignored.
