# CSAM Project: Modernization & Upgrades

This document details the critical architectural upgrades made to the legacy codebase. We have transitioned from a simple synchronous prototype to a high-performance, asynchronous, multi-agent memory system.

## 🚀 Key Upgrades at a Glance

| Feature | Legacy Codebase | Modernized Codebase |
| :--- | :--- | :--- |
| **Execution Model** | Synchronous (blocking) | **Asynchronous (`asyncio`)** |
| **Compute** | CPU-only | **GPU-Accelerated (CUDA)** |
| **Memory Architecture**| Single Vector Store | **3-Layer Hierarchy (L1/L2/L3)** |
| **Forgetting** | Random / LRU | **Consolidation-Aware** |
| **Retrieval** | Simple Cosine Similarity | **Hybrid (Vector + Graph) + MMR** |
| **Context Scoping** | None (Global context) | **Metadata Filtering** |

---

## 🛠️ Detailed Changes

### 1. From Sync CPU to Async GPU Pipeline
**What we added:**
- **`AsyncGPUEmbeddingService`**: Replaced standard `sentence-transformers` usage with a specialized service that handles:
  - **Batching**: Automatically groups concurrent requests into larger batches for GPU efficiency.
  - **Non-blocking**: `await encode(text)` allows the server to handle other traffic while the GPU crunches numbers.
- **Why**: Legacy code blocked the entire server while calculating embeddings. The new system scales linear-ly with GPU memory.

### 2. The 3-Layer Memory Architecture
**What we added:**
The legacy system relied solely on a "flat" vector database. We introduced a biological hierarchy:

1.  **L1 Working Memory (`WorkingMemoryCache`)** **[NEW]**
    *   *Role*: Ultra-fast, transient cache (last ~20 items).
    *   *Implementation*: Injected directly into the **System Prompt** for immediate context. No vector search required.
    
2.  **L2 Episodic Memory (`PersistentFAISSRepository`)** **[UPGRADED]**
    *   *Role*: Long-term history.
    *   *Upgrade*: Switched from basic lists/HNSW to **Persistent FAISS**. Added disk serialization so memories survive restarts.
    
3.  **L3 Semantic Memory (`KnowledgeGraph`)** **[NEW]**
    *   *Role*: Distilled facts.
    *   *Usage*: Used in `HybridRetriever` to augment raw memories with conceptual links.

### 3. Smart "Consolidation-Aware" Forgetting
**What we added:**
- **Legacy**: Deleted the oldest memory when full (LRU).
- **New**: `ConsolidationAwareForgetting` strategy.
- **Logic**: The system checks if a memory has been *consolidated* (learned) into the Knowledge Graph (L3) before allowing it to be deleted.
- **Benefit**: "Important" memories are protected even if they are old.

### 4. Hybrid Retrieval with MMR
**What we added:**
- **Legacy**: Simple top-k retrieval.
- **New**: `HybridRetriever` that combines:
  - L2 Vectors (Raw similarity)
  - L3 Graph Nodes (Conceptual relevance)
  - **MMR (Maximal Marginal Relevance)**: Re-ranks results to ensure *diversity*, preventing the AI from looping on the same few memories.

### 5. Benchmark Capabilities
**What we added:**
- **`sustained_conversation.py`**: A new benchmark capable of running **multiple NPC pairs concurrently**.
- **Metrics**: We now track:
  - Memory System Latency (ms) (separated from LLM generation time)
  - Memory Growth & Forgetting Events
  - System Stability over long durations (e.g., 50+ turns)

### 6. Code Stability Fixes
- **Shape Mismatch Fix**: Fixed `numpy` shape errors (`(1,1,384)` vs `(384,)`) in the MMR module to ensure robustness with single-item batches.
- **Async/Sync Compatibility**: Updated `HybridRetriever` to support both legacy synchronous calls (`retrieve_sync`) and new async pipelines (`retrieve`).
