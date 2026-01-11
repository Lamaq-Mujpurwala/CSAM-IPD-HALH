# CSAM - Cognitive Sparse Access Memory

A hierarchical memory architecture for AI agents with **consolidation-aware forgetting**.

## 🎯 Research Contribution

Our novel contribution is **consolidation-aware forgetting**:

```
ForgetScore(m) = α·R(m) + β·(1-I(m)) + γ·C(m) + δ·D(m)
```

Where:
- `R(m)` = Recency decay
- `I(m)` = Importance score  
- `C(m)` = **Consolidation coverage** (how much of m is in L3) ⭐ NOVEL
- `D(m)` = **L3 redundancy** (similarity to any L3 node) ⭐ NOVEL

**Key insight:** If a memory's content has been "absorbed" into L3 (as a summary), 
the original memory is *redundant* and can be safely forgotten.

## 📁 Project Structure

```
csam_project/
├── csam_core/                    # Core Python modules
│   ├── __init__.py
│   ├── memory_repository.py      # L2 episodic store (HNSW)
│   ├── knowledge_graph.py        # L3 semantic store (SQLite)
│   ├── forgetting_engine.py      # ⭐ Novel forgetting strategies
│   ├── consolidation_tracker.py  # Tracks L2 → L3 mappings
│   ├── retrieval.py              # Hybrid L2+L3 retrieval
│   ├── mmr.py                    # Diversity mechanism
│   └── services/
│       ├── embedding.py          # Sentence-transformers wrapper
│       └── llm.py                # Ollama wrapper
├── tests/                        # Unit tests
├── evaluation/                   # Benchmarks and ablation studies
└── requirements.txt
```

## 🚀 Setup

### 1. Create Virtual Environment

```bash
cd "c:\Users\lamaq\OneDrive\Desktop\CSAM project\csam_project"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (for LLM)

Download from: https://ollama.com/download

Then pull a model:
```bash
ollama pull llama3.2:3b
```

### 4. Verify Installation

```bash
python -c "from csam_core import MemoryRepository; print('CSAM loaded successfully!')"
```

## 🧪 Quick Test

```python
from csam_core import MemoryRepository
from csam_core.services import EmbeddingService

# Initialize
embedder = EmbeddingService()
memory = MemoryRepository(embedding_dim=embedder.dimension)

# Add memories
embedding = embedder.encode("Player bought a sword for 100 gold")
memory.add("Player bought a sword for 100 gold", embedding, importance=0.7)

embedding = embedder.encode("Player likes exploring dungeons")
memory.add("Player likes exploring dungeons", embedding, importance=0.8)

# Retrieve
query = embedder.encode("What did the player buy?")
results = memory.retrieve(query, k=2)

for mem, score in results:
    print(f"[{score:.2f}] {mem.text}")
```

## 📊 Running Evaluation

```bash
# Run ablation study (compares forgetting strategies)
python evaluation/run_ablation.py

# Run E2E Benchmark (NPC-LoCoMo)
python benchmarks/benchmark_e2e.py --npcs 1 --memories 100 --strategy consolidation
```

## 🔧 Configuration

The forgetting engine can be configured:

```python
from csam_core import ConsolidationAwareForgetting

forgetting = ConsolidationAwareForgetting(
    alpha=0.2,    # Recency weight
    beta=0.2,     # Importance weight
    gamma=0.3,    # Consolidation coverage weight ⭐
    delta=0.3,    # L3 redundancy weight ⭐
    consolidation_threshold=0.5  # Protect unconsolidated memories
)
```

## 📝 License

Research use only. See LICENSE for details.
