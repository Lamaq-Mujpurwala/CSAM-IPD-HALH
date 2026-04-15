# Copilot Instructions

This file provides instructions to GitHub Copilot when working in this repository.

## Project

CSAM (Cognitive Sparse Access Memory) — A 3-tier hierarchical memory architecture for AI agents with consolidation-aware forgetting. Python 3.11+, pytest, NetworkX, HNSW, sentence-transformers.

## Rules

- Follow PEP 8, type annotations on all public functions
- Use `logging` module, never `print()` in production code
- Use frozen dataclasses and NamedTuples for data structures
- Maximum 50 lines per function, 800 lines per file
- Handle errors with specific exceptions, never bare `except:`
- No hardcoded secrets — use `.env` and `os.environ`
- Run `pytest csam_project/tests/ -v` before committing
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `bench:`

## Architecture (DO NOT BREAK)

- L1: Working Memory (LRU) — `csam_core/working_memory.py`
- L2: Long-Term Memory (HNSW) — `csam_core/memory_repository.py`
- L3: Knowledge Graph (NetworkX) — `csam_core/knowledge_graph.py`
- Forgetting formula: `0.25*R + 0.25*(1-I) + 0.25*C + 0.25*D` — preserve this in `forgetting_engine.py`
  (Note: 0.2/0.2/0.3/0.3 weighting is tracked as a separate experimental variant in grid_search_results.json)
