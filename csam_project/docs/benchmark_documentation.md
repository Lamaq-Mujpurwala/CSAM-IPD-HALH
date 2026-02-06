# Sustained Conversation Benchmark Documentation

## Overview
The `sustained_conversation.py` script is a high-performance benchmark designed to evaluate **long-term interaction quality** and **memory system stability** between AI agents. Unlike simple request-response tests, this benchmark simulates continuous, multi-turn conversations between pairs of NPCs to stress-test the memory architecture over time.

## Key Architectural Innovations
This implementation differs from standard benchmarks in four critical areas:

### 1. Multi-Agent Concurrency
Instead of running agents sequentially, this system uses `asyncio` to run multiple conversation pairs **concurrently**.
- **Parallel Execution**: Uses `asyncio.gather(*tasks)` to run N pairs simultaneously (e.g., `run_multi_pair_benchmark`).
- **Shared Resources**: A single `AsyncGPUEmbeddingService` and `AsyncLLMService` are shared across all agents to realistically simulate server load and resource contention.
- **Async Pipeline**: The entire pipeline—from embedding generation (`await self.embedding_service.encode`) to retrieval (`await self.retriever.retrieve`) to LLM generation—is fully asynchronous.

### 2. Three-Layer Memory Architecture
The benchmark implements a sophisticated hierarchy of memory types not found in typical RAG systems:

| Layer | Component | Purpose | Implementation Detail |
|-------|-----------|---------|-----------------------|
| **L1** | `WorkingMemoryCache` | **Short-term Context**. Stores the last K recent interactions. | Injected directly into the *System Prompt* to maintain immediate continuity (`RECENT WORKING MEMORY` section). |
| **L2** | `PersistentFAISSRepository` | **Episodic Memory**. Stores raw interaction logs using vector embeddings. | Uses **Consolidation-Aware Forgetting** to smartly manage storage limits rather than deleting randomly or by age. |
| **L3** | `KnowledgeGraph` | **Semantic Memory**. Stores distilled facts and relationships. | Used in Hybrid Retrieval to augment raw memories with conceptual understanding. |

### 3. Context-Aware Metadata Filtering
A critical differentiator in this implementation is strict **conversation scoping**.
- **The Problem**: In multi-user environments, agents often "hallucinate" memories from conversations with User A while talking to User B.
- **The Solution**: We implemented **Metadata Filtering** at the retrieval level.
  ```python
  retrieval_result = await self.retriever.retrieve(
      query_embedding=message_embedding,
      k=3,
      # ⭐ CRITICAL: Filters memories to ONLY those involving the current speaker
      metadata_filter={"player_name": speaker_name}  
  )
  ```
- **Result**: NPC A cleanly separates its history with NPC B from any other interactions.

### 4. Novel Forgetting Strategy
We implemented `ConsolidationAwareForgetting` instead of standard LRU (Least Recently Used).
- The system checks if a memory has been *consolidated* into the Knowledge Graph (L3) before allowing it to be forgotten from Episodic Memory (L2).
- This ensures that while raw details fade, the semantic "lessons" remain.

## Conversation Configuration

### Conversation Topics
To ensure varied and comparable benchmarks, conversations are seeded with specific rotation topics based on the Pair ID. These are distinct from generic "hello" messages to provoke memory usage immediately.

**Topics used:**
1. "What's your favorite memory?" (Tests retrieval of past events)
2. "Tell me about your day." (Tests recent L1 memory)
3. "What do you think about time?" (Tests abstract/L3 knowledge)
4. "Have we talked before?" (Tests existence checks)
5. "What interests you most?" (Tests personality traits)

**Mechanism**: `current_message = topics[pair_id % len(topics)]`

### NPC Personality Generation
NPCs are not just blank slates; they are generated with distinct roles ("bartender", "merchant") and traits to ensure the conversation has semantic substance to store.

## Performance Metrics
The benchmark tracks metrics that standard chat bots ignore:
- **Memory System Latency**: Pure retrieval/storage time (excluding LLM generation).
- **Memory Growth Rate**: How fast the vector index grows.
- **Forgetting Events**: Exact turns where memory limits effectively triggered the forgetting mechanism.
