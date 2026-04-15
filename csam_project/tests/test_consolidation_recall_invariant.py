"""
BUG-05 — Consolidation Recall Invariant.

Core publication guarantee:
    After M is ingested into L2, consolidated into L3, and then DELETED from L2,
    a semantic query for M's content must still return the L3 node.

This test validates that CSAM's forgetting is information-preserving:
data is not lost, it is promoted.

No LLM calls are made; consolidation is simulated directly to keep the test
fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from csam_project.csam_core.consolidation_tracker import ConsolidationTracker
from csam_project.csam_core.knowledge_graph import KnowledgeGraph
from csam_project.csam_core.memory_repository import MemoryRepository


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unit_vec(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Main invariant test ───────────────────────────────────────────────────────

class TestConsolidationRecallInvariant:
    """
    Verify: consolidate(M) → delete(M) does NOT lose information.

    The L3 node produced by consolidation must be retrievable via the same
    embedding that was used to retrieve M from L2.
    """

    def test_l3_retrieval_after_l2_deletion(self) -> None:
        """L3 node is found after its source L2 memory is deleted."""
        repo = MemoryRepository(embedding_dim=4, max_memories=100)
        kg = KnowledgeGraph(db_path=":memory:", embedding_dim=4)
        tracker = ConsolidationTracker()

        # ── Step 1: Ingest memory M into L2 ─────────────────────────────────
        mem_embedding = _unit_vec(4, seed=0)
        mem_id = repo.add(
            text="Alice attended the workshop on May 7.",
            embedding=mem_embedding,
            importance=0.6,
        )
        assert len(repo) == 1

        # ── Step 2: Simulate consolidation into L3 ──────────────────────────
        # The L3 node has a similar (but not identical) embedding to simulate
        # the summarisation process compressing but preserving the content.
        l3_embedding = _unit_vec(4, seed=1)  # different vector, same direction-ish
        l3_id = kg.add_node(
            content="Alice attended the May workshop.",
            embedding=l3_embedding,
            node_type="summary",
            source_memory_ids=[mem_id],
        )

        tracker.record_consolidation(
            l3_node_id=l3_id,
            source_memory_ids=[mem_id],
            l3_embedding=l3_embedding,
            memory_embeddings={mem_id: mem_embedding},
        )

        # Confirm coverage is recorded
        assert tracker.get_coverage(mem_id) > 0.0

        # ── Step 3: Delete M from L2 ────────────────────────────────────────
        deleted = repo.delete(mem_id)
        assert deleted is True
        assert len(repo) == 0

        # L2 must no longer return the memory
        l2_results = repo.retrieve(mem_embedding, k=5)
        l2_ids = [m.id for m, _ in l2_results]
        assert mem_id not in l2_ids

        # ── Step 4: L3 still returns the consolidated node ──────────────────
        l3_results = kg.query_by_embedding(mem_embedding, k=5)
        assert len(l3_results) > 0, "L3 must return at least one node after L2 deletion"
        l3_result_ids = [node.id for node, _ in l3_results]
        assert l3_id in l3_result_ids, (
            "The L3 node created during consolidation must be retrievable "
            "after the source L2 memory is deleted."
        )

    def test_multiple_memories_single_l3_node(self) -> None:
        """Multiple L2 memories consolidated to one L3 node: all can be deleted safely."""
        repo = MemoryRepository(embedding_dim=4, max_memories=100)
        kg = KnowledgeGraph(db_path=":memory:", embedding_dim=4)
        tracker = ConsolidationTracker()

        mem_embeddings = {f"mem{i}": _unit_vec(4, seed=i) for i in range(3)}
        mem_ids = []
        for key, emb in mem_embeddings.items():
            mid = repo.add(text=f"Fact: {key}", embedding=emb, importance=0.5)
            mem_ids.append(mid)

        l3_embedding = _unit_vec(4, seed=99)
        l3_id = kg.add_node(
            content="Summary of three facts.",
            embedding=l3_embedding,
            node_type="summary",
            source_memory_ids=mem_ids,
        )
        emb_by_id = {mid: repo.get(mid).embedding for mid in mem_ids}
        tracker.record_consolidation(
            l3_node_id=l3_id,
            source_memory_ids=mem_ids,
            l3_embedding=l3_embedding,
            memory_embeddings=emb_by_id,
        )

        # Delete all source memories
        for mid in mem_ids:
            repo.delete(mid)
        assert len(repo) == 0

        # L3 node is still retrievable using any source embedding
        for emb in mem_embeddings.values():
            results = kg.query_by_embedding(emb, k=5)
            result_ids = [n.id for n, _ in results]
            assert l3_id in result_ids

    def test_l3_node_content_matches_source(self) -> None:
        """The recalled L3 node must reference the deleted memory's ID."""
        repo = MemoryRepository(embedding_dim=4, max_memories=100)
        kg = KnowledgeGraph(db_path=":memory:", embedding_dim=4)
        tracker = ConsolidationTracker()

        emb = _unit_vec(4, seed=7)
        mid = repo.add("Alice went hiking on June 3.", emb, importance=0.7)

        l3_id = kg.add_node(
            "Alice went hiking.",
            emb.copy(),
            node_type="summary",
            source_memory_ids=[mid],
        )
        tracker.record_consolidation(l3_id, [mid], emb, {mid: emb})
        repo.delete(mid)

        results = kg.query_by_embedding(emb, k=1)
        assert results, "Expected at least one L3 result"
        node, _ = results[0]
        # The L3 node must reference the original memory
        assert mid in node.source_memory_ids

    def test_unrelated_query_does_not_return_node(self) -> None:
        """A query semantically unrelated to the L3 content returns low similarity."""
        repo = MemoryRepository(embedding_dim=4, max_memories=100)
        kg = KnowledgeGraph(db_path=":memory:", embedding_dim=4)
        tracker = ConsolidationTracker()

        # Use different seeds to guarantee genuinely different vectors
        emb = _unit_vec(4, seed=3)
        # Construct a vector orthogonal to emb using Gram-Schmidt with a
        # different seed to avoid the same-vector problem.
        base = _unit_vec(4, seed=13)  # different seed
        # Subtract projection onto emb, then normalise
        proj = float(np.dot(base, emb))
        orthogonal = base - proj * emb
        norm = np.linalg.norm(orthogonal)
        if norm < 1e-8:
            # In the rare event of numerical cancellation, use another base
            base2 = _unit_vec(4, seed=99)
            orthogonal = base2 - float(np.dot(base2, emb)) * emb
            norm = np.linalg.norm(orthogonal)
        orthogonal = (orthogonal / norm).astype(np.float32)

        # Verify we actually have an orthogonal-ish vector
        assert abs(float(np.dot(orthogonal, emb))) < 1e-5

        mid = repo.add("Alice met Bob.", emb, importance=0.5)
        l3_id = kg.add_node("Alice met Bob at the cafe.", emb.copy(),
                            node_type="summary", source_memory_ids=[mid])
        tracker.record_consolidation(l3_id, [mid], emb, {mid: emb})
        repo.delete(mid)

        results = kg.query_by_embedding(orthogonal, k=1)
        if results:
            _, score = results[0]
            assert score < 0.1, (
                "An orthogonal query should score near 0.0 against the L3 node, "
                f"got {score:.4f}"
            )
