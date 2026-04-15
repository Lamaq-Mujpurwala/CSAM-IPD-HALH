"""
Tests for forgetting_engine.py — gate logic, ranking, and FFR semantics.

Key assertions:
- CA gate (consolidation_threshold=0.3) gives score=0 for unconsolidated memories
- Unconsolidated memories are NEVER selected for deletion when gate is active
- CA formula-only (gate=0.0) CAN delete unconsolidated memories
- last_forgotten_ids is populated correctly
- NoForgetting never selects any memory
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from csam_project.csam_core.forgetting_engine import (
    ConsolidationAwareForgetting,
    ImportanceForgetting,
    LRUForgetting,
    NoForgetting,
    RecencyImportanceForgetting,
    create_forgetting_strategy,
)
from csam_project.csam_core.memory_repository import Memory


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_memory(
    importance: float = 0.5,
    age_days: float = 30.0,
    embedding: np.ndarray | None = None,
) -> Memory:
    """Create a test Memory with controllable age and importance."""
    if embedding is None:
        embedding = np.random.rand(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

    now = datetime.now()
    # Simulate age by adjusting last_accessed (used for R(m))
    import datetime as dt
    last_accessed = now - dt.timedelta(days=age_days)

    mem = Memory(
        id=str(uuid.uuid4()),
        text="test memory",
        embedding=embedding,
        importance=importance,
        last_accessed=last_accessed,
    )
    return mem


def make_consolidation_tracker(coverage_map: dict[str, float]) -> MagicMock:
    """Return a mock ConsolidationTracker with controlled coverage values."""
    tracker = MagicMock()
    tracker.get_coverage.side_effect = lambda mid: coverage_map.get(mid, 0.0)
    return tracker


# ── NoForgetting ──────────────────────────────────────────────────────────────

class TestNoForgetting:
    def test_scores_are_all_zero(self) -> None:
        strategy = NoForgetting()
        memories = [make_memory() for _ in range(5)]
        scores = strategy.compute_forget_scores(memories)
        assert all(v == 0.0 for v in scores.values())

    def test_select_returns_empty(self) -> None:
        strategy = NoForgetting()
        memories = [make_memory() for _ in range(5)]
        result = strategy.select_to_forget(memories, count=3)
        assert result == []

    def test_last_forgotten_ids_empty(self) -> None:
        strategy = NoForgetting()
        memories = [make_memory() for _ in range(3)]
        strategy.select_to_forget(memories, count=3)
        assert strategy.last_forgotten_ids == []


# ── LRUForgetting ─────────────────────────────────────────────────────────────

class TestLRUForgetting:
    def test_oldest_memory_scores_highest(self) -> None:
        strategy = LRUForgetting()
        old = make_memory(age_days=300.0)
        new = make_memory(age_days=1.0)
        scores = strategy.compute_forget_scores([old, new])
        assert scores[old.id] > scores[new.id]

    def test_last_forgotten_ids_populated(self) -> None:
        strategy = LRUForgetting()
        memories = [make_memory(age_days=float(i * 50)) for i in range(5)]
        result = strategy.select_to_forget(memories, count=2)
        assert strategy.last_forgotten_ids == result
        assert len(result) == 2


# ── ImportanceForgetting ──────────────────────────────────────────────────────

class TestImportanceForgetting:
    def test_low_importance_scores_highest(self) -> None:
        strategy = ImportanceForgetting()
        low = make_memory(importance=0.1)
        high = make_memory(importance=0.9)
        scores = strategy.compute_forget_scores([low, high])
        assert scores[low.id] > scores[high.id]

    def test_selects_lowest_importance_first(self) -> None:
        strategy = ImportanceForgetting()
        memories = [make_memory(importance=i / 10) for i in range(1, 6)]
        result = strategy.select_to_forget(memories, count=2)
        # Should pick the two lowest importance memories
        sorted_by_importance = sorted(memories, key=lambda m: m.importance)
        expected_ids = {sorted_by_importance[0].id, sorted_by_importance[1].id}
        assert set(result) == expected_ids


# ── CA gate protection (⭐ core publication claim) ────────────────────────────

class TestConsolidationAwareForgettingGate:
    """
    ⭐ These tests verify the core publication claim:
    The CA gate (C(m) < θ) prevents deletion of unconsolidated memories.
    """

    def test_unconsolidated_memory_gets_zero_score(self) -> None:
        """C(m)=0 < θ=0.3 → score must be 0.0."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        mem = make_memory(importance=0.1, age_days=300.0)
        tracker = make_consolidation_tracker({mem.id: 0.0})
        scores = strategy.compute_forget_scores([mem], consolidation_tracker=tracker)
        assert scores[mem.id] == 0.0

    def test_unconsolidated_memory_never_selected(self) -> None:
        """Gate ensures unconsolidated memories are never in the deletion list."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        unconsolidated = [make_memory(importance=0.0, age_days=365.0) for _ in range(5)]
        tracker = make_consolidation_tracker({m.id: 0.0 for m in unconsolidated})

        result = strategy.select_to_forget(
            unconsolidated, count=3, consolidation_tracker=tracker
        )
        assert result == []

    def test_consolidated_memory_can_be_deleted(self) -> None:
        """C(m)=0.9 ≥ θ=0.3 → memory is eligible for deletion."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        mem = make_memory(importance=0.1, age_days=300.0)
        tracker = make_consolidation_tracker({mem.id: 0.9})

        scores = strategy.compute_forget_scores([mem], consolidation_tracker=tracker)
        assert scores[mem.id] > 0.0

    def test_gate_boundary_exactly_at_threshold_is_protected(self) -> None:
        """C(m) = 0.299 < θ=0.3 → protected."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        mem = make_memory(importance=0.0, age_days=365.0)
        tracker = make_consolidation_tracker({mem.id: 0.299})

        scores = strategy.compute_forget_scores([mem], consolidation_tracker=tracker)
        assert scores[mem.id] == 0.0

    def test_gate_exactly_at_threshold_is_eligible(self) -> None:
        """C(m) = 0.3 == θ=0.3 → eligible (not protected)."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        mem = make_memory(importance=0.0, age_days=365.0)
        tracker = make_consolidation_tracker({mem.id: 0.3})

        scores = strategy.compute_forget_scores([mem], consolidation_tracker=tracker)
        assert scores[mem.id] > 0.0

    def test_gate_disabled_allows_unconsolidated_deletion(self) -> None:
        """Gate=0.0 (CA-Formula-Only ablation) → unconsolidated CAN be deleted."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.0)
        mem = make_memory(importance=0.0, age_days=365.0)
        tracker = make_consolidation_tracker({mem.id: 0.0})

        scores = strategy.compute_forget_scores([mem], consolidation_tracker=tracker)
        # With gate=0.0, C(m)=0.0 still counts, but not protected
        result = strategy.select_to_forget(
            [mem], count=1, consolidation_tracker=tracker
        )
        assert len(result) == 1

    def test_mixed_protection_selects_only_consolidated(self) -> None:
        """With mixed coverage, only memories above threshold are selected."""
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        protected = [make_memory(importance=0.0, age_days=365.0) for _ in range(3)]
        eligible = [make_memory(importance=0.0, age_days=365.0) for _ in range(3)]

        coverage = {m.id: 0.1 for m in protected}   # < 0.3 → protected
        coverage.update({m.id: 0.8 for m in eligible})  # ≥ 0.3 → eligible
        tracker = make_consolidation_tracker(coverage)

        result = strategy.select_to_forget(
            protected + eligible, count=3, consolidation_tracker=tracker
        )
        protected_ids = {m.id for m in protected}
        assert len(result) == 3
        assert not set(result) & protected_ids  # no protected memory deleted

    def test_l3_redundancy_raises_score(self) -> None:
        """D(m) component increases forget score when memory resembles L3 node."""
        strategy = ConsolidationAwareForgetting(
            alpha=0.0, beta=0.0, gamma=0.0, delta=1.0,  # pure D(m)
            consolidation_threshold=0.3,
        )
        embedding = np.ones(384, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        mem = make_memory(importance=0.0, age_days=0.0, embedding=embedding.copy())
        l3_embeddings = np.array([embedding])  # identical → D=1.0

        tracker = make_consolidation_tracker({mem.id: 1.0})  # eligible
        scores = strategy.compute_forget_scores(
            [mem], consolidation_tracker=tracker, l3_embeddings=l3_embeddings
        )
        assert scores[mem.id] == pytest.approx(1.0, abs=1e-5)

    def test_last_forgotten_ids_matches_return(self) -> None:
        strategy = ConsolidationAwareForgetting(consolidation_threshold=0.3)
        eligible = [make_memory(importance=0.0, age_days=300.0) for _ in range(5)]
        tracker = make_consolidation_tracker({m.id: 0.9 for m in eligible})

        result = strategy.select_to_forget(eligible, count=3, consolidation_tracker=tracker)
        assert strategy.last_forgotten_ids == result


# ── create_forgetting_strategy factory ───────────────────────────────────────

class TestFactory:
    @pytest.mark.parametrize("name,klass", [
        ("none", NoForgetting),
        ("lru", LRUForgetting),
        ("importance", ImportanceForgetting),
        ("recency_importance", RecencyImportanceForgetting),
        ("consolidation_aware", ConsolidationAwareForgetting),
    ])
    def test_factory_creates_correct_type(self, name: str, klass: type) -> None:
        strategy = create_forgetting_strategy(name)
        assert isinstance(strategy, klass)

    def test_factory_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_forgetting_strategy("nonexistent_strategy")
