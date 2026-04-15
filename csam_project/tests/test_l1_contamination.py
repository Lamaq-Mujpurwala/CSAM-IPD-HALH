"""
L1 contamination / isolation tests.

Verifies that WorkingMemoryCache:
- clear_all() empties ALL players' caches and facts
- clear_player() only removes one player's data
- Items from player A are never visible to player B
- LRU eviction respects per-player capacity
- Statistics reflect actual state after clears
"""

from __future__ import annotations

import pytest

from csam_project.csam_core.working_memory import WorkingMemoryCache


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def cache() -> WorkingMemoryCache:
    return WorkingMemoryCache(max_size=5, enable_facts=False)


# ── clear_all ─────────────────────────────────────────────────────────────────

class TestClearAll:
    def test_empty_after_clear_all(self, cache: WorkingMemoryCache) -> None:
        cache.add("hello world", player_name="alice")
        cache.add("foo bar", player_name="bob")
        cache.clear_all()
        assert len(cache) == 0

    def test_get_recent_returns_empty_after_clear_all(
        self, cache: WorkingMemoryCache
    ) -> None:
        cache.add("some text", player_name="alice")
        cache.clear_all()
        assert cache.get_recent("alice", k=5) == []

    def test_stats_show_zero_after_clear_all(
        self, cache: WorkingMemoryCache
    ) -> None:
        cache.add("text", player_name="alice")
        cache.clear_all()
        stats = cache.get_statistics()
        assert stats["total_cached_items"] == 0
        assert stats["total_players"] == 0

    def test_can_add_after_clear_all(self, cache: WorkingMemoryCache) -> None:
        cache.add("first", player_name="alice")
        cache.clear_all()
        cache.add("after clear", player_name="alice")
        items = cache.get_recent("alice", k=5)
        assert len(items) == 1
        assert items[0].text == "after clear"


# ── clear_player ──────────────────────────────────────────────────────────────

class TestClearPlayer:
    def test_target_player_removed(self, cache: WorkingMemoryCache) -> None:
        cache.add("alice data", player_name="alice")
        cache.add("bob data", player_name="bob")
        cache.clear_player("alice")
        assert cache.get_recent("alice", k=5) == []

    def test_other_player_unaffected(self, cache: WorkingMemoryCache) -> None:
        cache.add("alice data", player_name="alice")
        cache.add("bob data", player_name="bob")
        cache.clear_player("alice")
        bob_items = cache.get_recent("bob", k=5)
        assert len(bob_items) == 1
        assert bob_items[0].text == "bob data"

    def test_len_decremented_after_clear_player(
        self, cache: WorkingMemoryCache
    ) -> None:
        cache.add("a1", player_name="alice")
        cache.add("a2", player_name="alice")
        cache.add("b1", player_name="bob")
        cache.clear_player("alice")
        assert len(cache) == 1  # only bob's item remains


# ── Cross-player isolation ────────────────────────────────────────────────────

class TestCrossPlayerIsolation:
    def test_alice_items_not_visible_to_bob(
        self, cache: WorkingMemoryCache
    ) -> None:
        cache.add("alice secret", player_name="alice")
        bob_items = cache.get_recent("bob", k=5)
        assert bob_items == []

    def test_players_have_independent_histories(
        self, cache: WorkingMemoryCache
    ) -> None:
        for i in range(3):
            cache.add(f"alice msg {i}", player_name="alice")
        for i in range(2):
            cache.add(f"bob msg {i}", player_name="bob")

        alice_items = cache.get_recent("alice", k=10)
        bob_items = cache.get_recent("bob", k=10)

        alice_texts = {item.text for item in alice_items}
        bob_texts = {item.text for item in bob_items}

        # No cross-contamination
        assert not alice_texts & bob_texts


# ── LRU eviction per player ───────────────────────────────────────────────────

class TestLRUEviction:
    def test_capacity_capped_at_max_size(
        self, cache: WorkingMemoryCache
    ) -> None:
        for i in range(10):  # max_size=5
            cache.add(f"item {i}", player_name="alice")
        alice_items = cache.get_recent("alice", k=10)
        assert len(alice_items) == 5

    def test_oldest_item_evicted(self, cache: WorkingMemoryCache) -> None:
        for i in range(6):  # one over max_size=5
            cache.add(f"item {i}", player_name="alice")
        items = cache.get_recent("alice", k=10)
        texts = [item.text for item in items]
        # "item 0" is the oldest and must have been evicted
        assert "item 0" not in texts
        # Newest item must be present
        assert "item 5" in texts

    def test_most_recent_item_is_first_in_result(
        self, cache: WorkingMemoryCache
    ) -> None:
        cache.add("first", player_name="alice")
        cache.add("last", player_name="alice")
        items = cache.get_recent("alice", k=2)
        assert items[0].text == "last"

    def test_lru_is_per_player_not_global(
        self, cache: WorkingMemoryCache
    ) -> None:
        """Eviction in one player's cache must not affect another player's cache."""
        for i in range(6):
            cache.add(f"alice item {i}", player_name="alice")
        cache.add("bob item 0", player_name="bob")

        bob_items = cache.get_recent("bob", k=5)
        assert len(bob_items) == 1
        assert bob_items[0].text == "bob item 0"


# ── len() accuracy ────────────────────────────────────────────────────────────

class TestLen:
    def test_len_counts_all_players(self, cache: WorkingMemoryCache) -> None:
        cache.add("a", player_name="alice")
        cache.add("b", player_name="alice")
        cache.add("c", player_name="bob")
        assert len(cache) == 3

    def test_len_zero_initially(self) -> None:
        assert len(WorkingMemoryCache()) == 0
