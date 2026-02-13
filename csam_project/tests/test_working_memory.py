"""
Test L1 Working Memory

Verify that the working memory cache works correctly.
"""

import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csam_core.working_memory import WorkingMemoryCache


def test_working_memory():
    """Test L1 working memory functionality."""
    
    print("=" * 60)
    print("Testing L1 Working Memory")
    print("=" * 60)
    
    # Create cache
    cache = WorkingMemoryCache(max_size=5, enable_facts=True)
    
    # Test 1: Add items for Bob
    print("\n1. Adding items for Bob...")
    bob_interactions = [
        "Bob: Hi, I'm looking for weapons",
        "NPC: Welcome! I have swords and bows.",
        "Bob: I'm a warrior, so I need a sword",
        "NPC: Here's a steel sword for you.",
        "Bob: Perfect, thanks!"
    ]
    
    for text in bob_interactions:
        cache.add(
            text=text,
            player_name="Bob",
            metadata={"player_class": "Warrior"},
            importance=0.7
        )
        print(f"  [OK] Added: {text}")
    
    # Test 2: Retrieve recent for Bob
    print("\n2. Retrieving recent items for Bob (k=3)...")
    recent = cache.get_recent("Bob", k=3)
    print(f"  Found {len(recent)} items:")
    for item in recent:
        print(f"    - {item.text}")
    
    # Test 3: Check facts
    print("\n3. Checking extracted facts for Bob...")
    player_class = cache.get_fact("Bob", "player_class")
    print(f"  Player class: {player_class}")
    
    # Test 4: Add items for Alice
    print("\n4. Adding items for Alice...")
    alice_interactions = [
        "Alice: Hello, do you have potions?",
        "NPC: Yes, I have health and mana potions.",
        "Alice: I'll take 3 health potions"
    ]
    
    for text in alice_interactions:
        cache.add(
            text=text,
            player_name="Alice",
            importance=0.6
        )
    
    # Test 5: Verify player isolation
    print("\n5. Verifying player isolation...")
    bob_recent = cache.get_recent("Bob", k=10)
    alice_recent = cache.get_recent("Alice", k=10)
    
    print(f"  Bob's cache: {len(bob_recent)} items")
    print(f"  Alice's cache: {len(alice_recent)} items")
    
    # Verify no cross-contamination
    bob_only = all("Bob" in item.text or "NPC" in item.text for item in bob_recent)
    alice_only = all("Alice" in item.text or "NPC" in item.text for item in alice_recent)
    
    print(f"  Bob's items isolated: {bob_only}")
    print(f"  Alice's items isolated: {alice_only}")
    
    # Test 6: LRU eviction (max_size=5)
    print("\n6. Testing LRU eviction (max_size=5)...")
    print(f"  Adding 3 more items to Bob's cache...")
    for i in range(3):
        cache.add(
            text=f"Bob: Additional message {i+1}",
            player_name="Bob"
        )
    
    final_bob = cache.get_recent("Bob", k=10)
    print(f"  Bob's cache after additions: {len(final_bob)} items (should be 5)")
    print("  Oldest items should be evicted:")
    for item in final_bob:
        print(f"    - {item.text[:50]}...")
    
    # Test 7: Statistics
    print("\n7. Cache statistics...")
    stats = cache.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total cached items: {len(cache)}")
    print(f"  Players tracked: {stats['total_players']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  LRU eviction: {'[OK] Working' if len(final_bob) == 5 else '[FAIL] Failed'}")
    print(f"  Player isolation: {'[OK] Working' if (bob_only and alice_only) else '[FAIL] Failed'}")
    print("\n[OK] L1 Working Memory tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_working_memory()
