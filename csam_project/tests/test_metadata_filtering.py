"""
Test Metadata Filtering

Quick test to verify player-scoped memory retrieval works.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csam_core.memory_repository import MemoryRepository
from csam_core.services.embedding import EmbeddingService


def test_metadata_filtering():
    """Test that metadata filtering correctly scopes memories by player."""
    
    print("=" * 60)
    print("Testing Metadata Filtering")
    print("=" * 60)
    
    # Setup
    embedding_service = EmbeddingService()
    memory_repo = MemoryRepository(embedding_dim=384, max_memories=1000)
    
    # Add memories for different players
    print("\n1. Adding memories for different players...")
    
    # Bob's memories
    bob_memories = [
        "Bob: Hi, I'm looking for weapons",
        "Bob: I need a sword",
        "Bob: Thanks for the help!"
    ]
    
    # Alice's memories
    alice_memories = [
        "Alice: Hello, do you have potions?",
        "Alice: I'll take 3 health potions",
        "Alice: See you later!"
    ]
    
    # Charlie's memories
    charlie_memories = [
        "Charlie: Greetings, traveler",
        "Charlie: I'm just browsing",
    ]
    
    # Store all memories
    for text in bob_memories:
        embedding = embedding_service.encode(text)
        memory_repo.add(
            text=text,
            embedding=embedding,
            importance=0.7,
            metadata={"player_name": "Bob", "npc_name": "Shopkeeper"}
        )
        print(f"  ✓ Added: {text}")
    
    for text in alice_memories:
        embedding = embedding_service.encode(text)
        memory_repo.add(
            text=text,
            embedding=embedding,
            importance=0.7,
            metadata={"player_name": "Alice", "npc_name": "Shopkeeper"}
        )
        print(f"  ✓ Added: {text}")
    
    for text in charlie_memories:
        embedding = embedding_service.encode(text)
        memory_repo.add(
            text=text,
            embedding=embedding,
            importance=0.7,
            metadata={"player_name": "Charlie", "npc_name": "Shopkeeper"}
        )
        print(f"  ✓ Added: {text}")
    
    print(f"\nTotal memories stored: {len(memory_repo)}")
    
    # Test 1: Retrieve without filter (should get any memories)
    print("\n2. Test: Retrieve without filter (query: 'Hi')...")
    query = "Hi"
    query_embedding = embedding_service.encode(query)
    results = memory_repo.retrieve(query_embedding, k=3)
    
    print(f"  Found {len(results)} memories:")
    for memory, score in results:
        player = memory.metadata.get('player_name', 'Unknown')
        print(f"    [{player}] {memory.text[:50]}... (score: {score:.3f})")
    
    # Test 2: Retrieve with Bob filter
    print("\n3. Test: Retrieve with player_name='Bob' filter...")
    results_bob = memory_repo.retrieve(
        query_embedding,
        k=3,
        metadata_filter={"player_name": "Bob"}
    )
    
    print(f"  Found {len(results_bob)} memories:")
    for memory, score in results_bob:
        player = memory.metadata.get('player_name', 'Unknown')
        print(f"    [{player}] {memory.text[:50]}... (score: {score:.3f})")
    
    # Verify all results are from Bob
    bob_only = all(
        m.metadata.get('player_name') == 'Bob'
        for m, _ in results_bob
    )
    print(f"\n  ✓ All results from Bob: {bob_only}")
    
    # Test 3: Retrieve with Alice filter
    print("\n4. Test: Retrieve with player_name='Alice' filter...")
    results_alice = memory_repo.retrieve(
        query_embedding,
        k=3,
        metadata_filter={"player_name": "Alice"}
    )
    
    print(f"  Found {len(results_alice)} memories:")
    for memory, score in results_alice:
        player = memory.metadata.get('player_name', 'Unknown')
        print(f"    [{player}] {memory.text[:50]}... (score: {score:.3f})")
    
    # Verify all results are from Alice
    alice_only = all(
        m.metadata.get('player_name') == 'Alice'
        for m, _ in results_alice
    )
    print(f"\n  ✓ All results from Alice: {alice_only}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total memories: {len(memory_repo)}")
    print(f"  Without filter: {len(results)} results (mixed players)")
    print(f"  Bob filter: {len(results_bob)} results (Bob only: {bob_only})")
    print(f"  Alice filter: {len(results_alice)} results (Alice only: {alice_only})")
    
    if bob_only and alice_only:
        print("\n✅ Metadata filtering WORKING CORRECTLY!")
    else:
        print("\n❌ Metadata filtering NOT working as expected")
    
    print("=" * 60)


if __name__ == "__main__":
    test_metadata_filtering()
