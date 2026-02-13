"""
Test NPC L1 Integration

Verifies that the NPC class correctly uses the L1 Working Memory Cache.
"""

import sys
import os
import time
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.npc import NPC, NPCPersonality
from csam_core.services.embedding import EmbeddingService

class MockLLMService:
    def is_available(self):
        return False

def test_npc_l1_integration():
    print("=" * 60)
    print("Testing NPC L1 Integration")
    print("=" * 60)
    
    # Setup
    embedding_service = EmbeddingService()
    personality = NPCPersonality(
        name="Tester",
        role="Unit Tester",
        greeting="Ready to test."
    )
    
    npc = NPC(
        personality=personality,
        embedding_service=embedding_service,
        llm_service=MockLLMService()
    )
    
    print("\n1. Verifying L1 initialization...")
    if hasattr(npc, 'working_memory'):
        print("  + npc.working_memory exists")
    else:
        print("  X npc.working_memory MISSING")
        return
        
    print("\n2. Sending messages (filling L1)...")
    player_name = "Debra"
    
    # Turn 1
    npc.respond("Hello there", player_name=player_name)
    print("  + Sent: 'Hello there'")
    
    # Check L1
    l1_items = npc.working_memory.get_recent(player_name)
    print(f"  L1 items count: {len(l1_items)} (Expected >= 2, prompt + response)")
    
    if len(l1_items) > 0:
        print(f"  Latest L1 item: {l1_items[0].text}")
    
    # Turn 2
    npc.respond("My name is Debra", player_name=player_name)
    print("  + Sent: 'My name is Debra'")
    
    # Verify Fact Extraction
    fact_name = npc.working_memory.get_fact(player_name, "player_name")
    print(f"  Extracted Fact (player_name): {fact_name}")
    
    if fact_name == "Debra":
        print("  + Fact extraction WORKING")
    else:
        print(f"  X Fact extraction FAILED (Got: {fact_name})")
        
    print("\n3. Verifying Context Retrieval includes L1...")
    # This calls retrieve_context internally
    context = npc.retrieve_context("What is my name?", player_name=player_name)
    
    print("  Context retrieved:")
    print("-" * 20)
    print(context)
    print("-" * 20)
    
    if "Recent Coversation:" in context and "Debra" in context:
        print("  + L1 Context injection WORKING")
    else:
        print("  X L1 Context injection FAILED")

    print("\n4. Stats check...")
    stats = npc.get_stats()
    print(f"  L1 Items in stats: {stats.get('l1_items')}")
    
    if stats.get('l1_items') > 0:
        print("  + Stats update WORKING")
    else:
        print("  X Stats update FAILED")

    print("\n" + "=" * 60)
    print("Integration Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_npc_l1_integration()
