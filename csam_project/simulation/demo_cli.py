"""
Interactive CLI Demo - Real-World NPC Simulation

This demo lets you:
1. Talk to multiple NPCs with persistent memory
2. Skip ahead N dialogues (fill with random conversations)
3. Test memory recall across long conversations
4. View NPC stats and memory contents

Commands:
- talk <npc> <message>   : Talk to an NPC
- skip <npc> <N>         : Skip N random dialogues with NPC
- remember <npc> <fact>  : Tell NPC something specific to remember
- recall <npc> <question>: Ask NPC about something
- stats [npc]            : Show NPC stats
- memories <npc> [N]     : Show last N memories
- consolidate [npc]      : Force consolidation
- quit                   : Exit demo
"""

import sys
import os
import time
import random
import argparse
from typing import Dict, List, Optional
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.npc import NPC, NPCPersonality, TAVERN_NPCS
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm import LLMService


# Random conversation templates for skip functionality
RANDOM_DIALOGUES = [
    "What's the weather like today?",
    "Have you heard any news?",
    "Tell me about yourself.",
    "What do you recommend?",
    "How's business been?",
    "Any interesting travelers lately?",
    "What's your favorite drink?",
    "Tell me a story.",
    "What do you think about the king?",
    "Have you seen anything strange?",
    "What's your opinion on magic?",
    "Do you know any good jokes?",
    "What's the best thing to eat here?",
    "How long have you been here?",
    "What brings you to this tavern?",
    "Any quests available?",
    "Tell me about the local area.",
    "What's your greatest fear?",
    "Do you believe in dragons?",
    "What's your favorite memory?",
]


class TavernDemo:
    """
    Interactive tavern demo with multiple NPCs.
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "llama3.2:3b",
        player_name: str = "Traveler"
    ):
        """
        Initialize the demo.
        
        Args:
            use_llm: Whether to use LLM for responses
            llm_model: Ollama model to use
            player_name: Player's name
        """
        self.player_name = player_name
        
        print("=" * 60)
        print("CSAM Tavern Demo - Initializing...")
        print("=" * 60)
        
        # Initialize shared services
        print("\nLoading embedding model...")
        self.embedding_service = EmbeddingService()
        _ = self.embedding_service.dimension  # Force load
        print(f"  [OK] Loaded (dim={self.embedding_service.dimension})")
        
        # Initialize LLM
        self.llm_service = None
        if use_llm:
            print("\nConnecting to Ollama...")
            self.llm_service = LLMService(model=llm_model)
            if self.llm_service.is_available():
                print(f"  [OK] Connected (model={llm_model})")
            else:
                print("  ✗ Ollama not available, using fallback responses")
                self.llm_service = None
        
        # Initialize NPCs
        print("\nLoading NPCs...")
        self.npcs: Dict[str, NPC] = {}
        for personality in TAVERN_NPCS:
            npc = NPC(
                personality=personality,
                embedding_service=self.embedding_service,
                llm_service=self.llm_service,
                max_memories=10000,
                forget_threshold=500
            )
            self.npcs[personality.name.lower()] = npc
            print(f"  [OK] {personality.name} ({personality.role})")
        
        print(f"\n[OK] Demo ready! {len(self.npcs)} NPCs loaded.")
    
    def get_npc(self, name: str) -> Optional[NPC]:
        """Get NPC by name (case insensitive)."""
        return self.npcs.get(name.lower())
    
    def list_npcs(self) -> List[str]:
        """List all NPC names."""
        return [npc.personality.name for npc in self.npcs.values()]
    
    def talk(self, npc_name: str, message: str) -> Dict:
        """Talk to an NPC."""
        npc = self.get_npc(npc_name)
        if not npc:
            return {"error": f"Unknown NPC: {npc_name}. Available: {self.list_npcs()}"}
        
        result = npc.respond(message, player_name=self.player_name)
        return result
    
    def skip_dialogues(self, npc_name: str, count: int, verbose: bool = True) -> Dict:
        """
        Skip ahead N dialogues with random conversations.
        
        This fills the NPC's memory with random interactions to test
        long-term memory recall. Uses fast mode: memories are stored 
        directly (embedding only) without LLM response generation,
        then consolidation runs once at the end.
        """
        npc = self.get_npc(npc_name)
        if not npc:
            return {"error": f"Unknown NPC: {npc_name}"}
        
        if verbose:
            print(f"\nSkipping {count} dialogues with {npc.personality.name}...")
            print(f"  (fast mode: embedding-only, no LLM calls per turn)")
        
        start_time = time.time()
        
        # Save the original consolidation trigger interval
        original_conv_count = npc.conversation_count
        
        for i in range(count):
            # Random dialogue -- store directly as memories (no LLM call)
            message = random.choice(RANDOM_DIALOGUES)
            metadata = {
                "player_name": self.player_name,
                "npc_name": npc.personality.name,
                "timestamp": datetime.now().isoformat()
            }
            # Store player message
            npc.add_memory(
                f"{self.player_name} said: {message}",
                importance=0.5,
                metadata=metadata
            )
            # Store a simple NPC acknowledgment (no LLM generation)
            npc.add_memory(
                f"I ({npc.personality.name}) responded to: {message[:60]}",
                importance=0.3,
                metadata=metadata
            )
            npc.conversation_count += 1
            
            # Progress indicator
            if verbose and (i + 1) % 10 == 0:
                elapsed_so_far = time.time() - start_time
                print(f"  ... {i + 1}/{count} dialogues ({elapsed_so_far:.1f}s)")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  Memory fill done in {elapsed:.1f}s. Running consolidation...")
        
        # Use fast consolidation: no LLM calls, larger batches
        # Save original settings
        orig_use_llm = npc.consolidation_pipeline.use_llm_for_consolidation
        orig_max_batch = npc.consolidation_pipeline.max_memories_per_batch
        npc.consolidation_pipeline.use_llm_for_consolidation = False
        npc.consolidation_pipeline.max_memories_per_batch = 50
        
        consolidation_result = npc.run_consolidation()
        
        # Restore original settings for future real conversations
        npc.consolidation_pipeline.use_llm_for_consolidation = orig_use_llm
        npc.consolidation_pipeline.max_memories_per_batch = orig_max_batch
        
        stats = npc.get_stats()
        
        if verbose:
            print(f"  [OK] Completed in {elapsed:.1f}s")
            print(f"  Memories: {stats['memory_count']}, Consolidated: {stats['consolidation_ratio']:.1%}")
        
        return {
            "dialogues_added": count,
            "elapsed_seconds": elapsed,
            "memory_count": stats['memory_count'],
            "consolidation_result": consolidation_result
        }
    
    def remember(self, npc_name: str, fact: str) -> Dict:
        """
        Tell an NPC something specific to remember.
        
        This stores the fact with high importance so it's less likely to be forgotten.
        """
        npc = self.get_npc(npc_name)
        if not npc:
            return {"error": f"Unknown NPC: {npc_name}"}
        
        metadata = {
            "player_name": self.player_name,
            "npc_name": npc.personality.name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store the fact itself with very high importance
        full_message = f"{self.player_name} said: {fact}"
        npc.add_memory(full_message, importance=0.95, metadata=metadata)
        
        # Also store the NPC acknowledgment
        acknowledgment = f"I will remember that {fact}"
        npc.add_memory(f"I ({npc.personality.name}) thought: {acknowledgment}", importance=0.9, metadata=metadata)
        
        return {
            "stored": True,
            "fact": fact,
            "npc": npc.personality.name,
            "memory_count": len(npc.memory_repo)
        }
    
    def recall(self, npc_name: str, question: str) -> Dict:
        """
        Ask an NPC about something they should remember.
        
        Uses QA mode: direct L2 retrieval (k=20), strict QA prompt,
        no MMR diversity penalty, no L1 noise. Does NOT save the 
        question as a memory (prevents recall pollution).
        This matches the benchmark pipeline that achieves published results.
        """
        npc = self.get_npc(npc_name)
        if not npc:
            return {"error": f"Unknown NPC: {npc_name}. Available: {self.list_npcs()}"}
        
        start_time = time.time()
        
        # Use QA mode for accurate retrieval (matches benchmark pipeline)
        context = npc.retrieve_context(question, k=20, player_name=self.player_name, mode="qa")
        
        # Generate response with strict QA prompt (no persona fluff)
        if npc.llm_service and npc.llm_service.is_available():
            response = npc.llm_service.generate_response(
                context=context,
                user_message=question,
                persona=None,  # No persona -- strict QA
                mode="qa"
            )
        else:
            response = npc._generate_fallback_response(question, context)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Do NOT save the question/answer as memory (prevents pollution)
        return {
            "response": response,
            "latency_ms": elapsed_ms,
            "context_used": context,
            "mode": "qa"
        }
    
    def get_stats(self, npc_name: Optional[str] = None) -> Dict:
        """Get stats for one or all NPCs."""
        if npc_name:
            npc = self.get_npc(npc_name)
            if not npc:
                return {"error": f"Unknown NPC: {npc_name}"}
            return npc.get_stats()
        else:
            return {name: npc.get_stats() for name, npc in self.npcs.items()}
    
    def get_memories(self, npc_name: str, count: int = 10) -> List[Dict]:
        """Get recent memories from an NPC."""
        npc = self.get_npc(npc_name)
        if not npc:
            return []
        
        memories = npc.memory_repo.get_all()
        # Sort by timestamp, newest first
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        
        return [
            {
                "text": m.text[:100] + "..." if len(m.text) > 100 else m.text,
                "importance": m.importance,
                "consolidated": m.consolidated,
                "timestamp": m.timestamp.strftime("%H:%M:%S")
            }
            for m in memories[:count]
        ]
    
    def run_interactive(self):
        """Run the interactive CLI."""
        print("\n" + "=" * 60)
        print("Welcome to the Golden Dragon Tavern!")
        print("=" * 60)
        print(f"\nYou are: {self.player_name}")
        print(f"Available NPCs: {', '.join(self.list_npcs())}")
        print("\nCommands:")
        print("  talk <npc> <message>    - Talk to an NPC")
        print("  skip <npc> <N>          - Skip N random dialogues")
        print("  remember <npc> <fact>   - Store a fact with high importance")
        print("  recall <npc> <question> - Ask about something")
        print("  stats [npc]             - Show stats")
        print("  memories <npc> [N]      - Show last N memories")
        print("  name <your_name>        - Set your name")
        print("  help                    - Show this help")
        print("  quit                    - Exit")
        print()
        
        while True:
            try:
                cmd = input(f"\n[{self.player_name}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not cmd:
                continue
            
            parts = cmd.split(maxsplit=2)
            action = parts[0].lower()
            
            # ================== TALK ==================
            if action == "talk" and len(parts) >= 3:
                npc_name = parts[1]
                message = parts[2]
                
                result = self.talk(npc_name, message)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"\n{npc_name.title()}: {result['response']}")
                    print(f"  [Latency: {result['latency_ms']:.0f}ms | Memories: {result['memory_count']}]")
            
            # ================== SKIP ==================
            elif action == "skip" and len(parts) >= 3:
                npc_name = parts[1]
                try:
                    count = int(parts[2])
                    self.skip_dialogues(npc_name, count)
                except ValueError:
                    print("Error: Count must be a number")
            
            # ================== REMEMBER ==================
            elif action == "remember" and len(parts) >= 3:
                npc_name = parts[1]
                fact = parts[2]
                
                result = self.remember(npc_name, fact)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"[OK] Stored fact with {result['npc']} (high importance)")
                    print(f"  \"{result['fact']}\"")
            
            # ================== RECALL ==================
            elif action == "recall" and len(parts) >= 3:
                npc_name = parts[1]
                question = parts[2]
                
                result = self.recall(npc_name, question)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"\n{npc_name.title()}: {result['response']}")
                    print(f"  [Latency: {result['latency_ms']:.0f}ms]")
                    if result.get('context_used') and result['context_used'] != "No relevant memories.":
                        print(f"  [Context retrieved:]")
                        for line in result['context_used'].split('\n')[:3]:
                            print(f"    {line}")
            
            # ================== STATS ==================
            elif action == "stats":
                npc_name = parts[1] if len(parts) > 1 else None
                stats = self.get_stats(npc_name)
                
                if "error" in stats:
                    print(f"Error: {stats['error']}")
                elif npc_name:
                    print(f"\n{stats['name']} ({stats['role']}):")
                    print(f"  Memories: {stats['memory_count']}")
                    print(f"  Consolidated: {stats['consolidation_ratio']:.1%}")
                    print(f"  L3 Nodes: {stats['l3_nodes']}")
                    print(f"  Conversations: {stats['conversation_count']}")
                    print(f"  Avg Response Time: {stats['avg_response_time_ms']:.0f}ms")
                else:
                    print("\nNPC Statistics:")
                    print("-" * 50)
                    for name, s in stats.items():
                        print(f"  {s['name']:12} | Mem: {s['memory_count']:4} | "
                              f"Consol: {s['consolidation_ratio']:4.0%} | "
                              f"L3: {s['l3_nodes']:3}")
            
            # ================== MEMORIES ==================
            elif action == "memories" and len(parts) >= 2:
                npc_name = parts[1]
                count = int(parts[2]) if len(parts) > 2 else 10
                
                memories = self.get_memories(npc_name, count)
                if not memories:
                    print(f"No memories found for {npc_name}")
                else:
                    print(f"\nRecent memories ({npc_name}):")
                    for i, m in enumerate(memories, 1):
                        status = "[OK]" if m['consolidated'] else " "
                        print(f"  {i}. [{status}] [{m['importance']:.1f}] {m['text']}")
            
            # ================== NAME ==================
            elif action == "name" and len(parts) >= 2:
                self.player_name = parts[1]
                print(f"Your name is now: {self.player_name}")
            
            # ================== CONSOLIDATE ==================
            elif action == "consolidate":
                npc_name = parts[1] if len(parts) > 1 else None
                
                if npc_name:
                    npc = self.get_npc(npc_name)
                    if npc:
                        result = npc.run_consolidation()
                        print(f"Consolidation result: {result}")
                    else:
                        print(f"Unknown NPC: {npc_name}")
                else:
                    for name, npc in self.npcs.items():
                        result = npc.run_consolidation()
                        print(f"{name}: {result.get('status', 'done')}")
            
            # ================== HELP ==================
            elif action == "help":
                print("\nCommands:")
                print("  talk <npc> <message>    - Talk to an NPC")
                print("  skip <npc> <N>          - Skip N random dialogues")
                print("  remember <npc> <fact>   - Store a fact (high importance)")
                print("  recall <npc> <question> - Ask about something")
                print("  stats [npc]             - Show stats")
                print("  memories <npc> [N]      - Show last N memories")
                print("  consolidate [npc]       - Force consolidation")
                print("  name <your_name>        - Set your name")
                print("  quit                    - Exit")
                print(f"\nAvailable NPCs: {', '.join(self.list_npcs())}")
            
            # ================== QUIT ==================
            elif action in ["quit", "exit", "q"]:
                print("\nGoodbye, traveler!")
                break
            
            # ================== DEMO ==================
            elif action == "demo":
                self._run_demo_sequence()
            
            else:
                print(f"Unknown command: {action}. Type 'help' for commands.")
    
    def _run_demo_sequence(self):
        """Run a pre-defined demo sequence showing memory capabilities."""
        print("\n" + "=" * 60)
        print("Running Demo Sequence: Memory Persistence Test")
        print("=" * 60)
        
        npc_name = "greta"
        
        # Step 1: Introduce ourselves and share a secret
        print("\n[STEP 1] Introducing ourselves with a unique fact...")
        secret = "my mother's name is Celestia and she was a famous wizard"
        
        result = self.talk(npc_name, f"Hello! I'm {self.player_name}. I want to tell you something - {secret}")
        print(f"Greta: {result['response']}")
        print(f"  [Memories: {result['memory_count']}]")
        
        # Step 2: Skip 100 dialogues
        print("\n[STEP 2] Simulating 100 random conversations...")
        self.skip_dialogues(npc_name, 100)
        
        # Step 3: Ask about the secret
        print("\n[STEP 3] Testing recall of the original fact...")
        result = self.talk(npc_name, "Do you remember what I told you about my mother?")
        print(f"Greta: {result['response']}")
        print(f"  [Context used: {result['context_used'][:200]}...]")
        
        # Step 4: Show stats
        print("\n[STEP 4] Final Statistics:")
        stats = self.get_stats(npc_name)
        print(f"  Memories: {stats['memory_count']}")
        print(f"  Consolidated: {stats['consolidation_ratio']:.1%}")
        print(f"  L3 Knowledge Nodes: {stats['l3_nodes']}")
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CSAM Tavern Demo")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model")
    parser.add_argument("--name", type=str, default="Traveler", help="Player name")
    parser.add_argument("--demo", action="store_true", help="Run demo sequence")
    
    args = parser.parse_args()
    
    demo = TavernDemo(
        use_llm=not args.no_llm,
        llm_model=args.model,
        player_name=args.name
    )
    
    if args.demo:
        demo._run_demo_sequence()
    else:
        demo.run_interactive()


if __name__ == "__main__":
    main()
