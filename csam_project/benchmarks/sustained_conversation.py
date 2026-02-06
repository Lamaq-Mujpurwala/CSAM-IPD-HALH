"""
1-on-1 Sustained Conversation Benchmark

Tests sustained conversations between pairs of NPCs over X turns.
Each pair maintains their conversation for the full duration.

Usage:
    python benchmarks/sustained_conversation.py --pairs 5 --turns 100
"""

import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.multi_agent_orchestrator import Player
from simulation.npc import NPCPersonality
from csam_core.services.embedding_gpu import AsyncGPUEmbeddingService
from csam_core.services.llm_async import AsyncLLMService
from csam_core.memory_repository_gpu import FAISSGPUMemoryRepository
from csam_core.memory_repository_persistent import PersistentFAISSRepository  # Persistent FAISS
from csam_core.working_memory import WorkingMemoryCache  # L1 Working Memory
from csam_core.knowledge_graph import KnowledgeGraph
from csam_core.retrieval import HybridRetriever
from csam_core.forgetting_engine import ConsolidationAwareForgetting  # ⭐ Novel forgetting!
import shutil  # For cleanup


@dataclass
class ConversationMetrics:
    """Metrics for one sustained conversation."""
    pair_id: int
    npc_a_name: str
    npc_b_name: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory system metrics (excluding LLM)
    memory_latencies_ms: List[float] = field(default_factory=list)  # Embed + Retrieve + Store
    embedding_latencies_ms: List[float] = field(default_factory=list)
    retrieval_latencies_ms: List[float] = field(default_factory=list)
    storage_latencies_ms: List[float] = field(default_factory=list)
    
    # LLM metrics (separate)
    llm_latencies_ms: List[float] = field(default_factory=list)
    
    # Memory growth
    memory_growth_a: List[int] = field(default_factory=list)
    memory_growth_b: List[int] = field(default_factory=list)
    
    # Forgetting events
    forgetting_events_a: List[int] = field(default_factory=list)  # Turn numbers when forgetting occurred
    forgetting_events_b: List[int] = field(default_factory=list)


class AsyncNPC:
    """Async-enabled NPC for sustained conversations."""
    
    def __init__(
        self,
        personality: NPCPersonality,
        embedding_service: AsyncGPUEmbeddingService,
        llm_service: AsyncLLMService,
        use_gpu: bool = False,
        npc_id: str = "npc"  # For unique storage paths
    ):
        self.personality = personality
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.npc_id = npc_id
        
        # Initialize THREE-LAYER memory architecture
        embedding_dim = 384  # MiniLM
        
        # Create storage directory
        storage_dir = Path("sustained_temp")
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # L1: Working Memory (fastest, recent cache)
        self.working_memory = WorkingMemoryCache(
            max_size=20,
            enable_facts=True
        )
        
        # L2: Vector Memory (episodic, FAISS) - WITH PERSISTENCE
        # Use ConsolidationAwareForgetting for sophisticated forgetting
        forgetting_strategy = ConsolidationAwareForgetting(
            alpha=0.2,  # Recency weight
            beta=0.2,   # Inverse importance weight
            gamma=0.3,  # Consolidation coverage weight ⭐
            delta=0.3   # Redundancy with L3 weight ⭐
        )
        
        self.memory_repo = PersistentFAISSRepository(
            embedding_dim=embedding_dim,
            max_memories=500,  
            use_gpu=use_gpu,
            index_type="HNSW32",
            index_path=f"sustained_temp/{npc_id}_index.faiss",
            metadata_path=f"sustained_temp/{npc_id}_metadata.pkl",
            auto_save_interval=100,  # Save every 100 additions
            forgetting_strategy=forgetting_strategy  # ⭐ Pass the strategy!
        )
        
        # L3: Knowledge Graph (semantic relationships) - WITH PERSISTENCE
        self.knowledge_graph = KnowledgeGraph(
            embedding_dim=embedding_dim,
            db_path=f"sustained_temp/{npc_id}_kg.db"  # Disk-based SQLite
        )
        
        # Hybrid retrieval (L2 + L3)
        self.retriever = HybridRetriever(
            memory_repository=self.memory_repo,
            knowledge_graph=self.knowledge_graph
        )
        
        self.total_conversations = 0
    
    async def respond(self, message: str, speaker_name: str = "Partner") -> Dict[str, Any]:
        """Generate async response with 3-layer memory."""
        timings = {}
        
        # 0. Check L1 first (fastest)
        t0 = time.time()
        l1_items = self.working_memory.get_recent(speaker_name, k=3)
        timings['l1_lookup'] = (time.time() - t0) * 1000
        
        # 1. Encode message
        t0 = time.time()
        message_embedding = await self.embedding_service.encode(message)
        timings['embed_query'] = (time.time() - t0) * 1000
        
        # 2. HYBRID RETRIEVAL: L2 (episodic) + L3 (semantic graph)
        # Now with proper metadata filtering for L2!
        t0 = time.time()
        retrieval_result = await self.retriever.retrieve(
            query_embedding=message_embedding,
            k=3,  # Total results from both L2 and L3
            metadata_filter={"player_name": speaker_name}  # ⭐ Proper L2 filtering!
        )
        timings['retrieve'] = (time.time() - t0) * 1000
        
        # Extract text from hybrid results (L2 Memory + L3 Node)
        # L2 is already filtered by player_name, L3 is shared knowledge
        l2_context = []
        l3_context = []
        
        for item, score in retrieval_result.final_results:
            # Check if it's a Memory (L2) or L3Node
            if hasattr(item, 'text'):  # Memory from L2
                l2_context.append(f"- {item.text}")
            elif hasattr(item, 'content'):  # L3Node (Knowledge)
                l3_context.append(f"- [Knowledge] {item.content}")
        
        # Combine L1 with L2+L3 context
        l1_context_str = [f"- {item.text}" for item in l1_items]
        l2_l3_context = l2_context + l3_context
        
        # 3. Generate response with separated L1 / (L2+L3) context
        # L1 → System Prompt (recent working memory)
        enhanced_system = f"""{self.personality.system_prompt}

RECENT WORKING MEMORY:
{chr(10).join(l1_context_str) if l1_context_str else "- No recent interactions"}

Use this to maintain continuity."""
        
        # L2+L3 → User Message (long-term memories + knowledge)
        context_section = f"""
Relevant memories and knowledge:
{chr(10).join(l2_l3_context)}

Current message: {message}""" if l2_l3_context else message
        
        t0 = time.time()
        response = await self.llm_service.generate_response(
            context_section,
            message,
            persona=enhanced_system  # L1 in system prompt!
        )
        timings['llm'] = (time.time() - t0) * 1000
        
        if not response:
            response = f"(As {self.personality.name}) That's interesting, {speaker_name}!"
        
        # 4. Store interaction
        interaction_text = f"{speaker_name}: {message}\n{self.personality.name}: {response}"
        
        t0 = time.time()
        interaction_embedding = await self.embedding_service.encode(interaction_text)
        timings['embed_store'] = (time.time() - t0) * 1000
        
        
        t0 = time.time()
        prev_count = len(self.memory_repo)
        
        # Store with metadata for player-scoped retrieval
        metadata = {
            "player_name": speaker_name,
            "npc_name": self.personality.name,
            "timestamp": time.time()
        }
        
        # Store in L1
        self.working_memory.add(
            text=interaction_text,
            player_name=speaker_name,
            metadata=metadata
        )
        
        # Store in L2
        self.memory_repo.add(
            text=interaction_text,
            embedding=interaction_embedding,
            importance=0.5,
            metadata=metadata
        )
        new_count = len(self.memory_repo)
        timings['store'] = (time.time() - t0) * 1000
        timings['forgetting_triggered'] = new_count < prev_count + 1
        
        self.total_conversations += 1
        
        # Total memory system time (exclude LLM)
        memory_system_time = (timings.get('l1_lookup', 0) + timings['embed_query'] + 
                             timings['retrieve'] + timings['embed_store'] + timings['store'])
        
        return {
            "response": response,
            "memory_count": len(self.memory_repo),
            "npc_name": self.personality.name,
            "timings": timings,
            "memory_system_ms": memory_system_time
        }


class SustainedConversationBenchmark:
    """Benchmark sustained 1-on-1 NPC conversations."""
    
    def __init__(self, results_dir: str = "benchmark_results/sustained"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def create_npc(
        self,
        name: str,
        role: str,
        embedding_service: AsyncGPUEmbeddingService,
        llm_service: AsyncLLMService,
        use_gpu: bool = False
    ) -> AsyncNPC:
        """Create a single NPC."""
        personality = NPCPersonality(
            name=name,
            role=role,
            traits=["conversational", "memorable"],
            background=f"I am {name}, a {role}",
            speaking_style="engaging and thoughtful",
            greeting=f"Hello, I'm {name}!"
        )
        
        return AsyncNPC(
            personality, 
            embedding_service, 
            llm_service, 
            use_gpu, 
            npc_id=name.lower().replace(" ", "_")  # Unique ID for storage
        )
    
    async def run_sustained_conversation(
        self,
        npc_a: AsyncNPC,
        npc_b: AsyncNPC,
        turns: int,
        pair_id: int,
        verbose: bool = True
    ) -> ConversationMetrics:
        """Run a sustained conversation between two NPCs."""
        metrics = ConversationMetrics(
            pair_id=pair_id,
            npc_a_name=npc_a.personality.name,
            npc_b_name=npc_b.personality.name
        )
        
        if verbose:
            print(f"\n[Pair {pair_id}] {npc_a.personality.name} ↔ {npc_b.personality.name}")
        
        # Conversation starters
        topics = [
            "What's your favorite memory?",
            "Tell me about your day.",
            "What do you think about time?",
            "Have we talked before?",
            "What interests you most?",
        ]
        
        current_message = topics[pair_id % len(topics)]
        start_time = time.time()
        
        for turn in range(turns):
            # A responds to B's message (or starter)
            result_a = await npc_a.respond(current_message, npc_b.personality.name)
            
            # B responds to A's response
            result_b = await npc_b.respond(result_a["response"], npc_a.personality.name)
            
            # Track memory system metrics (exclude LLM)
            mem_time_a = result_a['memory_system_ms']
            mem_time_b = result_b['memory_system_ms']
            avg_memory_time = (mem_time_a + mem_time_b) / 2
            
            metrics.memory_latencies_ms.append(avg_memory_time)
            metrics.embedding_latencies_ms.append(
                (result_a['timings']['embed_query'] + result_a['timings']['embed_store'] +
                 result_b['timings']['embed_query'] + result_b['timings']['embed_store']) / 2
            )
            metrics.retrieval_latencies_ms.append(
                (result_a['timings']['retrieve'] + result_b['timings']['retrieve']) / 2
            )
            metrics.storage_latencies_ms.append(
                (result_a['timings']['store'] + result_b['timings']['store']) / 2
            )
            metrics.llm_latencies_ms.append(
                (result_a['timings']['llm'] + result_b['timings']['llm']) / 2
            )
            
            # Track memory growth and forgetting
            prev_a = metrics.memory_growth_a[-1] if metrics.memory_growth_a else 0
            prev_b = metrics.memory_growth_b[-1] if metrics.memory_growth_b else 0
            
            metrics.memory_growth_a.append(result_a["memory_count"])
            metrics.memory_growth_b.append(result_b["memory_count"])
            
            # Detect forgetting events
            if result_a['timings']['forgetting_triggered']:
                metrics.forgetting_events_a.append(turn)
            if result_b['timings']['forgetting_triggered']:
                metrics.forgetting_events_b.append(turn)
            
            # B's response becomes next message for A
            current_message = result_b["response"]
            
            if verbose and (turn + 1) % 5 == 0:  # Print every 5 turns
                print(f"  Turn {turn + 1}/{turns}: MemSys={avg_memory_time:.0f}ms, "
                      f"LLM={(result_a['timings']['llm'] + result_b['timings']['llm'])/2:.0f}ms, "
                      f"Mem A:{result_a['memory_count']}, B:{result_b['memory_count']}")
        
        total_time = (time.time() - start_time) * 1000
        
        if verbose:
            avg_mem = np.mean(metrics.memory_latencies_ms)
            avg_llm = np.mean(metrics.llm_latencies_ms)
            print(f"  ✓ Completed {turns} turns in {total_time/1000:.1f}s")
            print(f"    Avg Memory System: {avg_mem:.0f}ms, Avg LLM: {avg_llm:.0f}ms")
        
        return metrics
    
    async def run_multi_pair_benchmark(
        self,
        num_pairs: int,
        turns_per_pair: int,
        use_gpu: bool = False,
        verbose: bool = True
    ) -> List[ConversationMetrics]:
        """Run multiple sustained conversations in parallel."""
        print(f"\n{'='*60}")
        print(f"Sustained Conversation Benchmark")
        print(f"  Pairs: {num_pairs}")
        print(f"  Turns per pair: {turns_per_pair}")
        print(f"  Device: {'cuda' if use_gpu else 'cpu'}")
        print(f"{'='*60}")
        
        # Shared services
        embedding_service = AsyncGPUEmbeddingService(
            model_name="all-MiniLM-L6-v2",
            device="cuda" if use_gpu else "cpu"
        )
        
        llm_service = AsyncLLMService(
            model="llama3.2:1b",
            max_concurrent=num_pairs
        )
        
        # Warmup
        print("\nWarming up services...")
        await embedding_service.warmup()
        
        # Create NPC pairs
        print(f"\nCreating {num_pairs} NPC pairs...")
        pairs = []
        for i in range(num_pairs):
            npc_a = await self.create_npc(
                f"NPC-{i}A", f"role-{i}a", 
                embedding_service, llm_service, use_gpu
            )
            npc_b = await self.create_npc(
                f"NPC-{i}B", f"role-{i}b",
                embedding_service, llm_service, use_gpu
            )
            pairs.append((npc_a, npc_b))
        
        # Run all conversations concurrently
        print(f"\nRunning {num_pairs} concurrent conversations...")
        tasks = [
            self.run_sustained_conversation(a, b, turns_per_pair, i, verbose=verbose)
            for i, (a, b) in enumerate(pairs)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return results
    
    def plot_results(
        self,
        results: List[ConversationMetrics],
        save_path: str = None
    ):
        """Generate publication-quality research visualization."""
        # Create figure with 12 panels (4x3 grid)
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Color palette for pairs
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        # ===== ROW 1: PAIR-WISE MEMORY GROWTH =====
        
        # Plot 1: Individual Pair Memory Trajectories
        ax1 = fig.add_subplot(gs[0, :2])
        for idx, r in enumerate(results):
            turns = list(range(len(r.memory_growth_a)))
            ax1.plot(turns, r.memory_growth_a, color=colors[idx], linewidth=2,
                    label=f"Pair {r.pair_id}", marker='o', markersize=3, markevery=max(1, len(turns)//10))
            
            # Mark forgetting events
            for event_turn in r.forgetting_events_a:
                ax1.axvline(event_turn, color=colors[idx], alpha=0.2, linestyle='--', linewidth=1)
        
        ax1.set_xlabel('Conversation Turn', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Memory Count', fontsize=12, fontweight='bold')
        ax1.set_title('A. Memory Growth Trajectories by Pair (with Forgetting Events)', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=9, ncol=min(3, len(results)), loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # Plot 2: Memory Growth Rate (derivatives)
        ax2 = fig.add_subplot(gs[0, 2])
        for idx, r in enumerate(results):
            if len(r.memory_growth_a) > 1:
                growth_rate = np.diff(r.memory_growth_a)
                ax2.plot(growth_rate, color=colors[idx], alpha=0.7, linewidth=1.5)
        ax2.axhline(1, color='black', linestyle='--', alpha=0.5, label='No forgetting')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5, label='Forgetting threshold')
        ax2.set_xlabel('Turn', fontsize=11)
        ax2.set_ylabel('Δ Memories/Turn', fontsize=11)
        ax2.set_title('B. Memory Growth Rate', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ===== ROW 2: MEMORY SYSTEM PERFORMANCE (NO LLM) =====
        
        # Plot 3: Memory System Latency Over Time
        ax3 = fig.add_subplot(gs[1, 0])
        for idx, r in enumerate(results):
            turns = list(range(len(r.memory_latencies_ms)))
            ax3.plot(turns, r.memory_latencies_ms, color=colors[idx], alpha=0.6, linewidth=1.5)
        avg_mem_lat = np.mean([lat for r in results for lat in r.memory_latencies_ms])
        ax3.axhline(avg_mem_lat, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {avg_mem_lat:.1f}ms')
        ax3.set_xlabel('Turn', fontsize=11)
        ax3.set_ylabel('Latency (ms)', fontsize=11)
        ax3.set_title('C. Memory System Latency\n(Embed + Retrieve + Store)', 
                     fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Component Breakdown (stacked area)
        ax4 = fig.add_subplot(gs[1, 1])
        # Average across all pairs for each turn
        max_turns = max(len(r.embedding_latencies_ms) for r in results)
        avg_embed = np.zeros(max_turns)
        avg_retrieve = np.zeros(max_turns)
        avg_store = np.zeros(max_turns)
        counts = np.zeros(max_turns)
        
        for r in results:
            for i, (e, ret, s) in enumerate(zip(r.embedding_latencies_ms, 
                                                r.retrieval_latencies_ms,
                                                r.storage_latencies_ms)):
                avg_embed[i] += e
                avg_retrieve[i] += ret
                avg_store[i] += s
                counts[i] += 1
        
        avg_embed = avg_embed / np.maximum(counts, 1)
        avg_retrieve = avg_retrieve / np.maximum(counts, 1)
        avg_store = avg_store / np.maximum(counts, 1)
        
        turns = list(range(max_turns))
        ax4.fill_between(turns, 0, avg_embed, alpha=0.6, label='Embedding', color='#1f77b4')
        ax4.fill_between(turns, avg_embed, avg_embed + avg_retrieve, 
                        alpha=0.6, label='Retrieval', color='#ff7f0e')
        ax4.fill_between(turns, avg_embed + avg_retrieve, 
                        avg_embed + avg_retrieve + avg_store,
                        alpha=0.6, label='Storage', color='#2ca02c')
        ax4.set_xlabel('Turn', fontsize=11)
        ax4.set_ylabel('Latency (ms)', fontsize=11)
        ax4.set_title('D. Memory System Component Breakdown\n(Averaged Across Pairs)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Memory vs LLM Latency Comparison
        ax5 = fig.add_subplot(gs[1, 2])
        all_mem_lats = [lat for r in results for lat in r.memory_latencies_ms]
        all_llm_lats = [lat for r in results for lat in r.llm_latencies_ms]
        
        data_to_plot = [all_mem_lats, all_llm_lats]
        bp = ax5.boxplot(data_to_plot, tick_labels=['Memory\nSystem', 'LLM'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        ax5.set_ylabel('Latency (ms)', fontsize=11)
        ax5.set_title('E. Memory System vs LLM Latency', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # ===== ROW 3: STATISTICAL ANALYSIS =====
        
        # Plot 6: Latency Distribution (Histogram)
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(all_mem_lats, bins=40, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
        p50 = np.percentile(all_mem_lats, 50)
        p95 = np.percentile(all_mem_lats, 95)
        p99 = np.percentile(all_mem_lats, 99)
        ax6.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.0f}ms')
        ax6.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95:.0f}ms')
        ax6.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'P99: {p99:.0f}ms')
        ax6.set_xlabel('Memory System Latency (ms)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('F. Memory System Latency Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 7: Pair-wise Performance Comparison (Bar chart)
        ax7 = fig.add_subplot(gs[2, 1])
        pair_ids = [r.pair_id for r in results]
        avg_mem_by_pair = [np.mean(r.memory_latencies_ms) for r in results]
        bars = ax7.bar(pair_ids, avg_mem_by_pair, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax7.axhline(np.mean(avg_mem_by_pair), color='red', linestyle='--', linewidth=2,
                   label=f'Overall Mean: {np.mean(avg_mem_by_pair):.0f}ms')
        ax7.set_xlabel('Pair ID', fontsize=11)
        ax7.set_ylabel('Avg Memory Latency (ms)', fontsize=11)
        ax7.set_title('G. Average Memory System Performance by Pair', 
                     fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_xticks(pair_ids)
        
        # Plot 8: Forgetting Events Timeline
        ax8 = fig.add_subplot(gs[2, 2])
        for idx, r in enumerate(results):
            events_a = r.forgetting_events_a
            events_b = r.forgetting_events_b
            if events_a or events_b:
                ax8.scatter([r.pair_id] * len(events_a), events_a, 
                          color=colors[idx], marker='o', s=60, alpha=0.7)
                ax8.scatter([r.pair_id] * len(events_b), events_b,
                          color=colors[idx], marker='x', s=60, alpha=0.7)
        ax8.set_xlabel('Pair ID', fontsize=11)
        ax8.set_ylabel('Turn Number', fontsize=11)
        ax8.set_title('H. Forgetting Events Timeline\n(○ = NPC A, × = NPC B)', 
                     fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.set_xticks(pair_ids)
        
        # ===== ROW 4: SCALING & SUMMARY =====
        
        # Plot 9: Retrieval Latency Scaling vs Memory Size (like image-2.png)
        ax9 = fig.add_subplot(gs[3, 0])
        
        # Collect all (memory_count, retrieval_latency) pairs
        memory_counts = []
        retrieval_lats = []
        for r in results:
            for i, (mem_count, ret_lat) in enumerate(zip(r.memory_growth_a, r.retrieval_latencies_ms)):
                memory_counts.append(mem_count)
                retrieval_lats.append(ret_lat)
        
        # Scatter plot
        ax9.scatter(memory_counts, retrieval_lats, alpha=0.4, s=20, color='#9b59b6', edgecolors='none')
        
        # Calculate average latency at different memory sizes for trend line
        unique_sizes = sorted(set(memory_counts))
        avg_lats_by_size = []
        p99_lats_by_size = []
        for size in unique_sizes:
            lats_at_size = [lat for mc, lat in zip(memory_counts, retrieval_lats) if mc == size]
            if lats_at_size:
                avg_lats_by_size.append(np.mean(lats_at_size))
                p99_lats_by_size.append(np.percentile(lats_at_size, 99))
        
        # Plot trend lines
        ax9.plot(unique_sizes, avg_lats_by_size, 'o-', color='#3498db', linewidth=2.5, 
                markersize=8, label='Avg Latency', markeredgecolor='white', markeredgewidth=1)
        ax9.plot(unique_sizes, p99_lats_by_size, 's--', color='#e74c3c', linewidth=2, 
                markersize=7, label='P99 Latency', markeredgecolor='white', markeredgewidth=1)
        
        # Add annotation for scalability
        if len(unique_sizes) > 1:
            ax9.annotate('Scalable O(log N) Trend', 
                        xy=(unique_sizes[len(unique_sizes)//2], avg_lats_by_size[len(avg_lats_by_size)//2]),
                        xytext=(unique_sizes[len(unique_sizes)//2] + 2, max(avg_lats_by_size) * 1.15),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        fontsize=10, fontweight='bold')
        
        ax9.set_xlabel('Number of Memories', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Retrieval Latency (ms)', fontsize=12, fontweight='bold')
        ax9.set_title('I. Retrieval Latency Scaling vs. Memory Size', 
                     fontsize=13, fontweight='bold', pad=10)
        ax9.legend(fontsize=10, loc='upper left')
        ax9.grid(True, alpha=0.3, linestyle=':')
        
        # Add value labels on trend line
        for i, (size, lat) in enumerate(zip(unique_sizes[::max(1, len(unique_sizes)//3)], 
                                            avg_lats_by_size[::max(1, len(avg_lats_by_size)//3)])):
            ax9.text(size, lat + max(avg_lats_by_size) * 0.05, f'{lat:.2f}', 
                    ha='center', fontsize=9, fontweight='bold')
        
        # Plot 10: Final Memory Counts (Paired bars)
        ax10 = fig.add_subplot(gs[3, 1])
        final_a = [r.memory_growth_a[-1] if r.memory_growth_a else 0 for r in results]
        final_b = [r.memory_growth_b[-1] if r.memory_growth_b else 0 for r in results]
        x = np.arange(len(results))
        width = 0.35
        ax10.bar(x - width/2, final_a, width, label='NPC A', alpha=0.8, 
                edgecolor='black', linewidth=1, color='#3498db')
        ax10.bar(x + width/2, final_b, width, label='NPC B', alpha=0.8, 
                edgecolor='black', linewidth=1, color='#e74c3c')
        ax10.set_xlabel('Pair ID', fontsize=11, fontweight='bold')
        ax10.set_ylabel('Final Memory Count', fontsize=11, fontweight='bold')
        ax10.set_title('J. Final Memory Usage Comparison', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=9)
        ax10.grid(True, alpha=0.3, axis='y')
        ax10.set_xticks(x)
        ax10.set_xticklabels(pair_ids)
        
        # Plot 11: Performance Summary Table
        ax11 = fig.add_subplot(gs[3, 2])
        ax11.axis('tight')
        ax11.axis('off')
        
        # Calculate stats
        total_turns = sum(len(r.memory_latencies_ms) for r in results)
        total_forgetting = sum(len(r.forgetting_events_a) + len(r.forgetting_events_b) for r in results)
        
        stats_data = [
            ['Metric', 'Value'],
            ['─' * 30, '─' * 20],
            ['Total Pairs', f'{len(results)}'],
            ['Turns per Pair', f'{len(results[0].memory_latencies_ms) if results else 0}'],
            ['Total Turns', f'{total_turns}'],
            ['', ''],
            ['Memory System Latency', ''],
            ['  Mean', f'{np.mean(all_mem_lats):.1f} ms'],
            ['  Median', f'{np.median(all_mem_lats):.1f} ms'],
            ['  P95', f'{np.percentile(all_mem_lats, 95):.1f} ms'],
            ['  P99', f'{np.percentile(all_mem_lats, 99):.1f} ms'],
            ['', ''],
            ['LLM Latency (Reference)', ''],
            ['  Mean', f'{np.mean(all_llm_lats):.0f} ms'],
            ['  Speedup', f'{np.mean(all_llm_lats)/np.mean(all_mem_lats):.1f}x faster'],
            ['', ''],
            ['Memory Management', ''],
            ['  Total Forgetting Events', f'{total_forgetting}'],
            ['  Avg per Pair', f'{total_forgetting/len(results):.1f}'],
            ['  Avg Final Memory', f'{np.mean(final_a + final_b):.0f}'],
        ]
        
        table = ax11.table(cellText=stats_data, cellLoc='left', loc='center',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax11.set_title('K. Performance Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Main title
        plt.suptitle('Sustained 1-on-1 NPC Conversations: Memory System Performance Analysis',
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Research plots saved: {save_path}")
        
        plt.close()


async def main():
    """Run the sustained conversation benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sustained 1-on-1 NPC Conversation Benchmark'
    )
    parser.add_argument('--pairs', type=int, default=3,
                       help='Number of NPC pairs (default: 3)')
    parser.add_argument('--turns', type=int, default=50,
                       help='Turns per conversation (default: 50)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for embeddings')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output')
    
    args = parser.parse_args()
    
    benchmark = SustainedConversationBenchmark()
    
    # Run benchmark
    results = await benchmark.run_multi_pair_benchmark(
        num_pairs=args.pairs,
        turns_per_pair=args.turns,
        use_gpu=args.gpu,
        verbose=not args.quiet
    )
    
    # Generate plots
    plot_path = benchmark.results_dir / f"sustained_{benchmark.timestamp}.png"
    benchmark.plot_results(results, save_path=str(plot_path))
    
    print(f"\n✓ Benchmark complete!")
    print(f"  Results saved in: {benchmark.results_dir}")


if __name__ == "__main__":
    asyncio.run(main())
