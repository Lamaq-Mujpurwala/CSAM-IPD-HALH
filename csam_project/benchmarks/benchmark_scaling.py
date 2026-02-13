"""
Scalability Benchmark - Test CSAM with 1 to 100 NPCs

This benchmark measures:
1. Initialization time for N NPCs
2. Memory retrieval latency at scale
3. Total memory footprint
4. Response quality consistency

Run:
    python benchmarks/benchmark_scaling.py --max-npcs 100
"""

import sys
import os
import time
import gc
import json
import argparse
from typing import Dict, List
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.npc import NPC, NPCPersonality
from csam_core.services.embedding import EmbeddingService


def generate_npc_personalities(count: int) -> List[NPCPersonality]:
    """Generate N unique NPC personalities."""
    roles = ["guard", "merchant", "farmer", "blacksmith", "baker", "tailor", 
             "innkeeper", "hunter", "fisherman", "miner", "herbalist", "scholar"]
    traits_pool = ["friendly", "grumpy", "curious", "cautious", "talkative", 
                   "quiet", "wise", "naive", "brave", "timid"]
    
    personalities = []
    for i in range(count):
        role = roles[i % len(roles)]
        name = f"NPC_{i:03d}"
        traits = [traits_pool[i % len(traits_pool)], traits_pool[(i + 3) % len(traits_pool)]]
        
        personalities.append(NPCPersonality(
            name=name,
            role=role,
            traits=traits,
            background=f"A {role} in the village.",
            speaking_style="casual",
            greeting=f"Hello, I'm {name}, the {role}."
        ))
    
    return personalities


def run_scaling_benchmark(
    max_npcs: int = 100,
    memories_per_npc: int = 100,
    queries_per_npc: int = 10,
    output_file: str = None
) -> Dict:
    """
    Run the scalability benchmark.
    
    Args:
        max_npcs: Maximum number of NPCs to test
        memories_per_npc: Memories to add to each NPC
        queries_per_npc: Queries to run per NPC
        output_file: Optional JSON output file
    """
    print("=" * 70)
    print("CSAM SCALABILITY BENCHMARK")
    print("=" * 70)
    print(f"Max NPCs: {max_npcs}")
    print(f"Memories per NPC: {memories_per_npc}")
    print(f"Queries per NPC: {queries_per_npc}")
    print()
    
    # Shared embedding service (single model for all NPCs)
    print("Loading embedding model...")
    embedding_service = EmbeddingService()
    _ = embedding_service.dimension
    print("  [OK] Loaded\n")
    
    # Test points
    test_points = [1, 5, 10, 25, 50, 100]
    test_points = [p for p in test_points if p <= max_npcs]
    
    results = {
        "config": {
            "max_npcs": max_npcs,
            "memories_per_npc": memories_per_npc,
            "queries_per_npc": queries_per_npc,
            "timestamp": datetime.now().isoformat()
        },
        "measurements": []
    }
    
    # Generate all personalities upfront
    personalities = generate_npc_personalities(max_npcs)
    
    for n_npcs in test_points:
        print(f"\n{'='*60}")
        print(f"Testing with N={n_npcs} NPCs")
        print(f"{'='*60}")
        
        gc.collect()  # Clean up before test
        
        # ============== INITIALIZATION ==============
        print(f"\n[1] Initialization...")
        init_start = time.time()
        
        npcs: List[NPC] = []
        for i in range(n_npcs):
            npc = NPC(
                personality=personalities[i],
                embedding_service=embedding_service,
                llm_service=None,  # No LLM for benchmark
                max_memories=10000,
                forget_threshold=500
            )
            npcs.append(npc)
        
        init_time_sec = time.time() - init_start
        print(f"    Init time: {init_time_sec:.2f}s ({init_time_sec/n_npcs*1000:.1f}ms per NPC)")
        
        # ============== MEMORY POPULATION ==============
        print(f"\n[2] Populating memories ({memories_per_npc} per NPC)...")
        pop_start = time.time()
        
        test_facts = []  # Track facts for later recall
        
        for i, npc in enumerate(npcs):
            # Add a unique fact to remember
            unique_fact = f"my favorite number is {i * 7 + 13}"
            npc.add_memory(f"Player said: {unique_fact}", importance=0.95)
            test_facts.append((npc, unique_fact))
            
            # Add random memories
            for j in range(memories_per_npc - 1):
                npc.add_memory(f"Random interaction {j} with NPC {i}", importance=0.5)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"    ... {i + 1}/{n_npcs} NPCs populated")
        
        pop_time_sec = time.time() - pop_start
        total_memories = n_npcs * memories_per_npc
        print(f"    Population time: {pop_time_sec:.2f}s ({total_memories} total memories)")
        
        # ============== RETRIEVAL LATENCY ==============
        print(f"\n[3] Retrieval latency test ({queries_per_npc} queries per NPC)...")
        
        latencies = []
        query_start = time.time()
        
        for npc in npcs:
            for _ in range(queries_per_npc):
                q_start = time.perf_counter()
                context = npc.retrieve_context("What did the player tell you?", k=5)
                latencies.append((time.perf_counter() - q_start) * 1000)
        
        query_time_sec = time.time() - query_start
        avg_latency_ms = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"    Total query time: {query_time_sec:.2f}s")
        print(f"    Avg latency: {avg_latency_ms:.2f}ms")
        print(f"    P50 latency: {p50_latency:.2f}ms")
        print(f"    P99 latency: {p99_latency:.2f}ms")
        
        # ============== RECALL ACCURACY ==============
        print(f"\n[4] Recall accuracy test...")
        
        correct_recalls = 0
        for npc, fact in test_facts:
            context = npc.retrieve_context("What is your favorite number?", k=5)
            if fact in context.lower() or str(int(fact.split()[-1])) in context:
                correct_recalls += 1
        
        recall_accuracy = correct_recalls / len(test_facts)
        print(f"    Recall accuracy: {recall_accuracy:.1%} ({correct_recalls}/{len(test_facts)})")
        
        # ============== MEMORY FOOTPRINT ==============
        print(f"\n[5] Memory footprint...")
        
        total_l2_memories = sum(len(npc.memory_repo) for npc in npcs)
        total_l3_nodes = sum(len(npc.knowledge_graph) for npc in npcs)
        
        # Rough estimate: 384 floats * 4 bytes + 200 bytes metadata per memory
        estimated_bytes = total_l2_memories * (384 * 4 + 200)
        estimated_mb = estimated_bytes / (1024 * 1024)
        
        print(f"    Total L2 memories: {total_l2_memories}")
        print(f"    Total L3 nodes: {total_l3_nodes}")
        print(f"    Estimated memory: {estimated_mb:.1f} MB")
        
        # Store results
        measurement = {
            "n_npcs": n_npcs,
            "init_time_sec": init_time_sec,
            "init_time_per_npc_ms": init_time_sec / n_npcs * 1000,
            "population_time_sec": pop_time_sec,
            "total_memories": total_l2_memories,
            "avg_latency_ms": avg_latency_ms,
            "p50_latency_ms": p50_latency,
            "p99_latency_ms": p99_latency,
            "recall_accuracy": recall_accuracy,
            "estimated_memory_mb": estimated_mb
        }
        results["measurements"].append(measurement)
        
        # Clean up
        del npcs
        gc.collect()
    
    # ============== SUMMARY ==============
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'NPCs':>8} | {'Init (s)':>10} | {'Latency (ms)':>12} | {'Recall':>8} | {'Memory (MB)':>12}")
    print("-" * 60)
    
    for m in results["measurements"]:
        print(f"{m['n_npcs']:>8} | {m['init_time_sec']:>10.2f} | "
              f"{m['avg_latency_ms']:>12.2f} | {m['recall_accuracy']:>7.0%} | "
              f"{m['estimated_memory_mb']:>12.1f}")
    
    # Check O(log N) scaling
    if len(results["measurements"]) >= 2:
        first = results["measurements"][0]
        last = results["measurements"][-1]
        
        # Latency should not grow linearly with N
        npc_ratio = last["n_npcs"] / first["n_npcs"]
        latency_ratio = last["avg_latency_ms"] / first["avg_latency_ms"]
        
        print(f"\nScaling Analysis:")
        print(f"  NPC count increased: {first['n_npcs']} -> {last['n_npcs']} ({npc_ratio:.0f}x)")
        print(f"  Latency increased: {first['avg_latency_ms']:.2f}ms -> {last['avg_latency_ms']:.2f}ms ({latency_ratio:.1f}x)")
        
        if latency_ratio < npc_ratio * 0.5:
            print("  [OK] Sub-linear scaling achieved (O(log N) behavior)")
        else:
            print("  ⚠ Linear or worse scaling")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSAM Scalability Benchmark")
    parser.add_argument("--max-npcs", type=int, default=100, help="Max NPCs to test")
    parser.add_argument("--memories", type=int, default=100, help="Memories per NPC")
    parser.add_argument("--queries", type=int, default=10, help="Queries per NPC")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    run_scaling_benchmark(
        max_npcs=args.max_npcs,
        memories_per_npc=args.memories,
        queries_per_npc=args.queries,
        output_file=args.output
    )
