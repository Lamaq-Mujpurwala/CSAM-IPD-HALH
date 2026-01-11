"""
End-to-End CSAM Benchmark

Tests the COMPLETE hypothesis:
1. Memory persistence across long conversations
2. Consolidation-aware forgetting effectiveness
3. LLM response quality with memory context
4. Scalability from 1 to N NPCs
5. Real-world recall accuracy

Parameters (toggleable):
- Number of NPCs
- Memories before recall test
- Forgetting threshold
- With/without consolidation
- With/without LLM

Run:
    python benchmarks/benchmark_e2e.py --help
"""

import sys
import os
import time
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.npc import NPC, NPCPersonality, TAVERN_NPCS
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm import LLMService
from csam_core.forgetting_engine import (
    NoForgetting, LRUForgetting, ImportanceForgetting, ConsolidationAwareForgetting
)


@dataclass
class TestCase:
    """A single test case for memory recall."""
    id: str
    fact_stored: str
    question: str
    expected_keywords: List[str]
    importance: float = 0.9


@dataclass 
class TestResult:
    """Result of a single test case."""
    test_id: str
    npc_name: str
    fact_stored: str
    question: str
    response: str
    context_retrieved: str
    fact_in_context: bool
    keywords_in_response: List[str]
    recall_success: bool
    response_latency_ms: float
    memories_at_test: int


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    num_npcs: int = 1
    memories_before_test: int = 100
    forget_threshold: int = 200
    forgetting_strategy: str = "consolidation"  # none, lru, importance, consolidation
    use_llm: bool = True
    llm_model: str = "llama3.2:3b"
    run_consolidation: bool = True
    seed: int = 42


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    config: BenchmarkConfig
    timestamp: str
    total_tests: int
    passed_tests: int
    recall_accuracy: float
    avg_response_latency_ms: float
    avg_memories_at_test: int
    test_results: List[TestResult]
    summary: Dict[str, Any]


# Test cases for memory recall
TEST_CASES = [
    TestCase(
        id="name_recall",
        fact_stored="my name is Alexander and I come from the Northern Kingdom",
        question="Do you remember my name and where I'm from?",
        expected_keywords=["alexander", "northern", "kingdom"],
        importance=0.95
    ),
    TestCase(
        id="secret_recall",
        fact_stored="I have a secret - my mother Celestia was a powerful wizard who disappeared 10 years ago",
        question="What did I tell you about my mother?",
        expected_keywords=["celestia", "wizard", "mother", "disappeared"],
        importance=0.95
    ),
    TestCase(
        id="preference_recall",
        fact_stored="my favorite drink is honey mead and I always order it when I visit",
        question="What's my favorite drink?",
        expected_keywords=["honey", "mead"],
        importance=0.85
    ),
    TestCase(
        id="quest_recall",
        fact_stored="I'm searching for the legendary Sword of Dawn which was lost in the Crystal Caves",
        question="What quest am I on? What am I looking for?",
        expected_keywords=["sword", "dawn", "crystal", "caves"],
        importance=0.9
    ),
    TestCase(
        id="number_recall",
        fact_stored="my lucky number is 42 and I always bet on it at the tavern games",
        question="What's my lucky number?",
        expected_keywords=["42"],
        importance=0.8
    ),
]

# Random filler conversations
FILLER_CONVERSATIONS = [
    "What's the weather like today?",
    "Tell me about the local area.",
    "Any news from the capital?",
    "What's your best-selling item?",
    "Have you seen any travelers lately?",
    "Tell me a story about this place.",
    "What do you recommend?",
    "How long have you worked here?",
    "Any rumors going around?",
    "What's the history of this tavern?",
    "Do you know any good jokes?",
    "What's your opinion on the king?",
    "Have you heard about the dragon sightings?",
    "What's the best way to the mountain pass?",
    "Any quests available?",
    "What creatures lurk in the forest?",
    "Tell me about the local guild.",
    "What's your favorite memory?",
    "Any festivals coming up?",
    "What's the most valuable item you've seen?",
]


class EndToEndBenchmark:
    """
    Comprehensive end-to-end benchmark for CSAM.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        random.seed(config.seed)
        
        print("=" * 70)
        print("CSAM END-TO-END BENCHMARK")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  NPCs: {config.num_npcs}")
        print(f"  Memories before test: {config.memories_before_test}")
        print(f"  Forget threshold: {config.forget_threshold}")
        print(f"  Forgetting strategy: {config.forgetting_strategy}")
        print(f"  Use LLM: {config.use_llm} ({config.llm_model})")
        print(f"  Run consolidation: {config.run_consolidation}")
        print()
        
        # Initialize services
        print("Initializing services...")
        self.embedding_service = EmbeddingService()
        _ = self.embedding_service.dimension
        print(f"  ✓ Embedding model loaded")
        
        self.llm_service = None
        if config.use_llm:
            self.llm_service = LLMService(model=config.llm_model)
            if self.llm_service.is_available():
                print(f"  ✓ LLM connected ({config.llm_model})")
            else:
                print(f"  ✗ LLM not available, using fallback")
                self.llm_service = None
        
        # Initialize NPCs
        self.npcs: List[NPC] = []
        self._init_npcs()
    
    def _get_forgetting_strategy(self):
        """Get forgetting strategy based on config."""
        if self.config.forgetting_strategy == "none":
            return NoForgetting()
        elif self.config.forgetting_strategy == "lru":
            return LRUForgetting()
        elif self.config.forgetting_strategy == "importance":
            return ImportanceForgetting()
        else:  # consolidation
            return ConsolidationAwareForgetting(
                alpha=0.2, beta=0.2, gamma=0.3, delta=0.3
            )
    
    def _init_npcs(self):
        """Initialize NPCs with the configured settings."""
        print(f"\nInitializing {self.config.num_npcs} NPCs...")
        
        personalities = TAVERN_NPCS[:self.config.num_npcs]
        
        # If we need more NPCs than pre-defined, generate them
        while len(personalities) < self.config.num_npcs:
            i = len(personalities)
            personalities.append(NPCPersonality(
                name=f"NPC_{i}",
                role="villager",
                traits=["helpful"],
                background="A friendly villager.",
                speaking_style="casual",
                greeting="Hello there!"
            ))
        
        for personality in personalities:
            npc = NPC(
                personality=personality,
                embedding_service=self.embedding_service,
                llm_service=self.llm_service,
                max_memories=10000,
                forget_threshold=self.config.forget_threshold
            )
            # Override forgetting strategy
            npc.forgetting_strategy = self._get_forgetting_strategy()
            self.npcs.append(npc)
            print(f"  ✓ {personality.name} ({personality.role})")
    
    def run_test_case(self, npc: NPC, test_case: TestCase, player_name: str = "Traveler") -> TestResult:
        """Run a single test case on an NPC."""
        
        # Step 1: Store the fact
        fact_message = f"I want to tell you something important - {test_case.fact_stored}"
        npc.add_memory(f"{player_name} said: {fact_message}", importance=test_case.importance)
        npc.add_memory(f"I ({npc.personality.name}) acknowledged: I will remember that.", importance=test_case.importance - 0.1)
        
        # Step 2: Fill with random conversations
        print(f"    Filling {self.config.memories_before_test} filler conversations...")
        for i in range(self.config.memories_before_test):
            filler = random.choice(FILLER_CONVERSATIONS)
            # Add player message
            npc.add_memory(f"{player_name} asked: {filler}", importance=0.4)
            # Add simple NPC response (no LLM to speed up)
            npc.add_memory(f"I ({npc.personality.name}) responded to a question about {filler[:20]}...", importance=0.3)
        
        # Step 3: Run consolidation if enabled
        if self.config.run_consolidation:
            npc.run_consolidation()
        
        # Step 4: Ask the recall question
        start_time = time.time()
        result = npc.respond(test_case.question, player_name=player_name)
        latency_ms = (time.time() - start_time) * 1000
        
        response = result["response"]
        context = result["context_used"]
        
        # Step 5: Analyze results
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Check if fact is in retrieved context
        fact_in_context = any(kw in context_lower for kw in test_case.expected_keywords)
        
        # Check which keywords appear in response
        keywords_in_response = [kw for kw in test_case.expected_keywords if kw in response_lower]
        
        # Recall is successful if at least half the keywords are in response
        recall_success = len(keywords_in_response) >= len(test_case.expected_keywords) / 2
        
        return TestResult(
            test_id=test_case.id,
            npc_name=npc.personality.name,
            fact_stored=test_case.fact_stored,
            question=test_case.question,
            response=response,
            context_retrieved=context[:500],  # Truncate for logging
            fact_in_context=fact_in_context,
            keywords_in_response=keywords_in_response,
            recall_success=recall_success,
            response_latency_ms=latency_ms,
            memories_at_test=len(npc.memory_repo)
        )
    
    def run(self) -> BenchmarkResults:
        """Run the complete benchmark."""
        print("\n" + "=" * 70)
        print("RUNNING TESTS")
        print("=" * 70)
        
        all_results: List[TestResult] = []
        
        for npc in self.npcs:
            print(f"\n[NPC: {npc.personality.name}]")
            
            for test_case in TEST_CASES:
                print(f"  Test: {test_case.id}")
                result = self.run_test_case(npc, test_case)
                all_results.append(result)
                
                status = "✓ PASS" if result.recall_success else "✗ FAIL"
                print(f"    {status} | Keywords found: {result.keywords_in_response}")
                print(f"    Latency: {result.response_latency_ms:.0f}ms | Memories: {result.memories_at_test}")
                
                if result.recall_success:
                    print(f"    Response: {result.response[:100]}...")
                else:
                    print(f"    Response: {result.response[:100]}...")
                    print(f"    Context: {result.context_retrieved[:100]}...")
        
        # Calculate summary statistics
        passed = sum(1 for r in all_results if r.recall_success)
        total = len(all_results)
        avg_latency = sum(r.response_latency_ms for r in all_results) / total
        avg_memories = sum(r.memories_at_test for r in all_results) / total
        
        # Per-test-case analysis
        test_case_results = {}
        for test_case in TEST_CASES:
            tc_results = [r for r in all_results if r.test_id == test_case.id]
            test_case_results[test_case.id] = {
                "passed": sum(1 for r in tc_results if r.recall_success),
                "total": len(tc_results),
                "accuracy": sum(1 for r in tc_results if r.recall_success) / len(tc_results) if tc_results else 0
            }
        
        # Per-NPC analysis
        npc_results = {}
        for npc in self.npcs:
            npc_r = [r for r in all_results if r.npc_name == npc.personality.name]
            npc_results[npc.personality.name] = {
                "passed": sum(1 for r in npc_r if r.recall_success),
                "total": len(npc_r),
                "accuracy": sum(1 for r in npc_r if r.recall_success) / len(npc_r) if npc_r else 0,
                "final_memories": len(npc.memory_repo),
                "l3_nodes": len(npc.knowledge_graph),
                "consolidated_ratio": sum(1 for m in npc.memory_repo.get_all() if m.consolidated) / len(npc.memory_repo) if len(npc.memory_repo) > 0 else 0
            }
        
        results = BenchmarkResults(
            config=self.config,
            timestamp=datetime.now().isoformat(),
            total_tests=total,
            passed_tests=passed,
            recall_accuracy=passed / total if total > 0 else 0,
            avg_response_latency_ms=avg_latency,
            avg_memories_at_test=avg_memories,
            test_results=all_results,
            summary={
                "by_test_case": test_case_results,
                "by_npc": npc_results
            }
        )
        
        return results
    
    def print_summary(self, results: BenchmarkResults):
        """Print a summary of the results."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\n Overall Recall Accuracy: {results.recall_accuracy:.1%} ({results.passed_tests}/{results.total_tests})")
        print(f" Average Response Latency: {results.avg_response_latency_ms:.0f}ms")
        print(f" Average Memories at Test: {results.avg_memories_at_test:.0f}")
        
        print("\n By Test Case:")
        print("-" * 50)
        for tc_id, tc_stats in results.summary["by_test_case"].items():
            status = "✓" if tc_stats["accuracy"] >= 0.5 else "✗"
            print(f"  {status} {tc_id}: {tc_stats['accuracy']:.0%} ({tc_stats['passed']}/{tc_stats['total']})")
        
        print("\n By NPC:")
        print("-" * 50)
        for npc_name, npc_stats in results.summary["by_npc"].items():
            print(f"  {npc_name}: {npc_stats['accuracy']:.0%} accuracy, "
                  f"{npc_stats['final_memories']} memories, "
                  f"{npc_stats['l3_nodes']} L3 nodes, "
                  f"{npc_stats['consolidated_ratio']:.0%} consolidated")
        
        print("\n Configuration:")
        print("-" * 50)
        print(f"  NPCs: {results.config.num_npcs}")
        print(f"  Filler memories: {results.config.memories_before_test}")
        print(f"  Forgetting strategy: {results.config.forgetting_strategy}")
        print(f"  Forget threshold: {results.config.forget_threshold}")
        print(f"  LLM: {results.config.use_llm}")
    
    def save_results(self, results: BenchmarkResults, output_path: str):
        """Save results to JSON file."""
        # Convert to dict for JSON serialization
        data = {
            "config": asdict(results.config),
            "timestamp": results.timestamp,
            "total_tests": results.total_tests,
            "passed_tests": results.passed_tests,
            "recall_accuracy": results.recall_accuracy,
            "avg_response_latency_ms": results.avg_response_latency_ms,
            "avg_memories_at_test": results.avg_memories_at_test,
            "summary": results.summary,
            "test_results": [asdict(r) for r in results.test_results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CSAM End-to-End Benchmark")
    parser.add_argument("--npcs", type=int, default=1, help="Number of NPCs to test")
    parser.add_argument("--memories", type=int, default=100, help="Filler memories before recall test")
    parser.add_argument("--threshold", type=int, default=200, help="Forget threshold")
    parser.add_argument("--strategy", type=str, default="consolidation",
                        choices=["none", "lru", "importance", "consolidation"],
                        help="Forgetting strategy")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model")
    parser.add_argument("--no-consolidation", action="store_true", help="Disable consolidation")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        num_npcs=args.npcs,
        memories_before_test=args.memories,
        forget_threshold=args.threshold,
        forgetting_strategy=args.strategy,
        use_llm=not args.no_llm,
        llm_model=args.model,
        run_consolidation=not args.no_consolidation,
        seed=args.seed
    )
    
    benchmark = EndToEndBenchmark(config)
    results = benchmark.run()
    benchmark.print_summary(results)
    
    if args.output:
        benchmark.save_results(results, args.output)
    else:
        # Default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"e2e_results_{timestamp}.json"
        benchmark.save_results(results, output_path)
    
    # Return exit code based on success
    return 0 if results.recall_accuracy >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
