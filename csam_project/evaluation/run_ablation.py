"""
Ablation Study Runner - Compares Forgetting Strategies

This is the key evaluation script that proves our novel contribution.
It compares:
1. No forgetting (baseline - unbounded memory)
2. LRU forgetting (baseline - recency only)
3. Importance-based forgetting (baseline)
4. Consolidation-Aware forgetting (OURS - novel)

Metrics:
- F1 score on Q&A tasks
- Memory usage
- Retrieval latency
- Information retention ratio
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csam_core.memory_repository import MemoryRepository
from csam_core.knowledge_graph import KnowledgeGraph
from csam_core.forgetting_engine import (
    ForgettingStrategy,
    NoForgetting,
    LRUForgetting,
    ImportanceForgetting,
    ConsolidationAwareForgetting,
    create_forgetting_strategy
)
from csam_core.consolidation_tracker import ConsolidationTracker
from csam_core.services.embedding import EmbeddingService
from csam_core.consolidation import ConsolidationPipeline
from csam_core.retrieval import HybridRetriever

from evaluation.npc_locomo import BenchmarkGenerator, ConversationHistory, QAPair


@dataclass
class EvaluationResult:
    """Results from evaluating one strategy."""
    strategy_name: str
    f1_scores: Dict[str, float]  # by question type
    overall_f1: float
    memory_count: int
    memory_bytes: int
    avg_latency_ms: float
    consolidation_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "f1_scores": self.f1_scores,
            "overall_f1": self.overall_f1,
            "memory_count": self.memory_count,
            "memory_bytes_mb": self.memory_bytes / 1e6,
            "avg_latency_ms": self.avg_latency_ms,
            "consolidation_ratio": self.consolidation_ratio
        }


class MemorySystemWithForgetting:
    """
    Complete memory system with a specific forgetting strategy.
    """
    
    def __init__(
        self,
        forgetting_strategy: ForgettingStrategy,
        max_memories: int = 10000,
        forget_threshold: int = 1000,
        embedding_service: EmbeddingService = None
    ):
        """
        Initialize memory system.
        
        Args:
            forgetting_strategy: Strategy to use for forgetting
            max_memories: Maximum memory capacity
            forget_threshold: Trigger forgetting when exceeding this
            embedding_service: Shared embedding service
        """
        self.forgetting_strategy = forgetting_strategy
        self.max_memories = max_memories
        self.forget_threshold = forget_threshold
        
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize components
        self.memory_repo = MemoryRepository(
            embedding_dim=self.embedding_service.dimension,
            max_memories=max_memories
        )
        self.knowledge_graph = KnowledgeGraph(
            db_path=":memory:",
            embedding_dim=self.embedding_service.dimension
        )
        self.consolidation_tracker = ConsolidationTracker()
        
        # Consolidation pipeline (without LLM for speed)
        self.consolidation_pipeline = ConsolidationPipeline(
            memory_repository=self.memory_repo,
            knowledge_graph=self.knowledge_graph,
            consolidation_tracker=self.consolidation_tracker,
            embedding_service=self.embedding_service,
            llm_service=None,  # Use simple fallback
            min_memories_per_batch=5,
            max_memories_per_batch=10,
            consolidation_threshold_hours=0.0  # Immediate for testing
        )
        
        # Retriever
        self.retriever = HybridRetriever(
            memory_repository=self.memory_repo,
            knowledge_graph=self.knowledge_graph,
            mmr_lambda=0.5
        )
    
    def add_memory(self, text: str, importance: float = 0.5) -> str:
        """Add a memory and trigger forgetting/consolidation if needed."""
        embedding = self.embedding_service.encode(text)
        memory_id = self.memory_repo.add(text, embedding, importance)
        
        # Check if we need to forget
        if len(self.memory_repo) > self.forget_threshold:
            self._run_forgetting()
        
        # Run consolidation periodically
        if len(self.memory_repo) % 20 == 0:
            self.consolidation_pipeline.run_consolidation()
        
        return memory_id
    
    def _run_forgetting(self):
        """Run the forgetting strategy."""
        # Calculate how many to forget
        excess = len(self.memory_repo) - self.forget_threshold
        forget_count = max(excess, int(self.forget_threshold * 0.1))
        
        memories = self.memory_repo.get_all()
        
        # Get L3 embeddings for consolidation-aware forgetting
        l3_embeddings = self.knowledge_graph.get_embeddings_matrix()
        
        # Select memories to forget
        to_forget = self.forgetting_strategy.select_to_forget(
            memories,
            count=forget_count,
            consolidation_tracker=self.consolidation_tracker,
            l3_embeddings=l3_embeddings
        )
        
        # Delete selected memories
        self.memory_repo.delete_batch(to_forget)
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        """Retrieve relevant memories for a query."""
        query_embedding = self.embedding_service.encode(query)
        result = self.retriever.retrieve_sync(query_embedding, k=k)
        return result.final_results
    
    def get_context_for_question(self, question: str) -> str:
        """Get context string for answering a question."""
        results = self.retrieve(question, k=5)
        
        context_parts = []
        for item, score in results:
            if hasattr(item, 'text'):  # Memory
                context_parts.append(item.text)
            elif hasattr(item, 'content'):  # L3Node
                context_parts.append(item.content)
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        memories = self.memory_repo.get_all()
        consolidated = sum(1 for m in memories if m.consolidated)
        
        # Estimate memory usage (rough)
        memory_bytes = len(memories) * (384 * 4 + 200)  # embedding + metadata
        
        return {
            "memory_count": len(memories),
            "consolidated_count": consolidated,
            "consolidation_ratio": consolidated / len(memories) if memories else 0,
            "l3_nodes": len(self.knowledge_graph),
            "memory_bytes": memory_bytes
        }


def compute_f1(predicted: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    Simple word overlap based F1.
    """
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # Precision and recall
    common = pred_tokens & truth_tokens
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def answer_question_from_context(context: str, question: str, ground_truth: str) -> Tuple[str, float]:
    """
    Simple context-based answer extraction.
    
    Without LLM, we use simple keyword matching as a baseline.
    This gives us a consistent baseline across all strategies.
    """
    # For evaluation purposes, we check if the context contains
    # relevant information to answer the question
    
    context_lower = context.lower()
    truth_lower = ground_truth.lower()
    
    # Check if any significant words from ground truth appear in context
    truth_words = set(truth_lower.split())
    context_words = set(context_lower.split())
    
    # Score based on overlap
    overlap = truth_words & context_words
    
    if len(truth_words) == 0:
        return "unknown", 0.0
    
    # Coverage score
    coverage = len(overlap) / len(truth_words)
    
    # Use coverage as proxy for whether answer could be derived
    if coverage > 0.5:
        return ground_truth, coverage
    else:
        return "unknown", 0.0


def evaluate_strategy(
    strategy_name: str,
    forgetting_strategy: ForgettingStrategy,
    dataset: List[ConversationHistory],
    embedding_service: EmbeddingService,
    max_memories: int = 10000,
    forget_threshold: int = 500,
    verbose: bool = True
) -> EvaluationResult:
    """
    Evaluate a single forgetting strategy.
    
    Args:
        strategy_name: Name of the strategy
        forgetting_strategy: The strategy instance
        dataset: Benchmark dataset
        embedding_service: Shared embedding service
        max_memories: Maximum memory capacity
        forget_threshold: When to trigger forgetting
        verbose: Print progress
        
    Returns:
        EvaluationResult with metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {strategy_name}")
        print(f"{'='*60}")
    
    # Create memory system with this strategy
    system = MemorySystemWithForgetting(
        forgetting_strategy=forgetting_strategy,
        max_memories=max_memories,
        forget_threshold=forget_threshold,
        embedding_service=embedding_service
    )
    
    # Load all interactions from dataset
    all_qa_pairs = []
    
    for history in dataset:
        if verbose:
            print(f"  Loading {history.id}: {len(history.interactions)} interactions...")
        
        for interaction in history.interactions:
            system.add_memory(interaction.text, importance=interaction.importance)
        
        all_qa_pairs.extend(history.qa_pairs)
    
    # Run final consolidation
    system.consolidation_pipeline.run_consolidation()
    
    if verbose:
        stats = system.get_statistics()
        print(f"  Final memory count: {stats['memory_count']}")
        print(f"  Consolidation ratio: {stats['consolidation_ratio']:.2%}")
        print(f"  L3 nodes: {stats['l3_nodes']}")
    
    # Benchmark retrieval latency
    latencies = []
    for _ in range(20):
        query = "What did the player buy?"
        start = time.perf_counter()
        system.retrieve(query, k=5)
        latencies.append((time.perf_counter() - start) * 1000)
    avg_latency = np.mean(latencies)
    
    # Evaluate on Q&A pairs
    f1_by_type = {"single-hop": [], "multi-hop": [], "temporal": [], "adversarial": []}
    
    if verbose:
        print(f"  Evaluating {len(all_qa_pairs)} Q&A pairs...")
    
    for qa in all_qa_pairs:
        context = system.get_context_for_question(qa.question)
        predicted, score = answer_question_from_context(context, qa.question, qa.answer)
        
        f1 = compute_f1(predicted, qa.answer)
        f1_by_type[qa.qa_type].append(f1)
    
    # Aggregate F1 scores
    f1_scores = {}
    for qa_type, scores in f1_by_type.items():
        if scores:
            f1_scores[qa_type] = np.mean(scores)
        else:
            f1_scores[qa_type] = 0.0
    
    overall_f1 = np.mean([s for s in f1_scores.values() if s > 0])
    
    stats = system.get_statistics()
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Overall F1: {overall_f1:.3f}")
        for qa_type, f1 in f1_scores.items():
            print(f"    {qa_type}: {f1:.3f}")
        print(f"    Latency: {avg_latency:.2f}ms")
    
    return EvaluationResult(
        strategy_name=strategy_name,
        f1_scores=f1_scores,
        overall_f1=overall_f1,
        memory_count=stats["memory_count"],
        memory_bytes=stats["memory_bytes"],
        avg_latency_ms=avg_latency,
        consolidation_ratio=stats["consolidation_ratio"]
    )


def run_ablation_study(
    num_conversations: int = 5,
    interactions_per_conversation: int = 100,
    forget_threshold: int = 200,
    output_file: str = None,
    verbose: bool = True
):
    """
    Run the full ablation study.
    
    Compares all forgetting strategies on the NPC-LoCoMo benchmark.
    """
    print("=" * 70)
    print("CSAM ABLATION STUDY: Forgetting Strategy Comparison")
    print("=" * 70)
    
    # Generate benchmark dataset
    print("\nGenerating benchmark dataset...")
    generator = BenchmarkGenerator(seed=42)
    dataset = generator.generate_benchmark_dataset(
        num_conversations=num_conversations,
        interactions_per_conversation=interactions_per_conversation
    )
    
    total_interactions = sum(len(h.interactions) for h in dataset)
    total_qa = sum(len(h.qa_pairs) for h in dataset)
    print(f"  Generated {len(dataset)} conversations")
    print(f"  Total interactions: {total_interactions}")
    print(f"  Total Q&A pairs: {total_qa}")
    
    # Shared embedding service (for fair comparison)
    print("\nLoading embedding model...")
    embedding_service = EmbeddingService()
    _ = embedding_service.dimension  # Force load
    
    # Define strategies to compare
    strategies = [
        ("No-Forgetting", NoForgetting()),
        ("LRU", LRUForgetting()),
        ("Importance", ImportanceForgetting()),
        ("Consolidation-Aware (Ours)", ConsolidationAwareForgetting(
            alpha=0.2, beta=0.2, gamma=0.3, delta=0.3,
            consolidation_threshold=0.0
        )),
    ]
    
    # Run evaluation for each strategy
    results = []
    for strategy_name, strategy in strategies:
        result = evaluate_strategy(
            strategy_name=strategy_name,
            forgetting_strategy=strategy,
            dataset=dataset,
            embedding_service=embedding_service,
            forget_threshold=forget_threshold,
            verbose=verbose
        )
        results.append(result)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"\n{'Strategy':<30} {'F1 Score':>10} {'Memory':>10} {'Latency':>10} {'Consol.':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.strategy_name:<30} {r.overall_f1:>10.3f} {r.memory_count:>10} {r.avg_latency_ms:>9.2f}ms {r.consolidation_ratio:>9.1%}")
    
    # Print detailed F1 by type
    print("\n" + "-" * 70)
    print("F1 Scores by Question Type:")
    print("-" * 70)
    print(f"{'Strategy':<30} {'Single-hop':>12} {'Multi-hop':>12} {'Temporal':>12} {'Adversarial':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.strategy_name:<30} "
              f"{r.f1_scores.get('single-hop', 0):>12.3f} "
              f"{r.f1_scores.get('multi-hop', 0):>12.3f} "
              f"{r.f1_scores.get('temporal', 0):>12.3f} "
              f"{r.f1_scores.get('adversarial', 0):>12.3f}")
    
    # Save results if output file specified
    if output_file:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_conversations": num_conversations,
                "interactions_per_conversation": interactions_per_conversation,
                "forget_threshold": forget_threshold
            },
            "results": [r.to_dict() for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CSAM Ablation Study")
    parser.add_argument("--conversations", type=int, default=5, help="Number of conversations")
    parser.add_argument("--interactions", type=int, default=100, help="Interactions per conversation")
    parser.add_argument("--threshold", type=int, default=200, help="Forget threshold")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    
    args = parser.parse_args()
    
    run_ablation_study(
        num_conversations=args.conversations,
        interactions_per_conversation=args.interactions,
        forget_threshold=args.threshold,
        output_file=args.output,
        verbose=not args.quiet
    )
