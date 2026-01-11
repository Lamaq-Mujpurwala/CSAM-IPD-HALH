"""
Quick test script to verify CSAM core modules are working.

Run this after installing dependencies:
    python tests/test_quick.py
"""

import sys
import os

# Add project root to path so we can import csam_core
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from csam_core.memory_repository import MemoryRepository, Memory
        print("  ✓ memory_repository")
    except ImportError as e:
        print(f"  ✗ memory_repository: {e}")
        return False
    
    try:
        from csam_core.knowledge_graph import KnowledgeGraph, L3Node
        print("  ✓ knowledge_graph")
    except ImportError as e:
        print(f"  ✗ knowledge_graph: {e}")
        return False
    
    try:
        from csam_core.forgetting_engine import (
            ConsolidationAwareForgetting,
            LRUForgetting,
            ImportanceForgetting
        )
        print("  ✓ forgetting_engine")
    except ImportError as e:
        print(f"  ✗ forgetting_engine: {e}")
        return False
    
    try:
        from csam_core.consolidation_tracker import ConsolidationTracker
        print("  ✓ consolidation_tracker")
    except ImportError as e:
        print(f"  ✗ consolidation_tracker: {e}")
        return False
    
    try:
        from csam_core.retrieval import HybridRetriever
        print("  ✓ retrieval")
    except ImportError as e:
        print(f"  ✗ retrieval: {e}")
        return False
    
    try:
        from csam_core.mmr import MaximalMarginalRelevance
        print("  ✓ mmr")
    except ImportError as e:
        print(f"  ✗ mmr: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_memory_repository():
    """Test basic memory repository operations."""
    print("Testing MemoryRepository...")
    
    from csam_core.memory_repository import MemoryRepository
    
    # Create repository
    repo = MemoryRepository(embedding_dim=384, max_memories=1000)
    print(f"  Created repository with dim={repo.embedding_dim}")
    
    # Add memories with random embeddings
    np.random.seed(42)
    
    for i in range(10):
        embedding = np.random.randn(384).astype(np.float32)
        memory_id = repo.add(f"Test memory {i}", embedding, importance=0.5)
    
    print(f"  Added 10 memories, total: {len(repo)}")
    
    # Retrieve
    query = np.random.randn(384).astype(np.float32)
    results = repo.retrieve(query, k=3)
    print(f"  Retrieved {len(results)} similar memories")
    
    # Benchmark
    stats = repo.benchmark_retrieval(n_queries=50)
    print(f"  Retrieval latency: {stats['avg_latency_ms']:.2f}ms avg")
    
    print("MemoryRepository OK!\n")
    return True


def test_forgetting_strategies():
    """Test forgetting strategies."""
    print("Testing Forgetting Strategies...")
    
    from csam_core.memory_repository import Memory
    from csam_core.forgetting_engine import (
        LRUForgetting,
        ImportanceForgetting,
        ConsolidationAwareForgetting
    )
    from datetime import datetime, timedelta
    
    # Create test memories
    memories = []
    for i in range(5):
        mem = Memory(
            id=f"mem_{i}",
            text=f"Memory {i}",
            embedding=np.random.randn(384).astype(np.float32),
            importance=0.1 * i,  # 0.0, 0.1, 0.2, 0.3, 0.4
        )
        mem.last_accessed = datetime.now() - timedelta(days=i)  # 0, 1, 2, 3, 4 days ago
        memories.append(mem)
    
    # Test LRU
    lru = LRUForgetting()
    lru_scores = lru.compute_forget_scores(memories)
    print(f"  LRU scores: {[f'{s:.2f}' for s in lru_scores.values()]}")
    
    # Test Importance
    imp = ImportanceForgetting()
    imp_scores = imp.compute_forget_scores(memories)
    print(f"  Importance scores: {[f'{s:.2f}' for s in imp_scores.values()]}")
    
    # Test Consolidation-Aware
    cons = ConsolidationAwareForgetting()
    cons_scores = cons.compute_forget_scores(memories)
    print(f"  Consolidation-Aware scores: {[f'{s:.2f}' for s in cons_scores.values()]}")
    
    # Select to forget
    to_forget = lru.select_to_forget(memories, count=2)
    print(f"  LRU would forget: {to_forget}")
    
    print("Forgetting Strategies OK!\n")
    return True


def test_knowledge_graph():
    """Test knowledge graph operations."""
    print("Testing KnowledgeGraph...")
    
    from csam_core.knowledge_graph import KnowledgeGraph
    
    # Create in-memory graph
    kg = KnowledgeGraph(db_path=":memory:", embedding_dim=384)
    print(f"  Created in-memory graph")
    
    # Add nodes
    np.random.seed(42)
    node_ids = []
    for i in range(5):
        embedding = np.random.randn(384).astype(np.float32)
        node_id = kg.add_node(
            content=f"Entity {i}",
            embedding=embedding,
            node_type="entity"
        )
        node_ids.append(node_id)
    
    print(f"  Added 5 nodes, total: {len(kg)}")
    
    # Add edges
    kg.add_edge(node_ids[0], node_ids[1], "related_to")
    kg.add_edge(node_ids[1], node_ids[2], "caused")
    
    # Query
    query = np.random.randn(384).astype(np.float32)
    results = kg.query_by_embedding(query, k=3)
    print(f"  Retrieved {len(results)} similar nodes")
    
    # Traverse
    traversed = kg.traverse(node_ids[0], max_hops=2)
    print(f"  Traversed {len(traversed)} nodes from start")
    
    print("KnowledgeGraph OK!\n")
    return True


def test_mmr():
    """Test MMR re-ranking."""
    print("Testing MMR...")
    
    from csam_core.mmr import MaximalMarginalRelevance, compute_intra_list_diversity
    
    mmr = MaximalMarginalRelevance(lambda_param=0.5)
    
    # Create test candidates
    np.random.seed(42)
    candidates = []
    for i in range(10):
        item = f"Item {i}"
        embedding = np.random.randn(384).astype(np.float32)
        relevance = 1.0 - (i * 0.1)  # Decreasing relevance
        candidates.append((item, embedding, relevance))
    
    # Re-rank
    query = np.random.randn(384).astype(np.float32)
    results = mmr.rerank(candidates, query, k=5)
    
    print(f"  Re-ranked {len(candidates)} candidates to top {len(results)}")
    print(f"  Selected: {[item for item, _ in results]}")
    
    # Compute diversity
    embeddings = [c[1] for c in candidates[:5]]
    diversity = compute_intra_list_diversity(embeddings)
    print(f"  Diversity score of top 5: {diversity:.3f}")
    
    print("MMR OK!\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CSAM Quick Tests")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Check your installation.")
        return 1
    
    # Test individual modules
    tests = [
        test_memory_repository,
        test_forgetting_strategies,
        test_knowledge_graph,
        test_mmr,
    ]
    
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"  ❌ {test_func.__name__} failed with error: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
