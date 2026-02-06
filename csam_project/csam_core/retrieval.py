"""
Hybrid Retrieval - Combines L2 (Episodic) and L3 (Semantic) retrieval with MMR.

This module orchestrates the full retrieval pipeline:
1. Query L2 (episodic memories) via HNSW
2. Query L3 (knowledge graph) via embedding similarity
3. Optionally expand via graph traversal
4. Combine and re-rank using MMR for diversity
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

from .memory_repository import Memory, MemoryRepository
from .knowledge_graph import L3Node, KnowledgeGraph
from .mmr import MaximalMarginalRelevance, compute_intra_list_diversity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result from hybrid retrieval.
    """
    # Final combined results (after MMR)
    final_results: List[Tuple[Any, float]]  # (Memory or L3Node, score)
    
    # Raw results from each source
    l2_results: List[Tuple[Memory, float]]
    l3_results: List[Tuple[L3Node, float]]
    
    # Statistics
    l2_count: int
    l3_count: int
    diversity_score: float
    
    def get_context_string(self, max_chars: int = 2000) -> str:
        """Convert results to a context string for LLM."""
        lines = []
        total_chars = 0
        
        for item, score in self.final_results:
            if isinstance(item, Memory):
                text = f"[Memory] {item.text}"
            elif isinstance(item, L3Node):
                text = f"[{item.node_type.title()}] {item.content}"
            else:
                text = str(item)
            
            if total_chars + len(text) > max_chars:
                break
            
            lines.append(text)
            total_chars += len(text)
        
        return "\n".join(lines)


class HybridRetriever:
    """
    Hybrid retrieval system combining L2 and L3 with MMR.
    """
    
    def __init__(
        self,
        memory_repository: MemoryRepository,
        knowledge_graph: KnowledgeGraph,
        mmr_lambda: float = 0.5,
        l2_weight: float = 0.5,
        l3_weight: float = 0.5,
        enable_graph_expansion: bool = True,
        max_expansion_hops: int = 1
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            memory_repository: L2 memory store
            knowledge_graph: L3 knowledge graph
            mmr_lambda: MMR diversity parameter
            l2_weight: Weight for L2 results in final ranking
            l3_weight: Weight for L3 results in final ranking
            enable_graph_expansion: Whether to expand L3 results via traversal
            max_expansion_hops: Maximum hops for graph expansion
        """
        self.memory_repo = memory_repository
        self.knowledge_graph = knowledge_graph
        self.mmr = MaximalMarginalRelevance(lambda_param=mmr_lambda)
        self.l2_weight = l2_weight
        self.l3_weight = l3_weight
        self.enable_graph_expansion = enable_graph_expansion
        self.max_expansion_hops = max_expansion_hops
    
    async def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        l2_k: Optional[int] = None,
        l3_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None  # ⭐ NEW: Filter for L2 queries
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval.
        
        Args:
            query_embedding: Query vector
            k: Total number of final results
            l2_k: Number of L2 candidates (default: 2*k)
            l3_k: Number of L3 seed candidates (default: k)
            metadata_filter: Metadata filter for L2 queries (e.g., {"player_name": "Alice"})
            
        Returns:
            RetrievalResult with combined and raw results
        """
        l2_k = l2_k or k * 2
        l3_k = l3_k or k
        
        # Step 1: Query L2 (episodic memories) WITH METADATA FILTER
        l2_results = await self.memory_repo.retrieve(
            query_embedding, 
            k=l2_k,
            metadata_filter=metadata_filter  # ⭐ Pass filter to L2!
        )
        
        # Step 2: Query L3 (knowledge graph)
        l3_seed_results = self.knowledge_graph.query_by_embedding(query_embedding, k=l3_k)
        
        # Step 3: Expand L3 via graph traversal
        l3_results = list(l3_seed_results)
        if self.enable_graph_expansion and l3_seed_results:
            seen_ids = set(node.id for node, _ in l3_results)
            for node, _ in l3_seed_results:
                expanded = self.knowledge_graph.traverse(
                    node.id, 
                    max_hops=self.max_expansion_hops,
                    max_nodes=3
                )
                for exp_node in expanded:
                    if exp_node.id not in seen_ids:
                        seen_ids.add(exp_node.id)
                        # Compute similarity for expanded nodes
                        sim = self._cosine_similarity(query_embedding, exp_node.embedding)
                        l3_results.append((exp_node, sim * 0.8))  # Discount expanded
        
        # Step 4: Combine candidates
        candidates = []
        
        for memory, score in l2_results:
            weighted_score = score * self.l2_weight
            candidates.append((memory, memory.embedding, weighted_score))
        
        for node, score in l3_results:
            weighted_score = score * self.l3_weight
            candidates.append((node, node.embedding, weighted_score))
        
        # Step 5: Re-rank with MMR
        if candidates:
            final_results = self.mmr.rerank(candidates, query_embedding, k)
        else:
            final_results = []
        
        # Compute diversity score
        if final_results:
            final_embeddings = []
            for item, _ in final_results:
                if isinstance(item, Memory):
                    final_embeddings.append(item.embedding)
                elif isinstance(item, L3Node):
                    final_embeddings.append(item.embedding)
            diversity = compute_intra_list_diversity(final_embeddings)
        else:
            diversity = 0.0
        
        return RetrievalResult(
            final_results=final_results,
            l2_results=l2_results,
            l3_results=l3_results,
            l2_count=len(l2_results),
            l3_count=len(l3_results),
            diversity_score=diversity
        )
    
    def retrieve_sync(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        l2_k: Optional[int] = None,
        l3_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Synchronous version of retrieve for non-async contexts.
        
        Use this when calling from synchronous code (e.g., regular NPC class).
        For async contexts (e.g., benchmark scripts), use retrieve() instead.
        
        Args:
            query_embedding: Query vector
            k: Total number of final results
            l2_k: Number of L2 candidates (default: 2*k)
            l3_k: Number of L3 seed candidates (default: k)
            metadata_filter: Metadata filter for L2 queries
            
        Returns:
            RetrievalResult with combined and raw results
        """
        l2_k = l2_k or k * 2
        l3_k = l3_k or k
        
        # Step 1: Query L2 (episodic memories) - Use sync method
        # Check if repository has async retrieve (FAISSGPUMemoryRepository)
        # If so, use retrieve_sync; otherwise use regular retrieve
        if hasattr(self.memory_repo, 'retrieve_sync'):
            l2_results = self.memory_repo.retrieve_sync(
                query_embedding, 
                k=l2_k,
                metadata_filter=metadata_filter
            )
        else:
            # Regular MemoryRepository has synchronous retrieve
            l2_results = self.memory_repo.retrieve(
                query_embedding, 
                k=l2_k,
                metadata_filter=metadata_filter
            )
        
        # Step 2: Query L3 (knowledge graph)
        l3_seed_results = self.knowledge_graph.query_by_embedding(query_embedding, k=l3_k)
        
        # Step 3: Expand L3 via graph traversal
        l3_results = list(l3_seed_results)
        if self.enable_graph_expansion and l3_seed_results:
            seen_ids = set(node.id for node, _ in l3_results)
            for node, _ in l3_seed_results:
                expanded = self.knowledge_graph.traverse(
                    node.id, 
                    max_hops=self.max_expansion_hops,
                    max_nodes=3
                )
                for exp_node in expanded:
                    if exp_node.id not in seen_ids:
                        seen_ids.add(exp_node.id)
                        # Compute similarity for expanded nodes
                        sim = self._cosine_similarity(query_embedding, exp_node.embedding)
                        l3_results.append((exp_node, sim * 0.8))  # Discount expanded
        
        # Step 4: Combine candidates
        candidates = []
        
        for memory, score in l2_results:
            weighted_score = score * self.l2_weight
            candidates.append((memory, memory.embedding, weighted_score))
        
        for node, score in l3_results:
            weighted_score = score * self.l3_weight
            candidates.append((node, node.embedding, weighted_score))
        
        # Step 5: Re-rank with MMR
        if candidates:
            final_results = self.mmr.rerank(candidates, query_embedding, k)
        else:
            final_results = []
        
        # Compute diversity score
        if final_results:
            final_embeddings = []
            for item, _ in final_results:
                if isinstance(item, Memory):
                    final_embeddings.append(item.embedding)
                elif isinstance(item, L3Node):
                    final_embeddings.append(item.embedding)
            diversity = compute_intra_list_diversity(final_embeddings)
        else:
            diversity = 0.0
        
        return RetrievalResult(
            final_results=final_results,
            l2_results=l2_results,
            l3_results=l3_results,
            l2_count=len(l2_results),
            l3_count=len(l3_results),
            diversity_score=diversity
        )
    
    async def retrieve_text(
        self,
        query_text: str,
        embedding_service,
        k: int = 5
    ) -> RetrievalResult:
        """
        Convenience method for text queries (async version).
        
        Args:
            query_text: Text query
            embedding_service: EmbeddingService instance
            k: Number of results
            
        Returns:
            RetrievalResult
        """
        query_embedding = embedding_service.encode(query_text)
        return await self.retrieve(query_embedding, k=k)
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
