"""
Maximal Marginal Relevance (MMR) - Diversity Mechanism.

MMR re-ranks retrieved results to balance relevance with diversity,
preventing redundant memories from dominating the context.

Formula:
    MMR = argmax[λ·sim(d, q) - (1-λ)·max(sim(d, d_i)) for d_i in S]

Where:
    q = query
    d = candidate document
    S = already selected documents
    λ = diversity parameter (0 = pure diversity, 1 = pure relevance)
"""

import numpy as np
from typing import List, Tuple, Any, TypeVar, Callable
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for items being ranked


class MaximalMarginalRelevance:
    """
    MMR re-ranker for balancing relevance and diversity.
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR.
        
        Args:
            lambda_param: Balance parameter
                         1.0 = pure relevance (no diversity)
                         0.0 = pure diversity (ignore relevance)
                         0.5 = equal balance (recommended)
        """
        self.lambda_param = lambda_param
    
    def rerank(
        self,
        candidates: List[Tuple[T, np.ndarray, float]],
        query_embedding: np.ndarray,
        k: int
    ) -> List[Tuple[T, float]]:
        """
        Re-rank candidates using MMR.
        
        Args:
            candidates: List of (item, embedding, relevance_score) tuples
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (item, mmr_score) tuples, re-ranked for diversity
        """
        if len(candidates) == 0:
            return []
        
        k = min(k, len(candidates))
        
        # Extract embeddings and scores
        items = [c[0] for c in candidates]
        embeddings = np.array([c[1] for c in candidates])
        relevance_scores = np.array([c[2] for c in candidates])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms[:, np.newaxis]
        
        # Track selected and remaining
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Precompute similarity matrix
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        results = []
        
        for _ in range(k):
            if not remaining_indices:
                break
            
            best_score = float('-inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Relevance term
                relevance = relevance_scores[idx]
                
                # Diversity term (max similarity to already selected)
                if selected_indices:
                    max_sim_to_selected = max(
                        similarity_matrix[idx, sel_idx]
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim_to_selected = 0.0
                
                # MMR score
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_sim_to_selected
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                results.append((items[best_idx], float(best_score)))
        
        return results
    
    def rerank_simple(
        self,
        items: List[T],
        embeddings: List[np.ndarray],
        relevance_scores: List[float],
        k: int
    ) -> List[Tuple[T, float]]:
        """
        Simplified interface for re-ranking.
        
        Args:
            items: List of items (any type)
            embeddings: List of embedding vectors
            relevance_scores: List of relevance scores
            k: Number of results
            
        Returns:
            Re-ranked (item, score) tuples
        """
        candidates = list(zip(items, embeddings, relevance_scores))
        # Query embedding doesn't matter for already-computed relevance
        return self.rerank(candidates, embeddings[0], k)


def compute_intra_list_diversity(embeddings: List[np.ndarray]) -> float:
    """
    Compute average pairwise dissimilarity of a list.
    
    Used to measure how diverse the results are.
    Higher = more diverse.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Average pairwise dissimilarity (0-1, higher = more diverse)
    """
    if len(embeddings) < 2:
        return 1.0  # Single item is maximally "diverse"
    
    embeddings = np.array(embeddings)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1)
    norms = np.where(norms == 0, 1, norms)
    embeddings_norm = embeddings / norms[:, np.newaxis]
    
    # Compute similarity matrix
    sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Get upper triangle (excluding diagonal)
    n = len(embeddings)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]
    
    # Average dissimilarity = 1 - average similarity
    avg_dissimilarity = 1.0 - np.mean(pairwise_sims)
    
    return float(avg_dissimilarity)
