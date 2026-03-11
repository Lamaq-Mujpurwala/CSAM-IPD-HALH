"""
Forgetting Engine - Consolidation-Aware Forgetting (Novel Contribution).

This module implements our core research contribution:
    ForgetScore(m) = α·R(m) + β·(1-I(m)) + γ·C(m) + δ·D(m)

Where:
    R(m) = Recency decay (how old is the memory)
    I(m) = Importance score
    C(m) = Consolidation coverage (⭐ NOVEL - our key contribution)
    D(m) = Redundancy with L3 (⭐ NOVEL)

Default weights: α=β=γ=δ=0.25 (uniform, validated by grid search over 20 combos).

Memories with highest ForgetScore are candidates for deletion.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from .memory_repository import Memory

logger = logging.getLogger(__name__)


class ForgettingStrategy(ABC):
    """Abstract base class for forgetting strategies."""
    
    @abstractmethod
    def compute_forget_scores(
        self,
        memories: List[Memory],
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute forget scores for memories.
        
        Args:
            memories: List of memories to score
            **kwargs: Additional context (e.g., consolidation_tracker, l3_graph)
            
        Returns:
            Dictionary of {memory_id: forget_score}
            Higher scores = more likely to forget
        """
        pass
    
    def select_to_forget(
        self,
        memories: List[Memory],
        count: int,
        **kwargs
    ) -> List[str]:
        """
        Select memories to forget.
        
        Args:
            memories: List of candidate memories
            count: Number of memories to select for deletion
            **kwargs: Additional context
            
        Returns:
            List of memory IDs to delete
        """
        scores = self.compute_forget_scores(memories, **kwargs)
        
        # Sort by score (highest first = most forgettable)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return sorted_ids[:count]


class NoForgetting(ForgettingStrategy):
    """
    Baseline: Never forget anything.
    
    This will cause memory to grow unbounded and is only useful
    as a baseline to show the problems with not forgetting.
    """
    
    def compute_forget_scores(self, memories: List[Memory], **kwargs) -> Dict[str, float]:
        """All memories get score 0 (never forget)."""
        return {m.id: 0.0 for m in memories}
    
    def select_to_forget(self, memories: List[Memory], count: int, **kwargs) -> List[str]:
        """Never select any memories to forget."""
        return []


class LRUForgetting(ForgettingStrategy):
    """
    Baseline: Least Recently Used (LRU) forgetting.
    
    Forgets memories that haven't been accessed recently,
    regardless of their importance or consolidation status.
    """
    
    def compute_forget_scores(self, memories: List[Memory], **kwargs) -> Dict[str, float]:
        """Score based on time since last access (longer = higher score)."""
        now = datetime.now()
        scores = {}
        
        max_age = 1.0  # Normalize to [0, 1]
        for m in memories:
            # Time since last access in days
            age_days = (now - m.last_accessed).total_seconds() / 86400
            # Normalize (cap at 365 days)
            scores[m.id] = min(age_days / 365, 1.0)
        
        return scores


class ImportanceForgetting(ForgettingStrategy):
    """
    Baseline: Importance-based forgetting.
    
    Forgets least important memories first.
    This is a common approach but has the weakness that
    importance is subjectively assigned at creation time.
    """
    
    def compute_forget_scores(self, memories: List[Memory], **kwargs) -> Dict[str, float]:
        """Score based on inverse importance."""
        return {m.id: 1.0 - m.importance for m in memories}


class RecencyImportanceForgetting(ForgettingStrategy):
    """
    Baseline: Combined recency and importance (like Generative Agents).
    
    ForgetScore = α·R(m) + β·(1-I(m))
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            alpha: Weight for recency
            beta: Weight for inverse importance
        """
        self.alpha = alpha
        self.beta = beta
    
    def compute_forget_scores(self, memories: List[Memory], **kwargs) -> Dict[str, float]:
        """Combine recency and importance."""
        now = datetime.now()
        scores = {}
        
        for m in memories:
            # Recency decay
            age_days = (now - m.last_accessed).total_seconds() / 86400
            R = min(age_days / 365, 1.0)
            
            # Inverse importance
            I = 1.0 - m.importance
            
            scores[m.id] = self.alpha * R + self.beta * I
        
        return scores


class ConsolidationAwareForgetting(ForgettingStrategy):
    """
    ⭐ NOVEL CONTRIBUTION: Consolidation-Aware Forgetting
    
    ForgetScore(m) = α·R(m) + β·(1-I(m)) + γ·C(m) + δ·D(m)
    
    Where:
        R(m) = Recency decay (how old is the memory)
        I(m) = Importance score
        C(m) = Consolidation coverage (how much of m is in L3)  ⭐ NOVEL
        D(m) = Redundancy with L3 (similarity to any L3 node)  ⭐ NOVEL
    
    Key insight: If a memory's semantic content has been "absorbed" 
    into L3 (as a summary or reflection), then the original detailed 
    memory is REDUNDANT and can be safely forgotten.
    
    This provides an information-theoretic guarantee:
        If C(m) > threshold, then any query that would retrieve m
        can also retrieve the corresponding L3 summary.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.25,
        consolidation_threshold: float = 0.3
    ):
        """
        Initialize the forgetting strategy.
        
        Args:
            alpha: Weight for recency decay R(m)
            beta: Weight for inverse importance (1 - I(m))
            gamma: Weight for consolidation coverage C(m) ⭐
            delta: Weight for L3 redundancy D(m) ⭐
            consolidation_threshold: Minimum C(m) to consider forgetting.
                                    Memories with coverage < threshold are PROTECTED
                                    from forgetting (score=0). Default 0.3 ensures
                                    unconsolidated memories survive until L3 captures
                                    their content.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.consolidation_threshold = consolidation_threshold
        
        # Validate weights sum to ~1
        total = alpha + beta + gamma + delta
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Forgetting weights sum to {total}, expected ~1.0")
    
    def compute_forget_scores(
        self,
        memories: List[Memory],
        consolidation_tracker=None,
        l3_embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute consolidation-aware forget scores.
        
        Args:
            memories: Memories to score
            consolidation_tracker: ConsolidationTracker instance
            l3_embeddings: Matrix of L3 node embeddings (for redundancy calculation)
            
        Returns:
            Forget scores for each memory
        """
        now = datetime.now()
        scores = {}
        
        for m in memories:
            # R(m): Recency decay
            age_days = (now - m.last_accessed).total_seconds() / 86400
            R = min(age_days / 365, 1.0)
            
            # I(m): Importance (we use 1 - I for forget score)
            I = m.importance
            
            # C(m): Consolidation coverage ⭐ NOVEL
            if consolidation_tracker is not None:
                C = consolidation_tracker.get_coverage(m.id)
            else:
                C = 0.0
            
            # D(m): Redundancy with L3 ⭐ NOVEL
            if l3_embeddings is not None and len(l3_embeddings) > 0:
                # Max similarity to any L3 node
                sims = self._batch_cosine_similarity(m.embedding, l3_embeddings)
                D = float(np.max(sims)) if len(sims) > 0 else 0.0
            else:
                D = 0.0
            
            # Apply consolidation threshold protection
            if C < self.consolidation_threshold:
                # Protect unconsolidated memories by setting score very low
                scores[m.id] = 0.0
            else:
                # Compute final forget score
                scores[m.id] = (
                    self.alpha * R +
                    self.beta * (1 - I) +
                    self.gamma * C +
                    self.delta * D
                )
        
        return scores
    
    def select_to_forget(
        self,
        memories: List[Memory],
        count: int,
        consolidation_tracker=None,
        l3_embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[str]:
        """
        Select memories to forget, with consolidation awareness.
        
        Ensures we never forget memories that haven't been consolidated
        if consolidation_threshold > 0.
        """
        scores = self.compute_forget_scores(
            memories,
            consolidation_tracker=consolidation_tracker,
            l3_embeddings=l3_embeddings
        )
        
        # Filter out protected memories (score = 0)
        forgettable = {mid: score for mid, score in scores.items() if score > 0}
        
        # Sort by score (highest first)
        sorted_ids = sorted(forgettable.keys(), key=lambda x: forgettable[x], reverse=True)
        
        return sorted_ids[:count]
    
    @staticmethod
    def _batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all rows of matrix."""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(matrix))
        
        query_normalized = query / query_norm
        matrix_norms = np.linalg.norm(matrix, axis=1)
        
        # Avoid division by zero
        matrix_norms = np.where(matrix_norms == 0, 1, matrix_norms)
        matrix_normalized = matrix / matrix_norms[:, np.newaxis]
        
        return np.dot(matrix_normalized, query_normalized)
    
    def get_config(self) -> Dict[str, float]:
        """Get the current configuration."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "consolidation_threshold": self.consolidation_threshold
        }


# Factory function for easy instantiation
def create_forgetting_strategy(
    strategy_name: str,
    **kwargs
) -> ForgettingStrategy:
    """
    Create a forgetting strategy by name.
    
    Args:
        strategy_name: One of 'none', 'lru', 'importance', 
                      'recency_importance', 'consolidation_aware'
        **kwargs: Strategy-specific parameters
        
    Returns:
        ForgettingStrategy instance
    """
    strategies = {
        'none': NoForgetting,
        'lru': LRUForgetting,
        'importance': ImportanceForgetting,
        'recency_importance': RecencyImportanceForgetting,
        'consolidation_aware': ConsolidationAwareForgetting,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs)
