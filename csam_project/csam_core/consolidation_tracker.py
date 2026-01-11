"""
Consolidation Tracker - Tracks which L2 memories contributed to which L3 nodes.

This module is essential for our novel contribution: consolidation-aware forgetting.
By tracking which memories have been "absorbed" into L3, we can safely forget
the original detailed memories without losing information.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationRecord:
    """
    Record of a consolidation event.
    
    Tracks which L2 memories were consolidated into which L3 node,
    and how semantically similar they are.
    """
    l3_node_id: str
    source_memory_ids: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    # Semantic coverage: how similar is the L3 summary to each source memory
    coverage_scores: Dict[str, float] = field(default_factory=dict)


class ConsolidationTracker:
    """
    Tracks consolidation relationships between L2 memories and L3 nodes.
    
    This is crucial for our novel contribution:
    - C(m) = Consolidation Coverage = max similarity to L3 nodes that used m
    - D(m) = Redundancy with L3 = max similarity to any L3 node
    """
    
    def __init__(self):
        """Initialize the tracker."""
        # memory_id -> list of L3 node IDs it contributed to
        self._memory_to_l3: Dict[str, List[str]] = {}
        
        # l3_node_id -> list of memory IDs that contributed to it
        self._l3_to_memories: Dict[str, List[str]] = {}
        
        # Consolidation records
        self._records: Dict[str, ConsolidationRecord] = {}
        
        # Cache of coverage scores: memory_id -> max coverage score
        self._coverage_cache: Dict[str, float] = {}
    
    def record_consolidation(
        self,
        l3_node_id: str,
        source_memory_ids: List[str],
        l3_embedding: np.ndarray,
        memory_embeddings: Dict[str, np.ndarray]
    ) -> ConsolidationRecord:
        """
        Record a consolidation event.
        
        Args:
            l3_node_id: ID of the created L3 node
            source_memory_ids: IDs of L2 memories that contributed
            l3_embedding: Embedding of the L3 node content
            memory_embeddings: Embeddings of source memories {id: embedding}
            
        Returns:
            The consolidation record
        """
        # Compute coverage scores (similarity between each memory and the L3 node)
        coverage_scores = {}
        for mem_id in source_memory_ids:
            if mem_id in memory_embeddings:
                similarity = self._cosine_similarity(
                    l3_embedding, 
                    memory_embeddings[mem_id]
                )
                coverage_scores[mem_id] = float(similarity)
        
        # Create record
        record = ConsolidationRecord(
            l3_node_id=l3_node_id,
            source_memory_ids=source_memory_ids,
            coverage_scores=coverage_scores
        )
        self._records[l3_node_id] = record
        
        # Update mappings
        self._l3_to_memories[l3_node_id] = source_memory_ids
        for mem_id in source_memory_ids:
            if mem_id not in self._memory_to_l3:
                self._memory_to_l3[mem_id] = []
            self._memory_to_l3[mem_id].append(l3_node_id)
            
            # Update coverage cache
            if mem_id in coverage_scores:
                current_max = self._coverage_cache.get(mem_id, 0.0)
                self._coverage_cache[mem_id] = max(current_max, coverage_scores[mem_id])
        
        logger.debug(
            f"Recorded consolidation: {len(source_memory_ids)} memories -> L3 node {l3_node_id[:8]}..."
        )
        return record
    
    def get_coverage(self, memory_id: str) -> float:
        """
        Get the consolidation coverage C(m) for a memory.
        
        C(m) = max{sim(embed(m), embed(s)) : s ∈ L3 nodes that used m}
        
        If the memory was never consolidated, returns 0.0
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Coverage score between 0.0 and 1.0
        """
        return self._coverage_cache.get(memory_id, 0.0)
    
    def get_coverage_batch(self, memory_ids: List[str]) -> Dict[str, float]:
        """
        Get coverage scores for multiple memories.
        
        Args:
            memory_ids: List of memory IDs
            
        Returns:
            Dictionary of {memory_id: coverage_score}
        """
        return {
            mem_id: self._coverage_cache.get(mem_id, 0.0)
            for mem_id in memory_ids
        }
    
    def is_consolidated(self, memory_id: str) -> bool:
        """Check if a memory has been consolidated at all."""
        return memory_id in self._memory_to_l3
    
    def get_l3_nodes_for_memory(self, memory_id: str) -> List[str]:
        """Get L3 nodes that this memory contributed to."""
        return self._memory_to_l3.get(memory_id, [])
    
    def get_memories_for_l3_node(self, l3_node_id: str) -> List[str]:
        """Get memories that contributed to an L3 node."""
        return self._l3_to_memories.get(l3_node_id, [])
    
    def get_statistics(self) -> Dict[str, any]:
        """Get tracker statistics."""
        if not self._memory_to_l3:
            return {
                "total_consolidated_memories": 0,
                "total_l3_nodes": 0,
                "avg_coverage": 0.0
            }
        
        coverages = list(self._coverage_cache.values())
        return {
            "total_consolidated_memories": len(self._memory_to_l3),
            "total_l3_nodes": len(self._l3_to_memories),
            "avg_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
            "max_coverage": max(coverages) if coverages else 0.0,
            "min_coverage": min(coverages) if coverages else 0.0
        }
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
