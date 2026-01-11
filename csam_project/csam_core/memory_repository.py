"""
Memory Repository - L2 Episodic Memory Store with HNSW Indexing.

This is the core episodic memory store that provides:
- O(log N) approximate nearest neighbor retrieval via HNSW
- Efficient storage of raw interaction memories
- Support for forgetting strategies

This module is central to CSAM's scalability claims.
"""

import numpy as np
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """
    A single episodic memory in L2.
    
    Attributes:
        id: Unique identifier
        text: Raw text of the memory
        embedding: Vector embedding of the text
        importance: Importance score (0.0 to 1.0)
        timestamp: When the memory was created
        last_accessed: When the memory was last retrieved
        access_count: Number of times retrieved
        consolidated: Whether this memory has been consolidated into L3
        consolidation_ids: IDs of L3 nodes this memory contributed to
        metadata: Additional metadata
    """
    id: str
    text: str
    embedding: np.ndarray
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    consolidated: bool = False
    consolidation_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary (for serialization)."""
        return {
            "id": self.id,
            "text": self.text,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "consolidated": self.consolidated,
            "consolidation_ids": self.consolidation_ids,
            "metadata": self.metadata
        }
    
    def mark_accessed(self):
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def mark_consolidated(self, l3_node_id: str):
        """Mark this memory as having contributed to an L3 node."""
        self.consolidated = True
        if l3_node_id not in self.consolidation_ids:
            self.consolidation_ids.append(l3_node_id)


class MemoryRepository:
    """
    L2 Episodic Memory Store with HNSW indexing.
    
    Provides O(log N) approximate nearest neighbor search for retrieving
    relevant memories based on semantic similarity.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        max_memories: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50
    ):
        """
        Initialize the memory repository.
        
        Args:
            embedding_dim: Dimension of embedding vectors (384 for MiniLM)
            max_memories: Maximum number of memories (for HNSW initialization)
            ef_construction: HNSW construction parameter (higher = better quality, slower build)
            M: HNSW connections per layer (higher = better quality, more memory)
            ef_search: HNSW search parameter (higher = better recall, slower search)
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        
        # Memory storage
        self._memories: Dict[str, Memory] = {}
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._next_index: int = 0
        
        # HNSW index (lazy initialization)
        self._index = None
    
    @property
    def index(self):
        """Lazy load HNSW index."""
        if self._index is None:
            try:
                import hnswlib
                self._index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
                self._index.init_index(
                    max_elements=self.max_memories,
                    ef_construction=self.ef_construction,
                    M=self.M
                )
                self._index.set_ef(self.ef_search)
                logger.info(f"Initialized HNSW index: dim={self.embedding_dim}, max={self.max_memories}")
            except ImportError:
                raise ImportError(
                    "hnswlib is required for HNSW indexing. "
                    "Install with: pip install hnswlib"
                )
        return self._index
    
    def __len__(self) -> int:
        """Return number of memories stored."""
        return len(self._memories)
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory to the repository.
        
        Args:
            text: Raw text of the memory
            embedding: Vector embedding of the text
            importance: Importance score (0.0 to 1.0)
            metadata: Optional additional metadata
            
        Returns:
            ID of the created memory
        """
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            text=text,
            embedding=embedding.astype(np.float32),
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {}
        )
        
        # Store memory
        self._memories[memory_id] = memory
        
        # Add to HNSW index
        index_pos = self._next_index
        self._id_to_index[memory_id] = index_pos
        self._index_to_id[index_pos] = memory_id
        self._next_index += 1
        
        # Add vector to index
        self.index.add_items(
            embedding.reshape(1, -1).astype(np.float32),
            np.array([index_pos])
        )
        
        logger.debug(f"Added memory {memory_id[:8]}... (total: {len(self)})")
        return memory_id
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        return self._memories.get(memory_id)
    
    def get_all(self) -> List[Memory]:
        """Get all memories."""
        return list(self._memories.values())
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        update_access: bool = True
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve the k most similar memories to the query.
        
        This is the core O(log N) operation using HNSW.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            update_access: Whether to update access tracking
            
        Returns:
            List of (Memory, similarity_score) tuples, sorted by similarity
        """
        if len(self) == 0:
            return []
        
        # Limit k to available memories
        k = min(k, len(self))
        
        # HNSW search - returns (indices, distances)
        # For cosine space, distance = 1 - similarity
        indices, distances = self.index.knn_query(
            query_embedding.reshape(1, -1).astype(np.float32),
            k=k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            memory_id = self._index_to_id.get(int(idx))
            if memory_id and memory_id in self._memories:
                memory = self._memories[memory_id]
                similarity = 1.0 - dist  # Convert distance to similarity
                
                if update_access:
                    memory.mark_accessed()
                
                results.append((memory, float(similarity)))
        
        return results
    
    def retrieve_by_text(
        self,
        query_text: str,
        embedding_service,
        k: int = 5
    ) -> List[Tuple[Memory, float]]:
        """
        Convenience method to retrieve using text query.
        
        Args:
            query_text: Text query
            embedding_service: EmbeddingService instance
            k: Number of results
            
        Returns:
            List of (Memory, similarity) tuples
        """
        query_embedding = embedding_service.encode(query_text)
        return self.retrieve(query_embedding, k=k)
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Note: HNSW doesn't support true deletion, so we just remove from our dict.
        The vector remains in the index but won't be returned.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if memory_id not in self._memories:
            return False
        
        # Remove from our storage
        del self._memories[memory_id]
        
        # Note: We can't remove from HNSW, but since we check _memories
        # in retrieve(), it effectively won't be returned
        logger.debug(f"Deleted memory {memory_id[:8]}... (remaining: {len(self)})")
        return True
    
    def delete_batch(self, memory_ids: List[str]) -> int:
        """
        Delete multiple memories.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            Number of memories deleted
        """
        deleted = 0
        for memory_id in memory_ids:
            if self.delete(memory_id):
                deleted += 1
        return deleted
    
    def get_memories_for_consolidation(
        self,
        min_count: int = 5,
        max_age_hours: float = 24.0,
        exclude_consolidated: bool = True
    ) -> List[Memory]:
        """
        Get memories that are candidates for consolidation.
        
        Args:
            min_count: Minimum number of memories to return
            max_age_hours: Maximum age in hours to consider
            exclude_consolidated: Whether to exclude already consolidated memories
            
        Returns:
            List of memories ready for consolidation
        """
        now = datetime.now()
        candidates = []
        
        for memory in self._memories.values():
            # Skip if already consolidated and we're excluding
            if exclude_consolidated and memory.consolidated:
                continue
            
            # Check age
            age_hours = (now - memory.timestamp).total_seconds() / 3600
            if age_hours <= max_age_hours:
                candidates.append(memory)
        
        # Sort by timestamp (oldest first)
        candidates.sort(key=lambda m: m.timestamp)
        
        return candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        if len(self) == 0:
            return {
                "total_memories": 0,
                "consolidated_count": 0,
                "avg_importance": 0.0,
                "avg_access_count": 0.0
            }
        
        consolidated = sum(1 for m in self._memories.values() if m.consolidated)
        avg_importance = sum(m.importance for m in self._memories.values()) / len(self)
        avg_access = sum(m.access_count for m in self._memories.values()) / len(self)
        
        return {
            "total_memories": len(self),
            "consolidated_count": consolidated,
            "consolidation_ratio": consolidated / len(self),
            "avg_importance": avg_importance,
            "avg_access_count": avg_access
        }
    
    def benchmark_retrieval(self, n_queries: int = 100) -> Dict[str, float]:
        """
        Benchmark retrieval performance.
        
        Args:
            n_queries: Number of queries to run
            
        Returns:
            Dictionary with timing statistics
        """
        if len(self) == 0:
            return {"error": "No memories to benchmark"}
        
        # Generate random query vectors
        queries = np.random.randn(n_queries, self.embedding_dim).astype(np.float32)
        
        # Warm up
        self.index.knn_query(queries[0].reshape(1, -1), k=5)
        
        # Benchmark
        latencies = []
        for query in queries:
            start = time.perf_counter()
            self.index.knn_query(query.reshape(1, -1), k=5)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            "n_memories": len(self),
            "n_queries": n_queries,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99)
        }
