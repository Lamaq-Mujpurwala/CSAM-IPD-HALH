"""
GPU-Accelerated Memory Repository using FAISS

Replaces HNSW with FAISS-GPU for faster retrieval at scale.
Maintains same API as original memory_repository.py for compatibility.
"""

import numpy as np
import time
import uuid
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# \u2b50 Import forgetting engine for sophisticated forgetting
try:
    from .forgetting_engine import ForgettingStrategy, ImportanceForgetting
except ImportError:
    # Fallback if module structure is different
    from csam_core.forgetting_engine import ForgettingStrategy, ImportanceForgetting



@dataclass
class Memory:
    """
    A single episodic memory in L2.
    (Identical to original - maintains compatibility)
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
        """Convert memory to dictionary."""
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
        """Mark this memory as consolidated."""
        self.consolidated = True
        if l3_node_id not in self.consolidation_ids:
            self.consolidation_ids.append(l3_node_id)


class FAISSGPUMemoryRepository:
    """
    GPU-accelerated L2 memory store using FAISS.
    
    Features:
    - FAISS-GPU index for fast similarity search
    - Async retrieval methods
    - Batch operations for efficiency
    - Automatic GPU/CPU fallback
    """
   
    def __init__(
        self,
        embedding_dim: int = 384,
        max_memories: int = 100000,
        use_gpu: bool = False,  # Default to CPU (faiss-cpu package)
        gpu_id: int = 0,
        index_type: str = "HNSW32",  # or "IVF1024,PQ64"
        forgetting_strategy: Optional[ForgettingStrategy] = None  # ⭐ NEW!
    ):
        """
        Initialize the FAISS-GPU memory repository.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            max_memories: Maximum capacity
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID (if multiple GPUs)
            index_type: FAISS index type (HNSW32 recommended for quality)
            forgetting_strategy: Strategy from forgetting_engine.py (defaults to ImportanceForgetting)
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.index_type = index_type
        self.forgetting_strategy = forgetting_strategy or ImportanceForgetting()  # ⭐ Use strategy!
        
        # Memory storage
        self._memories: Dict[str, Memory] = {}
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._next_index: int = 0
        
        # FAISS index (lazy initialization)
        self._index = None
        self._gpu_resources = None
        
        # Async executor
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def index(self):
        """Lazy load FAISS index."""
        if self._index is None:
            try:
                import faiss
                
                # Create base index
                if self.index_type.startswith("HNSW"):
                    # HNSW index (good quality, GPU compatible)
                    M = int(self.index_type.replace("HNSW", ""))
                    self._index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                    self._index.hnsw.efConstruction = 200
                    self._index.hnsw.efSearch = 50
                elif self.index_type.startswith("IVF"):
                    # IVF index (needs training, very fast)
                    index = faiss.index_factory(self.embedding_dim, self.index_type)
                    self._index = index
                else:
                    # Flat index (exact search, slower for large datasets)
                    self._index = faiss.IndexFlatL2(self.embedding_dim)
                
                # Move to GPU if requested
                if self.use_gpu:
                    if not hasattr(faiss, 'StandardGpuResources'):
                        logger.warning(
                            "FAISS-GPU not available. Install with: "
                            "conda install -c pytorch faiss-gpu"
                        )
                        self.use_gpu = False
                    else:
                        try:
                            self._gpu_resources = faiss.StandardGpuResources()
                            # Convert to GPU index
                            self._index = faiss.index_cpu_to_gpu(
                                self._gpu_resources,
                                self.gpu_id,
                                self._index
                            )
                            logger.info(
                                f"FAISS index on GPU {self.gpu_id}: "
                                f"type={self.index_type}, dim={self.embedding_dim}"
                            )
                        except RuntimeError as e:
                            logger.warning(f"GPU init failed, using CPU: {e}")
                            self.use_gpu = False
                
                if not self.use_gpu:
                    logger.info(
                        f"FAISS index on CPU: "
                        f"type={self.index_type}, dim={self.embedding_dim}"
                    )
                    
            except ImportError:
                raise ImportError(
                    "FAISS is required. Install with: pip install faiss-gpu (GPU) or faiss-cpu (CPU)"
                )
        
        return self._index
    
    def __len__(self) -> int:
        """Return number of memories."""
        return len(self._memories)
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory with automatic forgetting when limit is reached.
        
        Args:
            text: Memory text
            embedding: Vector embedding
            importance: Importance score (0-1)
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        # CHECK IF WE NEED TO FORGET (USE FORGETTING STRATEGY)
        if len(self._memories) >= self.max_memories:
            # Use forgetting strategy to select which memory to evict
            memories_list = list(self._memories.values())
            
            # Select 1 memory to forget
            to_forget = self.forgetting_strategy.select_to_forget(
                memories=memories_list,
                count=1
            )
            
            if to_forget:
                memory_id_to_forget = to_forget[0]
                logger.info(f"Forgetting memory {memory_id_to_forget[:8]}... (limit: {self.max_memories})")
                self._remove_memory(memory_id_to_forget)
        
        memory_id = str(uuid.uuid4())
        
        memory = Memory(
            id=memory_id,
            text=text,
            embedding=embedding.astype(np.float32),
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {}
        )
        
        self._memories[memory_id] = memory
        
        # Add to FAISS index
        index_pos = self._next_index
        self._id_to_index[memory_id] = index_pos
        self._index_to_id[index_pos] = memory_id
        self._next_index += 1
        
        # Add vector to index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        logger.debug(f"Added memory {memory_id[:8]}... (total: {len(self)})")
        return memory_id
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory from the repository."""
        if memory_id not in self._memories:
            return
        
        # Remove from mappings
        if memory_id in self._id_to_index:
            index_pos = self._id_to_index[memory_id]
            del self._id_to_index[memory_id]
            del self._index_to_id[index_pos]
        
        # Remove from memories
        del self._memories[memory_id]
        
        # Note: FAISS doesn't support individual deletions efficiently
        # The index will have a "hole" but it won't affect retrieval
        # For production, consider periodic full rebuild
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID."""
        return self._memories.get(memory_id)
    
    def get_all(self) -> List[Memory]:
        """Get all memories."""
        return list(self._memories.values())
    
    def _retrieve_sync(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        update_access: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Memory, float]]:
        """Synchronous retrieval (runs in thread pool)."""
        if len(self) == 0:
            return []
        
        # If filtering, retrieve more candidates
        k_search = min(k * 3, len(self)) if metadata_filter else min(k, len(self))
        
        # FAISS search (returns distances, indices)
        # For L2 distance, smaller = more similar
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            k_search
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            memory_id = self._index_to_id.get(int(idx))
            if memory_id and memory_id in self._memories:
                memory = self._memories[memory_id]
                
                # Apply metadata filter if provided
                if metadata_filter:
                    match = all(
                        memory.metadata.get(key) == value
                        for key, value in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                # Convert L2 distance to similarity (inverse)
                # Using: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + float(dist))
                
                if update_access:
                    memory.mark_accessed()
                
                results.append((memory, similarity))
                
                # Stop if we have enough results after filtering
                if len(results) >= k:
                    break
        
        return results
    
    async def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        update_access: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Async retrieve the k most similar memories.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            update_access: Whether to update access tracking
            metadata_filter: Optional filter dict (e.g., {"player_name": "Bob"})
            
        Returns:
            List of (Memory, similarity_score) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._retrieve_sync,
            query_embedding,
            k,
            update_access,
            metadata_filter
        )
    
    async def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> List[List[Tuple[Memory, float]]]:
        """
        Retrieve for multiple queries concurrently.
        
        Args:
            query_embeddings: Array of shape (n_queries, dim)
            k: Results per query
            
        Returns:
            List of retrieval results for each query
        """
        tasks = [
            self.retrieve(emb, k, update_access=False)
            for emb in query_embeddings
        ]
        return await asyncio.gather(*tasks)
    
    # Synchronous retrieve for backward compatibility
    def retrieve_sync(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        update_access: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Memory, float]]:
        """Synchronous retrieve (backward compatible)."""
        return self._retrieve_sync(query_embedding, k, update_access, metadata_filter)
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Note: FAISS doesn't support true deletion from the index,
        so we just remove from our dict storage.
        """
        if memory_id not in self._memories:
            return False
        
        del self._memories[memory_id]
        logger.debug(f"Deleted memory {memory_id[:8]}... (remaining: {len(self)})")
        return True
    
    def delete_batch(self, memory_ids: List[str]) -> int:
        """Delete multiple memories."""
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
        """Get memories ready for consolidation."""
        now = datetime.now()
        candidates = []
        
        for memory in self._memories.values():
            if exclude_consolidated and memory.consolidated:
                continue
            
            age_hours = (now - memory.timestamp).total_seconds() / 3600
            if age_hours <= max_age_hours:
                candidates.append(memory)
        
        candidates.sort(key=lambda m: m.timestamp)
        return candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        if len(self) == 0:
            return {
                "total_memories": 0,
                "consolidated_count": 0,
                "avg_importance": 0.0,
                "avg_access_count": 0.0,
                "using_gpu": self.use_gpu
            }
        
        consolidated = sum(1 for m in self._memories.values() if m.consolidated)
        avg_importance = sum(m.importance for m in self._memories.values()) / len(self)
        avg_access = sum(m.access_count for m in self._memories.values()) / len(self)
        
        return {
            "total_memories": len(self),
            "consolidated_count": consolidated,
            "consolidation_ratio": consolidated / len(self),
            "avg_importance": avg_importance,
            "avg_access_count": avg_access,
            "using_gpu": self.use_gpu,
            "index_type": self.index_type
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU/device info."""
        info = {
            "using_gpu": self.use_gpu,
            "gpu_id": self.gpu_id if self.use_gpu else None,
            "index_type": self.index_type,
            "total_memories": len(self)
        }
        
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(self.gpu_id)
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated(self.gpu_id) / 1e9
                    info["gpu_memory_reserved"] = torch.cuda.memory_reserved(self.gpu_id) / 1e9
            except:
                pass
        
        return info
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
