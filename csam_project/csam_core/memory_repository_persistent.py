"""
Persistent FAISS Memory Repository Wrapper

Adds save/load functionality to FAISSGPUMemoryRepository for production use.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .memory_repository_gpu import FAISSGPUMemoryRepository, Memory

logger = logging.getLogger(__name__)


class PersistentFAISSRepository(FAISSGPUMemoryRepository):
    """
    FAISS repository with disk persistence.
    
    Extends FAISSGPUMemoryRepository to add:
    - Save/load FAISS index to disk
    - Save/load metadata to disk  
    - Auto-save after N additions
    - Clean shutdown
    """
    
    def __init__(
        self,
        embedding_dim: int,
        max_memories: int = 10000,
        use_gpu: bool = True,
        index_type: str = "HNSW32",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        auto_save_interval: int = 100,
        forgetting_strategy: Optional[Any] = None  # \u2b50 Pass to parent
    ):
        """
        Initialize persistent repository.
        
        Args:
            embedding_dim: Embedding dimension
            max_memories: Max memories to store
            use_gpu: Use GPU acceleration
            index_type: FAISS index type
            index_path: Path to save FAISS index (None = no persistence)
            metadata_path: Path to save metadata (None = no persistence)
            auto_save_interval: Auto-save every N additions (0 = manual only)
            forgetting_strategy: ForgettingStrategy from forgetting_engine.py
        """
        # Store paths
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.auto_save_interval = auto_save_interval
        self._save_counter = 0
        
        # Try to load from disk first
        if self.index_path and self.index_path.exists():
            super().__init__(
                embedding_dim=embedding_dim,
                max_memories=max_memories,
                use_gpu=use_gpu,
                gpu_id=0,
                index_type=index_type,
                forgetting_strategy=forgetting_strategy
            )
            self._load_from_disk()
        else:
            # Create new
            super().__init__(
                embedding_dim=embedding_dim,
                max_memories=max_memories,
                use_gpu=use_gpu,
                gpu_id=0,
                index_type=index_type,
                forgetting_strategy=forgetting_strategy
            )
            logger.info("Created new persistent repository")
    
    def _load_from_disk(self):
        """Load FAISS index and metadata from disk."""
        try:
            import faiss
            
            logger.info(f"Loading FAISS index from {self.index_path}...")
            
            # Load CPU index
            cpu_index = faiss.read_index(str(self.index_path))
            
            # Move to GPU if needed
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        res = faiss.StandardGpuResources()
                        self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                        logger.info("Loaded index to GPU")
                    else:
                        self._index = cpu_index
                        logger.info("GPU not available, using CPU")
                except ImportError:
                    self._index = cpu_index
                    logger.info("PyTorch not available, using CPU")
            else:
                self._index = cpu_index
            
            # Load metadata
            if self.metadata_path and self.metadata_path.exists():
                logger.info(f"Loading metadata from {self.metadata_path}...")
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self._memories = data['memories']
                    self._id_to_index = data['id_to_index']
                    self._index_to_id = data['index_to_id']
                    self._next_index = data['next_index']
            
            logger.info(f"[OK] Loaded {len(self._memories)} memories from disk")
            
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
            logger.info("Starting with empty repository")
    
    def save_to_disk(self):
        """Save FAISS index and metadata to disk."""
        if not self.index_path:
            return  # No save path configured
        
        try:
            import faiss
            
            # Create directories
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            if self.metadata_path:
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get CPU index for saving
            if self.use_gpu and self._index is not None:
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self._index)
                except:
                    cpu_index = self._index  # Already on CPU
            else:
                cpu_index = self._index
            
            # Save FAISS index
            if cpu_index is not None:
                faiss.write_index(cpu_index, str(self.index_path))
                logger.debug(f"Saved index to {self.index_path}")
            
            # Save metadata
            if self.metadata_path:
                data = {
                    'memories': self._memories,
                    'id_to_index': self._id_to_index,
                    'index_to_id': self._index_to_id,
                    'next_index': self._next_index
                }
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug(f"Saved metadata to {self.metadata_path}")
            
            logger.info(f"[OK] Saved {len(self._memories)} memories to disk")
            
        except Exception as e:
            logger.error(f"Error saving to disk: {e}")
    
    def add(
        self,
        text: str,
        embedding: np.ndarray,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add memory with auto-save."""
        # Call parent add
        memory_id = super().add(text, embedding, importance, metadata)
        
        # Auto-save if interval reached
        if self.auto_save_interval > 0:
            self._save_counter += 1
            if self._save_counter >= self.auto_save_interval:
                self.save_to_disk()
                self._save_counter = 0
        
        return memory_id
    
    def shutdown(self):
        """Clean shutdown with final save."""
        logger.info("Shutting down persistent repository...")
        self.save_to_disk()
        
        # Cleanup executor
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("Shutdown complete")
    
    def cleanup(self):
        """Delete saved files from disk."""
        if self.index_path and self.index_path.exists():
            self.index_path.unlink()
            logger.info(f"Deleted {self.index_path}")
        
        if self.metadata_path and self.metadata_path.exists():
            self.metadata_path.unlink()
            logger.info(f"Deleted {self.metadata_path}")
        
        logger.info("Cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with auto-save."""
        self.shutdown()
        return False
