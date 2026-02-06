"""
Async GPU-Accelerated Embedding Service

This module provides GPU-accelerated text embeddings with async batching support.
- Uses PyTorch backend for GPU acceleration
- Async API for concurrent batching
- Automatic device management (GPU/CPU fallback)
"""

import numpy as np
import asyncio
from typing import List, Union, Optional
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncGPUEmbeddingService:
    """
    GPU-accelerated embedding service with async batching.
    
    Features:
    - GPU acceleration via PyTorch backend
    - Async batch encoding for high throughput
    - Automatic fallback to CPU if GPU unavailable
    - Thread-safe concurrent encoding
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 32,
        max_workers: int = 4
    ):
        """
        Initialize the GPU embedding service.
        
        Args:
            model_name: Sentence-transformer model name
            device: 'cuda' for GPU, 'cpu' for CPU
            batch_size: Default batch size for encoding
            max_workers: Thread pool size for async operations
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()
        
    @property
    def model(self):
        """Lazy load the model with GPU support."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                
                # Check CUDA availability
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    self.device = "cpu"
                
                # Load model to specified device
                self._model = SentenceTransformer(self.model_name, device=self.device)
                
                # Get dimension
                test_embedding = self._model.encode("test", convert_to_numpy=True)
                self._dimension = len(test_embedding)
                
                logger.info(
                    f"Model loaded on {self.device}. "
                    f"Embedding dimension: {self._dimension}"
                )
                
                # Log GPU info if available
                if self.device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                    
            except ImportError as e:
                raise ImportError(
                    f"Required packages not installed: {e}\n"
                    "Install with: pip install sentence-transformers torch"
                )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Force load
        return self._dimension
    
    def _encode_sync(
        self,
        text: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Synchronous encoding (runs in thread pool)."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        batch_size = batch_size or self.batch_size
        
        # Encode with model
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False  # Keep raw embeddings
        )
        
        # Return single embedding if input was single
        # Ensure 1D array for single inputs: (384,) not (1, 384)
        if is_single:
            return embeddings[0].astype(np.float32)
        return embeddings.astype(np.float32)
    
    async def encode(
        self,
        text: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Async encode text(s) into vector embedding(s).
        
        Args:
            text: Single string or list of strings
            batch_size: Override default batch size
            
        Returns:
            numpy array of embeddings
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._encode_sync,
            text,
            batch_size
        )
    
    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Async batch encoding (optimized for throughput).
        
        Args:
            texts: List of strings to encode
            batch_size: Batch size (larger = more GPU efficient)
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        # Larger batches for better GPU utilization
        batch_size = batch_size or min(len(texts), self.batch_size * 2)
        return await self.encode(texts, batch_size=batch_size)
    
    async def encode_concurrent(
        self,
        text_batches: List[List[str]]
    ) -> List[np.ndarray]:
        """
        Encode multiple batches concurrently.
        
        Args:
            text_batches: List of text batches
            
        Returns:
            List of embedding arrays
        """
        tasks = [self.encode_batch(batch) for batch in text_batches]
        return await asyncio.gather(*tasks)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def move_to_device(self, device: str):
        """Move model to different device (cuda/cpu)."""
        if self._model is not None:
            logger.info(f"Moving model to {device}")
            self.device = device
            self._model = self._model.to(device)
    
    def get_device_info(self) -> dict:
        """Get current device information."""
        info = {
            "current_device": self.device,
            "model_loaded": self._model is not None
        }
        
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    info["cuda_available"] = True
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9
                    info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) / 1e9
                else:
                    info["cuda_available"] = False
            except ImportError:
                info["cuda_available"] = False
        
        return info
    
    async def warmup(self, sample_texts: Optional[List[str]] = None):
        """
        Warmup the model with sample texts to initialize GPU kernels.
        
        Args:
            sample_texts: Optional sample texts (defaults to generic samples)
        """
        if sample_texts is None:
            sample_texts = [
                "Hello, this is a warmup text.",
                "The quick brown fox jumps over the lazy dog.",
                "GPU acceleration test message."
            ]
        
        logger.info("Warming up embedding model...")
        await self.encode_batch(sample_texts)
        logger.info("Warmup complete")
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Backward-compatible synchronous wrapper
class GPUEmbeddingService:
    """
    Synchronous wrapper around AsyncGPUEmbeddingService.
    
    Provides same API as original EmbeddingService but with GPU support.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self._async_service = AsyncGPUEmbeddingService(
            model_name=model_name,
            device=device
        )
        self.model_name = model_name
        self.device = device
    
    @property
    def model(self):
        return self._async_service.model
    
    @property
    def dimension(self) -> int:
        return self._async_service.dimension
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Synchronous encode (blocks until complete)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._async_service.encode(text))
            return result
        finally:
            loop.close()
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Synchronous batch encode."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._async_service.encode_batch(texts, batch_size)
            )
            return result
        finally:
            loop.close()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return self._async_service.similarity(embedding1, embedding2)
    
    def get_device_info(self) -> dict:
        return self._async_service.get_device_info()


# Factory function
def create_embedding_service(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda",
    async_mode: bool = True
) -> Union[AsyncGPUEmbeddingService, GPUEmbeddingService]:
    """
    Create an embedding service.
    
    Args:
        model_name: Model to use
        device: 'cuda' or 'cpu'
        async_mode: If True, return async service; if False, return sync wrapper
        
    Returns:
        Embedding service instance
    """
    if async_mode:
        return AsyncGPUEmbeddingService(model_name=model_name, device=device)
    else:
        return GPUEmbeddingService(model_name=model_name, device=device)
