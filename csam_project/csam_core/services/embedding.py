"""
Embedding Service - Sentence-Transformer wrapper for vector embeddings.

This module provides text-to-vector encoding using sentence-transformers,
which runs locally without any cloud dependencies.
"""

import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wrapper for sentence-transformers embedding model.
    
    Runs entirely locally - no API keys or cloud services needed.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model.
                       Default 'all-MiniLM-L6-v2' is fast and has 384 dimensions.
                       Alternative: 'all-mpnet-base-v2' (768 dims, better quality)
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get dimension from a test encoding
                test_embedding = self._model.encode("test")
                self._dimension = len(test_embedding)
                logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            # Force model load to get dimension
            _ = self.model
        return self._dimension
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into vector embedding(s).
        
        Args:
            text: Single string or list of strings to encode.
            
        Returns:
            numpy array of shape (dimension,) for single text,
            or (n_texts, dimension) for multiple texts.
        """
        if isinstance(text, str):
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings.astype(np.float32)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of texts efficiently.
        
        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            
        Returns:
            numpy array of shape (n_texts, dimension)
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.astype(np.float32)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


# Singleton instance for convenience
_default_service = None

def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Get the default embedding service instance."""
    global _default_service
    if _default_service is None or _default_service.model_name != model_name:
        _default_service = EmbeddingService(model_name)
    return _default_service
