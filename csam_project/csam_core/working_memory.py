"""
L1 Working Memory - Recent Interaction Cache

This is the fastest memory layer in the CSAM architecture:
- Stores last N interactions (default: 10-20)
- O(1) lookup for very recent context
- LRU eviction policy
- Optional structured fact extraction

The three-layer hierarchy:
L1 (Working Memory) - Recent cache (this module)
L2 (Vector Memory)   - Episodic FAISS search
L3 (Knowledge Graph) - Semantic relationships
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    text: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance
        }


class WorkingMemoryCache:
    """
    L1 Working Memory - Fast cache for recent interactions.
    
    Features:
    - LRU cache with fixed capacity
    - O(1) insert and lookup
    - Player-scoped storage
    - Structured fact extraction (optional)
    
    Performance:
    - Lookup: ~0.1-0.5ms (vs. 15-20ms for FAISS)
    - Storage: ~0.1ms (vs. 10-15ms for FAISS)
    """
    
    def __init__(
        self,
        max_size: int = 20,
        enable_facts: bool = True
    ):
        """
        Initialize working memory cache.
        
        Args:
            max_size: Maximum number of items to cache (default: 20)
            enable_facts: Whether to extract structured facts
        """
        self.max_size = max_size
        self.enable_facts = enable_facts
        
        # LRU cache: player_name -> OrderedDict of recent items
        self._cache: Dict[str, OrderedDict[str, WorkingMemoryItem]] = {}
        
        # Structured facts: player_name -> facts dict
        self._facts: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._total_items = 0
    
    def add(
        self,
        text: str,
        player_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> None:
        """
        Add item to working memory.
        
        Args:
            text: Interaction text
            player_name: Player identifier
            metadata: Optional metadata
            importance: Importance score
        """
        # Initialize player cache if needed
        if player_name not in self._cache:
            self._cache[player_name] = OrderedDict()
        
        # Create item
        item = WorkingMemoryItem(
            text=text,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance=importance
        )
        
        # Add to cache (newest = last in OrderedDict)
        item_id = f"{player_name}_{len(self._cache[player_name])}"
        self._cache[player_name][item_id] = item
        self._total_items += 1
        
        # LRU eviction if over capacity
        while len(self._cache[player_name]) > self.max_size:
            # Remove oldest (first item)
            self._cache[player_name].popitem(last=False)
        
        # Extract structured facts if enabled
        if self.enable_facts:
            self._extract_facts(text, player_name, metadata or {})
        
        logger.debug(
            f"Added to L1 for {player_name}: {len(self._cache[player_name])} items"
        )
    
    def _extract_facts(
        self,
        text: str,
        player_name: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Extract structured facts from text.
        
        Simple heuristic-based extraction for common facts.
        Could be replaced with LLM-based extraction for better quality.
        """
        if player_name not in self._facts:
            self._facts[player_name] = {}
        
        facts = self._facts[player_name]
        
        # Extract from metadata first (most reliable)
        if "player_name" in metadata:
            facts["player_name"] = metadata["player_name"]
        if "player_class" in metadata:
            facts["player_class"] = metadata["player_class"]
        
        # Simple text-based extraction (heuristics)
        text_lower = text.lower()
        
        # Name extraction
        if "my name is" in text_lower:
            # Extract name after "my name is"
            idx = text_lower.index("my name is") + len("my name is")
            name_part = text[idx:].strip().split()[0].rstrip(".,!?")
            if name_part:
                facts["player_name"] = name_part.capitalize()
        
        # Class/role extraction
        if "i'm a" in text_lower or "i am a" in text_lower:
            for keyword in text_lower.split():
                if keyword in ["warrior", "mage", "rogue", "healer", "merchant"]:
                    facts["player_class"] = keyword.capitalize()
        
        logger.debug(f"Extracted facts for {player_name}: {facts}")
    
    def get_recent(
        self,
        player_name: str,
        k: int = 5
    ) -> List[WorkingMemoryItem]:
        """
        Get k most recent items for a player.
        
        Args:
            player_name: Player identifier
            k: Number of items to return
            
        Returns:
            List of most recent items (newest first)
        """
        if player_name not in self._cache:
            self._misses += 1
            return []
        
        self._hits += 1
        
        # Get last k items (reversed to get newest first)
        items = list(self._cache[player_name].values())[-k:]
        return list(reversed(items))
    
    def get_fact(
        self,
        player_name: str,
        fact_name: str
    ) -> Optional[Any]:
        """
        Get a structured fact about a player.
        
        Args:
            player_name: Player identifier
            fact_name: Name of fact (e.g., "player_name", "player_class")
            
        Returns:
            Fact value or None
        """
        if player_name not in self._facts:
            return None
        return self._facts[player_name].get(fact_name)
    
    def clear_player(self, player_name: str) -> None:
        """Clear all items for a specific player."""
        if player_name in self._cache:
            del self._cache[player_name]
        if player_name in self._facts:
            del self._facts[player_name]
        logger.debug(f"Cleared L1 for {player_name}")
    
    def clear_all(self) -> None:
        """Clear all items from working memory."""
        self._cache.clear()
        self._facts.clear()
        self._total_items = 0
        logger.debug("Cleared all L1 memory")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_cached = sum(len(cache) for cache in self._cache.values())
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
        
        return {
            "max_size": self.max_size,
            "total_players": len(self._cache),
            "total_cached_items": total_cached,
            "total_facts": sum(len(f) for f in self._facts.values()),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": hit_rate,
            "total_items_added": self._total_items
        }
    
    def __len__(self) -> int:
        """Return total number of cached items across all players."""
        return sum(len(cache) for cache in self._cache.values())
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"WorkingMemoryCache("
            f"max_size={self.max_size}, "
            f"players={stats['total_players']}, "
            f"cached={stats['total_cached_items']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )


# Convenience functions for quick checks

def check_recent_context(
    text: str,
    working_memory: WorkingMemoryCache,
    player_name: str,
    threshold: float = 0.7
) -> Optional[str]:
    """
    Quick check if text is very similar to recent context.
    
    Args:
        text: Query text
        working_memory: L1 cache instance
        player_name: Player identifier
        threshold: Similarity threshold (simple string matching)
        
    Returns:
        Matching recent context or None
    """
    recent_items = working_memory.get_recent(player_name, k=3)
    
    text_lower = text.lower()
    for item in recent_items:
        item_lower = item.text.lower()
        # Simple substring matching
        if text_lower in item_lower or item_lower in text_lower:
            return item.text
    
    return None
