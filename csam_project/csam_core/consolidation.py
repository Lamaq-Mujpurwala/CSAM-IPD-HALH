"""
Consolidation Pipeline - Summarizes L2 memories into L3 knowledge.

This module handles:
1. Selecting memories for consolidation (based on age, count, etc.)
2. Summarizing groups of related memories using LLM
3. Extracting entities and relationships
4. Storing results in L3 knowledge graph
5. Updating consolidation tracker (for forgetting)

The consolidation process is what enables our novel forgetting mechanism -
once memories are consolidated, we can safely forget the originals.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .memory_repository import Memory, MemoryRepository
from .knowledge_graph import KnowledgeGraph
from .consolidation_tracker import ConsolidationTracker
from .services.llm import LLMService
from .services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class ConsolidationPipeline:
    """
    Pipeline for consolidating L2 memories into L3 knowledge.
    
    The consolidation process:
    1. Group related memories (by time or semantic similarity)
    2. Summarize each group using LLM
    3. Extract entities and relationships
    4. Store in L3 knowledge graph
    5. Track which memories contributed (for forgetting)
    """
    
    def __init__(
        self,
        memory_repository: MemoryRepository,
        knowledge_graph: KnowledgeGraph,
        consolidation_tracker: ConsolidationTracker,
        embedding_service: EmbeddingService,
        llm_service: Optional[LLMService] = None,
        min_memories_per_batch: int = 3,
        max_memories_per_batch: int = 10,
        consolidation_threshold_hours: float = 1.0,
        use_llm_for_consolidation: bool = True
    ):
        """
        Initialize the consolidation pipeline.
        
        Args:
            memory_repository: L2 memory store
            knowledge_graph: L3 knowledge graph
            consolidation_tracker: Tracker for consolidation relationships
            embedding_service: For embedding summaries
            llm_service: For summarization (optional - uses simple fallback if None)
            min_memories_per_batch: Minimum memories to trigger consolidation
            max_memories_per_batch: Maximum memories per consolidation batch
            consolidation_threshold_hours: Minimum age for memories to consolidate
            use_llm_for_consolidation: If False, skip LLM calls (summary/entity) for speed
        """
        self.memory_repo = memory_repository
        self.knowledge_graph = knowledge_graph
        self.tracker = consolidation_tracker
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.min_memories_per_batch = min_memories_per_batch
        self.max_memories_per_batch = max_memories_per_batch
        self.consolidation_threshold_hours = consolidation_threshold_hours
        self.use_llm_for_consolidation = use_llm_for_consolidation
    
    def should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        candidates = self._get_consolidation_candidates()
        return len(candidates) >= self.min_memories_per_batch
    
    def _get_consolidation_candidates(self) -> List[Memory]:
        """Get memories that are candidates for consolidation."""
        now = datetime.now()
        threshold = now - timedelta(hours=self.consolidation_threshold_hours)
        
        candidates = []
        for memory in self.memory_repo.get_all():
            # Skip already consolidated
            if memory.consolidated:
                continue
            # Skip too recent
            if memory.timestamp > threshold:
                continue
            candidates.append(memory)
        
        # Sort by timestamp (oldest first)
        candidates.sort(key=lambda m: m.timestamp)
        return candidates
    
    def run_consolidation(self) -> Dict[str, Any]:
        """
        Run the consolidation pipeline.
        
        Returns:
            Dictionary with consolidation statistics
        """
        candidates = self._get_consolidation_candidates()
        
        if len(candidates) < self.min_memories_per_batch:
            return {
                "status": "skipped",
                "reason": f"Not enough candidates ({len(candidates)} < {self.min_memories_per_batch})"
            }
        
        # Group memories into batches
        batches = self._group_memories(candidates)
        
        summaries_created = 0
        entities_created = 0
        memories_consolidated = 0
        
        for batch in batches:
            result = self._consolidate_batch(batch)
            summaries_created += result.get("summaries", 0)
            entities_created += result.get("entities", 0)
            memories_consolidated += result.get("memories", 0)
        
        return {
            "status": "completed",
            "batches_processed": len(batches),
            "summaries_created": summaries_created,
            "entities_created": entities_created,
            "memories_consolidated": memories_consolidated
        }
    
    def _group_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """
        Group memories into batches for consolidation.
        
        Uses simple time-based grouping. Could be enhanced with
        semantic clustering for better groupings.
        """
        batches = []
        current_batch = []
        
        for memory in memories:
            current_batch.append(memory)
            
            if len(current_batch) >= self.max_memories_per_batch:
                batches.append(current_batch)
                current_batch = []
        
        # Don't forget the last batch if it has enough memories
        if len(current_batch) >= self.min_memories_per_batch:
            batches.append(current_batch)
        
        return batches
    
    def _consolidate_batch(self, memories: List[Memory]) -> Dict[str, int]:
        """
        Consolidate a batch of memories.
        
        Creates:
        1. A summary node in L3
        2. Entity nodes for extracted entities
        3. Edges between summary and entities
        """
        result = {"summaries": 0, "entities": 0, "memories": 0}
        
        # Get memory texts
        memory_texts = [m.text for m in memories]
        memory_ids = [m.id for m in memories]
        memory_embeddings = {m.id: m.embedding for m in memories}
        
        # Generate summary
        summary_text = self._generate_summary(memory_texts)
        
        if not summary_text:
            logger.warning("Failed to generate summary")
            return result
        
        # Create summary embedding
        summary_embedding = self.embedding_service.encode(summary_text)
        
        # Add summary to L3
        summary_node_id = self.knowledge_graph.add_node(
            content=summary_text,
            embedding=summary_embedding,
            node_type="summary",
            source_memory_ids=memory_ids
        )
        result["summaries"] = 1
        
        # Record consolidation in tracker
        self.tracker.record_consolidation(
            l3_node_id=summary_node_id,
            source_memory_ids=memory_ids,
            l3_embedding=summary_embedding,
            memory_embeddings=memory_embeddings
        )
        
        # Mark memories as consolidated
        for memory in memories:
            memory.mark_consolidated(summary_node_id)
        result["memories"] = len(memories)
        
        # Extract entities (if LLM available AND enabled)
        if self.use_llm_for_consolidation and self.llm_service and self.llm_service.is_available():
            entities_result = self._extract_and_store_entities(
                memory_texts, 
                summary_node_id
            )
            result["entities"] = entities_result.get("count", 0)
        
        logger.info(
            f"Consolidated {len(memories)} memories into summary {summary_node_id[:8]}..."
        )
        
        return result
    
    def _generate_summary(self, memory_texts: List[str]) -> str:
        """Generate a summary from memory texts."""
        
        # Use LLM if available AND enabled
        if self.use_llm_for_consolidation and self.llm_service and self.llm_service.is_available():
            return self.llm_service.summarize(memory_texts)
        
        # Fallback: simple concatenation-based summary
        # (Just take first sentence of each memory)
        summary_parts = []
        for text in memory_texts[:5]:  # Limit to first 5
            # Get first sentence
            sentences = text.split('.')
            if sentences:
                summary_parts.append(sentences[0].strip())
        
        if summary_parts:
            return ". ".join(summary_parts) + "."
        
        return memory_texts[0] if memory_texts else ""
    
    def _extract_and_store_entities(
        self, 
        memory_texts: List[str],
        summary_node_id: str
    ) -> Dict[str, Any]:
        """Extract entities from memories and store in L3."""
        
        result = {"count": 0}
        
        # Combine texts for extraction
        combined_text = " ".join(memory_texts)
        
        # Extract entities using LLM
        extraction = self.llm_service.extract_entities(combined_text)
        
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])
        
        # Store entities
        entity_id_map = {}  # name -> node_id
        
        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "Concept")
            
            if not name:
                continue
            
            # Check if entity already exists
            existing = self._find_existing_entity(name)
            
            if existing:
                entity_id_map[name] = existing
            else:
                # Create new entity node
                entity_embedding = self.embedding_service.encode(name)
                entity_id = self.knowledge_graph.add_node(
                    content=name,
                    embedding=entity_embedding,
                    node_type="entity",
                    metadata={"entity_type": entity_type}
                )
                entity_id_map[name] = entity_id
                result["count"] += 1
            
            # Link entity to summary
            self.knowledge_graph.add_edge(
                summary_node_id,
                entity_id_map[name],
                edge_type="about"
            )
        
        # Store relationships between entities
        for rel in relationships:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            rel_type = rel.get("type", "related_to")
            
            if source_name in entity_id_map and target_name in entity_id_map:
                self.knowledge_graph.add_edge(
                    entity_id_map[source_name],
                    entity_id_map[target_name],
                    edge_type=rel_type
                )
        
        return result
    
    def _find_existing_entity(self, name: str) -> Optional[str]:
        """Find an existing entity node by name."""
        # Simple search - could be improved with fuzzy matching
        name_embedding = self.embedding_service.encode(name)
        results = self.knowledge_graph.query_by_embedding(name_embedding, k=1)
        
        if results:
            node, similarity = results[0]
            # High similarity threshold for entity matching
            if similarity > 0.9 and node.node_type == "entity":
                return node.id
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        all_memories = self.memory_repo.get_all()
        consolidated = sum(1 for m in all_memories if m.consolidated)
        candidates = len(self._get_consolidation_candidates())
        
        return {
            "total_memories": len(all_memories),
            "consolidated_memories": consolidated,
            "pending_candidates": candidates,
            "consolidation_ratio": consolidated / len(all_memories) if all_memories else 0,
            "l3_nodes": len(self.knowledge_graph),
            "tracker_stats": self.tracker.get_statistics()
        }
