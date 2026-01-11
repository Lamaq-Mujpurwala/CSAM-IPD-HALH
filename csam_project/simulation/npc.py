"""
NPC Class - Game NPC with CSAM Memory System

Each NPC has:
- Distinct personality (name, role, traits, speaking style)
- Independent CSAM memory (L2 episodic + L3 semantic)
- Consolidation-aware forgetting
- LLM-powered response generation
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from csam_core.memory_repository import MemoryRepository
from csam_core.knowledge_graph import KnowledgeGraph
from csam_core.forgetting_engine import ConsolidationAwareForgetting
from csam_core.consolidation_tracker import ConsolidationTracker
from csam_core.consolidation import ConsolidationPipeline
from csam_core.retrieval import HybridRetriever
from csam_core.services.embedding import EmbeddingService
from csam_core.services.llm import LLMService


@dataclass
class NPCPersonality:
    """Defines an NPC's personality and behavior."""
    name: str
    role: str  # e.g., "bartender", "merchant", "guard"
    traits: List[str] = field(default_factory=list)  # e.g., ["friendly", "gossips"]
    background: str = ""  # Brief backstory
    speaking_style: str = "casual"  # How they talk
    greeting: str = ""  # Default greeting
    
    @property
    def system_prompt(self) -> str:
        """Generate system prompt for LLM."""
        traits_str = ", ".join(self.traits) if self.traits else "helpful"
        return f"""You are {self.name}, a {self.role} in a fantasy tavern.

Personality: {traits_str}
Background: {self.background}
Speaking style: {self.speaking_style}

Guidelines:
- Stay in character at all times
- Reference past conversations when relevant
- Be brief and conversational (1-3 sentences max)
- If you remember something about the player, mention it naturally
- If asked about something you don't know, admit it honestly"""


class NPC:
    """
    An NPC with CSAM-powered memory.
    
    Each NPC maintains independent memory and can:
    - Remember player interactions over long conversations
    - Consolidate important memories into semantic knowledge
    - Forget irrelevant details while retaining key information
    - Generate personality-consistent responses
    """
    
    def __init__(
        self,
        personality: NPCPersonality,
        embedding_service: EmbeddingService,
        llm_service: Optional[LLMService] = None,
        max_memories: int = 10000,
        forget_threshold: int = 500
    ):
        """
        Initialize an NPC with CSAM memory.
        
        Args:
            personality: NPC's personality definition
            embedding_service: Shared embedding service
            llm_service: LLM for response generation (optional)
            max_memories: Maximum memory capacity
            forget_threshold: When to trigger forgetting
        """
        self.personality = personality
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.forget_threshold = forget_threshold
        
        # Initialize CSAM components
        self.memory_repo = MemoryRepository(
            embedding_dim=embedding_service.dimension,
            max_memories=max_memories
        )
        self.knowledge_graph = KnowledgeGraph(
            db_path=":memory:",
            embedding_dim=embedding_service.dimension
        )
        self.consolidation_tracker = ConsolidationTracker()
        self.forgetting_strategy = ConsolidationAwareForgetting(
            alpha=0.2, beta=0.2, gamma=0.3, delta=0.3
        )
        
        # Consolidation pipeline
        self.consolidation_pipeline = ConsolidationPipeline(
            memory_repository=self.memory_repo,
            knowledge_graph=self.knowledge_graph,
            consolidation_tracker=self.consolidation_tracker,
            embedding_service=embedding_service,
            llm_service=llm_service,
            min_memories_per_batch=5,
            max_memories_per_batch=10,
            consolidation_threshold_hours=0.0  # Immediate for demo
        )
        
        # Retriever
        self.retriever = HybridRetriever(
            memory_repository=self.memory_repo,
            knowledge_graph=self.knowledge_graph,
            mmr_lambda=0.5
        )
        
        # Stats
        self.conversation_count = 0
        self.total_response_time_ms = 0.0
    
    def add_memory(self, text: str, importance: float = 0.5) -> str:
        """Add a memory to this NPC's memory system."""
        embedding = self.embedding_service.encode(text)
        memory_id = self.memory_repo.add(text, embedding, importance)
        
        # Check if we need to forget
        if len(self.memory_repo) > self.forget_threshold:
            self._run_forgetting()
        
        return memory_id
    
    def _run_forgetting(self):
        """Run the forgetting strategy."""
        excess = len(self.memory_repo) - self.forget_threshold
        forget_count = max(excess, int(self.forget_threshold * 0.1))
        
        memories = self.memory_repo.get_all()
        l3_embeddings = self.knowledge_graph.get_embeddings_matrix()
        
        to_forget = self.forgetting_strategy.select_to_forget(
            memories,
            count=forget_count,
            consolidation_tracker=self.consolidation_tracker,
            l3_embeddings=l3_embeddings
        )
        
        self.memory_repo.delete_batch(to_forget)
    
    def run_consolidation(self):
        """Manually trigger consolidation."""
        return self.consolidation_pipeline.run_consolidation()
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context for a query."""
        query_embedding = self.embedding_service.encode(query)
        result = self.retriever.retrieve(query_embedding, k=k)
        
        context_parts = []
        for item, score in result.final_results:
            if hasattr(item, 'text'):  # Memory
                context_parts.append(f"- {item.text}")
            elif hasattr(item, 'content'):  # L3Node
                context_parts.append(f"- [Knowledge] {item.content}")
        
        return "\n".join(context_parts) if context_parts else "No relevant memories."
    
    def respond(self, player_message: str, player_name: str = "Player") -> Dict[str, Any]:
        """
        Generate a response to the player.
        
        Args:
            player_message: What the player said
            player_name: Player's name (if known)
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant context
        context = self.retrieve_context(player_message, k=5)
        
        # Step 2: Generate response
        if self.llm_service and self.llm_service.is_available():
            prompt = f"""Based on your memories and knowledge:
{context}

{player_name} says: "{player_message}"

Respond naturally as {self.personality.name}:"""
            
            response = self.llm_service.generate(
                prompt,
                system_prompt=self.personality.system_prompt,
                temperature=0.7,
                max_tokens=150
            )
        else:
            # Fallback without LLM
            response = self._generate_fallback_response(player_message, context)
        
        # Step 3: Save the interaction to memory
        # Save with higher importance for direct interactions
        self.add_memory(
            f"{player_name} said: {player_message}",
            importance=0.7
        )
        self.add_memory(
            f"I ({self.personality.name}) responded: {response}",
            importance=0.5
        )
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.conversation_count += 1
        self.total_response_time_ms += elapsed_ms
        
        # Run consolidation periodically
        if self.conversation_count % 20 == 0:
            self.run_consolidation()
        
        return {
            "response": response,
            "latency_ms": elapsed_ms,
            "memory_count": len(self.memory_repo),
            "context_used": context
        }
    
    def _generate_fallback_response(self, message: str, context: str) -> str:
        """Generate response without LLM (for testing)."""
        # Check for specific patterns in context
        message_lower = message.lower()
        
        if "remember" in message_lower:
            if "No relevant memories" not in context:
                return f"Yes, I remember that. {context.split('- ')[1] if '- ' in context else 'Let me think...'}"
            return "I don't recall that specifically, sorry."
        
        if "name" in message_lower and "my name" in message_lower:
            # Look for name in context
            if "name is" in context.lower():
                return "Of course I remember your name! It's good to see you again."
            return f"Nice to meet you! I'm {self.personality.name}."
        
        if "hello" in message_lower or "hi" in message_lower:
            return self.personality.greeting or f"Hello there! Welcome to my establishment. I'm {self.personality.name}."
        
        return f"I hear you. {self.personality.greeting or 'How can I help you today?'}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get NPC statistics."""
        consolidated = sum(1 for m in self.memory_repo.get_all() if m.consolidated)
        return {
            "name": self.personality.name,
            "role": self.personality.role,
            "memory_count": len(self.memory_repo),
            "consolidated_count": consolidated,
            "consolidation_ratio": consolidated / len(self.memory_repo) if len(self.memory_repo) > 0 else 0,
            "l3_nodes": len(self.knowledge_graph),
            "conversation_count": self.conversation_count,
            "avg_response_time_ms": self.total_response_time_ms / max(1, self.conversation_count)
        }


# Pre-defined NPC personalities for the Tavern scenario
TAVERN_NPCS = [
    NPCPersonality(
        name="Greta",
        role="bartender",
        traits=["friendly", "observant", "good listener", "remembers regulars"],
        background="Has run the Golden Dragon tavern for 20 years. Knows everyone in town.",
        speaking_style="warm and welcoming, uses nicknames",
        greeting="Welcome to the Golden Dragon! What can I get for you today?"
    ),
    NPCPersonality(
        name="Marcus",
        role="merchant",
        traits=["shrewd", "business-minded", "fair", "remembers deals"],
        background="Traveling merchant who visits the tavern weekly. Deals in rare goods.",
        speaking_style="direct and businesslike, but honest",
        greeting="Ah, a potential customer! Looking to buy or sell today?"
    ),
    NPCPersonality(
        name="Old Tom",
        role="regular patron",
        traits=["talkative", "tells stories", "slightly drunk", "exaggerates"],
        background="Retired adventurer. Spends most days at the tavern sharing tales.",
        speaking_style="rambling, enthusiastic, prone to tangents",
        greeting="*hiccup* Hey there, friend! Have I told you about the time I fought a dragon?"
    ),
    NPCPersonality(
        name="Elena",
        role="mysterious stranger",
        traits=["secretive", "observant", "speaks in riddles", "knows things"],
        background="Nobody knows where she came from. Appears to know about hidden dangers.",
        speaking_style="cryptic, thoughtful, measured words",
        greeting="*nods silently* You seek something. Perhaps I can help... if you're worthy."
    ),
    NPCPersonality(
        name="Finn",
        role="bard",
        traits=["creative", "cheerful", "remembers songs", "composes about patrons"],
        background="Traveling bard collecting stories and songs. Loves inspiration.",
        speaking_style="poetic, musical, references songs",
        greeting="♪ Ah, a new face! Every stranger has a story. Care to share yours? ♪"
    ),
]
