"""
NPC-LoCoMo Benchmark Generator

Generates synthetic NPC conversation histories with ground-truth Q&A pairs
for evaluating memory systems.

Based on LoCoMo (Long Context Conversation Memory) benchmark structure,
adapted for game NPC scenarios.
"""

import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import os


@dataclass
class Interaction:
    """A single player-NPC interaction."""
    id: int
    text: str
    timestamp: datetime
    speaker: str  # 'player' or 'npc'
    facts: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5


@dataclass
class QAPair:
    """A question-answer pair for evaluation."""
    question: str
    answer: str
    qa_type: str  # 'single-hop', 'multi-hop', 'temporal', 'adversarial'
    source_interaction_ids: List[int] = field(default_factory=list)
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'


@dataclass
class ConversationHistory:
    """A complete conversation history with Q&A pairs."""
    id: str
    interactions: List[Interaction]
    qa_pairs: List[QAPair]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Templates for generating diverse interactions
# Each tuple: (template_string, values_dict, base_importance)
# Importance assigned by fact category — NOT random
INTERACTION_TEMPLATES = [
    # Shopping/Trading
    ("Player bought {item} for {price} gold", {"item": ["sword", "shield", "potion", "armor", "bow"], "price": [50, 100, 150, 200, 300]}, 0.8),
    ("Player sold {item} to the merchant", {"item": ["old sword", "rare gem", "herbs", "leather", "iron ore"]}, 0.7),
    ("Player asked about {item} prices", {"item": ["weapons", "armor", "potions", "magical items", "food"]}, 0.4),
    
    # Personal information
    ("Player said their name is {name}", {"name": ["Alex", "Jordan", "Morgan", "Sam", "Taylor"]}, 0.9),
    ("Player mentioned they are from {place}", {"place": ["the northern village", "the capital city", "the coastal town", "the mountain fortress", "the forest settlement"]}, 0.8),
    ("Player said they like {hobby}", {"hobby": ["hunting", "fishing", "crafting", "exploring", "collecting rare items"]}, 0.6),
    
    # Quests/Tasks
    ("Player asked about {quest}", {"quest": ["the missing merchant", "the dragon sighting", "the haunted mine", "the stolen artifact", "the bandit problem"]}, 0.7),
    ("Player completed {task}", {"task": ["the delivery quest", "the escort mission", "the monster hunt", "the treasure map", "the investigation"]}, 0.8),
    ("Player failed {task}", {"task": ["to defeat the boss", "the stealth mission", "to save the villager", "the time trial"]}, 0.7),
    
    # Emotions/Reactions
    ("Player expressed {emotion} about {topic}", {"emotion": ["excitement", "concern", "frustration", "joy", "curiosity"], "topic": ["the recent battle", "the new equipment", "the village problems", "their progress"]}, 0.3),
    ("Player thanked the shopkeeper for {reason}", {"reason": ["the advice", "the discount", "the information", "the help", "the recommendation"]}, 0.3),
    
    # Events
    ("Player mentioned seeing {event}", {"event": ["a strange creature", "suspicious travelers", "a meteor", "unusual weather", "armed soldiers"]}, 0.5),
    ("Player talked about {past_event}", {"past_event": ["their first adventure", "when they met the king", "the great battle", "their training days", "their hometown"]}, 0.4),
]


class BenchmarkGenerator:
    """Generates NPC-LoCoMo benchmark data."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
    
    def generate_conversation_history(
        self,
        conversation_id: str,
        num_interactions: int = 100,
        time_span_days: int = 30
    ) -> ConversationHistory:
        """
        Generate a conversation history with interactions and Q&A pairs.
        
        Args:
            conversation_id: Unique ID for this conversation
            num_interactions: Number of interactions to generate
            time_span_days: Time span over which interactions occur
            
        Returns:
            ConversationHistory with interactions and Q&A pairs
        """
        interactions = []
        facts_database = {}  # Track facts for Q&A generation
        
        base_time = datetime.now() - timedelta(days=time_span_days)
        
        for i in range(num_interactions):
            # Select random template (now includes base importance)
            template, values_dict, base_importance = random.choice(
                INTERACTION_TEMPLATES
            )
            
            # Fill in template
            filled_values = {}
            for key, options in values_dict.items():
                filled_values[key] = random.choice(options)
            
            text = template.format(**filled_values)
            
            # Calculate timestamp
            time_offset = timedelta(
                days=random.uniform(0, time_span_days),
                hours=random.uniform(0, 24)
            )
            timestamp = base_time + time_offset
            
            # Importance from fact category + small noise (±0.05)
            importance = max(0.0, min(1.0,
                base_importance + random.uniform(-0.05, 0.05)
            ))
            
            # Create interaction
            interaction = Interaction(
                id=i,
                text=text,
                timestamp=timestamp,
                speaker="player",
                facts=filled_values,
                importance=importance
            )
            interactions.append(interaction)
            
            # Store facts for Q&A
            for key, value in filled_values.items():
                fact_key = f"{i}_{key}"
                facts_database[fact_key] = {
                    "value": value,
                    "interaction_id": i,
                    "template_key": key,
                    "text": text,
                    "timestamp": timestamp
                }
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x.timestamp)
        
        # Reassign IDs after sorting
        for i, interaction in enumerate(interactions):
            interaction.id = i
        
        # Generate Q&A pairs
        qa_pairs = self._generate_qa_pairs(interactions, facts_database)
        
        return ConversationHistory(
            id=conversation_id,
            interactions=interactions,
            qa_pairs=qa_pairs,
            metadata={
                "num_interactions": num_interactions,
                "time_span_days": time_span_days,
                "seed": self.seed
            }
        )
    
    def _generate_qa_pairs(
        self,
        interactions: List[Interaction],
        facts_database: Dict[str, Any]
    ) -> List[QAPair]:
        """Generate Q&A pairs from interactions."""
        qa_pairs = []
        
        # Single-hop questions (direct recall)
        qa_pairs.extend(self._generate_single_hop_qa(interactions))
        
        # Multi-hop questions (cross-interaction reasoning)
        qa_pairs.extend(self._generate_multi_hop_qa(interactions, facts_database))
        
        # Temporal questions (sequence/order)
        qa_pairs.extend(self._generate_temporal_qa(interactions))
        
        # Adversarial questions (test for hallucination)
        qa_pairs.extend(self._generate_adversarial_qa(interactions))
        
        return qa_pairs
    
    def _generate_single_hop_qa(self, interactions: List[Interaction]) -> List[QAPair]:
        """Generate single-hop recall questions."""
        qa_pairs = []
        
        # Sample some interactions for questions
        sample_size = min(10, len(interactions))
        sampled = random.sample(interactions, sample_size)
        
        for interaction in sampled:
            facts = interaction.facts
            
            if "name" in facts:
                qa_pairs.append(QAPair(
                    question="What is the player's name?",
                    answer=facts["name"],
                    qa_type="single-hop",
                    source_interaction_ids=[interaction.id],
                    difficulty="easy"
                ))
            
            if "item" in facts and "price" in facts:
                qa_pairs.append(QAPair(
                    question=f"How much did the {facts['item']} cost?",
                    answer=f"{facts['price']} gold",
                    qa_type="single-hop",
                    source_interaction_ids=[interaction.id],
                    difficulty="easy"
                ))
            
            if "hobby" in facts:
                qa_pairs.append(QAPair(
                    question="What hobbies did the player mention?",
                    answer=facts["hobby"],
                    qa_type="single-hop",
                    source_interaction_ids=[interaction.id],
                    difficulty="easy"
                ))
            
            if "place" in facts:
                qa_pairs.append(QAPair(
                    question="Where is the player from?",
                    answer=facts["place"],
                    qa_type="single-hop",
                    source_interaction_ids=[interaction.id],
                    difficulty="easy"
                ))
        
        return qa_pairs[:10]  # Limit to 10
    
    def _generate_multi_hop_qa(
        self,
        interactions: List[Interaction],
        facts_database: Dict[str, Any]
    ) -> List[QAPair]:
        """Generate multi-hop reasoning questions.

        Fixes:
        - "What did X buy?" now uses ALL items bought (not a single
          random pick), so the ground truth matches what the LLM sees.
        - "What was the most expensive item X bought?" is a true
          multi-hop question requiring name lookup + price comparison.
        """
        qa_pairs = []

        name_interactions = [i for i in interactions if "name" in i.facts]
        # Only "bought" interactions have both item and price
        buy_interactions = [
            i for i in interactions
            if "item" in i.facts and "price" in i.facts
        ]

        if name_interactions and buy_interactions:
            name_i = name_interactions[0]
            player_name = name_i.facts["name"]

            # Q1: What items did <name> buy?  (ground truth = all items)
            all_items = sorted({i.facts["item"] for i in buy_interactions})
            source_ids = [name_i.id] + [i.id for i in buy_interactions]
            qa_pairs.append(QAPair(
                question=f"What items did {player_name} buy?",
                answer=", ".join(all_items),
                qa_type="multi-hop",
                source_interaction_ids=source_ids,
                difficulty="medium"
            ))

            # Q2: Most expensive purchase (true reasoning over price)
            most_expensive = max(buy_interactions,
                                 key=lambda i: i.facts["price"])
            qa_pairs.append(QAPair(
                question=(
                    f"What was the most expensive item "
                    f"{player_name} bought?"
                ),
                answer=(
                    f"{most_expensive.facts['item']} for "
                    f"{most_expensive.facts['price']} gold"
                ),
                qa_type="multi-hop",
                source_interaction_ids=[name_i.id, most_expensive.id],
                difficulty="hard"
            ))

        # Hobby + completed-quest combination
        hobby_interactions = [i for i in interactions if "hobby" in i.facts]
        completed_quests = [
            i for i in interactions
            if "task" in i.facts and "completed" in i.text.lower()
        ]

        if hobby_interactions and completed_quests:
            hobby_i = hobby_interactions[0]
            quest_i = completed_quests[0]
            qa_pairs.append(QAPair(
                question=(
                    f"The player enjoys {hobby_i.facts['hobby']}. "
                    f"What quest did they complete?"
                ),
                answer=quest_i.facts["task"],
                qa_type="multi-hop",
                source_interaction_ids=[hobby_i.id, quest_i.id],
                difficulty="hard"
            ))

        return qa_pairs[:5]
    
    def _generate_temporal_qa(self, interactions: List[Interaction]) -> List[QAPair]:
        """Generate temporal reasoning questions."""
        qa_pairs = []
        
        if len(interactions) < 2:
            return qa_pairs
        
        # First interaction question
        first = interactions[0]
        qa_pairs.append(QAPair(
            question="What was the first thing the player did?",
            answer=first.text,
            qa_type="temporal",
            source_interaction_ids=[first.id],
            difficulty="medium"
        ))
        
        # Most recent question
        last = interactions[-1]
        qa_pairs.append(QAPair(
            question="What was the most recent interaction?",
            answer=last.text,
            qa_type="temporal",
            source_interaction_ids=[last.id],
            difficulty="medium"
        ))
        
        # Ordering question
        if len(interactions) >= 5:
            early = interactions[1]
            late = interactions[-2]
            
            qa_pairs.append(QAPair(
                question=f"Did '{early.text[:50]}...' happen before or after '{late.text[:50]}...'?",
                answer="before",
                qa_type="temporal",
                source_interaction_ids=[early.id, late.id],
                difficulty="hard"
            ))
        
        return qa_pairs[:5]
    
    def _generate_adversarial_qa(self, interactions: List[Interaction]) -> List[QAPair]:
        """Generate adversarial questions to test for hallucination."""
        qa_pairs = []
        
        # Questions about things that were never mentioned
        adversarial_questions = [
            ("What pet does the player have?", "not mentioned"),
            ("What is the player's favorite food?", "not mentioned"),
            ("How many children does the player have?", "not mentioned"),
            ("What is the player's profession?", "not mentioned"),
            ("What color is the player's armor?", "not mentioned"),
        ]
        
        # Check what was actually mentioned
        all_facts = set()
        for interaction in interactions:
            all_facts.update(interaction.facts.keys())
        
        for question, default_answer in adversarial_questions:
            # Only include if the fact wasn't actually mentioned
            keyword = question.lower().split()[2]  # Get key word from question
            if keyword not in str(all_facts).lower():
                qa_pairs.append(QAPair(
                    question=question,
                    answer=default_answer,
                    qa_type="adversarial",
                    source_interaction_ids=[],
                    difficulty="hard"
                ))
        
        return qa_pairs[:5]
    
    def generate_benchmark_dataset(
        self,
        num_conversations: int = 10,
        interactions_per_conversation: int = 100,
        output_dir: str = None
    ) -> List[ConversationHistory]:
        """
        Generate a full benchmark dataset.
        
        Args:
            num_conversations: Number of conversation histories to generate
            interactions_per_conversation: Interactions per conversation
            output_dir: Optional directory to save dataset
            
        Returns:
            List of ConversationHistory objects
        """
        dataset = []
        
        for i in range(num_conversations):
            # Use different seed for each conversation for variety
            random.seed(self.seed + i)
            
            history = self.generate_conversation_history(
                conversation_id=f"conv_{i:03d}",
                num_interactions=interactions_per_conversation
            )
            dataset.append(history)
        
        # Reset seed
        random.seed(self.seed)
        
        # Save if output_dir specified
        if output_dir:
            self._save_dataset(dataset, output_dir)
        
        return dataset
    
    def _save_dataset(self, dataset: List[ConversationHistory], output_dir: str):
        """Save dataset to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for history in dataset:
            # Convert to serializable format
            data = {
                "id": history.id,
                "metadata": history.metadata,
                "interactions": [
                    {
                        "id": i.id,
                        "text": i.text,
                        "timestamp": i.timestamp.isoformat(),
                        "speaker": i.speaker,
                        "facts": i.facts,
                        "importance": i.importance
                    }
                    for i in history.interactions
                ],
                "qa_pairs": [
                    {
                        "question": qa.question,
                        "answer": qa.answer,
                        "type": qa.qa_type,
                        "source_ids": qa.source_interaction_ids,
                        "difficulty": qa.difficulty
                    }
                    for qa in history.qa_pairs
                ]
            }
            
            filepath = os.path.join(output_dir, f"{history.id}.json")
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"Saved {len(dataset)} conversations to {output_dir}")


if __name__ == "__main__":
    # Generate sample dataset
    generator = BenchmarkGenerator(seed=42)
    dataset = generator.generate_benchmark_dataset(
        num_conversations=5,
        interactions_per_conversation=50,
        output_dir="npc_locomo_data"
    )
    
    print(f"\nGenerated {len(dataset)} conversations")
    for history in dataset:
        print(f"  {history.id}: {len(history.interactions)} interactions, {len(history.qa_pairs)} Q&A pairs")
