"""
Multi-Agent Orchestrator

Manages N NPCs × N Players concurrent interactions with GPU batching.
"""

import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Represents a player in the simulation."""
    id: str
    name: str
    messages: List[str] = field(default_factory=list)  # Conversation history
    
    def get_next_message(self) -> Optional[str]:
        """Get next message from player's script."""
        if self.messages:
            return self.messages.pop(0)
        return None


@dataclass
class InteractionResult:
    """Result of a single NPC-Player interaction."""
    npc_name: str
    player_name: str
    player_message: str
    npc_response: str
    latency_ms: float
    memory_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    result_data: Dict[str, Any] = field(default_factory=dict)  # Full response dict


@dataclass
class RoundResults:
    """Results from one round of concurrent interactions."""
    round_number: int
    interactions: List[InteractionResult]
    total_time_ms: float
    avg_latency_ms: float
    throughput: float  # interactions per second


class MultiAgentOrchestrator:
    """
    Orchestrates concurrent N×N NPC-Player interactions.
    
    Features:
    - Multiple pairing strategies
    - GPU-batched operations
    - Progress tracking
    - Result aggregation
    """
    
    def __init__(
        self,
        npcs: List[Any],  # List of async-enabled NPC objects
        players: List[Player],
        batch_embeddings: bool = True
    ):
        """
        Initialize orchestrator.
        
        Args:
            npcs: List of NPC objects (with async respond method)
            players: List of Player objects
            batch_embeddings: Whether to batch embedding operations
        """
        self.npcs = npcs
        self.players = players
        self.batch_embeddings = batch_embeddings
        
        logger.info(
            f"Orchestrator initialized: {len(npcs)} NPCs × {len(players)} Players"
        )
    
    def _create_pairs_random(self) -> List[Tuple[Any, Player]]:
        """Random pairing strategy."""
        shuffled_players = self.players.copy()
        random.shuffle(shuffled_players)
        
        pairs = []
        for i, npc in enumerate(self.npcs):
            player = shuffled_players[i % len(shuffled_players)]
            pairs.append((npc, player))
        
        return pairs
    
    def _create_pairs_fixed(self) -> List[Tuple[Any, Player]]:
        """Fixed pairing (NPC i with Player i)."""
        pairs = []
        for i, npc in enumerate(self.npcs):
            player = self.players[i % len(self.players)]
            pairs.append((npc, player))
        return pairs
    
    def _create_pairs_round_robin(self, round_num: int) -> List[Tuple[Any, Player]]:
        """Round-robin pairing (rotate each round)."""
        pairs = []
        for i, npc in enumerate(self.npcs):
            player_idx = (i + round_num) % len(self.players)
            player = self.players[player_idx]
            pairs.append((npc, player))
        return pairs
    
    async def _run_single_interaction(
        self,
        npc: Any,
        player: Player,
        message: str
    ) -> InteractionResult:
        """Run a single NPC-Player interaction."""
        start_time = time.time()
        
        # Call NPC's async respond method
        result = await npc.respond(message, player_name=player.name)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return InteractionResult(
            npc_name=npc.personality.name,
            player_name=player.name,
            player_message=message,
            npc_response=result.get("response", ""),
            latency_ms=latency_ms,
            memory_count=result.get("memory_count", 0),
            result_data=result  # Store full result for detailed metrics
        )
    
    async def run_concurrent_round(
        self,
        round_num: int,
        pairing_strategy: str = 'random',
        messages: Optional[List[str]] = None
    ) -> RoundResults:
        """
        Run one round of concurrent interactions.
        
        Args:
            round_num: Round number
            pairing_strategy: 'random', 'fixed', or 'round_robin'
            messages: Optional list of messages (random if None)
            
        Returns:
            RoundResults with all interaction data
        """
        # Create pairs
        if pairing_strategy == 'random':
            pairs = self._create_pairs_random()
        elif pairing_strategy == 'fixed':
            pairs = self._create_pairs_fixed()
        elif pairing_strategy == 'round_robin':
            pairs = self._create_pairs_round_robin(round_num)
        else:
            raise ValueError(f"Unknown strategy: {pairing_strategy}")
        
        # Generate messages if not provided
        if messages is None:
            default_messages = [
                "Hello, how are you today?",
                "Tell me about yourself.",
                "What do you remember about me?",
                "What's new?",
                "Any interesting stories?",
            ]
            messages = [random.choice(default_messages) for _ in pairs]
        
        # Run all interactions concurrently
        start_time = time.time()
        
        tasks = [
            self._run_single_interaction(npc, player, msg)
            for (npc, player), msg in zip(pairs, messages)
        ]
        
        interactions = await asyncio.gather(*tasks)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Calculate metrics
        avg_latency = sum(i.latency_ms for i in interactions) / len(interactions)
        throughput = len(interactions) / (total_time_ms / 1000)
        
        return RoundResults(
            round_number=round_num,
            interactions=interactions,
            total_time_ms=total_time_ms,
            avg_latency_ms=avg_latency,
            throughput=throughput
        )
    
    async def run_m_rounds(
        self,
        m: int,
        pairing_strategy: str = 'random',
        verbose: bool = True
    ) -> List[RoundResults]:
        """
        Run M rounds of concurrent interactions.
        
        Args:
            m: Number of rounds
            pairing_strategy: Pairing strategy to use
            verbose: Print progress
            
        Returns:
            List of RoundResults for each round
        """
        all_results = []
        
        for round_num in range(m):
            if verbose:
                print(f"\n[Round {round_num + 1}/{m}] Running {len(self.npcs)} concurrent interactions...")
            
            results = await self.run_concurrent_round(
                round_num,
                pairing_strategy=pairing_strategy
            )
            
            all_results.append(results)
            
            if verbose:
                print(f"  ✓ Completed in {results.total_time_ms:.0f}ms")
                print(f"    Avg latency: {results.avg_latency_ms:.0f}ms")
                print(f"    Throughput: {results.throughput:.2f} interactions/sec")
                
                # Show sample interactions
                for interaction in results.interactions[:3]:  # First 3
                    print(
                        f"      • {interaction.npc_name} ↔ {interaction.player_name}: "
                        f"{interaction.latency_ms:.0f}ms"
                    )
        
        return all_results
    
    def get_aggregate_metrics(
        self,
        all_results: List[RoundResults]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics across all rounds."""
        total_interactions = sum(len(r.interactions) for r in all_results)
        all_latencies = [
            i.latency_ms 
            for r in all_results 
            for i in r.interactions
        ]
        
        return {
            "total_rounds": len(all_results),
            "total_interactions": total_interactions,
            "avg_latency_ms": sum(all_latencies) / len(all_latencies),
            "p50_latency_ms": sorted(all_latencies)[len(all_latencies) // 2],
            "p95_latency_ms": sorted(all_latencies)[int(len(all_latencies) * 0.95)],
            "p99_latency_ms": sorted(all_latencies)[int(len(all_latencies) * 0.99)],
            "avg_throughput": sum(r.throughput for r in all_results) / len(all_results),
            "total_time_ms": sum(r.total_time_ms for r in all_results)
        }
    
    def generate_report(
        self,
        all_results: List[RoundResults],
        save_path: Optional[str] = None
    ) -> str:
        """Generate a text report of results."""
        metrics = self.get_aggregate_metrics(all_results)
        
        report = f"""
Multi-Agent Interaction Report
{'=' * 60}

Configuration:
  NPCs: {len(self.npcs)}
  Players: {len(self.players)}
  Rounds: {metrics['total_rounds']}
  Total Interactions: {metrics['total_interactions']}

Performance Metrics:
  Total Time: {metrics['total_time_ms'] / 1000:.2f}s
  Avg Latency: {metrics['avg_latency_ms']:.0f}ms
  P50 Latency: {metrics['p50_latency_ms']:.0f}ms
  P95 Latency: {metrics['p95_latency_ms']:.0f}ms
  P99 Latency: {metrics['p99_latency_ms']:.0f}ms
  Avg Throughput: {metrics['avg_throughput']:.2f} interactions/sec

Per-Round Breakdown:
"""
        
        for r in all_results:
            report += f"  Round {r.round_number + 1}: "
            report += f"{r.total_time_ms:.0f}ms, "
            report += f"{r.throughput:.2f} tps, "
            report += f"{len(r.interactions)} interactions\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
