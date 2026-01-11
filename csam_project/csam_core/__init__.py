"""
CSAM - Cognitive Sparse Access Memory

A hierarchical memory architecture for AI agents with 
consolidation-aware forgetting.
"""

__version__ = "0.1.0"
__author__ = "CSAM Research Team"

from .memory_repository import MemoryRepository, Memory
from .knowledge_graph import KnowledgeGraph, L3Node
from .forgetting_engine import (
    ForgettingStrategy,
    NoForgetting,
    LRUForgetting,
    ImportanceForgetting,
    ConsolidationAwareForgetting
)
from .consolidation_tracker import ConsolidationTracker
from .retrieval import HybridRetriever
from .mmr import MaximalMarginalRelevance

__all__ = [
    "MemoryRepository",
    "Memory",
    "KnowledgeGraph",
    "L3Node",
    "ForgettingStrategy",
    "NoForgetting",
    "LRUForgetting",
    "ImportanceForgetting",
    "ConsolidationAwareForgetting",
    "ConsolidationTracker",
    "HybridRetriever",
    "MaximalMarginalRelevance",
]
