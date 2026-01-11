"""
Knowledge Graph - L3 Semantic Memory Store.

This module implements the L3 layer which stores:
- Consolidated summaries from L2 memories
- Extracted entities and their relationships
- Reflections and higher-level insights

Uses SQLite for persistence + in-memory NetworkX for graph operations.
"""

import sqlite3
import numpy as np
import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class L3Node:
    """
    A node in the L3 Knowledge Graph.
    
    Can represent:
    - An entity (Person, Place, Object, Concept)
    - A summary (consolidated from L2 memories)
    - A reflection (higher-level insight)
    """
    id: str
    node_type: str  # 'entity', 'summary', 'reflection'
    content: str    # Natural language description
    embedding: np.ndarray
    source_memory_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "content": self.content,
            "source_memory_ids": self.source_memory_ids,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class L3Edge:
    """
    An edge (relationship) in the L3 Knowledge Graph.
    """
    source_id: str
    target_id: str
    edge_type: str  # e.g., 'related_to', 'caused', 'part_of', 'about'
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    L3 Knowledge Graph for semantic memory.
    
    Features:
    - Store entities, summaries, and reflections
    - Query by embedding similarity
    - Graph traversal for multi-hop reasoning
    - SQLite persistence
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_dim: int = 384
    ):
        """
        Initialize the knowledge graph.
        
        Args:
            db_path: Path to SQLite database (":memory:" for in-memory)
            embedding_dim: Dimension of embedding vectors
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # In-memory storage for fast access
        self._nodes: Dict[str, L3Node] = {}
        self._edges: Dict[str, List[L3Edge]] = {}  # source_id -> edges
        
        # Embedding cache for fast similarity search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._embeddings_dirty: bool = True
        
        # For in-memory databases, we need a persistent connection
        # (each new connection to :memory: creates a fresh database)
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection (persistent for in-memory, new for file-based)."""
        if self._persistent_conn is not None:
            return self._persistent_conn
        return sqlite3.connect(self.db_path)
    
    def _close_connection(self, conn: sqlite3.Connection):
        """Close connection if it's not the persistent one."""
        if conn != self._persistent_conn:
            conn.close()
    
    def _init_db(self):
        """Initialize SQLite database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT,
                content TEXT,
                embedding BLOB,
                source_memory_ids TEXT,
                created_at TEXT,
                last_accessed TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT,
                target_id TEXT,
                edge_type TEXT,
                weight REAL,
                metadata TEXT,
                PRIMARY KEY (source_id, target_id, edge_type)
            )
        """)
        
        conn.commit()
        self._close_connection(conn)
        
        # Load existing data if any (for file-based DBs)
        if self.db_path != ":memory:":
            self._load_from_db()
    
    def _load_from_db(self):
        """Load data from database into memory."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Load nodes
        cursor.execute("SELECT * FROM nodes")
        for row in cursor.fetchall():
            node_id, node_type, content, embedding_bytes, source_ids_json, created, accessed, meta_json = row
            
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32) if embedding_bytes else np.zeros(self.embedding_dim)
            source_ids = json.loads(source_ids_json) if source_ids_json else []
            metadata = json.loads(meta_json) if meta_json else {}
            
            node = L3Node(
                id=node_id,
                node_type=node_type,
                content=content,
                embedding=embedding,
                source_memory_ids=source_ids,
                created_at=datetime.fromisoformat(created) if created else datetime.now(),
                last_accessed=datetime.fromisoformat(accessed) if accessed else datetime.now(),
                metadata=metadata
            )
            self._nodes[node_id] = node
        
        # Load edges
        cursor.execute("SELECT * FROM edges")
        for row in cursor.fetchall():
            source_id, target_id, edge_type, weight, meta_json = row
            metadata = json.loads(meta_json) if meta_json else {}
            
            edge = L3Edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata
            )
            if source_id not in self._edges:
                self._edges[source_id] = []
            self._edges[source_id].append(edge)
        
        self._close_connection(conn)
        self._embeddings_dirty = True
        logger.info(f"Loaded {len(self._nodes)} nodes and {sum(len(e) for e in self._edges.values())} edges from DB")
    
    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self._nodes)
    
    def add_node(
        self,
        content: str,
        embedding: np.ndarray,
        node_type: str = "summary",
        source_memory_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a node to the knowledge graph.
        
        Args:
            content: Natural language content
            embedding: Vector embedding
            node_type: 'entity', 'summary', or 'reflection'
            source_memory_ids: L2 memories that contributed to this node
            metadata: Additional metadata
            
        Returns:
            ID of the created node
        """
        node_id = str(uuid.uuid4())
        
        node = L3Node(
            id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding.astype(np.float32),
            source_memory_ids=source_memory_ids or [],
            metadata=metadata or {}
        )
        
        self._nodes[node_id] = node
        self._embeddings_dirty = True
        
        # Save to database
        self._save_node(node)
        
        logger.debug(f"Added L3 node {node_id[:8]}... type={node_type}")
        return node_id
    
    def _save_node(self, node: L3Node):
        """Save a node to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO nodes 
            (id, node_type, content, embedding, source_memory_ids, created_at, last_accessed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            node.node_type,
            node.content,
            node.embedding.tobytes(),
            json.dumps(node.source_memory_ids),
            node.created_at.isoformat(),
            node.last_accessed.isoformat(),
            json.dumps(node.metadata)
        ))
        
        conn.commit()
        self._close_connection(conn)
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an edge between two nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning(f"Cannot add edge: node not found")
            return
        
        edge = L3Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        if source_id not in self._edges:
            self._edges[source_id] = []
        self._edges[source_id].append(edge)
        
        # Save to database
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (source_id, target_id, edge_type, weight, json.dumps(edge.metadata)))
        conn.commit()
        self._close_connection(conn)
    
    def get_node(self, node_id: str) -> Optional[L3Node]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_all_nodes(self) -> List[L3Node]:
        """Get all nodes."""
        return list(self._nodes.values())
    
    def get_embeddings_matrix(self) -> np.ndarray:
        """
        Get matrix of all node embeddings.
        
        Used for computing redundancy D(m) in forgetting.
        """
        if self._embeddings_dirty or self._embedding_matrix is None:
            if len(self._nodes) == 0:
                self._embedding_matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)
            else:
                embeddings = []
                self._id_to_idx = {}
                self._idx_to_id = {}
                
                for idx, (node_id, node) in enumerate(self._nodes.items()):
                    embeddings.append(node.embedding)
                    self._id_to_idx[node_id] = idx
                    self._idx_to_id[idx] = node_id
                
                self._embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            self._embeddings_dirty = False
        
        return self._embedding_matrix
    
    def query_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[L3Node, float]]:
        """
        Find the k most similar nodes to the query.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            List of (node, similarity) tuples
        """
        if len(self._nodes) == 0:
            return []
        
        embeddings = self.get_embeddings_matrix()
        
        # Compute cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        
        query_normalized = query_embedding / query_norm
        norms = np.linalg.norm(embeddings, axis=1)
        norms = np.where(norms == 0, 1, norms)
        embeddings_normalized = embeddings / norms[:, np.newaxis]
        
        similarities = np.dot(embeddings_normalized, query_normalized)
        
        # Get top k
        k = min(k, len(self._nodes))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            node_id = self._idx_to_id[idx]
            node = self._nodes[node_id]
            node.last_accessed = datetime.now()
            results.append((node, float(similarities[idx])))
        
        return results
    
    def traverse(
        self,
        start_id: str,
        max_hops: int = 2,
        max_nodes: int = 10
    ) -> List[L3Node]:
        """
        Traverse the graph from a starting node.
        
        Args:
            start_id: Starting node ID
            max_hops: Maximum number of hops
            max_nodes: Maximum nodes to return
            
        Returns:
            List of reachable nodes
        """
        if start_id not in self._nodes:
            return []
        
        visited = set([start_id])
        frontier = [start_id]
        result = [self._nodes[start_id]]
        
        for hop in range(max_hops):
            next_frontier = []
            for node_id in frontier:
                for edge in self._edges.get(node_id, []):
                    if edge.target_id not in visited:
                        visited.add(edge.target_id)
                        next_frontier.append(edge.target_id)
                        result.append(self._nodes[edge.target_id])
                        
                        if len(result) >= max_nodes:
                            return result
            
            frontier = next_frontier
        
        return result
    
    def compute_redundancy(self, memory_embedding: np.ndarray) -> float:
        """
        Compute D(m) = max similarity to any L3 node.
        
        Used in consolidation-aware forgetting.
        """
        if len(self._nodes) == 0:
            return 0.0
        
        embeddings = self.get_embeddings_matrix()
        
        # Compute similarities
        mem_norm = np.linalg.norm(memory_embedding)
        if mem_norm == 0:
            return 0.0
        
        mem_normalized = memory_embedding / mem_norm
        norms = np.linalg.norm(embeddings, axis=1)
        norms = np.where(norms == 0, 1, norms)
        embeddings_normalized = embeddings / norms[:, np.newaxis]
        
        similarities = np.dot(embeddings_normalized, mem_normalized)
        
        return float(np.max(similarities)) if len(similarities) > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        edge_count = sum(len(edges) for edges in self._edges.values())
        type_counts = {}
        for node in self._nodes.values():
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
        
        return {
            "total_nodes": len(self._nodes),
            "total_edges": edge_count,
            "node_types": type_counts
        }
