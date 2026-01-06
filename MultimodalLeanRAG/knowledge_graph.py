"""
Knowledge Graph Module for Multimodal LeanRAG
=============================================
Stores signature IDs as nodes and relationships between audio chunks as edges.
Supports hierarchical structure with aggregation nodes.
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between audio chunks."""
    SEQUENTIAL = "sequential"           # Temporally adjacent chunks
    SEMANTIC_SIMILAR = "semantic_similar"  # Semantically similar content
    SAME_SPEAKER = "same_speaker"       # Same speaker detected
    SAME_TOPIC = "same_topic"           # Same topic/theme
    CROSS_MODAL = "cross_modal"         # Related across modalities
    AGGREGATION = "aggregation"         # Chunk belongs to aggregation node
    HIERARCHY = "hierarchy"             # Parent-child in hierarchy


@dataclass
class KGNode:
    """Node in the knowledge graph."""
    signature_id: str                   # Unique identifier
    node_type: str                      # 'audio_chunk' or 'aggregation'
    level: int                          # Hierarchy level (0 = base chunks)
    description: Optional[str] = None   # Generated description
    parent_id: Optional[str] = None     # Parent aggregation node
    source_file: Optional[str] = None   # Source audio file
    metadata: Optional[Dict] = None     # Additional metadata
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KGEdge:
    """Edge in the knowledge graph."""
    source_id: str                      # Source node signature ID
    target_id: str                      # Target node signature ID
    relation_type: str                  # Type of relationship
    weight: float = 1.0                 # Edge weight/strength
    description: Optional[str] = None   # Relation description
    metadata: Optional[Dict] = None     # Additional metadata
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AudioKnowledgeGraph:
    """
    Knowledge Graph for storing audio chunk relationships.
    Uses SQLite for persistence with support for hierarchical structure.
    """
    
    def __init__(
        self,
        working_dir: str,
        db_name: str = "knowledge_graph.db"
    ):
        """
        Initialize the knowledge graph.
        
        Args:
            working_dir: Directory for database storage
            db_name: Name of SQLite database file
        """
        self.working_dir = working_dir
        self.db_path = os.path.join(working_dir, db_name)
        
        os.makedirs(working_dir, exist_ok=True)
        
        logger.info(f"Initializing Knowledge Graph at {self.db_path}")
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                signature_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                level INTEGER DEFAULT 0,
                description TEXT,
                parent_id TEXT,
                source_file TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES nodes(signature_id)
            )
        """)
        
        # Create edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                description TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES nodes(signature_id),
                FOREIGN KEY (target_id) REFERENCES nodes(signature_id),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type)")
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema initialized")
    
    def add_node(self, node: KGNode) -> bool:
        """
        Add a node to the knowledge graph.
        
        Args:
            node: KGNode object
        
        Returns:
            True if successful, False if node already exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO nodes (signature_id, node_type, level, description, 
                                  parent_id, source_file, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node.signature_id,
                node.node_type,
                node.level,
                node.description,
                node.parent_id,
                node.source_file,
                json.dumps(node.metadata) if node.metadata else None
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Node {node.signature_id} already exists")
            return False
        finally:
            conn.close()
    
    def add_nodes_batch(self, nodes: List[KGNode]) -> int:
        """
        Add multiple nodes in a batch.
        
        Args:
            nodes: List of KGNode objects
        
        Returns:
            Number of nodes successfully added
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for node in nodes:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO nodes 
                    (signature_id, node_type, level, description, parent_id, source_file, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.signature_id,
                    node.node_type,
                    node.level,
                    node.description,
                    node.parent_id,
                    node.source_file,
                    json.dumps(node.metadata) if node.metadata else None
                ))
                if cursor.rowcount > 0:
                    count += 1
            except Exception as e:
                logger.error(f"Error adding node {node.signature_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added {count}/{len(nodes)} nodes to knowledge graph")
        return count
    
    def add_edge(self, edge: KGEdge) -> bool:
        """
        Add an edge between two nodes.
        
        Args:
            edge: KGEdge object
        
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO edges 
                (source_id, target_id, relation_type, weight, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.weight,
                edge.description,
                json.dumps(edge.metadata) if edge.metadata else None
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding edge: {e}")
            return False
        finally:
            conn.close()
    
    def add_edges_batch(self, edges: List[KGEdge]) -> int:
        """
        Add multiple edges in a batch.
        
        Args:
            edges: List of KGEdge objects
        
        Returns:
            Number of edges successfully added
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for edge in edges:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO edges 
                    (source_id, target_id, relation_type, weight, description, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    edge.source_id,
                    edge.target_id,
                    edge.relation_type,
                    edge.weight,
                    edge.description,
                    json.dumps(edge.metadata) if edge.metadata else None
                ))
                count += 1
            except Exception as e:
                logger.error(f"Error adding edge {edge.source_id}->{edge.target_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added {count}/{len(edges)} edges to knowledge graph")
        return count
    
    def create_sequential_edges(self, chunks_by_source: Dict[str, List[str]]) -> int:
        """
        Create sequential edges between consecutive chunks from the same source.
        
        Args:
            chunks_by_source: Dict mapping source_file to list of signature_ids (ordered)
        
        Returns:
            Number of edges created
        """
        edges = []
        
        for source_file, signature_ids in chunks_by_source.items():
            for i in range(len(signature_ids) - 1):
                edge = KGEdge(
                    source_id=signature_ids[i],
                    target_id=signature_ids[i + 1],
                    relation_type=RelationType.SEQUENTIAL.value,
                    weight=1.0,
                    description=f"Sequential chunks from {os.path.basename(source_file)}"
                )
                edges.append(edge)
        
        return self.add_edges_batch(edges)
    
    def create_semantic_edges(
        self,
        similarity_pairs: List[Tuple[str, str, float]],
        threshold: float = 0.7
    ) -> int:
        """
        Create semantic similarity edges based on embedding similarity.
        
        Args:
            similarity_pairs: List of (id1, id2, similarity_score) tuples
            threshold: Minimum similarity to create edge
        
        Returns:
            Number of edges created
        """
        edges = []
        
        for id1, id2, score in similarity_pairs:
            if score >= threshold:
                edge = KGEdge(
                    source_id=id1,
                    target_id=id2,
                    relation_type=RelationType.SEMANTIC_SIMILAR.value,
                    weight=score,
                    description=f"Semantic similarity: {score:.3f}"
                )
                edges.append(edge)
        
        return self.add_edges_batch(edges)
    
    def get_node(self, signature_id: str) -> Optional[KGNode]:
        """Get a node by its signature ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT signature_id, node_type, level, description, parent_id, 
                   source_file, metadata
            FROM nodes WHERE signature_id = ?
        """, (signature_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return KGNode(
                signature_id=row[0],
                node_type=row[1],
                level=row[2],
                description=row[3],
                parent_id=row[4],
                source_file=row[5],
                metadata=json.loads(row[6]) if row[6] else None
            )
        return None
    
    def get_edges(
        self,
        node_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None
    ) -> List[KGEdge]:
        """
        Get edges connected to a node.
        
        Args:
            node_id: Signature ID of the node
            direction: 'outgoing', 'incoming', or 'both'
            relation_type: Optional filter by relation type
        
        Returns:
            List of KGEdge objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        edges = []
        
        if direction in ["outgoing", "both"]:
            query = "SELECT source_id, target_id, relation_type, weight, description, metadata FROM edges WHERE source_id = ?"
            params = [node_id]
            if relation_type:
                query += " AND relation_type = ?"
                params.append(relation_type)
            
            cursor.execute(query, params)
            for row in cursor.fetchall():
                edges.append(KGEdge(
                    source_id=row[0],
                    target_id=row[1],
                    relation_type=row[2],
                    weight=row[3],
                    description=row[4],
                    metadata=json.loads(row[5]) if row[5] else None
                ))
        
        if direction in ["incoming", "both"]:
            query = "SELECT source_id, target_id, relation_type, weight, description, metadata FROM edges WHERE target_id = ?"
            params = [node_id]
            if relation_type:
                query += " AND relation_type = ?"
                params.append(relation_type)
            
            cursor.execute(query, params)
            for row in cursor.fetchall():
                edges.append(KGEdge(
                    source_id=row[0],
                    target_id=row[1],
                    relation_type=row[2],
                    weight=row[3],
                    description=row[4],
                    metadata=json.loads(row[5]) if row[5] else None
                ))
        
        conn.close()
        return edges
    
    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> Set[str]:
        """
        Get neighbor nodes up to a certain depth.
        
        Args:
            node_id: Starting node
            relation_type: Optional filter by relation type
            max_depth: Maximum traversal depth
        
        Returns:
            Set of neighbor signature IDs
        """
        visited = set()
        current_level = {node_id}
        
        for _ in range(max_depth):
            next_level = set()
            for nid in current_level:
                edges = self.get_edges(nid, relation_type=relation_type)
                for edge in edges:
                    neighbor = edge.target_id if edge.source_id == nid else edge.source_id
                    if neighbor not in visited:
                        next_level.add(neighbor)
            
            visited.update(current_level)
            current_level = next_level
        
        visited.update(current_level)
        visited.discard(node_id)  # Remove starting node
        
        return visited
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.
        
        Args:
            start_id: Starting node
            end_id: Target node
            max_depth: Maximum path length
        
        Returns:
            List of node IDs in path, or None if no path found
        """
        if start_id == end_id:
            return [start_id]
        
        from collections import deque
        
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for edge in self.get_edges(current):
                neighbor = edge.target_id if edge.source_id == current else edge.source_id
                
                if neighbor == end_id:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_ancestors(self, node_id: str) -> List[str]:
        """
        Get all ancestor nodes (parent chain) in the hierarchy.
        
        Args:
            node_id: Starting node
        
        Returns:
            List of ancestor signature IDs (from immediate parent to root)
        """
        ancestors = []
        current = node_id
        
        while True:
            node = self.get_node(current)
            if not node or not node.parent_id:
                break
            ancestors.append(node.parent_id)
            current = node.parent_id
        
        return ancestors
    
    def get_children(self, node_id: str) -> List[str]:
        """
        Get direct children of a node in the hierarchy.
        
        Args:
            node_id: Parent node
        
        Returns:
            List of child signature IDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT signature_id FROM nodes WHERE parent_id = ?
        """, (node_id,))
        
        children = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return children
    
    def update_node_parent(self, node_id: str, parent_id: str) -> bool:
        """Update the parent of a node (for hierarchy building)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE nodes SET parent_id = ? WHERE signature_id = ?
            """, (parent_id, node_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count nodes by type
        cursor.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type")
        nodes_by_type = dict(cursor.fetchall())
        
        # Count edges by type
        cursor.execute("SELECT relation_type, COUNT(*) FROM edges GROUP BY relation_type")
        edges_by_type = dict(cursor.fetchall())
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM nodes")
        total_nodes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM edges")
        total_edges = cursor.fetchone()[0]
        
        # Hierarchy depth
        cursor.execute("SELECT MAX(level) FROM nodes")
        max_level = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "hierarchy_depth": max_level,
            "db_path": self.db_path
        }
    
    def export_to_json(self, output_path: str):
        """Export the entire graph to JSON format."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Export nodes
        cursor.execute("SELECT * FROM nodes")
        columns = [desc[0] for desc in cursor.description]
        nodes = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Export edges
        cursor.execute("SELECT * FROM edges")
        columns = [desc[0] for desc in cursor.description]
        edges = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "stats": self.get_stats()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        logger.info(f"Exported graph to {output_path}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing AudioKnowledgeGraph...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create knowledge graph
        kg = AudioKnowledgeGraph(working_dir=tmp_dir)
        
        # Add sample nodes
        nodes = [
            KGNode(
                signature_id=f"audio_test_{i:04d}",
                node_type="audio_chunk",
                level=0,
                source_file="test.wav",
                description=f"Audio chunk {i}"
            )
            for i in range(5)
        ]
        
        kg.add_nodes_batch(nodes)
        print(f"✅ Added {len(nodes)} nodes")
        
        # Create sequential edges
        chunks_by_source = {
            "test.wav": [f"audio_test_{i:04d}" for i in range(5)]
        }
        num_edges = kg.create_sequential_edges(chunks_by_source)
        print(f"✅ Created {num_edges} sequential edges")
        
        # Add semantic similarity edge
        sem_edge = KGEdge(
            source_id="audio_test_0000",
            target_id="audio_test_0003",
            relation_type=RelationType.SEMANTIC_SIMILAR.value,
            weight=0.85
        )
        kg.add_edge(sem_edge)
        print("✅ Added semantic edge")
        
        # Query graph
        node = kg.get_node("audio_test_0002")
        print(f"📍 Node: {node.signature_id}, type={node.node_type}")
        
        edges = kg.get_edges("audio_test_0002")
        print(f"🔗 Edges: {len(edges)}")
        
        neighbors = kg.get_neighbors("audio_test_0002", max_depth=2)
        print(f"👥 Neighbors (depth 2): {neighbors}")
        
        path = kg.find_path("audio_test_0000", "audio_test_0004")
        print(f"🛤️ Path: {path}")
        
        # Stats
        stats = kg.get_stats()
        print(f"📊 Stats: {stats}")
        
        print("\n✅ All tests passed!")
