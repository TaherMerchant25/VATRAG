"""
Query and Retrieval Module for Multimodal LeanRAG
=================================================
Handles hierarchical retrieval using the Knowledge Graph structure.
Implements bottom-up traversal from audio chunks to aggregation nodes.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np

from audio_embedding import get_embedder, AudioEmbedder, SimpleAudioEmbedder
from milvus_store import MilvusAudioStore
from knowledge_graph import AudioKnowledgeGraph, KGNode, KGEdge, RelationType

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval query."""
    signature_id: str
    source_file: Optional[str]
    start_time: Optional[float]
    end_time: Optional[float]
    score: float
    description: Optional[str]
    source: str  # 'vector_search', 'graph_expansion', 'aggregation'
    level: int = 0
    metadata: Optional[Dict] = None


class MultimodalRetriever:
    """
    Hierarchical retriever for multimodal audio data.
    
    Implements LeanRAG-style retrieval:
    1. Vector search to find most relevant audio chunks
    2. Bottom-up graph traversal to gather context
    3. Reasoning path construction between retrieved chunks
    """
    
    def __init__(
        self,
        working_dir: str,
        embedder: Optional[Union[AudioEmbedder, SimpleAudioEmbedder]] = None,
        embedding_dim: int = 512
    ):
        """
        Initialize the retriever.
        
        Args:
            working_dir: Working directory with Milvus DB and KG
            embedder: Audio/text embedder (will create if not provided)
            embedding_dim: Embedding dimension
        """
        self.working_dir = working_dir
        self.embedding_dim = embedding_dim
        
        # Initialize components
        if embedder is None:
            self.embedder = get_embedder(
                model_type="simple",
                embedding_dim=embedding_dim
            )
        else:
            self.embedder = embedder
        
        # Connect to stores
        self.milvus_store = MilvusAudioStore(
            working_dir=working_dir,
            embedding_dim=embedding_dim
        )
        self.kg = AudioKnowledgeGraph(working_dir=working_dir)
        
        logger.info(f"Initialized MultimodalRetriever for {working_dir}")
    
    def search_by_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        top_k: int = 10,
        expand_graph: bool = True
    ) -> List[RetrievalResult]:
        """
        Search using an audio query.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of audio
            top_k: Number of results
            expand_graph: Whether to expand using KG
        
        Returns:
            List of RetrievalResult objects
        """
        # Generate embedding for query audio
        query_embedding = self.embedder.embed_audio(audio_data, sample_rate)
        
        return self._search_by_embedding(
            query_embedding,
            top_k=top_k,
            expand_graph=expand_graph
        )
    
    def search_by_text(
        self,
        text_query: str,
        top_k: int = 10,
        expand_graph: bool = True
    ) -> List[RetrievalResult]:
        """
        Search using a text query (requires CLAP model).
        
        Args:
            text_query: Text description to search for
            top_k: Number of results
            expand_graph: Whether to expand using KG
        
        Returns:
            List of RetrievalResult objects
        """
        if not hasattr(self.embedder, 'embed_text'):
            raise ValueError("Text search requires CLAP embedder")
        
        # Generate embedding for text query
        query_embedding = self.embedder.embed_text(text_query)
        
        return self._search_by_embedding(
            query_embedding,
            top_k=top_k,
            expand_graph=expand_graph
        )
    
    def _search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        expand_graph: bool = True
    ) -> List[RetrievalResult]:
        """
        Internal search method using embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of vector search results
            expand_graph: Whether to expand using KG
        
        Returns:
            List of RetrievalResult objects
        """
        results = []
        seen_ids = set()
        
        # Step 1: Vector search in Milvus
        milvus_results = self.milvus_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        for mr in milvus_results:
            sig_id = mr["signature_id"]
            seen_ids.add(sig_id)
            
            # Enrich with KG info
            node = self.kg.get_node(sig_id)
            
            result = RetrievalResult(
                signature_id=sig_id,
                source_file=mr.get("source_file"),
                start_time=mr.get("start_time"),
                end_time=mr.get("end_time"),
                score=mr["score"],
                description=node.description if node else None,
                source="vector_search",
                level=node.level if node else 0,
                metadata=node.metadata if node else None
            )
            results.append(result)
        
        # Step 2: Graph expansion (optional)
        if expand_graph:
            expanded = self._expand_via_graph(
                seed_ids=[r.signature_id for r in results],
                seen_ids=seen_ids
            )
            results.extend(expanded)
        
        return results
    
    def _expand_via_graph(
        self,
        seed_ids: List[str],
        seen_ids: set,
        max_depth: int = 2
    ) -> List[RetrievalResult]:
        """
        Expand retrieval results using knowledge graph relationships.
        
        Args:
            seed_ids: Starting signature IDs
            seen_ids: Already seen IDs (to avoid duplicates)
            max_depth: Maximum expansion depth
        
        Returns:
            Additional RetrievalResult objects from expansion
        """
        expanded_results = []
        
        for seed_id in seed_ids:
            # Get neighbors
            neighbors = self.kg.get_neighbors(
                seed_id,
                max_depth=max_depth
            )
            
            for neighbor_id in neighbors:
                if neighbor_id not in seen_ids:
                    seen_ids.add(neighbor_id)
                    
                    node = self.kg.get_node(neighbor_id)
                    if node:
                        # Determine edge type connecting to seed
                        edges = self.kg.get_edges(seed_id)
                        edge_types = [
                            e.relation_type for e in edges
                            if e.target_id == neighbor_id or e.source_id == neighbor_id
                        ]
                        
                        result = RetrievalResult(
                            signature_id=neighbor_id,
                            source_file=node.source_file,
                            start_time=None,
                            end_time=None,
                            score=0.5,  # Default score for graph expansion
                            description=node.description,
                            source="graph_expansion",
                            level=node.level,
                            metadata={
                                "expanded_from": seed_id,
                                "edge_types": edge_types
                            }
                        )
                        expanded_results.append(result)
        
        return expanded_results
    
    def get_hierarchical_context(
        self,
        signature_ids: List[str],
        include_ancestors: bool = True,
        include_siblings: bool = True
    ) -> Dict[str, Any]:
        """
        Get hierarchical context for a set of signature IDs.
        
        This implements the "bottom-up" retrieval strategy of LeanRAG:
        - Start from fine-grained audio chunks
        - Traverse up to aggregation nodes
        - Collect context at multiple levels
        
        Args:
            signature_ids: List of starting signature IDs
            include_ancestors: Include parent/grandparent nodes
            include_siblings: Include sibling nodes (same parent)
        
        Returns:
            Dictionary with hierarchical context
        """
        context = {
            "base_chunks": [],
            "ancestors": [],
            "siblings": [],
            "relations": []
        }
        
        all_ancestor_ids = set()
        
        for sig_id in signature_ids:
            node = self.kg.get_node(sig_id)
            if not node:
                continue
            
            # Add base chunk
            context["base_chunks"].append(node.to_dict())
            
            # Get ancestors (bottom-up traversal)
            if include_ancestors:
                ancestors = self.kg.get_ancestors(sig_id)
                for anc_id in ancestors:
                    if anc_id not in all_ancestor_ids:
                        all_ancestor_ids.add(anc_id)
                        anc_node = self.kg.get_node(anc_id)
                        if anc_node:
                            context["ancestors"].append(anc_node.to_dict())
            
            # Get siblings
            if include_siblings and node.parent_id:
                siblings = self.kg.get_children(node.parent_id)
                for sib_id in siblings:
                    if sib_id != sig_id:
                        sib_node = self.kg.get_node(sib_id)
                        if sib_node:
                            context["siblings"].append(sib_node.to_dict())
            
            # Get relations
            edges = self.kg.get_edges(sig_id)
            for edge in edges:
                context["relations"].append(edge.to_dict())
        
        # Deduplicate
        context["ancestors"] = _deduplicate_by_key(context["ancestors"], "signature_id")
        context["siblings"] = _deduplicate_by_key(context["siblings"], "signature_id")
        
        return context
    
    def get_reasoning_paths(
        self,
        signature_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Construct reasoning paths between retrieved audio chunks.
        
        Args:
            signature_ids: List of signature IDs to connect
        
        Returns:
            Dictionary with paths and path descriptions
        """
        from itertools import combinations
        
        paths_info = {
            "paths": [],
            "path_relations": [],
            "common_ancestors": []
        }
        
        # Find paths between all pairs
        for id1, id2 in combinations(signature_ids, 2):
            path = self.kg.find_path(id1, id2, max_depth=6)
            
            if path:
                path_data = {
                    "from": id1,
                    "to": id2,
                    "path": path,
                    "length": len(path)
                }
                paths_info["paths"].append(path_data)
                
                # Get relations along path
                for i in range(len(path) - 1):
                    edges = self.kg.get_edges(path[i], direction="both")
                    for edge in edges:
                        other = edge.target_id if edge.source_id == path[i] else edge.source_id
                        if other == path[i + 1]:
                            paths_info["path_relations"].append({
                                "source": path[i],
                                "target": path[i + 1],
                                "type": edge.relation_type,
                                "description": edge.description
                            })
            
            # Find common ancestors
            anc1 = set(self.kg.get_ancestors(id1))
            anc2 = set(self.kg.get_ancestors(id2))
            common = anc1 & anc2
            
            if common:
                paths_info["common_ancestors"].append({
                    "nodes": [id1, id2],
                    "common_ancestors": list(common)
                })
        
        return paths_info
    
    def build_retrieval_context(
        self,
        query_results: List[RetrievalResult],
        max_context_items: int = 20
    ) -> str:
        """
        Build a text context from retrieval results for LLM consumption.
        
        Args:
            query_results: List of RetrievalResult objects
            max_context_items: Maximum items to include
        
        Returns:
            Formatted context string
        """
        # Sort by score
        sorted_results = sorted(
            query_results,
            key=lambda x: x.score,
            reverse=True
        )[:max_context_items]
        
        # Get hierarchical context
        sig_ids = [r.signature_id for r in sorted_results]
        hier_context = self.get_hierarchical_context(sig_ids)
        
        # Get reasoning paths
        path_context = self.get_reasoning_paths(sig_ids[:5])  # Top 5
        
        # Build formatted context
        context_parts = []
        
        # Audio chunk information
        context_parts.append("=== RETRIEVED AUDIO CHUNKS ===")
        for i, result in enumerate(sorted_results, 1):
            chunk_info = f"""
Chunk {i}: {result.signature_id}
  - Source: {result.source_file}
  - Time: {result.start_time:.1f}s - {result.end_time:.1f}s
  - Score: {result.score:.4f}
  - Description: {result.description or 'N/A'}
"""
            context_parts.append(chunk_info)
        
        # Aggregation/ancestor information
        if hier_context["ancestors"]:
            context_parts.append("\n=== AGGREGATION CONTEXT ===")
            for anc in hier_context["ancestors"]:
                context_parts.append(f"Level {anc.get('level', '?')}: {anc.get('description', 'N/A')}")
        
        # Relationship information
        if hier_context["relations"]:
            context_parts.append("\n=== RELATIONSHIPS ===")
            for rel in hier_context["relations"][:10]:
                context_parts.append(
                    f"{rel['source_id']} --[{rel['relation_type']}]--> {rel['target_id']}"
                )
        
        # Reasoning paths
        if path_context["paths"]:
            context_parts.append("\n=== REASONING PATHS ===")
            for path in path_context["paths"][:3]:
                context_parts.append(f"Path: {' -> '.join(path['path'])}")
        
        return "\n".join(context_parts)


def _deduplicate_by_key(items: List[Dict], key: str) -> List[Dict]:
    """Remove duplicates from list of dicts by key."""
    seen = set()
    unique = []
    for item in items:
        if item.get(key) not in seen:
            seen.add(item.get(key))
            unique.append(item)
    return unique


# ============================================================================
# High-Level Query Interface
# ============================================================================

class MultimodalQueryEngine:
    """
    High-level query engine combining retrieval with LLM generation.
    """
    
    def __init__(
        self,
        working_dir: str,
        llm_client: Optional[Any] = None,
        llm_model: str = "gpt-4"
    ):
        """
        Initialize query engine.
        
        Args:
            working_dir: Working directory with indexes
            llm_client: Optional OpenAI-compatible client
            llm_model: Model name for generation
        """
        self.retriever = MultimodalRetriever(working_dir)
        self.llm_client = llm_client
        self.llm_model = llm_model
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        generate_answer: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a query and optionally generate an answer.
        
        Args:
            query_text: Natural language query
            top_k: Number of results to retrieve
            generate_answer: Whether to generate LLM answer
        
        Returns:
            Dictionary with results and optional answer
        """
        # Retrieve relevant chunks
        try:
            results = self.retriever.search_by_text(query_text, top_k=top_k)
        except ValueError:
            # Fallback if text search not available
            logger.warning("Text search not available, using dummy results")
            results = []
        
        # Build context
        context = self.retriever.build_retrieval_context(results)
        
        response = {
            "query": query_text,
            "num_results": len(results),
            "results": [r.__dict__ for r in results],
            "context": context
        }
        
        # Generate answer if requested and LLM available
        if generate_answer and self.llm_client:
            prompt = f"""Based on the following retrieved audio context, answer the query.

Query: {query_text}

Context:
{context}

Please provide a comprehensive answer based on the retrieved audio information."""
            
            completion = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            response["answer"] = completion.choices[0].message.content
        
        return response


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query Multimodal LeanRAG")
    parser.add_argument("--working-dir", "-d", required=True, help="Working directory")
    parser.add_argument("--query", "-q", help="Text query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = MultimodalRetriever(args.working_dir)
    
    if args.query:
        try:
            results = retriever.search_by_text(args.query, top_k=args.top_k)
            
            print(f"\n🔍 Query: {args.query}")
            print(f"📊 Found {len(results)} results\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.signature_id}")
                print(f"   Score: {result.score:.4f}")
                print(f"   Source: {result.source}")
                print(f"   Description: {result.description}")
                print()
        except ValueError as e:
            print(f"⚠️ {e}")
            print("Text search requires CLAP model. Use audio query instead.")
    else:
        # Show stats
        stats = retriever.kg.get_stats()
        print(f"\n📊 Knowledge Graph Stats:")
        print(f"   Nodes: {stats['total_nodes']}")
        print(f"   Edges: {stats['total_edges']}")
        print(f"   By type: {stats['nodes_by_type']}")
