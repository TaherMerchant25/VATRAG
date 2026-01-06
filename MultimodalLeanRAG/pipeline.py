"""
Main Pipeline Orchestrator for Multimodal LeanRAG
=================================================
Coordinates the full pipeline: chunking -> embedding -> Milvus storage -> Knowledge Graph.
"""

import os
import json
import logging
import yaml
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from audio_chunking import AudioChunk, chunk_audio_file, chunk_audio_directory
from audio_embedding import get_embedder, AudioEmbedder, SimpleAudioEmbedder
from milvus_store import MilvusAudioStore, find_adjacent_chunks
from knowledge_graph import (
    AudioKnowledgeGraph, KGNode, KGEdge, RelationType
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the multimodal pipeline."""
    # Working directory
    working_dir: str
    
    # Audio chunking
    chunk_duration_sec: float = 10.0
    overlap_sec: float = 2.0
    sample_rate: int = 16000
    
    # Embedding
    embedding_model: str = "simple"  # 'clap' or 'simple'
    embedding_dim: int = 512
    
    # Milvus
    milvus_collection: str = "audio_chunks"
    
    # Knowledge Graph
    similarity_threshold: float = 0.7
    
    # Processing
    batch_size: int = 32


class MultimodalLeanRAGPipeline:
    """
    Main pipeline for processing multimodal audio data with LeanRAG.
    
    Pipeline stages:
    1. Audio Chunking - Split audio into overlapping chunks
    2. Embedding Generation - Create vector embeddings for each chunk
    3. Milvus Storage - Store embeddings with signature IDs
    4. Knowledge Graph - Store signature IDs and relationships
    5. Relation Extraction - Create edges between related chunks
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: PipelineConfig object
        """
        self.config = config
        
        # Ensure working directory exists
        os.makedirs(config.working_dir, exist_ok=True)
        
        logger.info(f"Initializing MultimodalLeanRAG Pipeline")
        logger.info(f"Working directory: {config.working_dir}")
        
        # Initialize components
        self._init_embedder()
        self._init_milvus()
        self._init_knowledge_graph()
        
        # Track processed data
        self.processed_chunks: List[AudioChunk] = []
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def _init_embedder(self):
        """Initialize the audio embedder."""
        logger.info(f"Initializing embedder: {self.config.embedding_model}")
        self.embedder = get_embedder(
            model_type=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim
        )
    
    def _init_milvus(self):
        """Initialize Milvus vector store."""
        logger.info("Initializing Milvus store")
        self.milvus_store = MilvusAudioStore(
            working_dir=self.config.working_dir,
            collection_name=self.config.milvus_collection,
            embedding_dim=self.config.embedding_dim
        )
    
    def _init_knowledge_graph(self):
        """Initialize the knowledge graph."""
        logger.info("Initializing Knowledge Graph")
        self.kg = AudioKnowledgeGraph(working_dir=self.config.working_dir)
    
    def process_audio_file(
        self,
        audio_path: str,
        generate_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single audio file through the full pipeline.
        
        Args:
            audio_path: Path to the audio file
            generate_relations: Whether to generate semantic relations
        
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        results = {
            "source_file": audio_path,
            "num_chunks": 0,
            "signature_ids": [],
            "num_edges": 0
        }
        
        # Step 1: Chunk the audio
        logger.info("Step 1: Chunking audio...")
        chunks = chunk_audio_file(
            file_path=audio_path,
            chunk_duration_sec=self.config.chunk_duration_sec,
            overlap_sec=self.config.overlap_sec,
            target_sr=self.config.sample_rate
        )
        results["num_chunks"] = len(chunks)
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 3: Store in Milvus
        logger.info("Step 3: Storing in Milvus...")
        signature_mapping = self.milvus_store.insert_chunks(chunks, embeddings)
        results["signature_ids"] = list(signature_mapping.keys())
        
        # Step 4: Create KG nodes
        logger.info("Step 4: Creating Knowledge Graph nodes...")
        kg_nodes = []
        for chunk in chunks:
            node = KGNode(
                signature_id=chunk.signature_id,
                node_type="audio_chunk",
                level=0,
                source_file=chunk.source_file,
                description=f"Audio chunk {chunk.chunk_index} from {os.path.basename(chunk.source_file)} "
                           f"({chunk.start_time:.1f}s - {chunk.end_time:.1f}s)",
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration
                }
            )
            kg_nodes.append(node)
        
        self.kg.add_nodes_batch(kg_nodes)
        
        # Step 5: Create sequential edges
        logger.info("Step 5: Creating sequential edges...")
        chunks_by_source = {
            audio_path: [chunk.signature_id for chunk in chunks]
        }
        num_seq_edges = self.kg.create_sequential_edges(chunks_by_source)
        results["num_edges"] += num_seq_edges
        
        # Step 6: Create semantic similarity edges
        if generate_relations:
            logger.info("Step 6: Creating semantic similarity edges...")
            similarity_pairs = self._compute_similarities(
                list(signature_mapping.keys()),
                embeddings
            )
            num_sem_edges = self.kg.create_semantic_edges(
                similarity_pairs,
                threshold=self.config.similarity_threshold
            )
            results["num_edges"] += num_sem_edges
        
        # Update tracking
        self.processed_chunks.extend(chunks)
        self.embeddings.update(embeddings)
        
        logger.info(f"✅ Processed {results['num_chunks']} chunks, "
                   f"created {results['num_edges']} edges")
        
        return results
    
    def process_audio_directory(
        self,
        directory_path: str,
        supported_formats: List[str] = [".wav", ".mp3", ".flac", ".ogg"],
        generate_cross_file_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Process all audio files in a directory.
        
        Args:
            directory_path: Path to directory containing audio files
            supported_formats: List of supported audio extensions
            generate_cross_file_relations: Whether to create relations across files
        
        Returns:
            Aggregated processing results
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Find all audio files
        audio_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    audio_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        all_results = {
            "directory": directory_path,
            "total_files": len(audio_files),
            "total_chunks": 0,
            "total_edges": 0,
            "file_results": []
        }
        
        # Process each file
        for audio_path in tqdm(audio_files, desc="Processing files"):
            try:
                result = self.process_audio_file(
                    audio_path,
                    generate_relations=True
                )
                all_results["file_results"].append(result)
                all_results["total_chunks"] += result["num_chunks"]
                all_results["total_edges"] += result["num_edges"]
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                continue
        
        # Create cross-file semantic relations
        if generate_cross_file_relations and len(self.embeddings) > 1:
            logger.info("Creating cross-file semantic relations...")
            all_sig_ids = list(self.embeddings.keys())
            cross_file_pairs = self._compute_similarities(
                all_sig_ids,
                self.embeddings,
                same_file_only=False
            )
            num_cross_edges = self.kg.create_semantic_edges(
                cross_file_pairs,
                threshold=self.config.similarity_threshold
            )
            all_results["total_edges"] += num_cross_edges
            logger.info(f"Created {num_cross_edges} cross-file edges")
        
        return all_results
    
    def _compute_similarities(
        self,
        signature_ids: List[str],
        embeddings: Dict[str, np.ndarray],
        same_file_only: bool = True,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Compute pairwise similarities between embeddings.
        
        Args:
            signature_ids: List of signature IDs
            embeddings: Dictionary of embeddings
            same_file_only: Only compare within same file
            top_k: Number of top similar pairs per chunk
        
        Returns:
            List of (id1, id2, similarity) tuples
        """
        similarity_pairs = []
        
        # Build embedding matrix
        valid_ids = [sid for sid in signature_ids if sid in embeddings]
        if len(valid_ids) < 2:
            return []
        
        emb_matrix = np.array([embeddings[sid] for sid in valid_ids])
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_matrix = emb_matrix / norms
        
        # Compute similarity matrix
        sim_matrix = emb_matrix @ emb_matrix.T
        
        # Extract top-k pairs for each chunk
        for i, id1 in enumerate(valid_ids):
            similarities = sim_matrix[i]
            
            # Get top-k indices (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            for j in top_indices:
                id2 = valid_ids[j]
                score = float(similarities[j])
                
                # Avoid duplicate pairs
                if id1 < id2:
                    similarity_pairs.append((id1, id2, score))
        
        # Remove duplicates and sort by score
        unique_pairs = list(set(similarity_pairs))
        unique_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return unique_pairs
    
    def get_related_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        expand_graph: bool = True,
        graph_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find related chunks using vector search and graph expansion.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results from vector search
            expand_graph: Whether to expand using KG relations
            graph_depth: Depth of graph expansion
        
        Returns:
            List of related chunk information
        """
        # Vector search
        milvus_results = self.milvus_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        results = []
        seen_ids = set()
        
        for mr in milvus_results:
            sig_id = mr["signature_id"]
            seen_ids.add(sig_id)
            
            # Get KG node info
            node = self.kg.get_node(sig_id)
            
            result = {
                "signature_id": sig_id,
                "source_file": mr["source_file"],
                "start_time": mr["start_time"],
                "end_time": mr["end_time"],
                "score": mr["score"],
                "description": node.description if node else None,
                "source": "vector_search"
            }
            results.append(result)
            
            # Graph expansion
            if expand_graph and node:
                neighbors = self.kg.get_neighbors(
                    sig_id,
                    max_depth=graph_depth
                )
                
                for neighbor_id in neighbors:
                    if neighbor_id not in seen_ids:
                        seen_ids.add(neighbor_id)
                        
                        neighbor_node = self.kg.get_node(neighbor_id)
                        if neighbor_node:
                            results.append({
                                "signature_id": neighbor_id,
                                "source_file": neighbor_node.source_file,
                                "description": neighbor_node.description,
                                "source": "graph_expansion"
                            })
        
        return results
    
    def get_reasoning_path(
        self,
        signature_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get reasoning paths between multiple signature IDs.
        
        Args:
            signature_ids: List of signature IDs to connect
        
        Returns:
            Dictionary with paths and relationship information
        """
        from itertools import combinations
        
        paths = []
        relations = []
        
        for id1, id2 in combinations(signature_ids, 2):
            path = self.kg.find_path(id1, id2, max_depth=5)
            if path:
                paths.append({
                    "from": id1,
                    "to": id2,
                    "path": path
                })
                
                # Get relations along path
                for i in range(len(path) - 1):
                    edges = self.kg.get_edges(path[i], direction="outgoing")
                    for edge in edges:
                        if edge.target_id == path[i + 1]:
                            relations.append({
                                "source": edge.source_id,
                                "target": edge.target_id,
                                "type": edge.relation_type,
                                "description": edge.description
                            })
        
        return {
            "paths": paths,
            "relations": relations
        }
    
    def export_results(self, output_path: str):
        """Export processing results to JSON."""
        results = {
            "config": {
                "working_dir": self.config.working_dir,
                "chunk_duration": self.config.chunk_duration_sec,
                "overlap": self.config.overlap_sec,
                "embedding_model": self.config.embedding_model
            },
            "milvus_stats": self.milvus_store.get_stats(),
            "kg_stats": self.kg.get_stats(),
            "processed_chunks": [
                chunk.to_dict() for chunk in self.processed_chunks
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Exported results to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_chunks_processed": len(self.processed_chunks),
            "total_embeddings": len(self.embeddings),
            "milvus": self.milvus_store.get_stats(),
            "knowledge_graph": self.kg.get_stats()
        }


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return PipelineConfig(
        working_dir=config_dict.get('working_dir', './multimodal_output'),
        chunk_duration_sec=config_dict.get('audio', {}).get('chunk_duration_sec', 10.0),
        overlap_sec=config_dict.get('audio', {}).get('overlap_sec', 2.0),
        sample_rate=config_dict.get('audio', {}).get('sample_rate', 16000),
        embedding_model=config_dict.get('embeddings', {}).get('audio', {}).get('model', 'simple'),
        embedding_dim=config_dict.get('embeddings', {}).get('audio', {}).get('dimension', 512),
        similarity_threshold=config_dict.get('processing', {}).get('similarity_threshold', 0.7),
        batch_size=config_dict.get('processing', {}).get('batch_size', 32)
    )


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal LeanRAG Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./multimodal_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        help="Chunk duration in seconds"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        help="Overlap between chunks in seconds"
    )
    parser.add_argument(
        "--embedding-model",
        choices=["clap", "simple"],
        default="simple",
        help="Embedding model to use"
    )
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig(
            working_dir=args.output_dir,
            chunk_duration_sec=args.chunk_duration,
            overlap_sec=args.overlap,
            embedding_model=args.embedding_model
        )
    
    # Initialize pipeline
    pipeline = MultimodalLeanRAGPipeline(config)
    
    # Process input
    if os.path.isdir(args.input):
        results = pipeline.process_audio_directory(args.input)
    else:
        results = pipeline.process_audio_file(args.input)
    
    # Export results
    output_file = os.path.join(args.output_dir, "pipeline_results.json")
    pipeline.export_results(output_file)
    
    # Print summary
    stats = pipeline.get_stats()
    print("\n" + "=" * 60)
    print("📊 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total chunks processed: {stats['total_chunks_processed']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"KG nodes: {stats['knowledge_graph']['total_nodes']}")
    print(f"KG edges: {stats['knowledge_graph']['total_edges']}")
    print(f"\n📁 Results saved to: {args.output_dir}")
