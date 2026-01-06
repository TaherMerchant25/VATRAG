"""
Milvus Database Utilities for Multimodal LeanRAG
================================================
Handles storage and retrieval of audio chunk embeddings in Milvus vector database.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

try:
    from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
except ImportError:
    raise ImportError("Please install pymilvus: pip install pymilvus milvus-lite")

from audio_chunking import AudioChunk

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunkRecord:
    """Record stored in Milvus for an audio chunk."""
    id: int                    # Auto-assigned by Milvus
    signature_id: str          # Unique signature identifier
    source_file: str           # Original audio file
    chunk_index: int           # Position in sequence
    start_time: float          # Start time in seconds
    end_time: float            # End time in seconds
    duration: float            # Chunk duration
    embedding: List[float]     # Vector embedding


class MilvusAudioStore:
    """
    Milvus vector database for storing and retrieving audio chunk embeddings.
    """
    
    def __init__(
        self,
        working_dir: str,
        collection_name: str = "audio_chunks",
        embedding_dim: int = 512,
        db_name: str = "milvus_audio.db"
    ):
        """
        Initialize Milvus audio store.
        
        Args:
            working_dir: Directory to store the Milvus database
            collection_name: Name of the collection
            embedding_dim: Dimension of audio embeddings
            db_name: Name of the database file
        """
        self.working_dir = working_dir
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.db_path = os.path.join(working_dir, db_name)
        
        # Ensure working directory exists
        os.makedirs(working_dir, exist_ok=True)
        
        # Initialize Milvus client
        logger.info(f"Connecting to Milvus at {self.db_path}")
        self.client = MilvusClient(uri=self.db_path)
        
        # Create collection if needed
        self._ensure_collection()
        
        # Track signature ID to Milvus ID mapping
        self._signature_to_id: Dict[str, int] = {}
        self._next_id = 0
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            # Load existing data count
            stats = self.client.get_collection_stats(self.collection_name)
            logger.info(f"Collection has {stats.get('row_count', 0)} records")
        else:
            logger.info(f"Creating collection '{self.collection_name}'")
            
            # Create index parameters
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_name="embedding_index",
                index_type="IVF_FLAT",
                metric_type="IP",  # Inner Product (cosine similarity for normalized vectors)
                params={"nlist": 128}
            )
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                index_params=index_params,
                metric_type="IP",
                consistency_level="Strong"
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
    
    def insert_chunks(
        self,
        chunks: List[AudioChunk],
        embeddings: Dict[str, np.ndarray],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Insert audio chunks with their embeddings into Milvus.
        
        Args:
            chunks: List of AudioChunk objects
            embeddings: Dictionary mapping signature_id to embedding
            batch_size: Number of records to insert per batch
        
        Returns:
            Dictionary mapping signature_id to Milvus record ID
        """
        logger.info(f"Inserting {len(chunks)} chunks into Milvus")
        
        records = []
        signature_to_id = {}
        
        for chunk in chunks:
            if chunk.signature_id not in embeddings:
                logger.warning(f"No embedding for chunk {chunk.signature_id}, skipping")
                continue
            
            record = {
                "id": self._next_id,
                "signature_id": chunk.signature_id,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "duration": chunk.duration,
                "embedding": embeddings[chunk.signature_id].tolist()
            }
            
            signature_to_id[chunk.signature_id] = self._next_id
            self._signature_to_id[chunk.signature_id] = self._next_id
            self._next_id += 1
            records.append(record)
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )
            total_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1}: {total_inserted}/{len(records)} records")
        
        logger.info(f"Successfully inserted {total_inserted} audio chunks")
        
        return signature_to_id
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_expr: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Search for similar audio chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_expr: Optional filter expression
        
        Returns:
            List of search results with metadata
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            params={"metric_type": "IP"},
            filter=filter_expr if filter_expr else None,
            output_fields=["signature_id", "source_file", "chunk_index", 
                          "start_time", "end_time", "duration"]
        )
        
        # Format results
        formatted = []
        for hit in results[0]:
            formatted.append({
                "signature_id": hit["entity"]["signature_id"],
                "source_file": hit["entity"]["source_file"],
                "chunk_index": hit["entity"]["chunk_index"],
                "start_time": hit["entity"]["start_time"],
                "end_time": hit["entity"]["end_time"],
                "duration": hit["entity"]["duration"],
                "score": hit["distance"]
            })
        
        return formatted
    
    def search_by_text(
        self,
        text_query: str,
        embedder: Any,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (requires cross-modal embedder like CLAP).
        
        Args:
            text_query: Text description to search for
            embedder: Embedder with embed_text method
            top_k: Number of results
        
        Returns:
            List of search results
        """
        text_embedding = embedder.embed_text(text_query)
        return self.search(text_embedding, top_k)
    
    def get_by_signature_id(self, signature_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its signature ID.
        
        Args:
            signature_id: Unique signature identifier
        
        Returns:
            Chunk metadata or None if not found
        """
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'signature_id == "{signature_id}"',
            output_fields=["signature_id", "source_file", "chunk_index",
                          "start_time", "end_time", "duration"]
        )
        
        if results:
            return results[0]
        return None
    
    def get_all_signature_ids(self) -> List[str]:
        """Get all signature IDs in the collection."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["signature_id"],
            limit=100000  # Adjust based on expected size
        )
        return [r["signature_id"] for r in results]
    
    def get_chunks_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific source file.
        
        Args:
            source_file: Path to the original audio file
        
        Returns:
            List of chunk records
        """
        # Escape quotes in file path
        escaped_path = source_file.replace('"', '\\"')
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'source_file == "{escaped_path}"',
            output_fields=["signature_id", "source_file", "chunk_index",
                          "start_time", "end_time", "duration"]
        )
        
        # Sort by chunk index
        results.sort(key=lambda x: x["chunk_index"])
        
        return results
    
    def delete_collection(self):
        """Delete the entire collection."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.client.has_collection(self.collection_name):
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "row_count": stats.get("row_count", 0),
                "embedding_dim": self.embedding_dim,
                "db_path": self.db_path
            }
        return {"error": "Collection does not exist"}
    
    def export_signature_mapping(self, output_path: str):
        """
        Export signature ID to Milvus ID mapping.
        
        Args:
            output_path: Path to save JSON file
        """
        mapping = {
            "collection_name": self.collection_name,
            "total_records": len(self._signature_to_id),
            "mappings": self._signature_to_id
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Exported {len(self._signature_to_id)} mappings to {output_path}")


def find_adjacent_chunks(
    store: MilvusAudioStore,
    signature_id: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find the previous and next chunks relative to a given chunk.
    
    Args:
        store: MilvusAudioStore instance
        signature_id: Signature ID of the reference chunk
    
    Returns:
        Tuple of (previous_chunk, next_chunk) or (None, None) if not found
    """
    chunk = store.get_by_signature_id(signature_id)
    if not chunk:
        return None, None
    
    source_file = chunk["source_file"]
    chunk_index = chunk["chunk_index"]
    
    # Get all chunks from same source
    all_chunks = store.get_chunks_by_source(source_file)
    
    prev_chunk = None
    next_chunk = None
    
    for c in all_chunks:
        if c["chunk_index"] == chunk_index - 1:
            prev_chunk = c
        elif c["chunk_index"] == chunk_index + 1:
            next_chunk = c
    
    return prev_chunk, next_chunk


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing MilvusAudioStore...")
    
    # Create a test store
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MilvusAudioStore(
            working_dir=tmp_dir,
            collection_name="test_audio",
            embedding_dim=512
        )
        
        # Create dummy chunks and embeddings
        from audio_chunking import AudioChunk
        
        chunks = [
            AudioChunk(
                signature_id=f"audio_test_{i:04d}",
                source_file="test.wav",
                chunk_index=i,
                start_time=i * 8.0,
                end_time=(i + 1) * 8.0 + 2.0,
                duration=10.0,
                sample_rate=16000
            )
            for i in range(5)
        ]
        
        embeddings = {
            chunk.signature_id: np.random.randn(512).astype(np.float32)
            for chunk in chunks
        }
        
        # Insert
        id_mapping = store.insert_chunks(chunks, embeddings)
        print(f"✅ Inserted {len(id_mapping)} chunks")
        
        # Search
        query = np.random.randn(512).astype(np.float32)
        results = store.search(query, top_k=3)
        print(f"🔍 Search returned {len(results)} results")
        for r in results:
            print(f"   - {r['signature_id']}: score={r['score']:.4f}")
        
        # Get stats
        stats = store.get_stats()
        print(f"📊 Collection stats: {stats}")
        
        print("\n✅ All tests passed!")
