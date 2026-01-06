"""
Multimodal LeanRAG Package
==========================
Knowledge-Graph-Based Generation with Semantic Aggregation for Multimodal Audio Data.
"""

from .audio_chunking import (
    AudioChunk,
    chunk_audio_file,
    chunk_audio_directory,
    generate_signature_id
)

from .audio_embedding import (
    AudioEmbedder,
    SimpleAudioEmbedder,
    get_embedder
)

from .milvus_store import (
    MilvusAudioStore,
    AudioChunkRecord
)

from .knowledge_graph import (
    AudioKnowledgeGraph,
    KGNode,
    KGEdge,
    RelationType
)

from .pipeline import (
    MultimodalLeanRAGPipeline,
    PipelineConfig
)

from .query import (
    MultimodalRetriever,
    MultimodalQueryEngine,
    RetrievalResult
)

__version__ = "0.1.0"
__author__ = "LeanRAG Team"

__all__ = [
    # Audio Chunking
    "AudioChunk",
    "chunk_audio_file",
    "chunk_audio_directory",
    "generate_signature_id",
    
    # Audio Embedding
    "AudioEmbedder",
    "SimpleAudioEmbedder",
    "get_embedder",
    
    # Milvus Store
    "MilvusAudioStore",
    "AudioChunkRecord",
    
    # Knowledge Graph
    "AudioKnowledgeGraph",
    "KGNode",
    "KGEdge",
    "RelationType",
    
    # Pipeline
    "MultimodalLeanRAGPipeline",
    "PipelineConfig",
    
    # Query
    "MultimodalRetriever",
    "MultimodalQueryEngine",
    "RetrievalResult",
]
