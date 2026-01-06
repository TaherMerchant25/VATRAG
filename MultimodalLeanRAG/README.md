# Multimodal LeanRAG

Knowledge-Graph-Based Generation with Semantic Aggregation for Multimodal Audio Data.

This module extends the LeanRAG framework to handle multimodal data, specifically audio content. It extracts audio chunks, stores them in Milvus vector database with unique signature IDs, and builds a knowledge graph to represent relationships between audio segments.

## 🎯 Key Features

- **Audio Chunking**: Split audio files into overlapping chunks with unique signature IDs
- **Vector Embeddings**: Generate embeddings using CLAP or simple mel-spectrogram features
- **Milvus Storage**: Store chunk embeddings for fast similarity search
- **Knowledge Graph**: Store signature IDs as nodes with relationship edges
- **Hierarchical Retrieval**: Bottom-up traversal from chunks to aggregations

## 🏗️ Architecture

```
Audio File
    ↓
┌─────────────────────────────────────┐
│  1. Audio Chunking                  │
│  - Split into 10s chunks            │
│  - 2s overlap (sliding window)      │
│  - Generate unique signature IDs    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Embedding Generation            │
│  - CLAP model (512-dim)             │
│  - Or simple mel-spectrogram        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Milvus Vector Store             │
│  - Store embeddings with metadata   │
│  - Map signature_id → vector        │
│  - Enable similarity search         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Knowledge Graph                 │
│  - Nodes: signature IDs             │
│  - Edges: chunk relationships       │
│    • Sequential (temporal)          │
│    • Semantic similarity            │
│    • Aggregation hierarchy          │
└─────────────────────────────────────┘
```

## 📁 Module Structure

```
MultimodalLeanRAG/
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies
├── audio_chunking.py     # Audio file chunking with signature ID generation
├── audio_embedding.py    # CLAP/mel-spectrogram embeddings
├── milvus_store.py       # Milvus vector database operations
├── knowledge_graph.py    # SQLite-based knowledge graph
├── pipeline.py           # Main pipeline orchestrator
├── query.py              # Retrieval and query module
└── README.md             # This file
```

## 🚀 Quick Start

### Installation

```bash
cd MultimodalLeanRAG
pip install -r requirements.txt
```

### Basic Usage

#### 1. Process Audio Files

```python
from pipeline import MultimodalLeanRAGPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    working_dir="./output",
    chunk_duration_sec=10.0,
    overlap_sec=2.0,
    embedding_model="simple"  # or "clap" for better quality
)

# Initialize and run
pipeline = MultimodalLeanRAGPipeline(config)

# Process single file
results = pipeline.process_audio_file("path/to/audio.wav")

# Or process directory
results = pipeline.process_audio_directory("path/to/audio_folder/")
```

#### 2. Query the Index

```python
from query import MultimodalRetriever

retriever = MultimodalRetriever(working_dir="./output")

# Search by text (requires CLAP model)
results = retriever.search_by_text("speech about machine learning")

# Get hierarchical context
context = retriever.get_hierarchical_context(
    signature_ids=[r.signature_id for r in results]
)
```

### Command Line Interface

```bash
# Process audio
python pipeline.py --input ./audio_files --output-dir ./output

# Query
python query.py --working-dir ./output --query "What topics are discussed?"
```

## 🔑 Signature ID System

Each audio chunk receives a unique signature ID based on:
- Audio content hash (first/last 1000 samples)
- Source file name
- Chunk index and timing

Format: `audio_{hash16}`

Example: `audio_a1b2c3d4e5f67890`

## 📊 Knowledge Graph Schema

### Nodes Table
| Field | Type | Description |
|-------|------|-------------|
| signature_id | TEXT (PK) | Unique chunk identifier |
| node_type | TEXT | 'audio_chunk' or 'aggregation' |
| level | INTEGER | Hierarchy level (0 = base) |
| description | TEXT | Generated description |
| parent_id | TEXT (FK) | Parent aggregation node |
| source_file | TEXT | Original audio file |
| metadata | JSON | Additional info |

### Edges Table
| Field | Type | Description |
|-------|------|-------------|
| source_id | TEXT (FK) | Source node |
| target_id | TEXT (FK) | Target node |
| relation_type | TEXT | Edge type |
| weight | REAL | Relationship strength |
| description | TEXT | Relation description |

### Relation Types
- `sequential` - Temporally adjacent chunks
- `semantic_similar` - Similar audio content
- `same_speaker` - Same speaker detected
- `same_topic` - Same topic/theme
- `aggregation` - Chunk to aggregation node
- `hierarchy` - Parent-child relationship

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
# Audio processing
audio:
  chunk_duration_sec: 10
  overlap_sec: 2
  sample_rate: 16000

# Embeddings
embeddings:
  audio:
    model: "laion/clap-htsat-unfused"  # or "simple"
    dimension: 512

# Knowledge Graph
knowledge_graph:
  use_sqlite: true

# Processing
processing:
  similarity_threshold: 0.7
  batch_size: 32
```

## 🔄 Pipeline Flow

```
1. Load audio file(s)
           ↓
2. Chunk into segments with overlap
           ↓
3. Generate signature ID for each chunk (SHA256 hash)
           ↓
4. Generate embedding vector (CLAP or mel-spectrogram)
           ↓
5. Store in Milvus: {signature_id, embedding, metadata}
           ↓
6. Create KG node: {signature_id, type, level, description}
           ↓
7. Create KG edges:
   - Sequential: chunk[i] → chunk[i+1]
   - Semantic: high similarity pairs
           ↓
8. Ready for retrieval!
```

## 🔍 Retrieval Strategy

Implements LeanRAG-style hierarchical retrieval:

1. **Vector Search**: Find top-K similar chunks via Milvus
2. **Graph Expansion**: Traverse KG edges to find related chunks
3. **Bottom-Up**: Walk up hierarchy to aggregation nodes
4. **Reasoning Paths**: Find paths connecting retrieved chunks
5. **Context Building**: Compile multi-level context for LLM

## 📝 Example Output

```json
{
  "signature_id": "audio_a1b2c3d4e5f67890",
  "source_file": "podcast_episode_1.mp3",
  "chunk_index": 5,
  "start_time": 40.0,
  "end_time": 50.0,
  "duration": 10.0,
  "relations": [
    {"type": "sequential", "target": "audio_b2c3d4e5f6789012"},
    {"type": "semantic_similar", "target": "audio_c3d4e5f67890123a", "weight": 0.85}
  ]
}
```

## 🧪 Testing

```bash
# Test audio chunking
python audio_chunking.py --input test.wav --output chunks.json

# Test Milvus store
python milvus_store.py

# Test knowledge graph
python knowledge_graph.py

# Full pipeline test
python pipeline.py --input test_audio/ --output-dir ./test_output
```

## 📚 Related

- [LeanRAG](../README.md) - Main LeanRAG framework
- [CLAP](https://github.com/LAION-AI/CLAP) - Contrastive Language-Audio Pretraining
- [Milvus](https://milvus.io/) - Vector database

## 📄 License

MIT License - See main repository LICENSE file.
