"""
Audio Chunking Module for Multimodal LeanRAG
============================================
Handles audio file loading, chunking with overlap, and signature ID generation.
"""

import os
import json
import logging
from hashlib import md5, sha256
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Generator
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    raise ImportError("Please install librosa and soundfile: pip install librosa soundfile")

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a single audio chunk with metadata."""
    signature_id: str          # Unique identifier for the chunk
    source_file: str           # Original audio file path
    chunk_index: int           # Position of chunk in sequence
    start_time: float          # Start time in seconds
    end_time: float            # End time in seconds
    duration: float            # Chunk duration in seconds
    sample_rate: int           # Audio sample rate
    audio_data: Optional[np.ndarray] = None  # Raw audio samples (optional, for processing)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding audio_data for serialization."""
        d = asdict(self)
        d.pop('audio_data', None)
        return d


def generate_signature_id(
    audio_data: np.ndarray, 
    source_file: str, 
    chunk_index: int,
    start_time: float
) -> str:
    """
    Generate a unique signature ID for an audio chunk.
    
    The signature is based on:
    - Hash of audio content (first 1000 samples + last 1000 samples)
    - Source file name
    - Chunk index and timing
    
    Returns:
        str: A unique signature ID in format 'audio_{hash}'
    """
    # Create content hash from audio samples
    content_bytes = audio_data[:1000].tobytes() + audio_data[-1000:].tobytes() if len(audio_data) > 2000 else audio_data.tobytes()
    
    # Combine with metadata for uniqueness
    metadata_str = f"{os.path.basename(source_file)}_{chunk_index}_{start_time:.3f}"
    combined = content_bytes + metadata_str.encode('utf-8')
    
    # Generate hash
    hash_value = sha256(combined).hexdigest()[:16]
    
    return f"audio_{hash_value}"


def load_audio(
    file_path: str, 
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 Hz)
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    logger.info(f"Loading audio file: {file_path}")
    
    # Load audio with librosa (automatically handles various formats)
    audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    logger.info(f"Loaded audio: {len(audio_data)/sr:.2f}s duration, {sr}Hz sample rate")
    
    return audio_data, sr


def chunk_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    source_file: str,
    chunk_duration_sec: float = 10.0,
    overlap_sec: float = 2.0,
) -> List[AudioChunk]:
    """
    Split audio data into overlapping chunks.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate of the audio
        source_file: Path to the original audio file
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between consecutive chunks in seconds
    
    Returns:
        List of AudioChunk objects
    """
    chunk_samples = int(chunk_duration_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    total_samples = len(audio_data)
    total_duration = total_samples / sample_rate
    
    logger.info(f"Chunking audio: {total_duration:.2f}s total, "
                f"{chunk_duration_sec}s chunks, {overlap_sec}s overlap")
    
    chunks = []
    chunk_index = 0
    start_sample = 0
    
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_data = audio_data[start_sample:end_sample]
        
        # Skip very short chunks (less than 1 second)
        if len(chunk_data) < sample_rate:
            break
        
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        duration = end_time - start_time
        
        # Generate unique signature ID
        signature_id = generate_signature_id(
            chunk_data, source_file, chunk_index, start_time
        )
        
        chunk = AudioChunk(
            signature_id=signature_id,
            source_file=source_file,
            chunk_index=chunk_index,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            sample_rate=sample_rate,
            audio_data=chunk_data
        )
        
        chunks.append(chunk)
        chunk_index += 1
        start_sample += step_samples
    
    logger.info(f"Created {len(chunks)} audio chunks")
    
    return chunks


def chunk_audio_file(
    file_path: str,
    chunk_duration_sec: float = 10.0,
    overlap_sec: float = 2.0,
    target_sr: int = 16000
) -> List[AudioChunk]:
    """
    Convenience function to load and chunk an audio file in one step.
    
    Args:
        file_path: Path to the audio file
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between consecutive chunks
        target_sr: Target sample rate
    
    Returns:
        List of AudioChunk objects
    """
    audio_data, sr = load_audio(file_path, target_sr)
    return chunk_audio(
        audio_data=audio_data,
        sample_rate=sr,
        source_file=file_path,
        chunk_duration_sec=chunk_duration_sec,
        overlap_sec=overlap_sec
    )


def chunk_audio_directory(
    directory_path: str,
    output_json_path: Optional[str] = None,
    chunk_duration_sec: float = 10.0,
    overlap_sec: float = 2.0,
    target_sr: int = 16000,
    supported_formats: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
) -> List[AudioChunk]:
    """
    Process all audio files in a directory.
    
    Args:
        directory_path: Path to directory containing audio files
        output_json_path: Optional path to save chunk metadata as JSON
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between consecutive chunks
        target_sr: Target sample rate
        supported_formats: List of supported audio file extensions
    
    Returns:
        List of all AudioChunk objects from all files
    """
    all_chunks = []
    
    # Find all audio files
    audio_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                audio_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(audio_files)} audio files in {directory_path}")
    
    for file_path in audio_files:
        try:
            chunks = chunk_audio_file(
                file_path=file_path,
                chunk_duration_sec=chunk_duration_sec,
                overlap_sec=overlap_sec,
                target_sr=target_sr
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Save metadata to JSON if requested
    if output_json_path:
        chunk_metadata = [chunk.to_dict() for chunk in all_chunks]
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved chunk metadata to {output_json_path}")
    
    return all_chunks


def save_chunks_to_disk(
    chunks: List[AudioChunk],
    output_dir: str,
    format: str = "wav"
) -> List[str]:
    """
    Save audio chunks as individual files.
    
    Args:
        chunks: List of AudioChunk objects
        output_dir: Directory to save chunk files
        format: Output audio format
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for chunk in chunks:
        if chunk.audio_data is None:
            logger.warning(f"Chunk {chunk.signature_id} has no audio data")
            continue
        
        file_name = f"{chunk.signature_id}.{format}"
        file_path = os.path.join(output_dir, file_name)
        
        sf.write(file_path, chunk.audio_data, chunk.sample_rate)
        saved_paths.append(file_path)
    
    logger.info(f"Saved {len(saved_paths)} chunk files to {output_dir}")
    
    return saved_paths


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Chunking for Multimodal LeanRAG")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="audio_chunks.json", help="Output JSON file for metadata")
    parser.add_argument("--chunk-duration", type=float, default=10.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap between chunks in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--save-chunks", action="store_true", help="Save individual chunk files")
    parser.add_argument("--chunks-dir", default="chunks", help="Directory to save chunk files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        chunks = chunk_audio_directory(
            directory_path=args.input,
            output_json_path=args.output,
            chunk_duration_sec=args.chunk_duration,
            overlap_sec=args.overlap,
            target_sr=args.sample_rate
        )
    else:
        chunks = chunk_audio_file(
            file_path=args.input,
            chunk_duration_sec=args.chunk_duration,
            overlap_sec=args.overlap,
            target_sr=args.sample_rate
        )
        # Save metadata
        chunk_metadata = [chunk.to_dict() for chunk in chunks]
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chunk_metadata, f, indent=2)
    
    if args.save_chunks:
        save_chunks_to_disk(chunks, args.chunks_dir)
    
    print(f"\n✅ Created {len(chunks)} chunks")
    print(f"📄 Metadata saved to: {args.output}")
    
    # Print sample signature IDs
    print("\n🔑 Sample Signature IDs:")
    for chunk in chunks[:5]:
        print(f"  - {chunk.signature_id} ({chunk.start_time:.1f}s - {chunk.end_time:.1f}s)")
