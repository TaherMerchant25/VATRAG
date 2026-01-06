"""
Audio Embedding Module for Multimodal LeanRAG
=============================================
Generates embeddings for audio chunks using CLAP (Contrastive Language-Audio Pretraining).
"""

import os
import logging
from typing import List, Dict, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import torch
    import torchaudio
except ImportError:
    raise ImportError("Please install torch and torchaudio: pip install torch torchaudio")

from audio_chunking import AudioChunk

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioEmbedder:
    """
    Audio embedding generator using CLAP model.
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: Optional[str] = None,
        enable_fusion: bool = False
    ):
        """
        Initialize the audio embedder.
        
        Args:
            model_name: Name of the CLAP model to use
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            enable_fusion: Whether to enable fusion for longer audio
        """
        self.model_name = model_name
        self.enable_fusion = enable_fusion
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing AudioEmbedder on {self.device}")
        
        # Load CLAP model
        self._load_model()
    
    def _load_model(self):
        """Load the CLAP model and processor."""
        try:
            from transformers import ClapModel, ClapProcessor
            
            logger.info(f"Loading CLAP model: {self.model_name}")
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model = ClapModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.projection_dim
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.warning(f"Failed to load CLAP model: {e}")
            logger.info("Falling back to laion-clap library")
            self._load_laion_clap()
    
    def _load_laion_clap(self):
        """Fallback: Load using laion-clap library."""
        try:
            import laion_clap
            
            self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion)
            self.model.load_ckpt()  # Load default checkpoint
            self.processor = None
            self.embedding_dim = 512
            self._use_laion_clap = True
            
            logger.info("Loaded laion-clap model successfully")
            
        except ImportError:
            raise ImportError("Please install laion-clap: pip install laion-clap")
    
    @torch.no_grad()
    def embed_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Generate embedding for a single audio segment.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of the audio
        
        Returns:
            NumPy array of shape (embedding_dim,)
        """
        if hasattr(self, '_use_laion_clap') and self._use_laion_clap:
            # Use laion-clap
            import tempfile
            import soundfile as sf
            
            # Save to temp file (laion-clap requires file path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio_data, sample_rate)
                embedding = self.model.get_audio_embedding_from_filelist([f.name])
                os.unlink(f.name)
            
            return embedding[0]
        else:
            # Use transformers CLAP
            # Ensure correct sample rate (CLAP expects 48kHz)
            if sample_rate != 48000:
                import torchaudio.transforms as T
                resampler = T.Resample(sample_rate, 48000)
                audio_tensor = torch.from_numpy(audio_data).float()
                audio_data = resampler(audio_tensor).numpy()
            
            inputs = self.processor(
                audios=audio_data,
                return_tensors="pt",
                sampling_rate=48000
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.get_audio_features(**inputs)
            embedding = outputs.cpu().numpy().flatten()
            
            return embedding
    
    def embed_chunks(
        self,
        chunks: List[AudioChunk],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple audio chunks.
        
        Args:
            chunks: List of AudioChunk objects
            batch_size: Number of chunks to process in parallel
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary mapping signature_id to embedding
        """
        embeddings = {}
        
        iterator = tqdm(chunks, desc="Generating embeddings") if show_progress else chunks
        
        for chunk in iterator:
            if chunk.audio_data is None:
                logger.warning(f"Chunk {chunk.signature_id} has no audio data, skipping")
                continue
            
            try:
                embedding = self.embed_audio(
                    audio_data=chunk.audio_data,
                    sample_rate=chunk.sample_rate
                )
                embeddings[chunk.signature_id] = embedding
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk.signature_id}: {e}")
                continue
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text (for cross-modal queries).
        
        Args:
            text: Text string to embed
        
        Returns:
            NumPy array of shape (embedding_dim,)
        """
        if hasattr(self, '_use_laion_clap') and self._use_laion_clap:
            embedding = self.model.get_text_embedding([text])
            return embedding[0]
        else:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            
            return outputs.cpu().numpy().flatten()


class SimpleAudioEmbedder:
    """
    Simplified audio embedder using mel-spectrogram features.
    Use this as a fallback when CLAP is not available.
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize simple embedder.
        
        Args:
            embedding_dim: Dimension of output embeddings
        """
        self.embedding_dim = embedding_dim
        logger.info(f"Using SimpleAudioEmbedder with dim={embedding_dim}")
    
    def embed_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Generate embedding using mel-spectrogram statistics.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate
        
        Returns:
            NumPy array of shape (embedding_dim,)
        """
        import librosa
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract various statistics
        features = []
        
        # Mean and std across time for each mel band
        features.append(np.mean(mel_db, axis=1))  # 128
        features.append(np.std(mel_db, axis=1))   # 128
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        features.append(np.mean(mfccs, axis=1))   # 40
        features.append(np.std(mfccs, axis=1))    # 40
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features.append(np.mean(chroma, axis=1))  # 12
        features.append(np.std(chroma, axis=1))   # 12
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)
        
        features.append([np.mean(spectral_centroid), np.std(spectral_centroid)])
        features.append([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        features.append([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.append([np.mean(zero_crossing), np.std(zero_crossing)])
        
        # Concatenate all features
        embedding = np.concatenate([np.atleast_1d(f).flatten() for f in features])
        
        # Pad or truncate to target dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def embed_chunks(
        self,
        chunks: List[AudioChunk],
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for multiple chunks."""
        embeddings = {}
        
        iterator = tqdm(chunks, desc="Generating embeddings") if show_progress else chunks
        
        for chunk in iterator:
            if chunk.audio_data is None:
                continue
            
            try:
                embedding = self.embed_audio(chunk.audio_data, chunk.sample_rate)
                embeddings[chunk.signature_id] = embedding
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk.signature_id}: {e}")
        
        return embeddings


def get_embedder(
    model_type: str = "clap",
    **kwargs
) -> Union[AudioEmbedder, SimpleAudioEmbedder]:
    """
    Factory function to get appropriate embedder.
    
    Args:
        model_type: 'clap' for CLAP model, 'simple' for mel-spectrogram features
        **kwargs: Additional arguments for embedder
    
    Returns:
        Embedder instance
    """
    if model_type == "clap":
        try:
            return AudioEmbedder(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to load CLAP model: {e}")
            logger.info("Falling back to simple embedder")
            return SimpleAudioEmbedder(**kwargs)
    else:
        return SimpleAudioEmbedder(**kwargs)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from audio_chunking import chunk_audio_file
    
    # Example: Embed chunks from an audio file
    test_audio = "test_audio.wav"  # Replace with actual file
    
    if os.path.exists(test_audio):
        # Chunk the audio
        chunks = chunk_audio_file(test_audio)
        
        # Get embedder
        embedder = get_embedder(model_type="simple")  # Use 'clap' for better quality
        
        # Generate embeddings
        embeddings = embedder.embed_chunks(chunks)
        
        print(f"\n✅ Generated {len(embeddings)} embeddings")
        print(f"📐 Embedding dimension: {embedder.embedding_dim}")
        
        # Show sample
        for sig_id, emb in list(embeddings.items())[:3]:
            print(f"  {sig_id}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    else:
        print("Please provide a test audio file")
