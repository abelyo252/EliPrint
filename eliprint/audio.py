"""Audio processing utilities with support for multiple formats."""

import numpy as np
from typing import Tuple
import logging
import os

# Try to import librosa for multi-format support
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Fallback to scipy for wav only if librosa is unavailable
import scipy.io.wavfile as wavfile
import scipy.signal as signal

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing for fingerprinting with multi-format support."""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize with target sample rate."""
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load and preprocess audio from various file formats (MP3, WAV, FLAC, etc.).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (preprocessed_audio, duration_in_seconds)
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Use librosa for multi-format support if available
        if LIBROSA_AVAILABLE:
            try:
                # Load audio with librosa (supports MP3, WAV, FLAC, etc.)
                audio_data, file_sample_rate = librosa.load(
                    file_path, 
                    sr=self.sample_rate,  # Resample during loading
                    mono=True  # Convert to mono
                )
                
                # Calculate duration
                duration = len(audio_data) / self.sample_rate
                
                # Preprocess audio
                audio_data = self.preprocess_audio(audio_data)
                
                logger.info(f"Loaded audio file {file_path} with librosa")
                return audio_data, duration
                
            except Exception as e:
                if ext != '.wav':
                    # If not a WAV file and librosa failed, we can't use scipy
                    raise RuntimeError(f"Failed to load audio file {file_path}: {e}")
                
                # For WAV files, fall back to scipy
                logger.warning(f"Librosa failed, falling back to scipy for WAV file: {e}")
        
        # Fallback to scipy for WAV files
        if ext != '.wav':
            raise ValueError(f"File format {ext} not supported without librosa. Please install librosa: pip install librosa")
            
        # Load WAV with scipy
        try:
            file_sample_rate, audio_data = wavfile.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Calculate duration
            duration = len(audio_data) / file_sample_rate
            
            # Resample if needed
            if file_sample_rate != self.sample_rate:
                audio_data = self._resample(audio_data, file_sample_rate)
            
            # Preprocess audio
            audio_data = self.preprocess_audio(audio_data)
            
            logger.info(f"Loaded WAV file {file_path} with scipy")
            return audio_data, duration
            
        except Exception as e:
            raise RuntimeError(f"Failed to load WAV file {file_path}: {e}")
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize and apply pre-emphasis filter."""
        # Convert to float if needed
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1]
        max_value = np.max(np.abs(audio_data))
        if max_value > 0:
            audio_data = audio_data / max_value
        
        # Apply pre-emphasis filter to boost high frequencies
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        return emphasized_audio
    
    def _resample(self, audio_data: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        # Calculate new length
        new_length = int(len(audio_data) * self.sample_rate / original_sample_rate)
        
        # Resample using scipy.signal
        resampled_audio = signal.resample(audio_data, new_length)
        
        return resampled_audio
