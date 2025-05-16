"""Audio processing utilities."""

import numpy as np
from typing import Tuple
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing for fingerprinting."""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize with target sample rate."""
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load and preprocess audio from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (preprocessed_audio, duration_in_seconds)
        """
        # Read audio file
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
        
        return audio_data, duration
    
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