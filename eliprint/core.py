"""Core fingerprinting algorithm."""

import numpy as np
import hashlib
from typing import List, Tuple, Dict, Optional
import logging

from .audio import AudioProcessor
from .models import AudioTrack, Fingerprint

logger = logging.getLogger(__name__)

class Fingerprinter:
    """
    Core fingerprinting engine similar to Shazam's algorithm.
    Extracts constellation map fingerprints from audio.
    """
    
    def __init__(self, sample_rate: int = 44100, window_size: int = 4096, 
                hop_size: int = 2048, peak_threshold: float = 0.3):
        """
        Initialize fingerprinter with configurable parameters.
        
        Args:
            sample_rate: Sample rate for audio processing
            window_size: Size of FFT window
            hop_size: Number of samples between consecutive FFT windows
            peak_threshold: Threshold for peak detection
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.peak_threshold = peak_threshold
        
        # Advanced parameters
        self.target_zone_size = 5         # Points in target zone
        self.target_zone_distance = 10    # Time frames to look ahead
        self.min_freq = 100               # Min frequency (Hz)
        self.max_freq = 8000              # Max frequency (Hz)
        self.peak_neighborhood_size = 20  # Size for peak finding
        self.max_hash_time_delta = 200    # Max frames between paired peaks
        
        # Create audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
    
    def fingerprint_file(self, file_path: str, track_info: Optional[Dict] = None) -> Tuple[AudioTrack, List[Fingerprint]]:
        """
        Extract fingerprints from an audio file.
        
        Args:
            file_path: Path to the audio file
            track_info: Optional metadata for the track
            
        Returns:
            Tuple of (AudioTrack, fingerprints)
        """
        # Generate track ID from file path
        track_id = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()
        
        # Create track info if not provided
        if track_info is None:
            track_info = {'title': file_path.split('/')[-1].split('.')[0]}
        
        # Process audio
        audio_data, duration = self.audio_processor.load_audio(file_path)
        track_info['duration'] = duration
        
        # Extract fingerprints
        fingerprints = self._extract_fingerprints(audio_data, track_id)
        
        # Create track object
        track = AudioTrack(id=track_id, **track_info)
        
        return track, fingerprints
    
    def _extract_fingerprints(self, audio_data: np.ndarray, track_id: str) -> List[Fingerprint]:
        """Extract fingerprints from preprocessed audio data."""
        # Compute spectrogram
        spectrogram = self._compute_spectrogram(audio_data)
        
        # Find peaks in spectrogram
        peaks = self._find_peaks(spectrogram)
        
        # Generate fingerprint hashes from peaks
        fingerprints = self._generate_hashes(peaks, track_id)
        
        return fingerprints
    
    def _compute_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """Compute spectrogram from audio data using STFT."""
        # Calculate number of frames
        num_frames = 1 + int((len(audio_data) - self.window_size) / self.hop_size)
        
        # Prepare window function
        window = np.hanning(self.window_size)
        
        # Allocate spectrogram array (frequency bins x time frames)
        spectrogram = np.zeros((int(self.window_size / 2) + 1, num_frames), dtype=np.complex128)
        
        # Compute STFT
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.window_size
            
            if end <= len(audio_data):
                # Apply window function
                frame = audio_data[start:end] * window
                
                # Compute FFT
                spectrum = np.fft.rfft(frame)
                
                # Store in spectrogram
                spectrogram[:, i] = spectrum
        
        # Convert to magnitude spectrogram
        spectrogram = np.abs(spectrogram)
        
        # Filter by frequency range
        freq_bins = np.fft.rfftfreq(self.window_size, 1.0/self.sample_rate)
        valid_bins = np.where((freq_bins >= self.min_freq) & (freq_bins <= self.max_freq))[0]
        
        return spectrogram[valid_bins, :]
    
    def _find_peaks(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """Find peaks in spectrogram (local maxima)."""
        # Apply logarithmic scaling
        log_spectrogram = np.log(np.maximum(spectrogram, 1e-10))
        
        # Normalize
        log_spectrogram = (log_spectrogram - np.mean(log_spectrogram)) / np.std(log_spectrogram)
        
        # Find local maxima
        peaks = []
        height, width = log_spectrogram.shape
        
        for i in range(self.peak_neighborhood_size, height - self.peak_neighborhood_size):
            for j in range(self.peak_neighborhood_size, width - self.peak_neighborhood_size):
                # Extract neighborhood
                neighborhood = log_spectrogram[
                    i - self.peak_neighborhood_size:i + self.peak_neighborhood_size + 1,
                    j - self.peak_neighborhood_size:j + self.peak_neighborhood_size + 1
                ]
                
                # Check if center point is maximum and above threshold
                if (log_spectrogram[i, j] == np.max(neighborhood) and 
                    log_spectrogram[i, j] > self.peak_threshold):
                    peaks.append((i, j))
        
        return peaks
    
    def _generate_hashes(self, peaks: List[Tuple[int, int]], track_id: str) -> List[Fingerprint]:
        """Generate fingerprint hashes using constellation map approach."""
        # Sort peaks by time
        peaks.sort(key=lambda p: p[1])
        
        # Store fingerprints
        fingerprints = []
        
        # Process each peak as an anchor point
        for i, anchor in enumerate(peaks):
            # Get anchor coordinates
            anchor_freq, anchor_time = anchor
            
            # Define target zone - next points within a specific time range
            target_points = []
            
            # Look ahead to find points in the target zone
            j = i + 1
            while j < len(peaks) and peaks[j][1] - anchor_time <= self.target_zone_distance:
                target_points.append(peaks[j])
                j += 1
                
                # Limit target zone size
                if len(target_points) >= self.target_zone_size:
                    break
            
            # Create hash for each point in the target zone
            for target in target_points:
                target_freq, target_time = target
                
                # Calculate time delta
                time_delta = target_time - anchor_time
                
                # Skip if time delta is too large
                if time_delta > self.max_hash_time_delta:
                    continue
                
                # Create a hash string
                hash_input = f"{anchor_freq}|{target_freq}|{time_delta}"
                hash_code = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                
                # Create fingerprint
                fingerprint = Fingerprint(
                    hash=hash_code,
                    track_id=track_id,
                    offset=anchor_time,
                    anchor_freq=anchor_freq,
                    target_freq=target_freq,
                    delta=time_delta
                )
                
                fingerprints.append(fingerprint)
        
        return fingerprints