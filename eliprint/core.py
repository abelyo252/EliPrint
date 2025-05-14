# eliprint/core.py
"""
Core fingerprinting functionality.
"""

import numpy as np
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Union, Any
import time
from dataclasses import dataclass
import logging

from .audio import AudioProcessor
from .models import AudioTrack, Fingerprint
from .utils import Timer

logger = logging.getLogger(__name__)

class Fingerprinter:
    """
    Core fingerprinting engine that extracts robust fingerprints from audio data.
    Uses a constellation map approach similar to Shazam's algorithm.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 window_size: int = 4096, 
                 hop_size: int = 2048,
                 target_zone_size: int = 5,
                 target_zone_distance: int = 10,
                 min_freq: int = 100,
                 max_freq: int = 8000,
                 peak_neighborhood_size: int = 20,
                 peak_threshold: float = 0.3,
                 min_hash_time_delta: int = 0,
                 max_hash_time_delta: int = 200,
                 fingerprint_reduction: bool = True):
        """
        Initialize the fingerprinter with configurable parameters.
        
        Args:
            sample_rate: Sample rate for audio processing
            window_size: Size of FFT window
            hop_size: Number of samples between consecutive FFT windows
            target_zone_size: Number of points to consider in each target zone
            target_zone_distance: Time distance between anchor and target zone
            min_freq: Minimum frequency to consider
            max_freq: Maximum frequency to consider
            peak_neighborhood_size: Size of neighborhood for peak finding
            peak_threshold: Threshold for peak detection
            min_hash_time_delta: Minimum time delta for hash pairs
            max_hash_time_delta: Maximum time delta for hash pairs
            fingerprint_reduction: Whether to apply fingerprint reduction
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.target_zone_size = target_zone_size
        self.target_zone_distance = target_zone_distance
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.peak_neighborhood_size = peak_neighborhood_size
        self.peak_threshold = peak_threshold
        self.min_hash_time_delta = min_hash_time_delta
        self.max_hash_time_delta = max_hash_time_delta
        self.fingerprint_reduction = fingerprint_reduction
        
        # Create audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Performance metrics
        self.last_fingerprint_time = 0
        self.last_peak_count = 0
        self.last_fingerprint_count = 0
    
    def fingerprint_file(self, file_path: str, 
                         track_id: Optional[str] = None,
                         track_info: Optional[Dict[str, Any]] = None) -> Tuple[AudioTrack, List[Fingerprint]]:
        """
        Extract fingerprints from an audio file.
        
        Args:
            file_path: Path to the audio file
            track_id: Optional track ID (will be generated if None)
            track_info: Optional metadata for the track
            
        Returns:
            Tuple of (AudioTrack, fingerprints)
        """
        try:
            with Timer() as timer:
                # Generate track ID if not provided
                if track_id is None:
                    track_id = self._generate_track_id(file_path)
                
                # Create track info if not provided
                if track_info is None:
                    track_info = {
                        'title': Path(file_path).stem,
                        'path': file_path,
                        'duration': 0  # Will be updated after processing
                    }
                
                # Process audio
                audio_data, duration = self.audio_processor.load_audio(file_path)
                track_info['duration'] = duration
                
                # Extract fingerprints
                fingerprints = self._extract_fingerprints(audio_data, track_id)
                
                # Save performance metrics
                self.last_fingerprint_time = timer.elapsed
                
                logger.info(f"Fingerprinted '{file_path}': {len(fingerprints)} fingerprints in {timer.elapsed:.2f}s")
            
            # Create track object
            track = AudioTrack(id=track_id, **track_info)
            
            return track, fingerprints
            
        except Exception as e:
            logger.error(f"Error fingerprinting file '{file_path}': {str(e)}")
            raise
    
    def fingerprint_audio(self, audio_data: np.ndarray, 
                          track_id: Optional[str] = None,
                          track_info: Optional[Dict[str, Any]] = None) -> Tuple[AudioTrack, List[Fingerprint]]:
        """
        Extract fingerprints from audio data.
        
        Args:
            audio_data: Audio data as NumPy array
            track_id: Optional track ID (will be generated if None)
            track_info: Optional metadata for the track
            
        Returns:
            Tuple of (AudioTrack, fingerprints)
        """
        with Timer() as timer:
            # Generate track ID if not provided
            if track_id is None:
                # Generate a unique ID based on a hash of the first few samples
                hash_input = str(audio_data[:min(1000, len(audio_data))].tobytes())
                track_id = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            
            # Create track info if not provided
            if track_info is None:
                track_info = {
                    'title': f"Track_{track_id[:8]}",
                    'duration': len(audio_data) / self.sample_rate
                }
            
            # Preprocess audio
            audio_data = self.audio_processor.preprocess_audio(audio_data)
            
            # Extract fingerprints
            fingerprints = self._extract_fingerprints(audio_data, track_id)
            
            # Save performance metrics
            self.last_fingerprint_time = timer.elapsed
            
            logger.info(f"Fingerprinted audio data: {len(fingerprints)} fingerprints in {timer.elapsed:.2f}s")
        
        # Create track object
        track = AudioTrack(id=track_id, **track_info)
        
        return track, fingerprints
    
    def fingerprint_batch(self, audio_batch: List[np.ndarray], 
                         track_ids: Optional[List[str]] = None,
                         track_infos: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[AudioTrack, List[Fingerprint]]]:
        """
        Extract fingerprints from a batch of audio data.
        
        Args:
            audio_batch: List of audio data arrays
            track_ids: Optional list of track IDs (will be generated if None)
            track_infos: Optional list of track metadata
            
        Returns:
            List of (AudioTrack, fingerprints) tuples
        """
        results = []
        
        # Generate track IDs if not provided
        if track_ids is None:
            track_ids = [None] * len(audio_batch)
        
        # Generate track infos if not provided
        if track_infos is None:
            track_infos = [None] * len(audio_batch)
        
        # Process each audio sample
        for i, (audio_data, track_id, track_info) in enumerate(zip(audio_batch, track_ids, track_infos)):
            try:
                result = self.fingerprint_audio(audio_data, track_id, track_info)
                results.append(result)
            except Exception as e:
                logger.error(f"Error fingerprinting batch item {i}: {str(e)}")
                # Continue with next item
        
        return results
    
    def _extract_fingerprints(self, audio_data: np.ndarray, track_id: str) -> List[Fingerprint]:
        """
        Extract fingerprints from preprocessed audio data.
        
        Args:
            audio_data: Preprocessed audio data
            track_id: Track ID to associate with fingerprints
            
        Returns:
            List of Fingerprint objects
        """
        # Compute spectrogram
        spectrogram = self._compute_spectrogram(audio_data)
        
        # Find peaks in spectrogram
        peaks = self._find_peaks(spectrogram)
        self.last_peak_count = len(peaks)
        
        # Generate fingerprint hashes from peaks
        fingerprints = self._generate_hashes(peaks, track_id)
        self.last_fingerprint_count = len(fingerprints)
        
        return fingerprints
    
    def _compute_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute spectrogram from audio data using Short-Time Fourier Transform.
        
        Args:
            audio_data: Preprocessed audio data
            
        Returns:
            Spectrogram as 2D NumPy array (frequency bins x time frames)
        """
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
        """
        Find peaks in spectrogram using a local maximum algorithm.
        
        Args:
            spectrogram: Magnitude spectrogram
            
        Returns:
            List of peak coordinates as (frequency_bin, time_frame) tuples
        """
        # Apply logarithmic scaling
        log_spectrogram = np.log(np.maximum(spectrogram, 1e-10))
        
        # Normalize
        log_spectrogram = (log_spectrogram - np.mean(log_spectrogram)) / np.std(log_spectrogram)
        
        # Find local maxima
        peaks = []
        height, width = log_spectrogram.shape
        
        # Vectorized peak finding (faster than nested loops)
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
        """
        Generate fingerprint hashes from peaks using a constellation map approach.
        Similar to Shazam's algorithm, but built from scratch.
        
        Args:
            peaks: List of peak coordinates as (frequency_bin, time_frame) tuples
            track_id: Track ID to associate with fingerprints
            
        Returns:
            List of Fingerprint objects
        """
        # Sort peaks by time
        peaks.sort(key=lambda p: p[1])
        
        # Store fingerprints
        fingerprints = []
        
        # Use a set to track unique hashes if fingerprint reduction is enabled
        hash_set = set() if self.fingerprint_reduction else None
        
        # Process each peak as an anchor point
        for i, anchor in enumerate(peaks):
            # Get anchor coordinates
            anchor_freq, anchor_time = anchor
            
            # Define target zone - next points within a specific time range
            # This is key to the Shazam-like algorithm
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
                
                # Skip if time delta is outside allowed range
                if not (self.min_hash_time_delta <= time_delta <= self.max_hash_time_delta):
                    continue
                
                # Create a hash string using anchor and target frequencies and time delta
                # This is the core of the fingerprinting algorithm
                hash_input = f"{anchor_freq}|{target_freq}|{time_delta}"
                hash_code = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                
                # Skip if we've already seen this hash and reduction is enabled
                if hash_set is not None and hash_code in hash_set:
                    continue
                
                # Create fingerprint
                fingerprint = Fingerprint(
                    hash=hash_code,
                    track_id=track_id,
                    offset=anchor_time,
                    anchor_freq=anchor_freq,
                    target_freq=target_freq,
                    delta=time_delta
                )
                
                # Add to results
                fingerprints.append(fingerprint)
                
                # Track unique hashes if reduction is enabled
                if hash_set is not None:
                    hash_set.add(hash_code)
        
        return fingerprints
    
    def _generate_track_id(self, file_path: str) -> str:
        """Generate a unique track ID from file path."""
        return hashlib.md5(str(file_path).encode('utf-8')).hexdigest()