# eliprint/utils.py
"""
Utility functions for the fingerprinting system.
"""

import time
from typing import Optional, Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json

# Setup logging
def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional path to log file
    """
    # Create logger
    logger = logging.getLogger('eliprint')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

def visualize_spectrogram(audio_data: np.ndarray, 
                          sample_rate: int, 
                          window_size: int = 4096, 
                          hop_size: int = 2048,
                          peaks: Optional[List[tuple]] = None,
                          title: str = "Audio Spectrogram",
                          save_path: Optional[str] = None) -> None:
    """
    Visualize audio spectrogram and optionally peaks.
    
    Args:
        audio_data: Audio data
        sample_rate: Sample rate
        window_size: Window size for STFT
        hop_size: Hop size for STFT
        peaks: Optional list of peak coordinates
        title: Plot title
        save_path: Optional path to save the visualization
    """
    # Calculate STFT
    window = np.hanning(window_size)
    num_frames = 1 + int((len(audio_data) - window_size) / hop_size)
    
    spectrogram = np.zeros((int(window_size / 2) + 1, num_frames), dtype=np.complex128)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        if end <= len(audio_data):
            frame = audio_data[start:end] * window
            spectrum = np.fft.rfft(frame)
            spectrogram[:, i] = spectrum
    
    # Convert to magnitude spectrogram and apply log scaling
    spectrogram = np.abs(spectrogram)
    log_spectrogram = np.log(np.maximum(spectrogram, 1e-10))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot spectrogram
    plt.imshow(log_spectrogram, aspect='auto', origin='lower', 
               extent=[0, len(audio_data) / sample_rate, 0, sample_rate / 2])
    
    # Add color bar
    plt.colorbar(format='%+2.0f dB')
    
    # Plot peaks if provided
    if peaks:
        # Convert peak coordinates to time and frequency
        peak_times = [p[1] * hop_size / sample_rate for p in peaks]
        freq_bins = np.linspace(0, sample_rate / 2, log_spectrogram.shape[0])
        peak_freqs = [freq_bins[p[0]] for p in peaks]
        
        plt.scatter(peak_times, peak_freqs, color='red', s=5, alpha=0.7)
    
    # Set labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def pretty_print_match(match_result, file_path=None):
    """
    Pretty print a match result.
    
    Args:
        match_result: MatchResult object
        file_path: Optional file path that was matched
    """
    if match_result is None:
        print("No match found" + (f" for '{file_path}'" if file_path else ""))
        return
    
    print("=" * 50)
    print(f"Match found" + (f" for '{file_path}'" if file_path else ""))
    print("-" * 50)
    print(f"Title: {match_result.title}")
    print(f"Artist: {match_result.artist}")
    print(f"Album: {match_result.album}" if match_result.album else "")
    print(f"Confidence: {match_result.confidence:.2%}")
    print(f"Time offset: {match_result.offset_seconds:.2f} seconds")
    print(f"Matched {match_result.match_count} of {match_result.query_fingerprint_count} fingerprints")
    print(f"Recognition took {match_result.time_taken:.3f} seconds")
    print("=" * 50)

def export_results(results, output_file):
    """
    Export match results to a JSON file.
    
    Args:
        results: List of MatchResult objects or a single MatchResult
        output_file: Path to output file
    """
    if not results:
        data = []
    elif not isinstance(results, list):
        # Single result
        data = [results.to_dict()]
    else:
        # List of results
        data = [r.to_dict() if r else None for r in results]
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

