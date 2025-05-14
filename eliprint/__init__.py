# eliprint/__init__.py
"""
EliPrint - Audio Fingerprinting Library
======================================

A powerful and efficient audio fingerprinting library named after Ethiopian music composer Elias Melka.
Similar to Shazam, EliPrint can identify audio tracks based on their acoustic fingerprints.

Main components:
- Audio processing and analysis
- Fingerprint extraction and matching
- Scalable database management
- Batch processing and training
- Plugin system for custom algorithms

Built with performance, flexibility, and ease of use in mind.
"""

from .core import Fingerprinter
from .database import Database
from .models import AudioTrack, MatchResult
from .recognizer import Recognizer
from .batch import BatchProcessor
from .config import Config

# High-level API for simplicity
from .api import (identify_song, add_song, batch_add_songs, 
                 remove_song, get_stats, setup_database)

__version__ = '1.0.0'


# eliprint/api.py
"""
High-level API for EliPrint.
Provides simplified functions for common operations.
"""

import os
from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
from pathlib import Path

from .core import Fingerprinter
from .database import Database
from .recognizer import Recognizer
from .models import AudioTrack, MatchResult
from .batch import BatchProcessor
from .config import Config

# Global instances
_config = Config()
_fingerprinter = None
_database = None
_recognizer = None
_batch_processor = None

def _ensure_initialized():
    """Ensure that the global instances are initialized."""
    global _fingerprinter, _database, _recognizer, _batch_processor
    
    if _fingerprinter is None:
        _fingerprinter = Fingerprinter(**_config.fingerprinter_params)
    
    if _database is None:
        _database = Database(**_config.database_params)
    
    if _recognizer is None:
        _recognizer = Recognizer(_fingerprinter, _database)
    
    if _batch_processor is None:
        _batch_processor = BatchProcessor(_fingerprinter, _database)

def setup_database(db_path: str = None, in_memory: bool = False, **kwargs):
    """
    Set up the database with custom parameters.
    
    Args:
        db_path: Path to the database file
        in_memory: Whether to use in-memory database
        **kwargs: Additional configuration parameters
    """
    global _config, _database
    
    # Update configuration
    _config.update(database_params={
        'db_path': db_path,
        'in_memory': in_memory
    })
    
    # Update other config parameters
    _config.update(**kwargs)
    
    # Reset database
    if _database is not None:
        _database.close()
        _database = None
    
    # Initialize components
    _ensure_initialized()
    
    return _database

def identify_song(audio_file_or_data: Union[str, np.ndarray]) -> Optional[MatchResult]:
    """
    Identify a song from an audio file or audio data.
    
    Args:
        audio_file_or_data: Path to audio file or audio data as NumPy array
        
    Returns:
        MatchResult if match found, None otherwise
    """
    _ensure_initialized()
    
    if isinstance(audio_file_or_data, str):
        return _recognizer.recognize_file(audio_file_or_data)
    else:
        return _recognizer.recognize_audio(audio_file_or_data)

def add_song(audio_file: str, metadata: Optional[Dict[str, Any]] = None) -> AudioTrack:
    """
    Add a song to the database.
    
    Args:
        audio_file: Path to audio file
        metadata: Optional metadata for the track
        
    Returns:
        AudioTrack that was added
    """
    _ensure_initialized()
    
    # Extract track info from filename if not provided
    if metadata is None:
        metadata = {}
        
    # Try to extract artist and title from filename
    if 'title' not in metadata:
        filename = Path(audio_file).stem
        parts = filename.split(' - ', 1)
        
        if len(parts) > 1:
            metadata['artist'] = parts[0]
            metadata['title'] = parts[1]
        else:
            metadata['title'] = filename
    
    # Add path to metadata
    metadata['path'] = audio_file
    
    # Fingerprint file
    track, fingerprints = _fingerprinter.fingerprint_file(
        file_path=audio_file,
        track_info=metadata
    )
    
    # Add to database
    _database.add_track(track)
    _database.add_fingerprints(fingerprints)
    
    return track

def batch_add_songs(directory_or_files: Union[str, List[str]], 
                   metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
                   max_workers: int = None, 
                   progress_callback=None) -> List[AudioTrack]:
    """
    Add multiple songs to the database.
    
    Args:
        directory_or_files: Directory path or list of file paths
        metadata_map: Optional mapping of file paths to metadata
        max_workers: Maximum number of worker processes
        progress_callback: Optional callback function for progress reporting
        
    Returns:
        List of AudioTrack objects that were added
    """
    _ensure_initialized()
    
    # Process directory or file list
    if isinstance(directory_or_files, str) and os.path.isdir(directory_or_files):
        files = [os.path.join(directory_or_files, f) for f in os.listdir(directory_or_files)
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a'))]
    else:
        files = directory_or_files if isinstance(directory_or_files, list) else [directory_or_files]
    
    # Create metadata map if not provided
    if metadata_map is None:
        metadata_map = {}
    
    # Process files in batch
    return _batch_processor.process_files(
        files=files,
        metadata_map=metadata_map,
        max_workers=max_workers,
        progress_callback=progress_callback
    )

def remove_song(track_id: str) -> bool:
    """
    Remove a song from the database.
    
    Args:
        track_id: Track ID
        
    Returns:
        True if removed, False if not found
    """
    _ensure_initialized()
    return _database.delete_track(track_id)

def get_stats() -> Dict[str, Any]:
    """
    Get database statistics.
    
    Returns:
        Dictionary of statistics
    """
    _ensure_initialized()
    return {
        'tracks': _database.get_track_count(),
        'fingerprints': _database.get_fingerprint_count(),
        'fingerprints_per_track': _database.get_fingerprint_count() / max(1, _database.get_track_count()),
        'database_type': 'in-memory' if _database.in_memory else 'sqlite',
        'database_path': _database.db_path if not _database.in_memory else None