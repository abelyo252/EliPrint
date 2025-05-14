
# eliprint/recognizer.py
"""
Audio recognition engine.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any
import time
import logging

from .core import Fingerprinter
from .database import Database
from .models import AudioTrack, Fingerprint, MatchResult
from .utils import Timer

logger = logging.getLogger(__name__)

class Recognizer:
    """
    Audio recognition engine that matches audio fingerprints against a database.
    """
    
    def __init__(self, fingerprinter: Fingerprinter, database: Database):
        """
        Initialize the recognizer.
        
        Args:
            fingerprinter: Fingerprinter instance
            database: Database instance
        """
        self.fingerprinter = fingerprinter
        self.database = database
        
        # Recognition parameters
        self.min_matches = 5
        self.min_confidence = 0.05
        self.time_tolerance = 0.5  # seconds
        
        # Performance metrics
        self.last_recognition_time = 0
    
    def recognize_file(self, file_path: str) -> Optional[MatchResult]:
        """
        Recognize an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            MatchResult if match found, None otherwise
        """
        try:
            with Timer() as timer:
                # Load and process audio
                audio_data, _ = self.fingerprinter.audio_processor.load_audio(file_path)
                
                # Recognize audio
                result = self._recognize_audio(audio_data)
                
                # Set time taken
                if result:
                    result.time_taken = timer.elapsed
                    
                # Save performance metrics
                self.last_recognition_time = timer.elapsed
                
                # Log result
                if result:
                    logger.info(f"Recognized '{file_path}' as '{result.title}' by '{result.artist}' with {result.confidence:.2%} confidence in {timer.elapsed:.2f}s")
                else:
                    logger.info(f"No match found for '{file_path}' in {timer.elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error recognizing file '{file_path}': {str(e)}")
            raise
    
    def recognize_audio(self, audio_data: np.ndarray) -> Optional[MatchResult]:
        """
        Recognize audio data.
        
        Args:
            audio_data: Audio data as NumPy array
            
        Returns:
            MatchResult if match found, None otherwise
        """
        with Timer() as timer:
            # Preprocess audio
            audio_data = self.fingerprinter.audio_processor.preprocess_audio(audio_data)
            
            # Recognize audio
            result = self._recognize_audio(audio_data)
            
            # Set time taken
            if result:
                result.time_taken = timer.elapsed
                
            # Save performance metrics
            self.last_recognition_time = timer.elapsed
            
            # Log result
            if result:
                logger.info(f"Recognized audio as '{result.title}' by '{result.artist}' with {result.confidence:.2%} confidence in {timer.elapsed:.2f}s")
            else:
                logger.info(f"No match found for audio in {timer.elapsed:.2f}s")
        
        return result
    
    def recognize_batch(self, audio_batch: List[np.ndarray]) -> List[Optional[MatchResult]]:
        """
        Recognize a batch of audio data.
        
        Args:
            audio_batch: List of audio data arrays
            
        Returns:
            List of MatchResult objects (None for no match)
        """
        results = []
        
        # Process each audio sample
        for i, audio_data in enumerate(audio_batch):
            try:
                result = self.recognize_audio(audio_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error recognizing batch item {i}: {str(e)}")
                results.append(None)
        
        return results
    
    def _recognize_audio(self, audio_data: np.ndarray) -> Optional[MatchResult]:
        """
        Core recognition function.
        
        Args:
            audio_data: Preprocessed audio data
            
        Returns:
            MatchResult if match found, None otherwise
        """
        # Extract fingerprints
        temp_track_id = "query"
        fingerprints = self.fingerprinter._extract_fingerprints(audio_data, temp_track_id)
        
        # Get hashes to look up
        hashes = [fp.hash for fp in fingerprints]
        
        # Look up matching fingerprints
        matches = self.database.get_matching_fingerprints(hashes)
        
        # Create a mapping of fingerprint hash to query offset
        query_offsets = {fp.hash: fp.offset for fp in fingerprints}
        
        # Group matches by track
        track_matches = defaultdict(list)
        
        # For each matching hash
        for hash_str, hash_matches in matches.items():
            # Get query offset for this hash
            query_offset = query_offsets[hash_str]
            
            # For each matching track
            for track_id, db_offset in hash_matches:
                # Calculate time offset between query and database track
                # This is key to the Shazam-like algorithm
                offset_diff = db_offset - query_offset
                
                # Store the match with its offset difference
                track_matches[track_id].append(offset_diff)
        
        # Find best matching track
        best_match = None
        best_count = 0
        best_offset = 0
        
        for track_id, offsets in track_matches.items():
            # Count the most common offset difference
            offset_counts = Counter(offsets)
            
            # Get the most common offset and its count
            (offset, count), = offset_counts.most_common(1) or [(0, 0)]
            
            # Update best match if this is better
            if count > best_count:
                best_count = count
                best_match = track_id
                best_offset = offset
        
        # Return None if no good match
        if best_match is None or best_count < self.min_matches:
            return None
        
        # Calculate confidence score
        confidence = best_count / len(fingerprints)
        
        # Skip if confidence is too low
        if confidence < self.min_confidence:
            return None
        
        # Get matching track
        track = self.database.get_track(best_match)
        
        # Skip if track not found (shouldn't happen)
        if track is None:
            return None
        
        # Convert offset from frames to seconds
        hop_size = self.fingerprinter.hop_size
        sample_rate = self.fingerprinter.sample_rate
        offset_seconds = best_offset * hop_size / sample_rate
        
        # Return match result
        return MatchResult(
            track_id=best_match,
            track=track,
            confidence=confidence,
            offset_seconds=offset_seconds,
            match_count=best_count,
            query_fingerprint_count=len(fingerprints),
            total_fingerprints=self.database.get_fingerprint_count()
        )

