"""High-level API for EliPrint audio fingerprinting."""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Union, Any
from collections import Counter, defaultdict
from pathlib import Path

from .core import Fingerprinter
from .database import Database
from .models import AudioTrack, Fingerprint, MatchResult

class EliPrint:
    """Main API for EliPrint audio fingerprinting operations."""
    
    def __init__(self, db_host="localhost", db_port=3306, db_user="root", 
                 db_password="", db_name="eliprint"):
        """Initialize with database connection parameters."""
        self.fingerprinter = Fingerprinter()
        self.database = Database(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
    
    def add_song(self, audio_file: str, metadata: Optional[Dict[str, Any]] = None) -> AudioTrack:
        """
        Add a song to the database.
        
        Args:
            audio_file: Path to audio file
            metadata: Optional metadata (artist, title, etc.)
            
        Returns:
            AudioTrack that was added
        """
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
        track, fingerprints = self.fingerprinter.fingerprint_file(
            file_path=audio_file,
            track_info=metadata
        )
        
        # Add to database
        self.database.add_track(track)
        self.database.add_fingerprints(fingerprints)
        
        return track
    
    def identify_song(self, audio_file: str) -> Optional[MatchResult]:
        """
        Identify a song from an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            MatchResult if match found, None otherwise
        """
        start_time = time.time()
        
        # Load and process audio
        audio_data, _ = self.fingerprinter.audio_processor.load_audio(audio_file)
        
        # Extract fingerprints from sample
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
        
        # Return None if no good match or too few matches
        min_matches = 5
        if best_match is None or best_count < min_matches:
            return None
        
        # Calculate confidence score
        confidence = best_count / len(fingerprints)
        
        # Skip if confidence is too low
        min_confidence = 0.05
        if confidence < min_confidence:
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
        result = MatchResult(
            track_id=best_match,
            track=track,
            confidence=confidence,
            offset_seconds=offset_seconds,
            match_count=best_count,
            time_taken=time.time() - start_time
        )
        
        return result
    
    def batch_add_songs(self, directory: str) -> List[AudioTrack]:
        """
        Add multiple songs from a directory.
        
        Args:
            directory: Directory containing audio files
            
        Returns:
            List of added AudioTrack objects
        """
        audio_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        tracks = []
        
        # Find all audio files in directory
        for filename in os.listdir(directory):
            if filename.lower().endswith(audio_extensions):
                file_path = os.path.join(directory, filename)
                try:
                    track = self.add_song(file_path)
                    tracks.append(track)
                    print(f"Added: {track.artist} - {track.title}")
                except Exception as e:
                    print(f"Error adding {filename}: {e}")
        
        return tracks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'tracks': self.database.get_track_count(),
            'fingerprints': self.database.get_fingerprint_count(),
            'avg_fingerprints_per_track': self.database.get_fingerprint_count() / max(1, self.database.get_track_count())
        }
    
    def close(self):
        """Close database connection."""
        self.database.close()