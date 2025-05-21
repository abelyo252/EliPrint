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
                 db_password="", db_name="eliprint", db_type=None, 
                 sqlite_path=None):
        """
        Initialize with database connection parameters.
        
        Args:
            db_host: Database host for MariaDB
            db_port: Database port for MariaDB
            db_user: Database username for MariaDB
            db_password: Database password for MariaDB
            db_name: Database name
            db_type: Database type ('mariadb' or 'sqlite'), auto-detected if None
            sqlite_path: Path to SQLite database file, defaults to [db_name].db in current directory
        """
        self.fingerprinter = Fingerprinter()
        self.database = Database(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            db_type=db_type,
            sqlite_path=sqlite_path
        )
    
    def add_song(self, audio_file: str, metadata: Optional[Dict[str, Any]] = None) -> AudioTrack:
        """
        Add a song to the database.
        
        Args:
            audio_file: Path to audio file
            metadata: Optional metadata including:
                - artist: Artist name
                - title: Song title
                - album: Album name
                - lyrics: Song lyrics
                - history: Song history/background information
                - youtube_url: URL to YouTube video
                - picture_url: URL to album art or artist image
                
        Returns:
            AudioTrack that was added
        """
        # Initialize metadata dict if not provided
        if metadata is None:
            metadata = {}
            
        # Try to extract artist and title from filename if not provided
        if 'title' not in metadata or 'artist' not in metadata:
            filename = Path(audio_file).stem
            parts = filename.split(' - ', 1)
            
            if len(parts) > 1 and 'artist' not in metadata:
                metadata['artist'] = parts[0]
                
            if 'title' not in metadata:
                metadata['title'] = parts[1] if len(parts) > 1 else filename
        
        # Extract standard fields from metadata
        track_info = {
            'title': metadata.get('title', ''),
            'artist': metadata.get('artist', ''),
            'album': metadata.get('album', ''),
            'lyrics': metadata.get('lyrics', ''),
            'history': metadata.get('history', ''),
            'youtube_url': metadata.get('youtube_url', ''),
            'picture_url': metadata.get('picture_url', ''),
        }
        
        # Store file path and remaining fields in extra_metadata
        extra_metadata = {k: v for k, v in metadata.items() 
                          if k not in ['title', 'artist', 'album', 'lyrics', 
                                      'history', 'youtube_url', 'picture_url']}
        
        # Add file path to extra_metadata
        extra_metadata['file_path'] = audio_file
        
        # Fingerprint file
        track, fingerprints = self.fingerprinter.fingerprint_file(
            file_path=audio_file,
            track_info=track_info,
            extra_metadata=extra_metadata
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
    
    def batch_add_songs(self, directory: str, metadata_fn=None) -> List[AudioTrack]:
        """
        Add multiple songs from a directory.
        
        Args:
            directory: Directory containing audio files
            metadata_fn: Optional function that takes a file path and returns a metadata dict
            
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
                    # Get metadata if function provided
                    metadata = None
                    if metadata_fn is not None:
                        metadata = metadata_fn(file_path)
                    
                    track = self.add_song(file_path, metadata)
                    tracks.append(track)
                    print(f"Added: {track.artist} - {track.title}")
                except Exception as e:
                    print(f"Error adding {filename}: {e}")
        
        return tracks
    
    def update_song_metadata(self, track_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing song.
        
        Args:
            track_id: Track ID to update
            metadata: Metadata to update including any of:
                - artist: Artist name
                - title: Song title
                - album: Album name
                - lyrics: Song lyrics
                - history: Song history/background information
                - youtube_url: URL to YouTube video
                - picture_url: URL to album art or artist image
                
        Returns:
            True if successfully updated, False if track not found
        """
        # Get existing track
        track = self.database.get_track(track_id)
        if track is None:
            return False
            
        # Update fields if present in metadata
        if 'title' in metadata:
            track.title = metadata['title']
        if 'artist' in metadata:
            track.artist = metadata['artist']
        if 'album' in metadata:
            track.album = metadata['album']
        if 'lyrics' in metadata:
            track.lyrics = metadata['lyrics']
        if 'history' in metadata:
            track.history = metadata['history']
        if 'youtube_url' in metadata:
            track.youtube_url = metadata['youtube_url']
        if 'picture_url' in metadata:
            track.picture_url = metadata['picture_url']
            
        # Update other metadata
        for key, value in metadata.items():
            if key not in ['title', 'artist', 'album', 'lyrics', 'history', 
                           'youtube_url', 'picture_url']:
                track.metadata[key] = value
                
        # Save to database
        self.database.add_track(track)
        return True
    
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
