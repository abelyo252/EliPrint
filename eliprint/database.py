
# eliprint/database.py
"""
Database management for audio fingerprints.
"""

import pickle
import sqlite3
import json
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union, Any

from .models import AudioTrack, Fingerprint

logger = logging.getLogger(__name__)

class Database:
    """
    Database for storing and retrieving audio fingerprints.
    Supports both in-memory and SQLite storage backends with optimized batch operations.
    """
    
    def __init__(self, db_path: Optional[str] = None, in_memory: bool = False, 
                batch_size: int = 1000, auto_commit: bool = True):
        """
        Initialize the database.
        
        Args:
            db_path: Path to SQLite database file (None for in-memory)
            in_memory: Whether to use in-memory storage instead of SQLite
            batch_size: Size of batches for bulk operations
            auto_commit: Whether to commit after each operation
        """
        self.in_memory = in_memory
        self.db_path = db_path
        self.batch_size = batch_size
        self.auto_commit = auto_commit
        
        # In-memory storage
        self.fingerprints = {}  # hash -> [(track_id, offset), ...]
        self.tracks = {}  # track_id -> AudioTrack
        
        # SQLite connection
        self.conn = None
        
        # Initialize database
        if not in_memory:
            self._init_sqlite(db_path)
    
    def _init_sqlite(self, db_path: Optional[str]):
        """Initialize SQLite database."""
        # Use in-memory database if no path provided
        if db_path is None:
            db_path = ":memory:"
        
        # Create directory if needed
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(db_path)
        
        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Optimize performance
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA cache_size = 10000")
        
        # Create tables if they don't exist
        self._create_tables()
        
        logger.info(f"Initialized database at '{db_path}'")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create tracks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            title TEXT,
            artist TEXT,
            album TEXT,
            duration REAL,
            metadata TEXT
        )
        """)
        
        # Create fingerprints table with index
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT NOT NULL,
            track_id TEXT NOT NULL,
            offset INTEGER NOT NULL,
            FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
        )
        """)
        
        # Create index on hash for fast lookups
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints(hash)
        """)
        
        # Create index on track_id for fast track lookups
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprints_track_id ON fingerprints(track_id)
        """)
        
        self.conn.commit()
    
    def add_track(self, track: AudioTrack) -> None:
        """
        Add a track to the database.
        
        Args:
            track: AudioTrack object
        """
        if self.in_memory:
            self.tracks[track.id] = track
        else:
            cursor = self.conn.cursor()
            
            # Convert metadata to JSON
            metadata = json.dumps(track.metadata) if track.metadata else None
            
            # Insert track
            cursor.execute("""
            INSERT OR REPLACE INTO tracks (id, title, artist, album, duration, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (track.id, track.title, track.artist, track.album, track.duration, metadata))
            
            if self.auto_commit:
                self.conn.commit()
    
    def add_fingerprints(self, fingerprints: List[Fingerprint]) -> None:
        """
        Add fingerprints to the database.
        
        Args:
            fingerprints: List of Fingerprint objects
        """
        if not fingerprints:
            return
        
        if self.in_memory:
            # Group by hash for efficient storage
            for fp in fingerprints:
                if fp.hash not in self.fingerprints:
                    self.fingerprints[fp.hash] = []
                self.fingerprints[fp.hash].append((fp.track_id, fp.offset))
        else:
            cursor = self.conn.cursor()
            
            # Prepare data for bulk insert
            data = [(fp.hash, fp.track_id, fp.offset) for fp in fingerprints]
            
            # Use batching for large datasets
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                
                # Use executemany for efficiency
                cursor.executemany("""
                INSERT INTO fingerprints (hash, track_id, offset)
                VALUES (?, ?, ?)
                """, batch)
            
            if self.auto_commit:
                self.conn.commit()
    
    def add_tracks_with_fingerprints(self, tracks_and_fingerprints: List[Tuple[AudioTrack, List[Fingerprint]]]) -> None:
        """
        Add multiple tracks with their fingerprints in a single transaction.
        
        Args:
            tracks_and_fingerprints: List of (track, fingerprints) tuples
        """
        if not tracks_and_fingerprints:
            return
        
        if self.in_memory:
            # Add each track and its fingerprints
            for track, fingerprints in tracks_and_fingerprints:
                self.add_track(track)
                self.add_fingerprints(fingerprints)
        else:
            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            try:
                cursor = self.conn.cursor()
                
                # Add tracks
                for track, _ in tracks_and_fingerprints:
                    metadata = json.dumps(track.metadata) if track.metadata else None
                    
                    cursor.execute("""
                    INSERT OR REPLACE INTO tracks (id, title, artist, album, duration, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (track.id, track.title, track.artist, track.album, track.duration, metadata))
                
                # Add fingerprints
                all_fingerprints = []
                for _, fingerprints in tracks_and_fingerprints:
                    all_fingerprints.extend([(fp.hash, fp.track_id, fp.offset) for fp in fingerprints])
                
                # Use batching for large datasets
                for i in range(0, len(all_fingerprints), self.batch_size):
                    batch = all_fingerprints[i:i + self.batch_size]
                    
                    cursor.executemany("""
                    INSERT INTO fingerprints (hash, track_id, offset)
                    VALUES (?, ?, ?)
                    """, batch)
                
                # Commit transaction
                self.conn.commit()
                
            except Exception as e:
                # Rollback on error
                self.conn.rollback()
                logger.error(f"Error adding tracks with fingerprints: {str(e)}")
                raise
    
    def get_track(self, track_id: str) -> Optional[AudioTrack]:
        """
        Get a track by ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            AudioTrack if found, None otherwise
        """
        if self.in_memory:
            return self.tracks.get(track_id)
        else:
            cursor = self.conn.cursor()
            
            cursor.execute("""
            SELECT id, title, artist, album, duration, metadata
            FROM tracks
            WHERE id = ?
            """, (track_id,))
            
            row = cursor.fetchone()
            
            if row:
                id, title, artist, album, duration, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                return AudioTrack(
                    id=id,
                    title=title,
                    artist=artist,
                    album=album,
                    duration=duration,
                    metadata=metadata
                )
            
            return None
    
    def get_matching_fingerprints(self, hashes: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get matching fingerprints for a list of hashes.
        
        Args:
            hashes: List of hash strings
            
        Returns:
            Dictionary mapping hash to list of (track_id, offset) tuples
        """
        if not hashes:
            return {}
        
        if self.in_memory:
            # Filter fingerprints that match the hashes
            return {h: self.fingerprints.get(h, []) for h in hashes}
        else:
            results = {}
            
            # Use batching for large queries
            for i in range(0, len(hashes), self.batch_size):
                batch = hashes[i:i + self.batch_size]
                
                # Prepare placeholders for IN clause
                placeholders = ','.join(['?'] * len(batch))
                
                # Execute query
                cursor = self.conn.cursor()
                cursor.execute(f"""
                SELECT hash, track_id, offset
                FROM fingerprints
                WHERE hash IN ({placeholders})
                """, batch)
                
                # Group by hash
                for hash_str, track_id, offset in cursor.fetchall():
                    if hash_str not in results:
                        results[hash_str] = []
                    results[hash_str].append((track_id, offset))
            
            return results
    
    def get_all_tracks(self) -> List[AudioTrack]:
        """
        Get all tracks in the database.
        
        Returns:
            List of AudioTrack objects
        """
        if self.in_memory:
            return list(self.tracks.values())
        else:
            cursor = self.conn.cursor()
            
            cursor.execute("""
            SELECT id, title, artist, album, duration, metadata
            FROM tracks
            """)
            
            tracks = []
            for row in cursor.fetchall():
                id, title, artist, album, duration, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                track = AudioTrack(
                    id=id,
                    title=title,
                    artist=artist,
                    album=album,
                    duration=duration,
                    metadata=metadata
                )
                tracks.append(track)
            
            return tracks
    
    def get_track_count(self) -> int:
        """Get the number of tracks in the database."""
        if self.in_memory:
            return len(self.tracks)
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            return cursor.fetchone()[0]
    
    def get_fingerprint_count(self) -> int:
        """Get the number of fingerprints in the database."""
        if self.in_memory:
            return sum(len(matches) for matches in self.fingerprints.values())
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fingerprints")
            return cursor.fetchone()[0]
    
    def delete_track(self, track_id: str) -> bool:
        """
        Delete a track and its fingerprints.
        
        Args:
            track_id: Track ID
            
        Returns:
            True if track was deleted, False if not found
        """
        if self.in_memory:
            if track_id not in self.tracks:
                return False
            
            # Remove track
            del self.tracks[track_id]
            
            # Remove fingerprints
            for hash_str, matches in list(self.fingerprints.items()):
                # Filter out matches for this track
                self.fingerprints[hash_str] = [m for m in matches if m[0] != track_id]
                
                # Remove empty entries
                if not self.fingerprints[hash_str]:
                    del self.fingerprints[hash_str]
            
            return True
        else:
            cursor = self.conn.cursor()
            
            # Check if track exists
            cursor.execute("SELECT 1 FROM tracks WHERE id = ?", (track_id,))
            if not cursor.fetchone():
                return False
            
            # Delete track (fingerprints will be deleted by CASCADE)
            cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            
            if self.auto_commit:
                self.conn.commit()
            
            return True
    
    def save(self, file_path: str) -> None:
        """
        Save in-memory database to a file.
        
        Args:
            file_path: Path to save the database
        """
        if not self.in_memory:
            raise ValueError("Cannot save SQLite database with this method")
        
        data = {
            'fingerprints': self.fingerprints,
            'tracks': self.tracks
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved in-memory database to '{file_path}'")
    
    def load(self, file_path: str) -> None:
        """
        Load database from a file.
        
        Args:
            file_path: Path to the database file
        """
        if not self.in_memory:
            raise ValueError("Cannot load SQLite database with this method")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.fingerprints = data['fingerprints']
        self.tracks = data['tracks']
        
        logger.info(f"Loaded in-memory database from '{file_path}'")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

