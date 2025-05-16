"""Database management for audio fingerprints using MariaDB."""

import json
import logging
from typing import List, Dict, Tuple, Optional, Any
import mysql.connector
from mysql.connector import Error as MySQLError

from .models import AudioTrack, Fingerprint, MatchResult

logger = logging.getLogger(__name__)

class Database:
    """Database for storing and retrieving audio fingerprints using MariaDB."""
    
    def __init__(self, host: str = "localhost", port: int = 3306, 
                 user: str = "root", password: str = "", 
                 database: str = "eliprint", batch_size: int = 1000):
        """
        Initialize the database connection.
        
        Args:
            host: MariaDB host
            port: MariaDB port
            user: MariaDB username
            password: MariaDB password
            database: MariaDB database name
            batch_size: Size of batches for bulk operations
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        
        # Initialize connection
        self.conn = None
        self._connect()
        
    def _connect(self):
        """Connect to MariaDB and initialize the database."""
        try:
            # Connect to MariaDB
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            
            cursor = self.conn.cursor()
            
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE `{self.database}`")
            
            # Create tables
            self._create_tables()
            
            logger.info(f"Connected to MariaDB database '{self.database}' on {self.host}:{self.port}")
            
        except MySQLError as e:
            logger.error(f"Error connecting to MariaDB: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create tracks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(255),
            artist VARCHAR(255),
            album VARCHAR(255),
            duration FLOAT,
            metadata TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create fingerprints table with index
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            hash VARCHAR(255) NOT NULL,
            track_id VARCHAR(255) NOT NULL,
            offset INT NOT NULL,
            FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # Create index on hash for fast lookups
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints(hash)
        """)
        
        self.conn.commit()
    
    def add_track(self, track: AudioTrack) -> None:
        """Add a track to the database."""
        cursor = self.conn.cursor()
        
        # Convert metadata to JSON
        metadata = json.dumps(track.metadata) if track.metadata else None
        
        # Insert track with UPSERT logic
        cursor.execute("""
        INSERT INTO tracks (id, title, artist, album, duration, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        artist = VALUES(artist),
        album = VALUES(album),
        duration = VALUES(duration),
        metadata = VALUES(metadata)
        """, (track.id, track.title, track.artist, track.album, track.duration, metadata))
        
        self.conn.commit()
    
    def add_fingerprints(self, fingerprints: List[Fingerprint]) -> None:
        """Add fingerprints to the database using batching for efficiency."""
        if not fingerprints:
            return
        
        cursor = self.conn.cursor()
        
        # Prepare data for bulk insert
        data = [(fp.hash, fp.track_id, fp.offset) for fp in fingerprints]
        
        # Use batching for large datasets
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            
            cursor.executemany("""
            INSERT INTO fingerprints (hash, track_id, offset)
            VALUES (%s, %s, %s)
            """, batch)
        
        self.conn.commit()
    
    def get_track(self, track_id: str) -> Optional[AudioTrack]:
        """Get a track by ID."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT id, title, artist, album, duration, metadata
        FROM tracks
        WHERE id = %s
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
        """Get matching fingerprints for a list of hashes."""
        if not hashes:
            return {}
        
        results = {}
        
        # Use batching for large queries
        for i in range(0, len(hashes), self.batch_size):
            batch = hashes[i:i + self.batch_size]
            
            # Prepare placeholders for IN clause
            placeholders = ','.join(['%s'] * len(batch))
            
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
    
    def get_track_count(self) -> int:
        """Get the number of tracks in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        return cursor.fetchone()[0]
    
    def get_fingerprint_count(self) -> int:
        """Get the number of fingerprints in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fingerprints")
        return cursor.fetchone()[0]
    
    def delete_track(self, track_id: str) -> bool:
        """Delete a track and its fingerprints."""
        cursor = self.conn.cursor()
        
        # Check if track exists
        cursor.execute("SELECT 1 FROM tracks WHERE id = %s", (track_id,))
        if not cursor.fetchone():
            return False
        
        # Delete track (fingerprints will be deleted by CASCADE)
        cursor.execute("DELETE FROM tracks WHERE id = %s", (track_id,))
        self.conn.commit()
        
        return True
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()