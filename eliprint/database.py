"""Database management for audio fingerprints with support for MariaDB and SQLite."""

import json
import logging
import os
import sqlite3
from typing import List, Dict, Tuple, Optional, Any, Union

# Handle possible absence of MySQL Connector
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    class MySQLError(Exception):
        pass

from .models import AudioTrack, Fingerprint, MatchResult

logger = logging.getLogger(__name__)

class Database:
    """Database for storing and retrieving audio fingerprints with support for MariaDB and SQLite."""
    
    def __init__(self, host: str = "localhost", port: int = 3306, 
                 user: str = "root", password: str = "", 
                 database: str = "eliprint", batch_size: int = 1000,
                 db_type: str = None, sqlite_path: str = None):
        """
        Initialize the database connection.
        
        Args:
            host: MariaDB host
            port: MariaDB port
            user: MariaDB username
            password: MariaDB password
            database: MariaDB database name
            batch_size: Size of batches for bulk operations
            db_type: Database type ('mariadb' or 'sqlite'), auto-detected if None
            sqlite_path: Path to SQLite database file, defaults to [database].db in current directory
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        
        # Determine database type
        if db_type is None:
            # First choice is explicitly set db_type
            # Second choice is MariaDB if available and can connect
            # Default fallback is SQLite
            self.db_type = 'sqlite'
            
            if MYSQL_AVAILABLE:
                # Try to connect to MariaDB
                try:
                    test_conn = mysql.connector.connect(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        connection_timeout=3  # Short timeout for quick checking
                    )
                    test_conn.close()
                    self.db_type = 'mariadb'
                except Exception as e:
                    logger.warning(f"Error connecting to MariaDB database: {e}")
                    logger.info("Falling back to SQLite")
        else:
            # User specified db_type
            self.db_type = db_type.lower()
            
            # Validate db_type
            if self.db_type == 'mariadb' and not MYSQL_AVAILABLE:
                logger.warning("MariaDB requested but mysql.connector not available. Falling back to SQLite.")
                self.db_type = 'sqlite'
        
        # Set SQLite path
        if self.db_type == 'sqlite':
            if sqlite_path is None:
                self.sqlite_path = f"{self.database}.db"
            else:
                self.sqlite_path = sqlite_path
            
            logger.info(f"Using SQLite database at {self.sqlite_path}")
        else:
            logger.info(f"Using MariaDB database {self.database} on {self.host}:{self.port}")
        
        # Initialize connection
        self.conn = None
        self._connect()
        
    def _connect(self):
        """Connect to database and initialize schema."""
        try:
            if self.db_type == 'mariadb':
                self._connect_mariadb()
            else:  # sqlite
                self._connect_sqlite()
                
            # Create tables
            self._create_tables()
            
            logger.info(f"Connected to {self.db_type} database '{self.database}'")
            
        except Exception as e:
            if self.db_type == 'mariadb':
                logger.error(f"Error connecting to MariaDB database: {e}")
                logger.info("Trying SQLite as fallback...")
                
                # Fall back to SQLite
                self.db_type = 'sqlite'
                if not hasattr(self, 'sqlite_path') or self.sqlite_path is None:
                    self.sqlite_path = f"{self.database}.db"
                
                try:
                    self._connect_sqlite()
                    self._create_tables()
                    logger.info(f"Successfully connected to SQLite database at {self.sqlite_path}")
                except Exception as sqlite_error:
                    logger.error(f"Error connecting to SQLite database: {sqlite_error}")
                    raise
            else:
                logger.error(f"Error connecting to SQLite database: {e}")
                raise
    
    def _connect_mariadb(self):
        """Connect to MariaDB database."""
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
    
    def _connect_sqlite(self):
        """Connect to SQLite database."""
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)
        
        # Connect to SQLite
        self.conn = sqlite3.connect(self.sqlite_path)
        
        # Enable foreign keys
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        if self.db_type == 'mariadb':
            # Create tracks table for MariaDB
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id VARCHAR(255) PRIMARY KEY,
                title VARCHAR(255),
                artist VARCHAR(255),
                album VARCHAR(255),
                duration FLOAT,
                lyrics TEXT,
                history TEXT,
                youtube_url VARCHAR(255),
                picture_url VARCHAR(255),
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
            
        else:  # sqlite
            # Create tracks table for SQLite
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                duration REAL,
                lyrics TEXT,
                history TEXT,
                youtube_url TEXT,
                picture_url TEXT,
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
        
        self.conn.commit()
    
    def add_track(self, track: AudioTrack) -> None:
        """Add a track to the database."""
        cursor = self.conn.cursor()
        
        # Convert metadata to JSON
        metadata = json.dumps(track.metadata) if track.metadata else None
        
        if self.db_type == 'mariadb':
            # Insert track with UPSERT logic
            cursor.execute("""
            INSERT INTO tracks (id, title, artist, album, duration, lyrics, history, youtube_url, picture_url, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            artist = VALUES(artist),
            album = VALUES(album),
            duration = VALUES(duration),
            lyrics = VALUES(lyrics),
            history = VALUES(history),
            youtube_url = VALUES(youtube_url),
            picture_url = VALUES(picture_url),
            metadata = VALUES(metadata)
            """, (track.id, track.title, track.artist, track.album, track.duration, 
                  track.lyrics, track.history, track.youtube_url, track.picture_url, metadata))
        else:  # sqlite
            # SQLite UPSERT syntax
            cursor.execute("""
            INSERT INTO tracks (id, title, artist, album, duration, lyrics, history, youtube_url, picture_url, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            artist = excluded.artist,
            album = excluded.album,
            duration = excluded.duration,
            lyrics = excluded.lyrics,
            history = excluded.history,
            youtube_url = excluded.youtube_url,
            picture_url = excluded.picture_url,
            metadata = excluded.metadata
            """, (track.id, track.title, track.artist, track.album, track.duration, 
                  track.lyrics, track.history, track.youtube_url, track.picture_url, metadata))
        
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
            
            # Insert with different placeholders based on db_type
            if self.db_type == 'mariadb':
                cursor.executemany("""
                INSERT INTO fingerprints (hash, track_id, offset)
                VALUES (%s, %s, %s)
                """, batch)
            else:  # sqlite
                cursor.executemany("""
                INSERT INTO fingerprints (hash, track_id, offset)
                VALUES (?, ?, ?)
                """, batch)
        
        self.conn.commit()
    
    def get_track(self, track_id: str) -> Optional[AudioTrack]:
        """Get a track by ID."""
        cursor = self.conn.cursor()
        
        if self.db_type == 'mariadb':
            cursor.execute("""
            SELECT id, title, artist, album, duration, lyrics, history, youtube_url, picture_url, metadata
            FROM tracks
            WHERE id = %s
            """, (track_id,))
        else:  # sqlite
            cursor.execute("""
            SELECT id, title, artist, album, duration, lyrics, history, youtube_url, picture_url, metadata
            FROM tracks
            WHERE id = ?
            """, (track_id,))
        
        row = cursor.fetchone()
        
        if row:
            id, title, artist, album, duration, lyrics, history, youtube_url, picture_url, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            return AudioTrack(
                id=id,
                title=title,
                artist=artist,
                album=album,
                duration=duration,
                lyrics=lyrics,
                history=history,
                youtube_url=youtube_url,
                picture_url=picture_url,
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
            
            cursor = self.conn.cursor()
            
            if self.db_type == 'mariadb':
                # Prepare placeholders for IN clause
                placeholders = ','.join(['%s'] * len(batch))
                
                # Execute query
                cursor.execute(f"""
                SELECT hash, track_id, offset
                FROM fingerprints
                WHERE hash IN ({placeholders})
                """, batch)
            else:  # sqlite
                # Prepare placeholders for IN clause
                placeholders = ','.join(['?'] * len(batch))
                
                # Execute query
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
        if self.db_type == 'mariadb':
            cursor.execute("SELECT 1 FROM tracks WHERE id = %s", (track_id,))
        else:  # sqlite
            cursor.execute("SELECT 1 FROM tracks WHERE id = ?", (track_id,))
        
        if not cursor.fetchone():
            return False
        
        # Delete track (fingerprints will be deleted by CASCADE)
        if self.db_type == 'mariadb':
            cursor.execute("DELETE FROM tracks WHERE id = %s", (track_id,))
        else:  # sqlite
            cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
        
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
