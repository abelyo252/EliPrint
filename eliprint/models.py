"""Data models for the EliPrint fingerprinting system."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Fingerprint:
    """Audio fingerprint hash with metadata."""
    hash: str
    track_id: str
    offset: int
    
    # Optional additional data (useful for debugging)
    anchor_freq: Optional[int] = None
    target_freq: Optional[int] = None
    delta: Optional[int] = None


@dataclass
class AudioTrack:
    """Metadata for an audio track."""
    id: str
    title: str = ""
    artist: str = ""
    album: str = ""
    duration: float = 0.0
    lyrics: str = ""
    history: str = ""
    youtube_url: str = ""
    picture_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Result of an audio matching operation."""
    track_id: str
    confidence: float
    offset_seconds: float
    match_count: int
    track: AudioTrack = None
    time_taken: float = 0.0
    
    @property
    def title(self) -> str:
        """Get track title."""
        return self.track.title if self.track else ""
    
    @property
    def artist(self) -> str:
        """Get track artist."""
        return self.track.artist if self.track else ""
    
    @property
    def album(self) -> str:
        """Get track album."""
        return self.track.album if self.track else ""
    
    @property
    def lyrics(self) -> str:
        """Get track lyrics."""
        return self.track.lyrics if self.track else ""
    
    @property
    def history(self) -> str:
        """Get track history."""
        return self.track.history if self.track else ""
    
    @property
    def youtube_url(self) -> str:
        """Get track YouTube URL."""
        return self.track.youtube_url if self.track else ""
    
    @property
    def picture_url(self) -> str:
        """Get track picture URL."""
        return self.track.picture_url if self.track else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "lyrics": self.lyrics,
            "history": self.history,
            "youtube_url": self.youtube_url,
            "picture_url": self.picture_url,
            "confidence": self.confidence,
            "offset_seconds": self.offset_seconds,
            "match_count": self.match_count,
            "time_taken": self.time_taken
        }
