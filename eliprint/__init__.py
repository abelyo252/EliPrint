"""
EliPrint - Audio Fingerprinting Library
======================================

A powerful and efficient audio fingerprinting library named after Ethiopian 
music composer Elias Melka. Similar to Shazam, EliPrint can identify 
audio tracks based on their acoustic fingerprints.

Features:
- Robust audio fingerprinting using constellation maps
- Fast matching algorithm for accurate identification
- MariaDB database storage for scalability
- Simple API for adding and identifying songs
"""

from .models import AudioTrack, MatchResult
from .api import EliPrint

# Version
__version__ = '1.0.0'

# Export high-level API
__all__ = ['EliPrint', 'AudioTrack', 'MatchResult']