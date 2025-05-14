
# eliprint/batch.py
"""
Batch processing functionality.
"""

import os
import logging
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import time

from .core import Fingerprinter
from .database import Database
from .models import AudioTrack, Fingerprint
from .utils import Timer

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Batch processor for fingerprinting multiple audio files efficiently.
    """
    
    def __init__(self, fingerprinter: Fingerprinter, database: Database):
        """
        Initialize the batch processor.
        
        Args:
            fingerprinter: Fingerprinter instance
            database: Database instance
        """
        self.fingerprinter = fingerprinter
        self.database = database
    
    def process_files(self, files: List[str], 
                    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
                    max_workers: Optional[int] = None,
                    progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[AudioTrack]:
        """
        Process multiple audio files in parallel.
        
        Args:
            files: List of file paths
            metadata_map: Optional mapping of file paths to metadata
            max_workers: Maximum number of worker processes (None for CPU count)
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            List of AudioTrack objects
        """
        if not files:
            return []
        
        # Create metadata map if not provided
        if metadata_map is None:
            metadata_map = {}
        
        # Filter valid files
        valid_files = []
        for file_path in files:
            if os.path.exists(file_path) and file_path.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                valid_files.append(file_path)
            else:
                logger.warning(f"Skipping invalid file: {file_path}")
        
        if not valid_files:
            logger.warning("No valid audio files found")
            return []
        
        # Process files
        if max_workers is None or max_workers > 1:
            # Use multi-processing for better performance
            return self._process_files_parallel(
                valid_files, 
                metadata_map,
                max_workers,
                progress_callback
            )
        else:
            # Use single-threaded processing
            return self._process_files_sequential(
                valid_files,
                metadata_map,
                progress_callback
            )
    
    def _process_files_sequential(self, files: List[str],
                                 metadata_map: Dict[str, Dict[str, Any]],
                                 progress_callback: Optional[Callable[[int, int, str], None]]) -> List[AudioTrack]:
        """Process files sequentially."""
        results = []
        tracks_and_fingerprints = []
        
        # Process each file
        for i, file_path in enumerate(files):
            try:
                # Report progress
                if progress_callback:
                    progress_callback(i, len(files), file_path)
                
                # Get metadata if available
                metadata = metadata_map.get(file_path)
                
                # Fingerprint file
                track, fingerprints = self.fingerprinter.fingerprint_file(
                    file_path=file_path,
                    track_info=metadata
                )
                
                # Add to results
                results.append(track)
                tracks_and_fingerprints.append((track, fingerprints))
                
                logger.info(f"Processed {i+1}/{len(files)}: '{file_path}' - {len(fingerprints)} fingerprints")
                
            except Exception as e:
                logger.error(f"Error processing file '{file_path}': {str(e)}")
        
        # Add tracks and fingerprints to database in a single transaction
        with Timer() as timer:
            self.database.add_tracks_with_fingerprints(tracks_and_fingerprints)
            logger.info(f"Added {len(tracks_and_fingerprints)} tracks to database in {timer.elapsed:.2f}s")
        
        return results
    
    def _process_files_parallel(self, files: List[str],
                               metadata_map: Dict[str, Dict[str, Any]],
                               max_workers: Optional[int],
                               progress_callback: Optional[Callable[[int, int, str], None]]) -> List[AudioTrack]:
        """Process files in parallel."""
        processed_count = 0
        results = []
        tracks_and_fingerprints = []
        
        # Create a thread-safe counter for progress reporting
        def update_progress(file_path, track, fingerprints):
            nonlocal processed_count
            processed_count += 1
            
            if progress_callback:
                progress_callback(processed_count, len(files), file_path)
            
            if track and fingerprints:
                results.append(track)
                tracks_and_fingerprints.append((track, fingerprints))
                
            logger.info(f"Processed {processed_count}/{len(files)}: '{file_path}' - {len(fingerprints) if fingerprints else 0} fingerprints")
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for file_path in files:
                metadata = metadata_map.get(file_path)
                
                # Submit task
                future = executor.submit(
                    self._fingerprint_file_worker,
                    file_path,
                    metadata
                )
                futures[future] = file_path
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                
                try:
                    track, fingerprints = future.result()
                    update_progress(file_path, track, fingerprints)
                except Exception as e:
                    logger.error(f"Error processing file '{file_path}': {str(e)}")
                    update_progress(file_path, None, None)
        
        # Add tracks and fingerprints to database in a single transaction
        with Timer() as timer:
            self.database.add_tracks_with_fingerprints(tracks_and_fingerprints)
            logger.info(f"Added {len(tracks_and_fingerprints)} tracks to database in {timer.elapsed:.2f}s")
        
        return results
    
    @staticmethod
    def _fingerprint_file_worker(file_path: str, metadata: Optional[Dict[str, Any]]) -> Tuple[AudioTrack, List[Fingerprint]]:
        """Worker function for parallel processing."""
        # Create a new fingerprinter instance for each worker
        fingerprinter = Fingerprinter()
        
        # Process file
        return fingerprinter.fingerprint_file(
            file_path=file_path,
            track_info=metadata
        )

