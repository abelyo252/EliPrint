# Usage examples
def example_usage():
    """Example usage of the library."""
    from .api import setup_database, add_song, batch_add_songs, identify_song, get_stats
    import os
    
    # Setup logging
    setup_logging(level=logging.INFO, log_file="eliprint.log")
    
    # Setup database
    setup_database(db_path="fingerprints.db")
    
    # Fingerprint a collection of audio files
    music_dir = "music_library"
    print(f"Adding songs from {music_dir}...")
    
    tracks = batch_add_songs(
        directory_or_files=music_dir,
        max_workers=4,
        progress_callback=lambda current, total, path: print(f"Processing {current}/{total}: {os.path.basename(path)}")
    )
    
    print(f"Added {len(tracks)} tracks to the database")
    
    # Print database stats
    stats = get_stats()
    print(f"Database contains {stats['tracks']} tracks and {stats['fingerprints']} fingerprints")
    print(f"Average fingerprints per track: {stats['fingerprints_per_track']:.1f}")
    
    # Recognize an unknown sample
    sample_path = "unknown_sample.wav"
    print(f"\nRecognizing {sample_path}...")
    
    result = identify_song(sample_path)
    pretty_print_match(result, sample_path)
    
    # Export results
    if result:
        export_results(result, "match_result.json")
        print(f"Results exported to match_result.json")


if __name__ == "__main__":
    example_usage()