# Basic usage
from eliprint import EliPrint, setup_logging

# Set up logging
setup_logging(log_file="eliprint.log")

# Initialize EliPrint with MariaDB connection
eli = EliPrint(
    host="localhost",
    port=3306,
    user="root",
    password="your_password",
    database="eliprint"
)

# Add a song to the database
track = eli.add_song("path/to/song.mp3")
print(f"Added: {track.artist} - {track.title}")

# Add an entire directory of songs
tracks = eli.batch_add_songs("path/to/music/directory")
print(f"Added {len(tracks)} tracks")

# Identify a song
result = eli.identify_song("path/to/sample.mp3")
if result:
    print(f"Match found: {result.artist} - {result.title}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Time offset: {result.offset_seconds:.2f} seconds")
else:
    print("No match found")

# Get database statistics
stats = eli.get_stats()
print(f"Database contains {stats['tracks']} tracks with {stats['fingerprints']} fingerprints")

# Clean up
eli.close()

# Benchmark usage
from eliprint import run_analysis, DatasetManager

# Create a test dataset
dataset_manager = DatasetManager("./datasets")
dataset_manager.create_dataset(
    name="test_dataset",
    source_dir="path/to/audio/files",
    split_ratio=0.8
)

# Run benchmark analysis
results = run_analysis(
    dataset="test_dataset",
    conditions=["clean", "noisy(-10dB)", "clip(30%)"],
    plot=True,
    plot_title="EliPrint Benchmark Results",
    plot_path="benchmark_results.png"
)

# Expected output:
# | Condition   | Precision | Recall | Songs/Min |
# |-------------|-----------|--------|-----------|
# | Clean       | 0.992     | 0.988  | 42        |
# | Noisy(-10dB)| 0.963     | 0.951  | 38        |
# | Clip(30%)   | 0.942     | 0.930  | 35        |
