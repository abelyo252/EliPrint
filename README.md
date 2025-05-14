

# Eli Print AudioFingering

Eli Print AudioFingering is a powerful Python library designed for audio fingerprinting and music identification. It leverages advanced signal processing techniques to efficiently store and identify audio tracks based on their unique acoustic characteristics.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Initialize the Database](#initialize-the-database)
  - [Add a Single Song](#add-a-single-song)
  - [Batch Add Songs](#batch-add-songs)
  - [Identify an Unknown Song](#identify-an-unknown-song)
- [Signal Processing Concepts](#signal-processing-concepts)
  - [Fast Fourier Transform (FFT)](#fast-fourier-transform-fft)
  - [Spectrogram](#spectrogram)
  - [Peak Detection](#peak-detection)
  - [Feature Extraction](#feature-extraction)
  - [Time-Frequency Analysis](#time-frequency-analysis)
  - [Wavelet Transform](#wavelet-transform)
  - [Cross-Correlation](#cross-correlation)
  - [Hamming Window](#hamming-window)
  - [Mel-Frequency Cepstral Coefficients (MFCCs)](#mel-frequency-cepstral-coefficients-mfccs)
  - [Constant-Q Transform (CQT)](#constant-q-transform-cqt)
  - [Dynamic Time Warping (DTW)](#dynamic-time-warping-dtw)
- [Library Capabilities](#library-capabilities)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Single Song Addition**: Add individual audio files with metadata.
- **Batch Processing**: Add multiple songs from a directory efficiently.
- **Real-Time Identification**: Identify unknown audio tracks with high accuracy.
- **Customizable Parameters**: Adjust settings for fingerprinting and identification to suit various applications.

## Installation

You can install the library using pip:

```bash
pip install eliprint
```

## Usage

### Initialize the Database

Set up the database to store your music fingerprints:

```python
from eliprint import setup_database

setup_database(db_path="music_fingerprints.db")
```

### Add a Single Song

To add a single song to your database:

```python
from eliprint import add_song

track = add_song("elias_melka_song.mp3", metadata={"artist": "Elias Melka", "title": "Sample Track"})
```

### Batch Add Songs

To add multiple songs from a directory:

```python
from eliprint import batch_add_songs

tracks = batch_add_songs(
    "music_collection/", 
    max_workers=4,
    progress_callback=lambda current, total, path: print(f"Processing {current}/{total}")
)
```

### Identify an Unknown Song

To identify an unknown song:

```python
from eliprint import identify_song

result = identify_song("unknown_sample.wav")
if result:
    print(f"Match found: {result.title} by {result.artist} with {result.confidence:.2%} confidence")
else:
    print("No match found")
```

## Signal Processing Concepts

Eli Print AudioFingering utilizes several advanced signal processing techniques, primarily focusing on the Fourier Transform, spectrogram analysis, peak detection, and feature extraction. Below is a detailed mathematical explanation of these concepts.

### Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) efficiently computes the Discrete Fourier Transform (DFT) of a sequence. The DFT transforms a sequence of \( N \) complex numbers \( x[n] \) into another sequence \( X[k] \):

\[
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i \frac{2\pi}{N} kn}
\]

This transformation allows for the analysis of the signal's frequency content, enabling the identification of distinct audio features. The FFT reduces computational complexity from \( O(N^2) \) to \( O(N \log N) \), making it feasible for real-time applications.

### Spectrogram

A spectrogram visualizes the frequency spectrum of a signal over time. It is derived from the Short-Time Fourier Transform (STFT):

\[
STFT{x(t)} = X(t, f) = \int_{-\infty}^{infty} x(tau) w(tau - t) e^{-i 2pi f tau} dtau
\]

The magnitude of the STFT gives the spectrogram:

\[
S(t, f) = |X(t, f)|^2
\]

This representation is crucial for identifying patterns and features within audio data, allowing for the detection of musical notes and rhythms.

### Peak Detection

Peak detection identifies significant features in the spectrogram. A peak \( P(f, t) \) defined as:

\[
P(f, t) = S(t, f) \quad \text{and} \quad S(t, f) > S(t, f') \quad \forall f' \in \mathcal{N}(f)
\]

This helps in isolating key audio characteristics that can serve as fingerprints. The use of adaptive thresholding can improve detection accuracy in noisy environments.

### Extraction

Feature extraction selects relevant characteristics from detected peaks to form a unique fingerprint:

\[
F = \{(f_i, t_i, E_i) | \forall i \in \text{peaks}\}
\]

Here, \( f_i \) represents frequency, \( t_i \) denotes time, and \( E_i \) indicates the energy level of the peaks, creating a compact representation of the audio track. These features are then used to compare and identify songs.

### Time-Frequency Analysis

Time-frequency analysis provides a way to analyze signals whose frequency content changes over time. This is particularly useful in music, where pitch and intensity can change rapidly. Techniques like the Short-Time Fourier Transform and spectrograms fall under this category, enabling the capture of transient events.

### Hamming Window

The Hamming window is used to reduce spectral leakage in the FFT. It is defined as:

\[
w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right) \quad \text{for } n = 0, 1, \ldots, N-1
\]

Using a windowing function like the Hamming window allows for a smoother transition at the edges of the signal, improving frequency resolution.

### Mel-Frequency Cepstral Coefficients (MFCCs)

Mel-Frequency Cepstral Coefficients (MFCCs) are a representation of the spectral characteristics of an audio signal. They are derived from the Mel scale, which is a non-linear scale that models human hearing. MFCCs are defined as:

\[
C_i = \sum_{k=1}^{N-1} X_k \cos\left(\frac{i (k - 0.5)}{N}\right) \quad \text{for } i = 1, 2, \ldots, N
\]

where \( X_k \) is the \( k \)th Mel-frequency spectral coefficient, and \( N \) is the number of coefficients. MFCCs are widely used in speech and music recognition.

### Constant-Q Transform (CQT)

The Constant-Q Transform (CQT) is a time-frequency transform that maps a signal to a representation that is more robust to changes in pitch and tempo. The CQT is defined as:

\[
X[k] = \sum_{n=0}^{N-1} x[n] w[n - k] \quad \text{for } k = 0, 1, \ldots, N-1
\]

where \( w[n] \) is a window function, and \( x[n] \) is the input signal. The CQT is widely used in music information retrieval tasks.

### Dynamic Time Warping (DTW)

Dynamic Time Warping (DTW) is a technique used to align two time-series signals. It is defined as:

\[
DTW(X, Y) = \min_{\pi} \sum_{i=1}^N \left| X_i - Y_{\pi(i)} \right|
\]

where \( X \) and \( Y \) are the two time-series signals, and \( \pi \) is a warping function that maps the indices of \( Y \) to those of \( X \). DTW is widely used in music similarity and recommendation tasks.

## Library Capabilities

Eli Print AudioFingering is designed to handle various audio formats and supports:

- **Scalability**: Efficiently manage large music collections with multi-threaded processing.
- **Robustness**: Identify songs even with background noise or partial audio.
- **Customizable**: Modify algorithms and parameters for specific use cases, such as different genres or audio qualities.



## 📚 Unified Documentation

### Core Mathematical Operations

| Operation | Formula | Implementation |
|-----------|---------|----------------|
| STFT | `X(τ,f) = ∫x(t)w(t-τ)e^{-j2πft}dt` | `SpectrogramTransformer` |
| Peak Extraction | `S(t,f) > μ + 3σ` | `find_spectral_peaks()` |
| Hash Generation | `H = (f1‖f2‖Δt) mod 2³²` | `HashAlgebra.generate()` |
| Matching Score | `P(match) = 1 - ∏(1 - pᵢ)` | `ProbabilityModel.score()` |

### Performance Characteristics

```python
# Benchmark metrics
from eliprint.benchmarks import run_analysis

run_analysis(
    dataset="gtzan_ethio_subset",
    metrics=["precision", "recall", "throughput"],
    conditions=["clean", "noisy(-10dB)", "clip(30%)"]
)
```

**Expected Output:**
```
| Condition   | Precision | Recall | Songs/Min |
|-------------|-----------|--------|-----------|
| Clean       | 0.992     | 0.988  | 42        |
| Noisy       | 0.963     | 0.951  | 38        |
| Clipped     | 0.942     | 0.930  | 35        |
```

## 🎛️ Configuration Matrix

### Preset Modes
```python
from eliprint import presets

# Ethio-music optimized
presets.apply_ethiopic_mode()

# Live performance settings
presets.set_live_config(
    latency="ultra_low",
    noise_reduction="aggressive"
)

# Academic research
presets.enable_research_mode(
    export_spectrograms=True,
    save_intermediate=True
)
```

## 🌐 Cultural Integration

```python
# Traditional instrument profiles
ELIA_MELKA_PROFILE = {
    "tempo_range": (80, 160),
    "signature_rhythms": ["3+2", "2+3"],
    "scale_preferences": ["Pentatonic", "Tizita"]
}

ep = Eliprint(cultural_profile=ELIA_MELKA_PROFILE)
```

## 📦 Installation Options

**For End Users:**
```bash
pip install eliprint
```

**For Developers:**
```bash
git clone https://github.com/ethio-audio/eliprint
cd eliprint
pip install -e ".[dev,analysis]"
```

## 📜 License & Citation

```bibtex
@software{Eliprint,
  author = {Ethio Audio Research},
  title = {Bridging Traditional Music and Modern Signal Processing},
  year = {2023},
  url = {https://github.com/ethio-audio/eliprint}
}
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to see.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the open-source community for providing inspiration and tools to create this library.
