
# Eliprint AudioFingering

Eliprint AudioFingering is a powerful Python library designed for audio fingerprinting and music identification. It leverages advanced signal processing techniques to efficiently store and identify audio tracks based on their unique acoustic characteristics.

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
  - [Constellation Map](#constellation-map)
  - [Fingerprint Hashing](#fingerprint-hashing)
  - [Matching Algorithm](#matching-algorithm)
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

Set up the database to store your music fingerprints with customizable parameters:

```python
from eliprint import setup_database

# Initialize the fingerprint database with customizable parameters
setup_database(
    db_path="music_fingerprints.db",
    hash_algorithm="constellation",  # Options: "constellation", "wavelet", "mfcc_hash"
    peak_neighborhood_size=20,       # Controls peak density
    min_peak_amplitude=10,           # Minimum amplitude for peak detection
    fan_value=15,                    # Number of point pairs per anchor point
    min_hash_time_delta=0,           # Minimum time between paired points
    max_hash_time_delta=200          # Maximum time between paired points
)
```

### Add a Single Song

To add a single song to your database:

```python
from eliprint import add_song

track = add_song(
    "elias_melka_song.mp3", 
    metadata={"artist": "Elias Melka", "title": "Sample Track"},
    fft_window_size=4096,           # FFT window size for frequency resolution
    hop_length=512,                 # Hop length between frames
    frequency_bands=(20, 8000),     # Min and max frequency to analyze
    energy_threshold=0.75           # Minimum energy for considering a frame
)
```

### Batch Add Songs

To add multiple songs from a directory:

```python
from eliprint import batch_add_songs

tracks = batch_add_songs(
    "music_collection/", 
    max_workers=4,
    progress_callback=lambda current, total, path: print(f"Processing {current}/{total}"),
    sample_rate=44100,              # Target sample rate for processing
    window_function="hamming",      # Window function type
    peak_pruning_algorithm="adaptive_threshold"  # Algorithm for pruning peaks
)
```

### Identify an Unknown Song

To identify an unknown song:

```python
from eliprint import identify_song

result = identify_song(
    "unknown_sample.wav",
    confidence_threshold=0.65,      # Minimum confidence for a match
    time_stretch_range=(-5, 5),     # Percentage range for time stretch invariance
    match_algorithm="probabilistic", # Options: "probabilistic", "geometric", "hybrid"
    noise_reduction=True            # Apply noise reduction filter
)
if result:
    print(f"Match found: {result.title} by {result.artist} with {result.confidence:.2%} confidence")
    print(f"Match details: {result.time_offset}s offset, {result.score} hash matches")
else:
    print("No match found")
```

## Signal Processing Concepts

Eli Print AudioFingering utilizes several advanced signal processing techniques for robust audio fingerprinting. Below is a detailed mathematical explanation of these concepts.

### Fast Fourier Transform (FFT)

The Fast Fourier Transform efficiently computes the Discrete Fourier Transform (DFT) of a signal, converting time-domain audio data into the frequency domain. For a discrete-time signal $x[n]$ of length $N$, the DFT $X[k]$ is defined as:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i \frac{2\pi}{N} kn}, \quad k = 0, 1, \ldots, N-1$$

The inverse DFT is given by:

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i \frac{2\pi}{N} kn}, \quad n = 0, 1, \ldots, N-1$$

The FFT reduces computational complexity from $O(N^2)$ to $O(N \log N)$ through divide-and-conquer techniques like the Cooley-Tukey algorithm:

$$X[k] = \sum_{m=0}^{N/2-1} x[2m] e^{-i \frac{2\pi}{N/2} mk} + e^{-i \frac{2\pi}{N} k} \sum_{m=0}^{N/2-1} x[2m+1] e^{-i \frac{2\pi}{N/2} mk}$$

This transformation allows for the analysis of the signal's frequency content, enabling the identification of distinct audio features.

### Spectrogram

A spectrogram provides a time-frequency representation of the signal through the Short-Time Fourier Transform (STFT). For a discrete signal $x[n]$ and a window function $w[n]$ of length $L$, the STFT is:

$$X[m, k] = \sum_{n=0}^{L-1} x[n+mH] w[n] e^{-i \frac{2\pi}{N} kn}$$

where $m$ is the frame index, $H$ is the hop size between consecutive frames, and $k$ is the frequency bin index. The spectrogram magnitude $S[m, k]$ is given by:

$$S[m, k] = |X[m, k]|^2$$

The power spectrogram in decibels is often used for visualization:

$$S_{dB}[m, k] = 10 \log_{10}(S[m, k])$$

Spectrograms are crucial for identifying patterns and features within audio data, allowing for the detection of musical notes, rhythms, and distinctive auditory events.

### Peak Detection

Peak detection identifies local maxima in the spectrogram that serve as robust reference points. A spectral peak at position $(m, k)$ satisfies:

$$S[m, k] > S[m+\Delta m, k+\Delta k] \quad \forall (\Delta m, \Delta k) \in \mathcal{N}(0,0)$$

where $\mathcal{N}(0,0)$ is a neighborhood around $(m, k)$.

In practice, adaptive thresholding is often used, where a point is a peak if:

$$S[m, k] > \mu_{local} + \alpha \cdot \sigma_{local}$$

where $\mu_{local}$ and $\sigma_{local}$ are the local mean and standard deviation in a neighborhood, and $\alpha$ is a sensitivity parameter (typically 3-5).

Morphological operations for noise-robust peak detection can be formulated as:

$$P = (S \ominus B_1) \cap (S \ominus B_2) \cap \ldots \cap (S \ominus B_n)$$

where $\ominus$ is the erosion operation and $B_i$ are structuring elements designed to detect peaks of different shapes.

### Constellation Map

After peak detection, a constellation map $C$ is formed as a set of time-frequency points:

$$C = \{(t_1, f_1), (t_2, f_2), \ldots, (t_n, f_n)\}$$

where $t_i$ is the time frame and $f_i$ is the frequency bin of each peak. This sparse representation captures the distinctive features of the audio while being robust to noise and distortion.

The constellation map is then filtered based on peak strength and density using a rank-order filter:

$$C_{filtered} = \{(t_i, f_i) \in C \mid S[t_i, f_i] \geq \text{rank}_k(S_{N(t_i, f_i)})\}$$

where $\text{rank}_k$ returns the $k$-th highest value in the neighborhood $N(t_i, f_i)$.

### Fingerprint Hashing

The fingerprint hashing process converts pairs of peaks from the constellation map into compact, robust hashes. Similar to Shazam's approach, for each anchor point $(t_a, f_a)$, a set of target points $(t_i, f_i)$ within a time window are selected:

$$T(t_a) = \{(t_i, f_i) \in C \mid t_a < t_i \leq t_a + \Delta t_{max}\}$$

For each pair of anchor and target points, a hash $h$ is computed:

$$h = \mathcal{H}(f_a, f_i, (t_i - t_a))$$

where $\mathcal{H}$ is a hash function that combines these values. A common implementation is:

$$h = (f_a \ll 20) | (f_i \ll 10) | (t_i - t_a)$$

where $\ll$ represents the bit shift operation and $|$ is the bitwise OR.

The complete fingerprint consists of the hash and the absolute time of the anchor point:

$$F = \{(h_1, t_{a1}), (h_2, t_{a2}), \ldots, (h_n, t_{an})\}$$

This approach creates a sparse and efficient representation that is robust to noise, time scaling, and pitch shifting.

### Matching Algorithm

The matching process uses a combinatorial approach to identify a song. For each hash $h_q$ from the query audio, matching hashes $h_r$ in the reference database are retrieved:

$$M(h_q) = \{(h_r, t_{ar}, song_{ID}) \mid h_r = h_q\}$$

For each match, the time offset $\delta_t$ between the query and reference is calculated:

$$\delta_t = t_{aq} - t_{ar}$$

A histogram of time offsets for each song $s$ is constructed:

$$H_s(\delta_t) = |\{(h_q, h_r) \mid h_q = h_r \land song_{ID}(h_r) = s \land (t_{aq} - t_{ar}) = \delta_t\}|$$

The best matching song is the one with the highest peak in its histogram:

$$match = \arg\max_s \max_{\delta_t} H_s(\delta_t)$$

The confidence score can be calculated as:

$$confidence(s) = \frac{\max_{\delta_t} H_s(\delta_t)}{\sum_i H_s(\delta_{t_i})} \cdot \frac{|matches(s)|}{|query\_hashes|}$$

where $|matches(s)|$ is the number of matching hashes for song $s$ and $|query\_hashes|$ is the total number of hashes in the query.

### Time-Frequency Analysis

Time-frequency analysis methods like the Gabor transform provide optimal time-frequency localization according to the Heisenberg uncertainty principle:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

The Gabor transform is defined as:

$$G_x(t, f) = \int_{-\infty}^{\infty} x(\tau) g(\tau - t) e^{-i2\pi f\tau} d\tau$$

where $g(t)$ is a Gaussian window:

$$g(t) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{t^2}{2\sigma^2}}$$

This provides better localization in both time and frequency domains compared to the conventional STFT.

### Wavelet Transform

The Continuous Wavelet Transform (CWT) decomposes a signal using scaled and shifted versions of a wavelet function:

$$CWT_x(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt$$

where $a$ is the scale parameter, $b$ is the translation parameter, and $\psi^*$ is the complex conjugate of the mother wavelet.

The Discrete Wavelet Transform (DWT) uses dyadic scales and positions:

$$DWT_x(j, k) = \frac{1}{\sqrt{2^j}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-2^j k}{2^j}\right) dt$$

Wavelet transforms provide multi-resolution analysis that adapts to the signal's local characteristics, making them suitable for analyzing non-stationary audio signals.

### Cross-Correlation

The cross-correlation between two signals $x[n]$ and $y[n]$ is defined as:

$$R_{xy}[m] = \sum_{n=-\infty}^{\infty} x[n] y[n+m]$$

The normalized cross-correlation is:

$$\rho_{xy}[m] = \frac{R_{xy}[m]}{\sqrt{R_{xx}[0] R_{yy}[0]}}$$

This provides a measure of similarity between the signals at different time lags, which is useful for audio matching.

### Hamming Window

The Hamming window is commonly used to reduce spectral leakage in the FFT. It is defined as:

$$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right) \quad \text{for } n = 0, 1, \ldots, N-1$$

The Hamming window has a main lobe width of approximately $8\pi/N$ radians and a first side lobe at about -43 dB.

The application of a window function modifies the spectral estimate:

$$P_{windowed}(f) = \int_{-\infty}^{\infty} |W(f - \lambda)|^2 P_{true}(\lambda) d\lambda$$

where $W(f)$ is the Fourier transform of the window function and $P_{true}(f)$ is the true power spectral density.

### Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs capture the spectral envelope of the signal in a perceptually relevant way. The mel scale conversion is:

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

The MFCCs are computed as:

1. Compute the power spectrum: $P[k] = |X[k]|^2$
2. Map to mel scale using a filterbank of $M$ triangular filters: $S[m] = \sum_{k=0}^{N-1} P[k] H_m[k]$
3. Take the logarithm: $\log(S[m])$
4. Apply the Discrete Cosine Transform (DCT):

$$c[i] = \sum_{m=0}^{M-1} \log(S[m]) \cos\left(\frac{\pi i (m+0.5)}{M}\right), \quad i = 0, 1, \ldots, L-1$$

MFCCs provide a compact representation of the spectral characteristics of audio signals, making them useful for fingerprinting.

### Constant-Q Transform (CQT)

The CQT provides a frequency analysis with logarithmically spaced frequency bins, which aligns with musical scales:

$$X_{CQ}[k] = \frac{1}{N[k]} \sum_{n=0}^{N[k]-1} x[n] w[k, n] e^{-i2\pi Q n / N[k]}$$

where $N[k] = \frac{Q f_s}{f_k}$ is the window length for the $k$-th bin, $f_k = f_0 \cdot 2^{k/b}$ is the center frequency, $f_0$ is the lowest frequency, $b$ is the number of bins per octave, and $Q$ is the quality factor.

The CQT offers better frequency resolution at lower frequencies and better time resolution at higher frequencies, making it particularly suitable for music analysis.

### Dynamic Time Warping (DTW)

DTW finds the optimal alignment between two time series by minimizing the cumulative distance:

$$D(i, j) = d(x_i, y_j) + \min\{D(i-1, j), D(i, j-1), D(i-1, j-1)\}$$

where $d(x_i, y_j)$ is the distance between points $x_i$ and $y_j$.

The warping path $W = \{w_1, w_2, \ldots, w_K\}$ with $w_k = (i_k, j_k)$ satisfies the constraints:
- Boundary conditions: $w_1 = (1, 1)$ and $w_K = (n, m)$
- Monotonicity: $i_{k-1} \leq i_k$ and $j_{k-1} \leq j_k$
- Continuity: $i_k - i_{k-1} \leq 1$ and $j_k - j_{k-1} \leq 1$

DTW allows for comparison of audio fingerprints with different time scales, making it robust to tempo variations.

## Library Capabilities

Eli Print AudioFingering implements these mathematical concepts with efficient algorithms to provide:

- **Scalability**: Efficiently manage large music collections with parallel processing and optimized data structures.
- **Robustness**: Identify songs even with background noise, time stretching, pitch shifting, or partial audio using advanced probabilistic models.
- **Customizability**: Modify algorithms and parameters for specific use cases, from low-latency live performances to high-accuracy archival applications.

The library employs a hierarchical matching approach:

1. **Coarse Matching**: Fast hash lookup with time offset histogram analysis
2. **Fine Matching**: Detailed verification of matched segments using DTW or cross-correlation
3. **Confidence Estimation**: Statistical models to evaluate match quality based on hash density and consistency

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
```
