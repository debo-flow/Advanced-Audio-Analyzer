# 🎧 Advanced Audio Analyzer

A Physics-Based Audio & Signal Processing Laboratory built with Streamlit, Librosa, SciPy, PyWavelets, and Plotly.

## 🚀 What's New in Version 3.0
* **📄 Automated PDF Lab Reports:** Generate comprehensive scientific PDF reports containing audio metadata, time-domain metrics (RMS, ZCR, RT60), and frequency-domain metrics (Dominant Frequency, THD) with a single click.

## ✨ Highlights from Version 2.0
* **🎙️ Live Audio Recording:** Directly record your voice or ambient sounds from your browser.
* **🎵 Musical Note Transcription:** Dynamically translates fundamental frequencies (Hz) into exact musical notes (e.g., C4, G#).

## 🔬 Core Features

### 1. Audio Sources & Generation
* Upload standard audio files (WAV, MP3, FLAC, OGG, M4A) or record live audio.
* Generate pure physical waveforms (Sine, Square, Sawtooth) and custom complex waves using Fourier Synthesis.

### 2. Signal Processing & Analysis
* **Time Domain:** Interactive waveforms and Hilbert Transform for Amplitude Envelope extraction.
* **Frequency Domain:** FFT Spectrum with Total Harmonic Distortion (THD) calculation and Welch's Power Spectral Density (PSD).
* **Acoustics:** Estimate RT60 (Reverberation Time) using energy decay extrapolation.

### 3. Advanced Visualization
* 2D Spectrogram Heatmaps and 3D Cumulative Spectral Decay (Waterfall) plots.
* Continuous Wavelet Transform (CWT) power scalograms using Morlet, Mexican Hat, and Complex Morlet wavelets.

### 4. DSP & Physics Simulators
* **Wave Kinematics:** Simulate the Doppler effect and amplitude drop-off based on source velocity and distance.
* **Chaos Theory:** Reconstruct 2D and 3D Phase Space Attractors (Strange Attractors) from audio signals.
* **Filters:** Apply customizable IIR Butterworth filters (Low-Pass, High-Pass, Band-Pass, Band-Stop) and visualize Pole-Zero (Z-Plane) stability maps.
* **Time Delay:** Estimate simulated echo delays using Cross-Correlation and basic spectral gating noise reduction.

### 5. Data Export
* Export FFT spectrum and time-series audio features (RMS, Zero-Crossing Rate) as CSV files.
* Download processed/filtered audio as WAV files or generate a full PDF Lab Report.

