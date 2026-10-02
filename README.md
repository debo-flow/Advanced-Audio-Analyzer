# 🎧 Advanced Audio Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deb-audio-analyzer.streamlit.app/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, physics-based audio and digital signal processing (DSP) laboratory built entirely with Python and Streamlit. 

🔗 **Live Demo:** [Click here to open the App](https://deb-audio-analyzer.streamlit.app/)

---

### 🚀 What's New in Version 7.0 (Wave Interference Update)
* **Acoustic Beats Simulator:** Added a wave superposition engine to the Oscilloscope tab. Users can now mix two frequencies to hear and visualize the pulsating "beats" caused by constructive and destructive interference, perfectly demonstrating the physical math behind beat frequencies ($f_{beat} = \vert{}f_1 - f_2\vert{}$).

### ✨ Highlights from Previous Versions
* **v6.0 (Live Stream):** Real-Time WebRTC integration for continuous live microphone streaming and interactive oscilloscopes.
* **v5.0 (Oscilloscope):** Visualized Simple Harmonic Motion (SHM) intersections and phase shifts via Lissajous curves.
* **v4.0 (Psychoacoustics):** A-Weighted Power Spectra and Mel-Spectrograms mirroring human auditory perception.
* **v3.0 (PDF Reports):** One-click automated scientific PDF lab reports.

---

## 🔬 Core Features

### 1. Audio Sources & Generation
* Upload standard audio files (WAV, MP3, FLAC), record live audio, or generate pure physical waveforms (Sine, Square, Sawtooth, Fourier Synthesis).
* Real-Time Live Streaming via WebRTC.

### 2. Signal Processing & Acoustics
* **Time Domain:** Interactive waveforms, Hilbert Transform (Amplitude Envelope), and Instantaneous Frequency.
* **Frequency Domain:** FFT Spectrum, Welch's PSD, and Total Harmonic Distortion (THD).
* **Acoustics & Psychoacoustics:** RT60 (Reverberation Time) estimation, A-Weighting perceptual loudness, and Mel-Spectrograms.

### 3. Advanced Visualization & Dynamics
* **Spectrograms:** 2D Heatmaps, 3D Cumulative Spectral Decay (Waterfall), and Continuous Wavelet Transform (CWT) scalograms.
* **Chaos Theory:** Reconstruct 2D and 3D Phase Space Attractors (Strange Attractors) from audio signals.
* **Oscilloscope:** Lissajous curves via orthogonal SHM superposition.

### 4. Physics Simulators
* **Wave Interference:** Acoustic Beats simulator demonstrating constructive and destructive wave superposition.
* **Wave Kinematics:** Simulate the Doppler effect and inverse square law amplitude drop-off.
* **Filters & Echo:** Customizable IIR Butterworth filters with Pole-Zero (Z-Plane) stability maps, Cross-Correlation for echo delay estimation, and Spectral Gating noise reduction.

### 5. Data Export
* Download processed audio (Mono WAV), extracted tabular data (CSV), and comprehensive PDF Lab Reports.

---

## 💻 Technologies Used

* **Frontend/UI:** Streamlit
* **Live Streaming:** Streamlit-WebRTC, PyAV
* **Audio Processing:** Librosa, SoundFile, Audio-Recorder-Streamlit
* **Signal Processing (DSP):** SciPy, NumPy, PyWavelets
* **Data Visualization:** Plotly
* **Data Management & Export:** Pandas, FPDF

---

## 🛠️ How to Run Locally

Follow these steps to set up and run the application on your own machine (PC, Mac, or Termux on Android).

### 1. Clone the repository
`git clone https://github.com/debo-flow/Advanced-Audio-Analyzer.git`

### 2. Open the project folder
`cd Advanced-Audio-Analyzer`

### 3. Create a Virtual Environment
`python -m venv .venv`

### 4. Activate the Virtual Environment
* **Windows:** `.venv\Scripts\activate`
* **macOS/Linux/Android (Termux):** `source .venv/bin/activate`

### 5. Install Dependencies
`pip install -r requirements.txt`

### 6. Run the Application
`streamlit run app.py`

---

## 👨‍💻 Author

**Deborudra De**

*Built and engineered directly from a mobile development environment.* 📱⚛️🛡️

⭐ If you find this project interesting or helpful, consider giving it a star!
