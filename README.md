# Advanced Audio Analyzer

A comprehensive, physics-based audio and digital signal processing (DSP) laboratory built with Python and Streamlit.

This interactive web application allows users to upload audio files or generate pure waveforms to perform deep acoustical analysis and visualizations.

## Features

- **Audio Management:** Upload standard audio files (WAV, MP3, FLAC) or synthesize pure waves (Sine, Square, Sawtooth, Fourier).
- **Time-Domain Analysis:** Interactive waveforms, Hilbert Transform, and RT60 Reverb Time estimation.
- **Frequency Domain (FFT):** Power spectrum analysis, Total Harmonic Distortion (THD) calculation, and Welch's Power Spectral Density.
- **Spectrogram & 3D Waterfall:** Short-Time Fourier Transform (STFT) with 2D heatmap and 3D cumulative spectral decay visualizations.
- **Wavelet Transform (CWT):** High-resolution Continuous Wavelet Transform scalograms.
- **Pitch & Rhythm:** YIN-based pitch estimation and dynamic beat/tempo tracking.
- **Advanced Features:** Spectral Centroid, Rolloff, Flatness, and MFCC extraction.
- **Chaos Dynamics:** 2D and 3D Phase Space Trajectory reconstruction.
- **Kinematics & DSP:** Doppler Effect Simulator, IIR Butterworth Digital Filters, Spectral Gating, and Cross-Correlation.
- **Data Export:** Download processed audio (Mono WAV) and extracted tabular data (CSV).

## Technologies Used

- **Frontend/UI:** Streamlit
- **Audio Processing:** Librosa, SoundFile
- **Signal Processing (DSP):** SciPy, NumPy, PyWavelets
- **Data Visualization:** Plotly
- **Data Management:** Pandas

## How to Run Locally

Follow these steps to set up and run the application on your own machine.

### 1. Clone the repository

`git clone https://github.com/debo-flow/Advanced-Audio-Analyzer.git`

### 2. Open the folder

`cd Advanced-Audio-Analyzer`

### 3. Create a Virtual Environment

`python -m venv .venv`

### 4. Activate the Virtual Environment

Windows:

`.venv\Scripts\activate`

macOS/Linux:

`source .venv/bin/activate`

### 5. Install Dependencies

`pip install -r requirements.txt`

### 6. Run the Application

`streamlit run app.py`

## Project Structure

- `app.py` — Main application
- `requirements.txt` — Required Python packages
- `README.md` — Project documentation
- `.gitignore` — Ignored files

## Author

**Deborudra De**

⭐ If you find this project interesting, consider giving it a star!
