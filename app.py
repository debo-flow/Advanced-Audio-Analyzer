"""
Advanced Audio Analyzer
A Physics-Based Audio & Signal Processing Laboratory.
Built with Streamlit, Librosa, SciPy, PyWavelets, Plotly, Audio Recorder, and FPDF.
"""

import io
import os
import tempfile
import warnings
import queue
from typing import Optional, Tuple
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pywt
import scipy.fft
import scipy.signal
import soundfile as sf
import streamlit as st

# --- NEW: Live Streaming Library ---
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# --- NEW: Live Audio Recording Library ---
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None

# --- NEW: PDF Report Generation Library ---
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Advanced Audio Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM HASHING FUNCTION FOR STREAMLIT CACHE
# ==========================================
def fast_hash_np(x: np.ndarray):
    """Generates a fast hash for numpy arrays to prevent stale data in cache."""
    if x.size == 0:
        return 0
    return hash((x.shape, x[0], np.sum(x[::100])))

FAST_NP_HASH = {np.ndarray: fast_hash_np}


# ==========================================
# CACHED CORE FUNCTIONS (Optimization)
# ==========================================
@st.cache_data(show_spinner=False)
def load_audio_file(
    file_bytes: bytes, file_extension: str
) -> Tuple[Optional[np.ndarray], Optional[int], str]:
  """Saves uploaded bytes to a temp file, loads with librosa, returns mono audio array and sample rate."""
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
  try:
    temp_file.write(file_bytes)
    temp_file.close()
    y, sr = librosa.load(temp_file.name, sr=None, mono=True)
    return y, sr, ""
  except Exception as e:
    return None, None, str(e)
  finally:
    os.unlink(temp_file.name)


@st.cache_data(hash_funcs=FAST_NP_HASH)
def downsample_array(
    array: np.ndarray, max_points: int = 10000
) -> np.ndarray:
  """Downsamples a 1D array for safe, fast Plotly rendering."""
  if len(array) <= max_points:
    return array
  factor = len(array) // max_points
  return array[::factor]


@st.cache_data(hash_funcs=FAST_NP_HASH)
def downsample_fft(
    freqs: np.ndarray, mags: np.ndarray, max_points: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
  """Downsamples frequency domain data using max pooling to preserve critical single-bin peaks."""
  if len(mags) <= max_points:
    return freqs, mags
  factor = len(mags) // max_points
  pad_size = (factor - len(mags) % factor) % factor
  mags_padded = np.pad(
      mags, (0, pad_size), mode="constant", constant_values=-np.inf
  )
  freqs_padded = np.pad(freqs, (0, pad_size), mode="edge")

  mags_pooled = mags_padded.reshape(-1, factor).max(axis=1)
  freqs_pooled = freqs_padded.reshape(-1, factor).mean(axis=1)

  return freqs_pooled, mags_pooled


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_fft(
    y: np.ndarray, sr: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes windowed FFT and returns frequencies, magnitude, and dB scale."""
  window = np.hanning(len(y))
  y_windowed = y * window
  fft_result = scipy.fft.rfft(y_windowed)
  freqs = scipy.fft.rfftfreq(len(y), 1 / sr)

  magnitude = np.abs(fft_result)
  magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

  return freqs, magnitude, magnitude_db


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_welch_psd(y: np.ndarray, sr: int, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
  """Computes Power Spectral Density using Welch's method for reduced variance."""
  actual_nperseg = min(nperseg, len(y))
  if actual_nperseg == 0:
      return np.array([]), np.array([])
      
  freqs, psd = scipy.signal.welch(y, sr, nperseg=actual_nperseg)
  psd_db = 10 * np.log10(psd + 1e-12)
  return freqs, psd_db


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_stft(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
  """Computes STFT and returns Magnitude in dB."""
  stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
  magnitude = np.abs(stft_result)
  return librosa.amplitude_to_db(magnitude, ref=np.max)


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_cwt(
    y: np.ndarray, sr: int, wavelet: str = "morl", num_scales: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes Continuous Wavelet Transform (CWT) and returns time, frequencies, and power spectrum."""
  scales = np.arange(1, num_scales + 1)
  coefficients, frequencies = pywt.cwt(
      y, scales, wavelet, sampling_period=1.0 / sr
  )
  power = np.abs(coefficients) ** 2
  time_axis = np.linspace(0, len(y) / sr, len(y))

  return time_axis, frequencies, power


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_hilbert(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Applies Hilbert Transform to extract amplitude envelope and instantaneous frequency."""
  analytic_signal = scipy.signal.hilbert(y)
  amplitude_envelope = np.abs(analytic_signal)
  instantaneous_phase = np.unwrap(np.angle(analytic_signal))
  instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * sr)
  instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])
  time_axis = np.arange(len(y)) / sr
  
  return time_axis, y, amplitude_envelope, instantaneous_frequency


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_filter(
    y: np.ndarray,
    sr: int,
    filter_type: str,
    cutoff: float,
    order: int,
    cutoff2: float = None,
) -> np.ndarray:
  """Applies a Butterworth filter to the audio signal."""
  nyquist = 0.5 * sr

  if filter_type in ["Low-Pass", "High-Pass"]:
    normal_cutoff = cutoff / nyquist
    btype = "low" if filter_type == "Low-Pass" else "high"
    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False)
  else:
    normal_cutoff1 = cutoff / nyquist
    normal_cutoff2 = cutoff2 / nyquist
    btype = "bandpass" if filter_type == "Band-Pass" else "bandstop"
    b, a = scipy.signal.butter(
        order, [normal_cutoff1, normal_cutoff2], btype=btype, analog=False
    )

  filtered_y = scipy.signal.filtfilt(b, a, y)
  return filtered_y

@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_filter_zpk(
    sr: int, filter_type: str, cutoff: float, order: int, cutoff2: float = None
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes Zeros and Poles (Z-Plane) for the selected digital filter."""
  nyquist = 0.5 * sr
  if filter_type in ["Low-Pass", "High-Pass"]:
    normal_cutoff = cutoff / nyquist
    btype = "low" if filter_type == "Low-Pass" else "high"
    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False)
  else:
    normal_cutoff1 = cutoff / nyquist
    normal_cutoff2 = cutoff2 / nyquist
    btype = "bandpass" if filter_type == "Band-Pass" else "bandstop"
    b, a = scipy.signal.butter(
        order, [normal_cutoff1, normal_cutoff2], btype=btype, analog=False
    )
  z, p, _ = scipy.signal.tf2zpk(b, a)
  return z, p


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_cross_correlation(y: np.ndarray, sr: int, delay_sec: float, noise_lvl: float) -> Tuple[np.ndarray, np.ndarray, float]:
  """Simulates an echo with noise, and uses Cross-Correlation to estimate the delay."""
  limit = sr * 5  
  y_sub = y[:limit] if len(y) > limit else y
  
  delay_samples = int(delay_sec * sr)
  y_delayed = np.pad(y_sub, (delay_samples, 0), mode='constant')[:len(y_sub)]
  y_delayed += noise_lvl * np.random.randn(len(y_delayed))
  
  correlation = scipy.signal.correlate(y_delayed, y_sub, mode='full')
  lags = np.arange(-len(y_sub) + 1, len(y_delayed)) / sr
  
  peak_idx = np.argmax(correlation)
  estimated_delay_sec = lags[peak_idx]
  
  return lags, correlation, estimated_delay_sec


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_spectral_gating(y: np.ndarray, threshold_db: float) -> np.ndarray:
  """Basic noise reduction using spectral gating."""
  S = librosa.stft(y)
  S_mag, S_phase = librosa.magphase(S)
  S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

  mask = (S_db > threshold_db).astype(float)

  S_clean = S_mag * mask * S_phase
  y_clean = librosa.istft(S_clean)
  return y_clean


@st.cache_data(hash_funcs=FAST_NP_HASH)
def estimate_rt60(
    y: np.ndarray, sr: int
) -> Tuple[
    float,
    Optional[np.ndarray],
    Optional[np.ndarray],
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
  """Estimates RT60 using linear regression on the energy decay curve."""
  rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
  times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
  rms_db = librosa.amplitude_to_db(rms, ref=np.max)

  peak_idx = np.argmax(rms_db)
  decay_db = rms_db[peak_idx:]
  decay_times = times[peak_idx:]

  if len(decay_db) < 10:
    return 0.0, None, None, 0.0, 0.0, decay_times, decay_db

  try:
    start_idx = np.where(decay_db <= -5)[0][0]
    end_idx = np.where(decay_db <= -25)[0][0]

    if start_idx >= end_idx:
      return 0.0, None, None, 0.0, 0.0, decay_times, decay_db

    region_db = decay_db[start_idx:end_idx]
    region_times = decay_times[start_idx:end_idx]

    slope, intercept = np.polyfit(region_times, region_db, 1)

    if slope >= 0:  
      return 0.0, None, None, 0.0, 0.0, decay_times, decay_db

    rt60 = -60.0 / slope

    if rt60 < 0 or rt60 > 20:
      return 0.0, None, None, 0.0, 0.0, decay_times, decay_db

    return rt60, region_times, region_db, slope, intercept, decay_times, decay_db
  except IndexError:
    return 0.0, None, None, 0.0, 0.0, decay_times, decay_db


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_doppler_effect(
    y: np.ndarray, sr: int, velocity_ms: float, closest_distance: float = 5.0, c: float = 343.0
) -> np.ndarray:
  """Simulates the Doppler effect and inverse square law amplitude dropoff."""
  velocity_ms = min(velocity_ms, c * 0.95)

  t_src = np.arange(len(y)) / sr
  t_mid = t_src[-1] / 2.0

  x_src = velocity_ms * (t_src - t_mid)
  r_src = np.sqrt(x_src**2 + closest_distance**2)

  t_obs = t_src + (r_src / c)
  t_obs_uniform = np.arange(0, np.max(t_obs), 1.0 / sr)

  y_obs = np.interp(t_obs_uniform, t_obs, y, left=0.0, right=0.0)
  r_obs_uniform = np.interp(t_obs_uniform, t_obs, r_src)
  amplitude_envelope = closest_distance / r_obs_uniform

  return y_obs * amplitude_envelope


# --- PSYCHOACOUSTICS FUNCTIONS (Version 4.0) ---
@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_mel_spec(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates the Mel-Spectrogram mimicking human ear frequency perception."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr)
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr/2.0)
    return times, mel_freqs, mel_db


# ==========================================
# PDF REPORT GENERATOR 
# ==========================================
def create_lab_report(file_name, sr, duration, num_samples, peak_amp, rms_mean, zcr_mean, rt60, dom_freq, thd):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Advanced Audio Analyzer - Lab Report", ln=True, align='C')
    pdf.ln(10)
    
    # 1. Audio File Information
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Audio File Information", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Filename: {file_name}", ln=True)
    pdf.cell(0, 8, f"Sample Rate: {sr} Hz", ln=True)
    pdf.cell(0, 8, f"Duration: {duration:.2f} seconds", ln=True)
    pdf.cell(0, 8, f"Total Samples: {num_samples:,}", ln=True)
    pdf.cell(0, 8, f"Peak Amplitude: {peak_amp:.4f}", ln=True)
    pdf.ln(5)
    
    # 2. Time Domain Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Time Domain Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Mean RMS Energy: {rms_mean:.4f}", ln=True)
    pdf.cell(0, 8, f"Mean Zero Crossing Rate: {zcr_mean:.4f}", ln=True)
    rt60_str = f"{rt60:.2f} seconds" if rt60 > 0 else "N/A (No clear decay tail found)"
    pdf.cell(0, 8, f"Estimated RT60 (Reverb Time): {rt60_str}", ln=True)
    pdf.ln(5)
    
    # 3. Frequency Domain Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Frequency Domain Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Dominant Frequency: {dom_freq:.2f} Hz", ln=True)
    pdf.cell(0, 8, f"Total Harmonic Distortion (THD): {thd:.2f}%", ln=True)
    pdf.ln(15)
    
    # Footer
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Generated by Advanced Audio Analyzer (v5.0)", ln=True, align='C')
    
    # Return as bytes
    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode('latin-1')
    return bytes(out)


# ==========================================
# UI COMPONENTS & MAIN LOGIC
# ==========================================
def main():
  # --- Sidebar ---
  with st.sidebar:
    st.title("🎧 Advanced Audio Analyzer")
    st.markdown("*Physics-Based Audio & Signal Processing Laboratory*")
    st.divider()

    # --- UPDATED: Added WebRTC option ---
    audio_source = st.radio(
        "Audio Source", ["Upload Audio File", "Record Live Audio", "Generate Pure Wave", "Real-Time WebRTC"]
    )

    file_bytes = None
    file_name = ""
    file_ext = ""

    if audio_source == "Upload Audio File":
      uploaded_file = st.file_uploader(
          "Upload Audio File", type=["wav", "mp3", "flac", "ogg", "m4a"]
      )
      if uploaded_file is not None:
          file_bytes = uploaded_file.read()
          file_ext = "." + uploaded_file.name.split(".")[-1].lower()
          file_name = uploaded_file.name

    elif audio_source == "Record Live Audio":
      if audio_recorder is None:
          st.error("Please install the required library: `pip install audio-recorder-streamlit`")
      else:
          st.markdown("### 🎙️ Live Recording")
          st.info("Click the microphone icon to start/stop recording.")
          recorded_audio = audio_recorder(text="", recording_color="#E91E63", neutral_color="#1DB954")
          if recorded_audio:
              file_bytes = recorded_audio
              file_ext = ".wav"
              file_name = "Live_Recording.wav"

    elif audio_source == "Generate Pure Wave":
      st.markdown("### 🌊 Wave Generator")
      wave_type = st.selectbox(
          "Waveform Type",
          ["Sine", "Square", "Sawtooth", "Fourier Synthesis"],
      )
      wave_freq = st.slider("Fundamental Freq (Hz)", 20.0, 2000.0, 440.0)
      wave_dur = st.slider("Duration (s)", 1.0, 10.0, 3.0)

      amp_1 = amp_2 = amp_3 = amp_4 = amp_5 = 0.0
      wave_amp = 0.5

      if wave_type == "Fourier Synthesis":
        st.markdown("#### Harmonic Amplitudes")
        st.caption("Build complex waves using the Principle of Superposition.")
        amp_1 = st.slider("Fundamental (f₀)", 0.0, 1.0, 1.0)
        amp_2 = st.slider("2nd Harmonic (2f₀)", 0.0, 1.0, 0.0)
        amp_3 = st.slider("3rd Harmonic (3f₀)", 0.0, 1.0, 0.0)
        amp_4 = st.slider("4th Harmonic (4f₀)", 0.0, 1.0, 0.0)
        amp_5 = st.slider("5th Harmonic (5f₀)", 0.0, 1.0, 0.0)
      else:
        wave_amp = st.slider("Amplitude", 0.0, 1.0, 0.5)

      with st.spinner("Synthesizing physical wave..."):
          sr_temp = 44100
          t = np.linspace(0, wave_dur, int(sr_temp * wave_dur), endpoint=False)
          
          if wave_type == "Sine":
              y_temp = wave_amp * np.sin(2 * np.pi * wave_freq * t)
          elif wave_type == "Square":
              y_temp = wave_amp * scipy.signal.square(2 * np.pi * wave_freq * t)
          elif wave_type == "Sawtooth":
              y_temp = wave_amp * scipy.signal.sawtooth(2 * np.pi * wave_freq * t)
          elif wave_type == "Fourier Synthesis":
              y_temp = (
                  amp_1 * np.sin(2 * np.pi * wave_freq * t)
                  + amp_2 * np.sin(2 * np.pi * (wave_freq * 2) * t)
                  + amp_3 * np.sin(2 * np.pi * (wave_freq * 3) * t)
                  + amp_4 * np.sin(2 * np.pi * (wave_freq * 4) * t)
                  + amp_5 * np.sin(2 * np.pi * (wave_freq * 5) * t)
              )
              max_val = np.max(np.abs(y_temp))
              if max_val > 1.0:
                  y_temp = y_temp / max_val
              elif max_val == 0.0:
                  y_temp = np.zeros_like(y_temp)
                  
          file_name = f"Generated_{wave_type.replace(' ', '_')}_{wave_freq}Hz.wav"
          file_ext = ".wav"
          
          buffer = io.BytesIO()
          sf.write(buffer, y_temp, sr_temp, format="WAV")
          file_bytes = buffer.getvalue()


    st.divider()
    st.markdown("### ⚙️ System Info")
    st.info(
        "Uses **Librosa** for extraction, **SciPy** for DSP, **PyWavelets** for"
        " CWT, and **Plotly** for interactive visualization."
    )

  # --- Main Content / Audio Loading ---
  
  # --- UPDATED: Live Animation Graph for WebRTC ---
  if audio_source == "Real-Time WebRTC":
      st.title("🔴 Live Real-Time Audio Streaming")
      st.write("Speak into your microphone. The audio frames are streaming to the server in real-time.")
      
      # Class to process audio frames instantly
      class AudioViewer(AudioProcessorBase):
          def __init__(self):
              self.audio_queue = queue.Queue(maxsize=10) # Store latest frames
              
          def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
              # Convert sound format to numpy array
              audio_data = frame.to_ndarray()
              if not self.audio_queue.full():
                  self.audio_queue.put(audio_data[0, :]) # Take channel 0 (Mono)
              return frame

      # Start connection
      webrtc_ctx = webrtc_streamer(
          key="live-audio-analyzer",
          mode=WebRtcMode.SENDONLY,
          audio_processor_factory=AudioViewer,
          media_stream_constraints={"audio": True, "video": False},
      )
      
      # Show live graph if playing
      if webrtc_ctx and webrtc_ctx.state.playing:
          st.success("🎙️ Microphone is LIVE! (Streaming active)")
          st.markdown("### 🌊 Live Oscilloscope (Waveform)")
          
          plot_spot = st.empty() # Create an empty container for the animation
          
          # Infinite loop to update the graph frame by frame
          while True:
              if webrtc_ctx.audio_processor:
                  try:
                      # Grab the latest chunk of audio
                      audio_chunk = webrtc_ctx.audio_processor.audio_queue.get(timeout=1.0)
                      
                      # Downsample slightly so browser doesn't freeze
                      plot_chunk = audio_chunk[::5] 
                      
                      # Draw the Graph
                      fig = go.Figure(go.Scatter(y=plot_chunk, mode='lines', line=dict(color='#1DB954', width=2)))
                      fig.update_layout(
                          margin=dict(l=20, r=20, t=20, b=20),
                          height=350,
                          yaxis=dict(range=[-32768, 32768], fixedrange=True, title="Amplitude (Int16)"),
                          xaxis=dict(showgrid=False, visible=False, fixedrange=True),
                          template="plotly_dark",
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)"
                      )
                      
                      # Overwrite the empty container with the new graph
                      plot_spot.plotly_chart(fig, use_container_width=True)
                      
                  except queue.Empty:
                      pass
      
      return # Stop execution here so it doesn't look for uploaded files
  
  if file_bytes is None:
      st.info(
          "👋 Welcome! Please upload an audio file, record your voice, or generate a"
          " wave to begin analysis."
      )
      return

  with st.spinner("Loading and decoding audio..."):
      y, sr, err = load_audio_file(file_bytes, file_ext)

  if err or y is None or len(y) == 0:
      st.error(f"Error loading audio: {err if err else 'Audio is empty.'}")
      return

  duration = librosa.get_duration(y=y, sr=sr)
  nyquist = sr / 2.0
  num_samples = len(y)

  # --- Application Tabs ---
  tabs = st.tabs([
      "📁 Audio",
      "🌊 Waveform",
      "⚡ FFT Spectrum",
      "🌈 Spectrogram",
      "📉 Wavelet CWT",
      "🎵 Pitch & Tempo",
      "🔬 Audio Features",
      "🌀 Chaos Dynamics",
      "🎛️ Kinematics & DSP",
      "🧠 Psychoacoustics",
      "📐 Oscilloscope & SHM",
      "🧩 3D Fourier Series",  # NEW VERSION 10 TAB
      "📊 Data Export",
  ])

  # ==========================================
  # TAB 1: Audio Metadata
  # ==========================================
  with tabs[0]:
    st.header("Audio File Information")
    st.audio(file_bytes)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filename", file_name)
    m2.metric("Sample Rate", f"{sr} Hz")
    m3.metric("Duration", f"{duration:.2f} s")
    m4.metric("Total Samples", f"{num_samples:,}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Nyquist Frequency", f"{nyquist:,.1f} Hz")
    m6.metric("File Size", f"{len(file_bytes) / (1024*1024):.2f} MB")
    m7.metric("Channels (Analyzed)", "1 (Mono)")

    peak_amp = np.max(np.abs(y))
    m8.metric("Peak Amplitude", f"{peak_amp:.4f}")

  # ==========================================
  # TAB 2: Waveform & Hilbert Transform
  # ==========================================
  with tabs[1]:
    st.header("Time Domain Analysis")
    st.markdown("Displays the instantaneous amplitude (sound pressure) over time.")

    with st.spinner("Generating Waveform..."):
      time_array = np.linspace(0, duration, num_samples)

      ds_points = 20000
      y_plot = downsample_array(y, ds_points)
      t_plot = downsample_array(time_array, ds_points)

      fig = go.Figure()
      fig.add_trace(
          go.Scatter(
              x=t_plot,
              y=y_plot,
              mode="lines",
              line=dict(color="#1DB954", width=1),
              name="Amplitude",
          )
      )
      fig.update_layout(
          title="Interactive Waveform",
          xaxis_title="Time (seconds)",
          yaxis_title="Amplitude",
          template="plotly_dark",
          hovermode="x",
      )
      st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Hilbert Transform (Analytic Signal & Envelope)"):
      st.markdown("Extracts the **Amplitude Envelope** and **Instantaneous Frequency** using the Analytic Signal.")
      st.warning("Note: Processing the Hilbert transform for very long files might take some time.")
      if st.button("Compute Hilbert Transform"):
        with st.spinner("Applying Hilbert Transform on the entire audio..."):
          h_time, h_y, h_env, h_inst_f = compute_hilbert(y, sr)
          
          ds_h = 5000
          ht_plot = downsample_array(h_time, ds_h)
          hy_plot = downsample_array(h_y, ds_h)
          henv_plot = downsample_array(h_env, ds_h)
          hinst_plot = downsample_array(h_inst_f, ds_h)
          
          fig_env = go.Figure()
          fig_env.add_trace(go.Scatter(x=ht_plot, y=hy_plot, mode='lines', line=dict(color='rgba(29, 185, 84, 0.4)', width=1), name='Original Signal'))
          fig_env.add_trace(go.Scatter(x=ht_plot, y=henv_plot, mode='lines', line=dict(color='#FF5722', width=2), name='Amplitude Envelope'))
          fig_env.update_layout(title="Signal & Amplitude Envelope", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark")
          st.plotly_chart(fig_env, use_container_width=True)
          
          fig_instf = go.Figure()
          threshold = np.max(h_env) * 0.05
          hinst_plot_masked = np.where(henv_plot > threshold, hinst_plot, np.nan)
          
          fig_instf.add_trace(go.Scatter(x=ht_plot, y=hinst_plot_masked, mode='markers', marker=dict(color='#00BCD4', size=2), name='Inst. Freq'))
          fig_instf.update_layout(title="Instantaneous Frequency (Filtered for noise)", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", template="plotly_dark")
          st.plotly_chart(fig_instf, use_container_width=True)

    with st.expander("Show Advanced Time-Domain Metrics & RT60 Acoustics"):
      col_t1, col_t2, col_t3, col_t4 = st.columns(4)
      rms = librosa.feature.rms(y=y)[0]
      zcr = librosa.feature.zero_crossing_rate(y)[0]

      col_t1.metric("Mean RMS Energy", f"{np.mean(rms):.4f}")
      col_t2.metric("Mean Zero Crossing Rate", f"{np.mean(zcr):.4f}")
      col_t3.metric(
          "Crest Factor",
          f"{peak_amp / np.maximum(1e-10, np.mean(rms)):.2f}",
      )

      with st.spinner("Estimating RT60..."):
        rt60, r_times, r_db, slope, intercept, d_times, d_db = estimate_rt60(
            y, sr
        )

      if rt60 > 0:
        col_t4.metric("Est. RT60 (Reverb Time)", f"{rt60:.2f} s")

        st.markdown("#### Energy Decay Curve & RT60 Extrapolation")
        st.caption(
            "Acoustic physics measures RT60 by tracking the logarithmic energy"
            " decay after a loud impulse."
        )

        fig_rt60 = go.Figure()

        plot_d_times = downsample_array(d_times, 2000)
        plot_d_db = downsample_array(d_db, 2000)

        fig_rt60.add_trace(
            go.Scatter(
                x=plot_d_times,
                y=plot_d_db,
                mode="lines",
                line=dict(color="#E91E63", width=2),
                name="Signal Decay (dB)",
            )
        )

        if r_times is not None:
          t_start = r_times[0]
          t_end = (-60 - intercept) / slope

          fit_times = [t_start, t_end]
          fit_db = [slope * t_start + intercept, -60]

          fig_rt60.add_trace(
              go.Scatter(
                  x=fit_times,
                  y=fit_db,
                  mode="lines",
                  line=dict(color="#00BCD4", width=2, dash="dash"),
                  name="RT60 Regression Fit",
              )
          )

        fig_rt60.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Energy (dB)",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_rt60, use_container_width=True)
      else:
        col_t4.metric("Est. RT60", "N/A")
        st.info(
            "Could not calculate RT60. This requires a sharp impulsive sound in"
            " a room with a clear decay tail."
        )

  # ==========================================
  # TAB 3: FFT Spectrum & Welch's Method
  # ==========================================
  with tabs[2]:
    st.header("Frequency Domain Analysis (FFT) & Harmonic Distortion")

    safe_max_freq = max(100, int(nyquist)) 
    max_freq = st.slider(
        "Maximum Frequency Display (Hz)",
        min_value=100,
        max_value=safe_max_freq,
        value=safe_max_freq,
        step=100,
    )

    with st.spinner("Computing FFT..."):
      freqs, mag, mag_db = compute_fft(y, sr)

      valid_idx = freqs <= max_freq
      f_plot, m_plot = downsample_fft(freqs[valid_idx], mag_db[valid_idx], 5000)

      fig_fft = go.Figure()
      fig_fft.add_trace(
          go.Scatter(
              x=f_plot,
              y=m_plot,
              mode="lines",
              line=dict(color="#FF5722", width=1),
              name="Magnitude (dB)",
          )
      )
      fig_fft.update_layout(
          title="Power Spectrum",
          xaxis_title="Frequency (Hz)",
          yaxis_title="Magnitude (dB)",
          template="plotly_dark",
          hovermode="x",
      )

      dom_idx = np.argmax(mag)
      dom_freq = freqs[dom_idx]

      for i in range(2, 6):
        harmonic_freq = dom_freq * i
        if harmonic_freq <= max_freq:
          fig_fft.add_vline(
              x=harmonic_freq,
              line_width=1.5,
              line_dash="dash",
              line_color="rgba(255, 255, 255, 0.5)",
              annotation_text=f"{i}f₀",
              annotation_position="top right",
              annotation_font=dict(color="rgba(255, 255, 255, 0.7)", size=10),
          )

      st.plotly_chart(fig_fft, use_container_width=True)

      harmonic_sq_sum = 0.0
      fund_mag = mag[dom_idx]

      for i in range(2, 11):
        h_freq = dom_freq * i
        if h_freq > freqs[-1]:
          break
        idx = np.argmin(np.abs(freqs - h_freq))
        window = mag[max(0, idx - 3) : min(len(mag), idx + 4)]
        h_mag = np.max(window) if len(window) > 0 else 0
        harmonic_sq_sum += h_mag**2

      thd = (
          (np.sqrt(harmonic_sq_sum) / fund_mag) * 100.0 if fund_mag > 0 else 0.0
      )

      col_f1, col_f2 = st.columns(2)
      col_f1.success(f"**Dominant Frequency:** {dom_freq:.2f} Hz")
      col_f2.info(f"**Total Harmonic Distortion (THD):** {thd:.2f}%")
                
    st.divider()
    st.subheader("Welch's Method (Power Spectral Density)")
    
    nperseg = st.selectbox("Window Size (nperseg)", [1024, 2048, 4096, 8192], index=2)
    if st.button("Compute PSD (Welch)"):
        with st.spinner("Computing Welch's PSD..."):
            w_freqs, w_psd_db = compute_welch_psd(y, sr, nperseg=nperseg)
            if len(w_freqs) > 0:
                valid_w_idx = w_freqs <= max_freq
                wf_plot, wpsd_plot = downsample_fft(w_freqs[valid_w_idx], w_psd_db[valid_w_idx], 5000)
                
                fig_welch = go.Figure()
                fig_welch.add_trace(go.Scatter(x=wf_plot, y=wpsd_plot, mode='lines', line=dict(color='#8BC34A', width=1.5), name="Welch PSD"))
                fig_welch.update_layout(title="Welch's Power Spectral Density", xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", template="plotly_dark")
                st.plotly_chart(fig_welch, use_container_width=True)
            else:
                st.error("Audio is too short for the selected window size.")

  # ==========================================
  # TAB 4: Spectrogram & 3D Waterfall
  # ==========================================
  with tabs[3]:
    st.header("Short-Time Fourier Transform (STFT) & 3D Waterfall")

    c1, c2, c3, c4 = st.columns(4)
    n_fft = c1.selectbox(
        "FFT Size (Resolution)", options=[512, 1024, 2048, 4096], index=2
    )
    hop_length = c2.selectbox(
        "Hop Length", options=[256, 512, 1024, 2048], index=1
    )
    y_scale = c3.radio("Frequency Scale", options=["Linear", "Logarithmic"])
    plot_type = c4.radio("Plot Dimension", options=["2D Heatmap", "3D Waterfall"])

    with st.spinner(f"Generating {plot_type}..."):
      S_db = compute_stft(y, n_fft, hop_length)

      max_time_bins = 400 if plot_type == "3D Waterfall" else 800
      max_freq_bins = 200 if plot_type == "3D Waterfall" else 400

      time_factor = max(1, S_db.shape[1] // max_time_bins)
      freq_factor = max(1, S_db.shape[0] // max_freq_bins)

      S_db_plot = S_db[::freq_factor, ::time_factor]

      t_axis = librosa.frames_to_time(
          np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length
      )[::time_factor]
      f_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[::freq_factor]

      if plot_type == "2D Heatmap":
        fig_stft = go.Figure(
            data=go.Heatmap(
                z=S_db_plot, x=t_axis, y=f_axis, colorscale="Inferno"
            )
        )

        if y_scale == "Logarithmic":
          fig_stft.update_layout(yaxis_type="log")

        fig_stft.update_layout(
            title="2D Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_dark",
        )
      else:
        fig_stft = go.Figure(
            data=[
                go.Surface(
                    z=S_db_plot, x=t_axis, y=f_axis, colorscale="Inferno"
                )
            ]
        )

        scene_dict = dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis_title="Magnitude (dB)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        )

        if y_scale == "Logarithmic":
          scene_dict["yaxis"] = dict(type="log", title="Frequency (Hz)")

        fig_stft.update_layout(
            title="3D Cumulative Spectral Decay (Waterfall)",
            scene=scene_dict,
            template="plotly_dark",
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

      st.plotly_chart(fig_stft, use_container_width=True)

  # ==========================================
  # TAB 5: Continuous Wavelet Transform (CWT)
  # ==========================================
  with tabs[4]:
    st.header("Continuous Wavelet Transform (CWT)")
    st.markdown(
        "CWT is computationally intensive. Select a specific time slice to analyze below to prevent browser crashes."
    )

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
      cwt_start = st.slider("Start Time (seconds)", 0.0, max(0.0, float(duration - 0.1)), 0.0)
    with col_c2:
      cwt_duration = st.slider("Duration to Analyze (seconds)", 0.1, 5.0, 1.0)
    with col_c3:
      wavelet_name = st.selectbox("Wavelet Family", ["morl", "mexh", "cmor"], index=0)
      
    num_scales = st.slider("Scale Resolution", min_value=16, max_value=128, value=64, step=16)

    if st.button("Compute CWT Spectrum"):
      with st.spinner("Calculating Continuous Wavelet Transform..."):
        
        start_sample = int(cwt_start * sr)
        end_sample = min(len(y), int((cwt_start + cwt_duration) * sr))
        y_cwt = y[start_sample:end_sample]

        if len(y_cwt) > 0:
            t_axis, f_axis, power_matrix = compute_cwt(
                y_cwt, sr, wavelet=wavelet_name, num_scales=num_scales
            )
            
            # Adjust time axis for correct display
            t_axis += cwt_start

            fig_cwt = go.Figure(
                data=go.Heatmap(
                    z=power_matrix,
                    x=t_axis,
                    y=f_axis,
                    colorscale="Viridis",
                )
            )

            fig_cwt.update_layout(
                title=f"CWT Power Scalogram ({wavelet_name} wavelet)",
                xaxis_title="Time (s)",
                yaxis_title="Pseudo-Frequency (Hz)",
                template="plotly_dark",
                height=500,
            )
            st.plotly_chart(fig_cwt, use_container_width=True)
        else:
            st.error("Invalid time selection.")

  # ==========================================
  # TAB 6: Pitch & Tempo
  # ==========================================
  with tabs[5]:
    st.header("Pitch & Rhythm Analysis")

    c_pitch, c_tempo = st.columns(2)

    with c_pitch:
      st.subheader("Pitch Estimation (YIN)")
      st.info("Calculates fundamental frequency ($f_0$) and translates to a Musical Note.")
      
      if st.button("Estimate Pitch"):
        with st.spinner("Calculating pitch over the entire track (this may take time for long audio)..."):
          # Fixed Target SR dynamically instead of hardcoding 11025
          target_sr = min(sr, 11025)
          y_pitch_resampled = librosa.resample(
              y, orig_sr=sr, target_sr=target_sr
          )

          f0, voiced_flag, voiced_probs = librosa.pyin(
              y_pitch_resampled,
              fmin=librosa.note_to_hz("C2"),
              fmax=librosa.note_to_hz("C7"),
              sr=target_sr,
          )
          times = librosa.times_like(f0, sr=target_sr)

          f0_voiced = np.where(voiced_flag, f0, np.nan)
          valid_f0 = f0_voiced[~np.isnan(f0_voiced)]

          if len(valid_f0) > 0:
            median_pitch = np.median(valid_f0)
            
            musical_note = librosa.hz_to_note(median_pitch)
            
            st.metric("Median Pitch", f"{median_pitch:.1f} Hz", f"Note: {musical_note}")

            fig_pitch = go.Figure()
            fig_pitch.add_trace(
                go.Scatter(
                    x=times,
                    y=f0_voiced,
                    mode="markers",
                    marker=dict(size=3, color="#00BCD4"),
                )
            )
            fig_pitch.update_layout(
                title="Pitch Tracking",
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_pitch, use_container_width=True)
          else:
            st.warning("No clear pitch detected (audio might be unvoiced/noise).")

    with c_tempo:
      st.subheader("Tempo & Beat Tracking")
      if st.button("Estimate Tempo"):
        with st.spinner("Tracking beats..."):
          tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
          beat_times = librosa.frames_to_time(beat_frames, sr=sr)

          tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
          st.metric("Estimated Tempo", f"{tempo_val:.1f} BPM")

          fig_beats = go.Figure()
          fig_beats.add_trace(
              go.Scatter(
                  x=beat_times,
                  y=np.ones_like(beat_times),
                  mode="markers",
                  marker=dict(
                      symbol="line-ns",
                      size=30,
                      color="#FFC107",
                      line=dict(width=2),
                  ),
              )
          )
          fig_beats.update_layout(
              title="Detected Beats",
              xaxis_title="Time (s)",
              yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
              template="plotly_dark",
              height=300,
          )
          st.plotly_chart(fig_beats, use_container_width=True)

  # ==========================================
  # TAB 7: Advanced Audio Features
  # ==========================================
  with tabs[6]:
    st.header("Timbral & Spectral Features")

    with st.spinner("Extracting features..."):
      spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
      spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
      spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

      frames = range(len(spectral_centroids))
      t_features = librosa.frames_to_time(frames, sr=sr)

      col_f1, col_f2 = st.columns(2)

      with col_f1:
        st.markdown("**Spectral Centroid**")
        fig_cent = px.line(x=t_features, y=spectral_centroids, template="plotly_dark")
        fig_cent.update_layout(xaxis_title="Time (s)", yaxis_title="Hz")
        st.plotly_chart(fig_cent, use_container_width=True)

      with col_f2:
        st.markdown("**Spectral Rolloff**")
        fig_roll = px.line(
            x=t_features,
            y=spectral_rolloff,
            template="plotly_dark",
            color_discrete_sequence=["#E91E63"],
        )
        fig_roll.update_layout(xaxis_title="Time (s)", yaxis_title="Hz")
        st.plotly_chart(fig_roll, use_container_width=True)

      st.markdown("**Spectral Flatness (Wiener Entropy)**")
      fig_flat = px.line(
          x=t_features,
          y=spectral_flatness,
          template="plotly_dark",
          color_discrete_sequence=["#00BCD4"],
      )
      fig_flat.update_layout(
          xaxis_title="Time (s)", yaxis_title="Flatness Ratio (0 to 1)"
      )
      st.plotly_chart(fig_flat, use_container_width=True)

    st.subheader("Mel-Frequency Cepstral Coefficients (MFCC)")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig_mfcc = go.Figure(
        data=go.Heatmap(
            z=mfccs, x=t_features, y=np.arange(1, 14), colorscale="Viridis"
        )
    )
    fig_mfcc.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="MFCC Coefficient",
        template="plotly_dark",
    )
    st.plotly_chart(fig_mfcc, use_container_width=True)

  # ==========================================
  # TAB 8: Chaos Theory & Non-Linear Dynamics
  # ==========================================
  with tabs[7]:
    st.header("Phase Space Trajectory (Attractor Reconstruction)")
    
    c_chaos1, c_chaos2 = st.columns([1, 2])

    with c_chaos1:
      st.markdown("#### Embedding Parameters")
      tau = st.slider(
          "Time Delay (τ) in samples", min_value=1, max_value=500, value=25, step=1
      )
      plot_dim = st.radio("Plot Dimension", ["2D Phase Space", "3D Phase Space"])
      max_pts = st.slider("Samples to Plot", 1000, 20000, 5000, step=1000)

    with c_chaos2:
      with st.spinner("Reconstructing Attractor..."):
        y_plot = y[:max_pts]

        if plot_dim == "2D Phase Space":
          y_t = y_plot[:-tau]
          y_t_tau = y_plot[tau:]

          fig_phase = go.Figure(
              go.Scattergl(
                  x=y_t,
                  y=y_t_tau,
                  mode="markers+lines",
                  marker=dict(
                      size=2,
                      color=np.arange(len(y_t)),
                      colorscale="Viridis",
                      showscale=False,
                  ),
                  line=dict(color="rgba(255,255,255,0.2)", width=1),
              )
          )
          fig_phase.update_layout(
              title=f"2D Phase Space (Delay τ = {tau})",
              xaxis_title="y(t)",
              yaxis_title="y(t + τ)",
              template="plotly_dark",
              height=500,
              width=500,
              yaxis=dict(scaleanchor="x", scaleratio=1),
          )
          st.plotly_chart(fig_phase, use_container_width=True)

        else:
          if len(y_plot) > 2 * tau:
            y_t = y_plot[: -2 * tau]
            y_t_tau1 = y_plot[tau:-tau]
            y_t_tau2 = y_plot[2 * tau :]

            fig_phase3d = go.Figure(
                go.Scatter3d(
                    x=y_t,
                    y=y_t_tau1,
                    z=y_t_tau2,
                    mode="lines",
                    line=dict(
                        color=np.arange(len(y_t)), colorscale="Plasma", width=3
                    ),
                )
            )

            fig_phase3d.update_layout(
                title="3D Phase Space Strange Attractor",
                scene=dict(
                    xaxis_title="y(t)",
                    yaxis_title="y(t + τ)",
                    zaxis_title="y(t + 2τ)",
                ),
                template="plotly_dark",
                height=600,
            )
            st.plotly_chart(fig_phase3d, use_container_width=True)

  # ==========================================
  # TAB 9: Kinematics & DSP
  # ==========================================
  with tabs[8]:
    st.header("Kinematics & Digital Signal Processing (DSP)")

    # --- 1. Thermodynamics & Speed of Sound ---
    st.subheader("1. Thermodynamics & Medium Kinetics")
    st.markdown("The speed of sound ($c$) is not constant. It depends heavily on the medium's density, elasticity, and in gases, the absolute temperature.")
    
    col_therm1, col_therm2, col_therm3 = st.columns(3)
    medium = col_therm1.selectbox("Acoustic Medium", ["Air (Ideal Gas)", "Water (Liquid)", "Seawater", "Steel (Solid)", "Helium (Gas)"])
    
    if medium == "Air (Ideal Gas)":
        temp_c = col_therm2.slider("Temperature (°C)", -50.0, 100.0, 20.0, 0.5)
        # Thermodynamic speed of sound formula
        c_speed = 331.3 * np.sqrt(1 + temp_c / 273.15)
        col_therm3.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
        st.caption(f"💡 Calculated using the thermodynamic formula for air: $c = 331.3 \sqrt{{1 + \\frac{{T}}{{273.15}}}}$")
    elif medium == "Water (Liquid)":
        c_speed = 1480.0
        col_therm2.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
        st.caption("💡 Sound travels roughly 4.3 times faster in water than in air.")
    elif medium == "Seawater":
        c_speed = 1530.0
        col_therm2.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
    elif medium == "Steel (Solid)":
        c_speed = 5100.0
        col_therm2.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
        st.caption("💡 In highly elastic solids like steel, sound travels extremely fast.")
    else: # Helium
        c_speed = 972.0
        col_therm2.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
        st.caption("💡 Helium is much less dense than air, so sound travels almost 3 times faster (which is why inhaling it makes your voice squeaky!).")

    st.divider()

    # --- 2. Doppler Simulator ---
    st.subheader("2. Wave Kinematics: Doppler Effect Simulator")
    st.markdown(f"Uses the calculated speed of sound (**{c_speed:.2f} m/s**) to simulate frequency shifts and distance attenuation.")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
      velocity_ms = st.slider("Source Velocity (m/s)", 5.0, 150.0, 30.0)
    with col_d2:
      closest_dist = st.slider("Closest Distance (meters)", 1.0, 50.0, 5.0)

    if st.button("Simulate Doppler Pass-by"):
      with st.spinner("Calculating time dilations and signal attenuation..."):
        y_doppler = apply_doppler_effect(y, sr, velocity_ms, closest_dist, c_speed)
        st.success("Kinematic simulation complete!")

        buffer_doppler = io.BytesIO()
        sf.write(buffer_doppler, y_doppler, sr, format="WAV")
        st.audio(buffer_doppler.getvalue(), format="audio/wav")

        time_dop = np.linspace(0, len(y_doppler) / sr, len(y_doppler))
        fig_dop = go.Figure(
            go.Scatter(
                x=downsample_array(time_dop, 5000),
                y=downsample_array(y_doppler, 5000),
                line=dict(color="#00BCD4", width=1),
            )
        )
        fig_dop.update_layout(
            title=f"Doppler Waveform (v = {velocity_ms} m/s)",
            xaxis_title="Time (s)",
            yaxis_title="Pressure Amplitude",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_dop, use_container_width=True)

    st.divider()

    # --- 3. Filters ---
    st.subheader("3. IIR Butterworth Filters")

    filt_type = st.selectbox(
        "Filter Type", ["Low-Pass", "High-Pass", "Band-Pass", "Band-Stop"]
    )
    filt_order = st.slider("Filter Order", min_value=1, max_value=10, value=4)

    cutoff1, cutoff2 = 1000.0, 3000.0

    if filt_type in ["Low-Pass", "High-Pass"]:
      default_cutoff = min(1000.0, float(nyquist - 1))
      cutoff1 = st.slider(
          "Cutoff Frequency (Hz)", 20.0, float(nyquist - 1), default_cutoff
      )
    else:
      default_high = min(3000.0, float(nyquist - 1))
      default_low = min(500.0, default_high - 1.0)
      c1, c2 = st.slider(
          "Frequency Band (Hz)",
          20.0,
          float(nyquist - 1),
          (default_low, default_high),
      )
      cutoff1, cutoff2 = c1, c2

    if st.button("Apply Filter"):
      with st.spinner("Filtering audio..."):
        try:
          y_filt = apply_filter(
              y, sr, filt_type, cutoff1, filt_order, cutoff2
          )
          st.success("Filter applied successfully!")

          buffer = io.BytesIO()
          sf.write(buffer, y_filt, sr, format="WAV")
          st.audio(buffer.getvalue(), format="audio/wav")

          fig_comp = go.Figure()
          fig_comp.add_trace(go.Scatter(y=y[:1000], name="Original", opacity=0.5))
          fig_comp.add_trace(
              go.Scatter(y=y_filt[:1000], name="Filtered", opacity=0.8)
          )
          fig_comp.update_layout(template="plotly_dark")
          st.plotly_chart(fig_comp, use_container_width=True)

        except ValueError as e:
          st.error(
              f"Filter Error: {e}. Try adjusting your cutoff frequencies."
          )

    # --- Pole-Zero Map Visualization ---
    show_pz = st.checkbox("Show Pole-Zero Map (Z-Plane) for this filter")
    if show_pz:
        with st.spinner("Calculating Poles and Zeros..."):
            z, p = compute_filter_zpk(sr, filt_type, cutoff1, filt_order, cutoff2)
            fig_pz = go.Figure()
            
            theta = np.linspace(0, 2*np.pi, 200)
            fig_pz.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='Unit Circle'))
            fig_pz.add_trace(go.Scatter(x=np.real(z), y=np.imag(z), mode='markers', marker=dict(symbol='circle-open', size=10, color='#00BCD4', line=dict(width=2)), name='Zeros'))
            fig_pz.add_trace(go.Scatter(x=np.real(p), y=np.imag(p), mode='markers', marker=dict(symbol='x', size=10, color='#E91E63'), name='Poles'))
            
            fig_pz.update_layout(
                title="Filter Stability: Pole-Zero Map", 
                xaxis_title="Real", 
                yaxis_title="Imaginary", 
                width=500, height=500, 
                yaxis=dict(scaleanchor="x", scaleratio=1), 
                template="plotly_dark"
            )
            fig_pz.update_xaxes(range=[-1.5, 1.5])
            fig_pz.update_yaxes(range=[-1.5, 1.5])
            st.plotly_chart(fig_pz, use_container_width=True)

    st.divider()
    st.subheader("4. Experimental Noise Reduction")

    thresh = st.slider("Noise Threshold (dB)", -80.0, 0.0, -40.0)
    if st.button("Apply Noise Reduction"):
      with st.spinner("Applying Spectral Gating..."):
        y_clean = apply_spectral_gating(y, thresh)
        st.success("Noise reduction applied!")

        buffer_nr = io.BytesIO()
        sf.write(buffer_nr, y_clean, sr, format="WAV")
        st.audio(buffer_nr.getvalue(), format="audio/wav")

    st.divider()
    
    # --- Cross-Correlation (Echo/Time Delay Estimation) ---
    st.subheader("5. Echolocation & SONAR (Cross-Correlation)")
    st.markdown(f"Uses Cross-Correlation to find the echo delay, then calculates the object's distance using the medium's sound speed (**{c_speed:.2f} m/s**).")
    
    c_delay, c_noise = st.columns(2)
    true_delay = c_delay.slider("Simulated Echo Delay (seconds)", 0.01, 1.0, 0.25, 0.01)
    noise_lvl = c_noise.slider("Simulation Noise Level", 0.0, 1.0, 0.1, 0.1)
    
    if st.button("Run Echolocation Radar"):
        with st.spinner("Correlating signals and calculating distance..."):
            lags, corr, est_delay = compute_cross_correlation(y, sr, true_delay, noise_lvl)
            
            # Physics Math: Distance = (Time * Speed) / 2
            est_distance = (est_delay * c_speed) / 2.0
            
            st.success(f"⏱️ **Detected Echo Delay:** {est_delay:.4f} seconds")
            st.info(f"📏 **Calculated Object Distance:** {est_distance:.2f} meters (using $d = \\frac{{t \\times c}}{{2}}$)")
            
            fig_xcorr = go.Figure()
            ds_x = 5000
            lags_plot = downsample_array(lags, ds_x)
            corr_plot = downsample_array(corr, ds_x)
            
            fig_xcorr.add_trace(go.Scatter(x=lags_plot, y=corr_plot, mode='lines', line=dict(color='#FFC107', width=1), name="Cross-Correlation"))
            fig_xcorr.add_vline(x=est_delay, line=dict(color='red', width=2, dash='dash'), annotation_text="Detected Peak")
            fig_xcorr.update_layout(title="Radar Cross-Correlation vs Time Lag", xaxis_title="Time Lag (s)", yaxis_title="Correlation Amplitude", template="plotly_dark")
            st.plotly_chart(fig_xcorr, use_container_width=True)

  # ==========================================
  # TAB 10: Psychoacoustics & Human Hearing
  # ==========================================
  with tabs[9]:
    st.header("Psychoacoustics & Human Hearing")
    st.markdown("Analyzes sound based on how the **human auditory system** perceives it, rather than pure physical metrics.")
    
    c_psyc1, c_psyc2 = st.columns(2)
    
    with c_psyc1:
      st.subheader("1. A-Weighted Power Spectrum")
      st.markdown("The human ear is less sensitive to very low and very high frequencies. **A-Weighting** mimics this non-linear perception.")
      
      with st.spinner("Applying perceptual weighting..."):
        freqs_p, _, mag_db_p = compute_fft(y, sr)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_weights = librosa.A_weighting(freqs_p)
            
        perceived_db = mag_db_p + a_weights
        
        valid_idx = freqs_p <= (sr / 2.0)
        fp_plot = downsample_array(freqs_p[valid_idx], 5000)
        mag_plot = downsample_array(mag_db_p[valid_idx], 5000)
        perc_plot = downsample_array(perceived_db[valid_idx], 5000)
        
        fig_aw = go.Figure()
        fig_aw.add_trace(go.Scatter(x=fp_plot, y=mag_plot, mode="lines", name="Physical (Linear)", line=dict(color="rgba(255, 255, 255, 0.4)", width=1.5)))
        fig_aw.add_trace(go.Scatter(x=fp_plot, y=perc_plot, mode="lines", name="Perceived (A-Weighted)", line=dict(color="#00BCD4", width=2)))
        
        fig_aw.update_layout(
            title="Physical vs Perceived Loudness",
            xaxis_title="Frequency (Hz) - Log Scale",
            yaxis_title="Magnitude (dB / dBA)",
            template="plotly_dark",
            xaxis_type="log",
            hovermode="x",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_aw, use_container_width=True)
        
    with c_psyc2:
      st.subheader("2. Mel-Spectrogram")
      st.markdown("The **Mel Scale** maps frequencies such that equal distances sound equally distant to the human ear.")
      
      with st.spinner("Generating Mel-Spectrogram..."):
        mel_times, mel_freqs, mel_db = compute_mel_spec(y, sr)
        
        time_factor = max(1, len(mel_times) // 400)
        mel_db_plot = mel_db[:, ::time_factor]
        mel_times_plot = mel_times[::time_factor]
        
        fig_mel = go.Figure(data=go.Heatmap(
            z=mel_db_plot, 
            x=mel_times_plot, 
            y=mel_freqs, 
            colorscale="Magma"
        ))
        fig_mel.update_layout(
            title="Perceptual Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Mel Frequency (Hz)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_mel, use_container_width=True)

  # ==========================================
  # TAB 11: Lissajous & Simple Harmonic Motion 
  # ==========================================
  with tabs[10]:
    st.header("Oscilloscope: Lissajous Figures & Damped Harmonic Motion")
    st.markdown("Visualizes the complex harmonic motion created by plotting two orthogonal sine waves (X and Y axes) against each other, now featuring **Exponential Damping** (Friction).")
    
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    freq_x = col_l1.slider("X-Axis Frequency ($f_x$) in Hz", 1.0, 1000.0, 440.0, 1.0)
    freq_y = col_l2.slider("Y-Axis Frequency ($f_y$) in Hz", 1.0, 1000.0, 440.0, 1.0)
    phase_delta = col_l3.slider("Phase Difference ($\delta$)", 0, 360, 90)
    damping = col_l4.slider("Damping Friction ($\gamma$)", 0.0, 2.0, 0.0, 0.1)

    with st.spinner("Generating Lissajous Curve..."):
        # Determine appropriate time range to show full patterns
        t_end = max(1.0/freq_x, 1.0/freq_y) * 20  # plotting 20 cycles for detailed curves
        t_lin = np.linspace(0, t_end, 5000)

        phase_rad = np.deg2rad(phase_delta)
        
        # Exponential Decay Envelope (Normalized time for consistent visual decay)
        decay_time = np.linspace(0, 5, 5000)
        decay = np.exp(-damping * decay_time)

        x_sig = decay * np.sin(2 * np.pi * freq_x * t_lin + phase_rad)
        y_sig = decay * np.sin(2 * np.pi * freq_y * t_lin)

        fig_liss = go.Figure(go.Scattergl(
            x=x_sig, y=y_sig, 
            mode='lines', 
            line=dict(color='#E91E63', width=1.5)
        ))
        
        fig_liss.update_layout(
            title=f"Lissajous Curve (Ratio {freq_x}:{freq_y}) | Damping: {damping}",
            xaxis_title="Amplitude (X-Axis)",
            yaxis_title="Amplitude (Y-Axis)",
            template="plotly_dark",
            width=600, height=600,
            yaxis=dict(scaleanchor="x", scaleratio=1), # Keeps it perfectly square
            xaxis=dict(range=[-1.1, 1.1]),
        )
        
        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(fig_liss, use_container_width=True)
            
        with col_info:
            st.info("💡 **Physics Insight:**")
            st.markdown(
                """
                A Lissajous figure is produced by the intersection of two Simple Harmonic Motions (SHM) at right angles. With damping, energy is lost over time:
                
                $x(t) = A e^{-\gamma t} \sin(2\pi f_x t + \delta)$
                
                $y(t) = B e^{-\gamma t} \sin(2\pi f_y t)$
                
                * **Damping ($\gamma = 0$):** Perfect SHM, eternal oscillation. Creates static circles or figures-of-eight.
                * **Damping ($\gamma > 0$):** Simulates real-world friction/air resistance. The trajectory spirals inward as kinetic energy decays exponentially!
                """
            )
            
    st.divider()
    st.subheader("Wave Interference: Acoustic Beats Simulator")
    st.markdown("When two sound waves of slightly different frequencies interfere, they produce a pulsating sound where the volume periodically increases and decreases. This physical phenomenon is known as a **Beat**.")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    f1 = col_b1.slider("Frequency 1 ($f_1$) Hz", 200.0, 1000.0, 440.0, 1.0)
    f2 = col_b2.slider("Frequency 2 ($f_2$) Hz", 200.0, 1000.0, 444.0, 1.0)
    beat_duration = col_b3.slider("Simulation Duration (s)", 1.0, 5.0, 3.0, 0.5)
    
    if st.button("Simulate Acoustic Beats"):
        with st.spinner("Calculating wave superposition..."):
            sr_beat = 44100
            t_beat = np.linspace(0, beat_duration, int(sr_beat * beat_duration), endpoint=False)
            
            # Superposition of two waves
            y1 = 0.5 * np.sin(2 * np.pi * f1 * t_beat)
            y2 = 0.5 * np.sin(2 * np.pi * f2 * t_beat)
            y_beat = y1 + y2
            
            # Audio playback
            buffer_beat = io.BytesIO()
            sf.write(buffer_beat, y_beat, sr_beat, format="WAV")
            st.audio(buffer_beat.getvalue(), format="audio/wav")
            
            # Calculate and display Beat Frequency
            beat_freq = abs(f1 - f2)
            st.success(f"**Beat Frequency ($f_{{beat}}$):** {beat_freq:.1f} Hz (You should hear exactly {beat_freq:.1f} volume pulses per second)")
            
            # Plotly Graph - Zoomed in to show the beat envelope clearly
            plot_limit = min(len(t_beat), int(sr_beat * (4.0 / beat_freq if beat_freq > 0 else 0.1))) 
            
            fig_beat = go.Figure()
            fig_beat.add_trace(go.Scatter(
                x=t_beat[:plot_limit:5], 
                y=y_beat[:plot_limit:5], 
                mode='lines', 
                line=dict(color='#00BCD4', width=1)
            ))
            fig_beat.update_layout(
                title=f"Superposition Envelope (Interference of {f1} Hz & {f2} Hz)",
                xaxis_title="Time (s)",
                yaxis_title="Pressure Amplitude",
                template="plotly_dark",
                height=350,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_beat, use_container_width=True)

  # ==========================================
  # NEW TAB 12: 3D Fourier Series Deconstruction (Version 10.0)
  # ==========================================
  with tabs[11]:
    st.header("3D Fourier Series Deconstruction")
    st.markdown("Joseph Fourier proved that any complex periodic waveform can be mathematically deconstructed into a sum of simple sine waves (harmonics). This simulator visually breaks apart a complex wave in 3D space to reveal its fundamental components.")

    col_f1, col_f2, col_f3 = st.columns(3)
    complex_wave_type = col_f1.selectbox("Complex Wave Type", ["Square Wave", "Sawtooth Wave"])
    fund_freq = col_f2.slider("Fundamental Frequency (Hz)", 1.0, 50.0, 5.0, 1.0)
    num_harmonics = col_f3.slider("Number of Harmonics (N)", 1, 50, 10)

    if st.button("Deconstruct Wave to 3D Harmonics"):
        with st.spinner("Calculating 3D Fourier Harmonics..."):
            t_fourier = np.linspace(0, 2.0/fund_freq, 1000) # Show 2 cycles
            
            fig_fourier = go.Figure()
            summed_wave = np.zeros_like(t_fourier)

            for n in range(1, num_harmonics + 1):
                if complex_wave_type == "Square Wave":
                    if n % 2 == 0:
                        continue # Square wave only has odd harmonics
                    amplitude = (4.0 / np.pi) * (1.0 / n)
                else: # Sawtooth wave
                    amplitude = (2.0 / np.pi) * ((-1.0)**(n+1) / n)

                harmonic_wave = amplitude * np.sin(2 * np.pi * (n * fund_freq) * t_fourier)
                summed_wave += harmonic_wave

                # Add individual harmonic in 3D (X=Time, Y=Harmonic Number, Z=Amplitude)
                fig_fourier.add_trace(go.Scatter3d(
                    x=t_fourier,
                    y=[n]*len(t_fourier),
                    z=harmonic_wave,
                    mode='lines',
                    name=f"Harmonic {n}",
                    line=dict(color=px.colors.sequential.Plasma[n % len(px.colors.sequential.Plasma)], width=3)
                ))

            # Add the resultant complex wave at the front (Y=0)
            fig_fourier.add_trace(go.Scatter3d(
                x=t_fourier,
                y=[0]*len(t_fourier),
                z=summed_wave,
                mode='lines',
                name="Resultant Wave",
                line=dict(color='#1DB954', width=6)
            ))

            fig_fourier.update_layout(
                title=f"3D Fourier Deconstruction ({complex_wave_type}, N={num_harmonics})",
                scene=dict(
                    xaxis_title="Time (s)",
                    yaxis_title="Harmonic Number (n)",
                    zaxis_title="Amplitude",
                    yaxis=dict(autorange="reversed") # Put resultant wave at the front
                ),
                template="plotly_dark",
                height=700,
                margin=dict(l=0, r=0, b=0, t=40),
                showlegend=False
            )
            st.plotly_chart(fig_fourier, use_container_width=True)
            
            st.info("💡 **Physics Insight (Gibbs Phenomenon):** Notice how summing more harmonic sine waves makes the resultant wave (green line at the front) look closer to a perfect Square/Sawtooth wave. The slight ringing at the sharp edges is a mathematical limitation known as the *Gibbs Phenomenon*.")

  # ==========================================
  # TAB 13: Data Export & PDF Report
  # ==========================================
  with tabs[12]:
    st.header("Export Analysis Data & Reports")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.subheader("📄 Automated PDF Lab Report")
        st.markdown("Generate a comprehensive PDF summary of the audio analysis.")
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Compiling scientific report..."):
                if FPDF is None:
                    st.error("FPDF library is not installed. Please run `pip install fpdf` in your terminal.")
                else:
                    # Fetching cached data for the report
                    freqs_exp, mag_exp, _ = compute_fft(y, sr)
                    dom_idx = np.argmax(mag_exp)
                    dom_freq = freqs_exp[dom_idx]
                    fund_mag = mag_exp[dom_idx]

                    # Recalculate THD quickly
                    harmonic_sq_sum = 0.0
                    for i in range(2, 11):
                        h_freq = dom_freq * i
                        if h_freq > freqs_exp[-1]:
                            break
                        idx = np.argmin(np.abs(freqs_exp - h_freq))
                        window = mag_exp[max(0, idx - 3) : min(len(mag_exp), idx + 4)]
                        h_mag = np.max(window) if len(window) > 0 else 0
                        harmonic_sq_sum += h_mag**2
                    thd = ((np.sqrt(harmonic_sq_sum) / fund_mag) * 100.0) if fund_mag > 0 else 0.0

                    rt60_exp, _, _, _, _, _, _ = estimate_rt60(y, sr)
                    rms_exp = librosa.feature.rms(y=y)[0]
                    zcr_exp = librosa.feature.zero_crossing_rate(y)[0]

                    # Generate PDF
                    pdf_bytes = create_lab_report(
                        file_name=file_name, sr=sr, duration=duration, num_samples=num_samples,
                        peak_amp=peak_amp, rms_mean=np.mean(rms_exp), zcr_mean=np.mean(zcr_exp),
                        rt60=rt60_exp, dom_freq=dom_freq, thd=thd
                    )

                    st.success("Report Generated Successfully!")
                    st.download_button(
                        label="⬇️ Download PDF Lab Report",
                        data=pdf_bytes,
                        file_name=f"Lab_Report_{file_name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )

    with col_exp2:
        st.subheader("📊 Raw Data Export")
        
        # FFT Data
        freqs_exp, mag_exp, mag_db_exp = compute_fft(y, sr)
        freqs_plot, mag_exp_plot = downsample_fft(freqs_exp, mag_exp, 5000)
        _, mag_db_exp_plot = downsample_fft(freqs_exp, mag_db_exp, 5000)

        df_fft = pd.DataFrame({
            "Frequency_Hz": freqs_plot,
            "Magnitude": mag_exp_plot,
            "Magnitude_dB": mag_db_exp_plot,
        })
        st.download_button("Download FFT Data (CSV)", df_fft.to_csv(index=False).encode("utf-8"), "fft_data.csv", "text/csv")

        # Audio Features Data
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        df_feat = pd.DataFrame({"Time_s": times, "RMS_Energy": rms, "Zero_Crossing_Rate": zcr})
        st.download_button("Download Audio Features (CSV)", df_feat.to_csv(index=False).encode("utf-8"), "audio_features.csv", "text/csv")

        # Mono Audio
        buffer_mono = io.BytesIO()
        sf.write(buffer_mono, y, sr, format="WAV")
        st.download_button(label="Download Mono WAV", data=buffer_mono.getvalue(), file_name="mono_converted.wav", mime="audio/wav")

if __name__ == "__main__":
  main()
