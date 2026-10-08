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
import time
from typing import Optional, Tuple
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
import scipy.fft
import scipy.signal
import soundfile as sf
import streamlit as st

# --- External Internal Labs ---
try:
    import blackhole_lab
    HAS_BLACKHOLE = True
except ImportError:
    HAS_BLACKHOLE = False

try:
    import quantum_lab
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

# --- NEW: Image Processing for Steganography ---
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# --- NEW: Live Streaming Library ---
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    class AudioProcessorBase: pass # Mock to prevent runtime class inheritance error

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
  """Computes windowed FFT and returns frequencies, normalized magnitude, and dB scale."""
  if len(y) == 0:
      return np.array([]), np.array([]), np.array([])
      
  window = np.hanning(len(y))
  y_windowed = y * window
  fft_result = scipy.fft.rfft(y_windowed)
  freqs = scipy.fft.rfftfreq(len(y), 1 / sr)

  # Normalize magnitude to physical amplitude
  magnitude = np.abs(fft_result) * 2.0 / (len(y) + 1e-10) 
  ref_val = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
  magnitude_db = librosa.amplitude_to_db(magnitude, ref=ref_val)

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
  if len(y) < n_fft:
      y = np.pad(y, (0, n_fft - len(y)), mode='constant')
  stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
  magnitude = np.abs(stft_result)
  ref_val = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
  return librosa.amplitude_to_db(magnitude, ref=ref_val)


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_cwt(
    y: np.ndarray, sr: int, wavelet: str = "morl", num_scales: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes Continuous Wavelet Transform (CWT) and returns time, frequencies, and power spectrum."""
  scales = np.arange(1, num_scales + 1)
  if len(y) == 0:
      return np.array([]), np.array([]), np.array([[]])
      
  coefficients, frequencies = pywt.cwt(
      y, scales, wavelet, sampling_period=1.0 / sr
  )
  power = np.abs(coefficients) ** 2
  time_axis = np.linspace(0, len(y) / sr, len(y))

  return time_axis, frequencies, power


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_hilbert(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Applies Hilbert Transform to extract amplitude envelope and instantaneous frequency."""
  if len(y) <= 1:
      return np.zeros_like(y), y, np.zeros_like(y), np.zeros_like(y)
      
  analytic_signal = scipy.signal.hilbert(y)
  amplitude_envelope = np.abs(analytic_signal)
  instantaneous_phase = np.unwrap(np.angle(analytic_signal))
  instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * sr)
  
  if len(instantaneous_frequency) > 0:
      instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])
  else:
      instantaneous_frequency = np.zeros_like(y)
      
  time_axis = np.arange(len(y)) / sr
  
  return time_axis, y, amplitude_envelope, instantaneous_frequency


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_filter(
    y: np.ndarray, sr: int, filter_type: str, cutoff: float, order: int, cutoff2: float = None
) -> np.ndarray:
  """Applies a Butterworth filter to the audio signal with safety bounds."""
  nyquist = 0.5 * sr
  
  # Ensure strict stability bounds
  cutoff = max(1.0, min(cutoff, nyquist - 1.0))
  if cutoff2 is not None:
      cutoff2 = max(cutoff + 1.0, min(cutoff2, nyquist - 0.5))

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

  if len(y) <= 3 * max(len(a), len(b)):
      raise ValueError("Audio segment is too short for the selected filter order. Try a lower order or longer audio.")

  filtered_y = scipy.signal.filtfilt(b, a, y)
  return filtered_y

@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_filter_zpk(
    sr: int, filter_type: str, cutoff: float, order: int, cutoff2: float = None
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes Zeros and Poles (Z-Plane) for the selected digital filter."""
  nyquist = 0.5 * sr
  cutoff = max(1.0, min(cutoff, nyquist - 1.0))
  if cutoff2 is not None:
      cutoff2 = max(cutoff + 1.0, min(cutoff2, nyquist - 0.5))
      
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
  if len(y_sub) == 0:
      return np.array([]), np.array([]), 0.0
  
  delay_samples = int(delay_sec * sr)
  y_delayed = np.pad(y_sub, (delay_samples, 0), mode='constant')[:len(y_sub)]
  
  # Normalize noise relative to the signal strength
  signal_amp = np.max(np.abs(y_sub)) if np.max(np.abs(y_sub)) > 0 else 1.0
  y_delayed += noise_lvl * np.random.randn(len(y_delayed)) * signal_amp
  
  correlation = scipy.signal.correlate(y_delayed, y_sub, mode='full')
  lags = np.arange(-len(y_sub) + 1, len(y_delayed)) / sr
  
  peak_idx = np.argmax(correlation)
  estimated_delay_sec = lags[peak_idx] if peak_idx < len(lags) else 0.0
  
  return lags, correlation, max(0.0, estimated_delay_sec)


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_spectral_gating(y: np.ndarray, threshold_db: float) -> np.ndarray:
  """Basic noise reduction using spectral gating."""
  S = librosa.stft(y)
  S_mag, S_phase = librosa.magphase(S)
  ref_val = np.max(S_mag) if np.max(S_mag) > 0 else 1.0
  S_db = librosa.amplitude_to_db(S_mag, ref=ref_val)

  mask = (S_db > threshold_db).astype(float)

  S_clean = S_mag * mask * S_phase
  y_clean = librosa.istft(S_clean, length=len(y))
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
  if len(y) < 2048:
      return 0.0, None, None, 0.0, 0.0, np.array([]), np.array([])
      
  rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
  if len(rms) == 0:
      return 0.0, None, None, 0.0, 0.0, np.array([]), np.array([])
      
  times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
  ref_val = np.max(rms) if np.max(rms) > 0 else 1.0
  rms_db = librosa.amplitude_to_db(rms, ref=ref_val)

  peak_idx = int(np.argmax(rms_db))
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
  closest_distance = max(closest_distance, 0.1)

  t_src = np.arange(len(y)) / sr
  t_mid = t_src[-1] / 2.0 if len(t_src) > 0 else 0.0

  x_src = velocity_ms * (t_src - t_mid)
  r_src = np.sqrt(x_src**2 + closest_distance**2)

  t_obs = t_src + (r_src / c)
  t_obs_uniform = np.arange(0, np.max(t_obs) if len(t_obs) > 0 else 0, 1.0 / sr)

  y_obs = np.interp(t_obs_uniform, t_obs, y, left=0.0, right=0.0)
  r_obs_uniform = np.interp(t_obs_uniform, t_obs, r_src)
  r_obs_uniform = np.maximum(r_obs_uniform, 0.1)
  
  amplitude_envelope = closest_distance / r_obs_uniform

  return y_obs * amplitude_envelope


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_3d_spatial_audio(y: np.ndarray, sr: int, mode: str, static_angle: float, radius: float, rot_hz: float, c: float = 343.0) -> np.ndarray:
    """Applies Interaural Time Difference (ITD) and Level Difference (ILD) for Binaural Panning."""
    T = np.arange(len(y)) / sr
    head_radius = 0.0875  # Average human head radius (8.75 cm)
    
    if mode == "Static 3D Position":
        theta = np.deg2rad(static_angle)
        theta_array = np.full_like(T, theta)
    else:
        # 8D Auto-Rotation
        theta_array = 2 * np.pi * rot_hz * T

    # Source Cartesian Coordinates
    x_s = radius * np.sin(theta_array)
    y_s = radius * np.cos(theta_array)
    
    # Ear Coordinates (Left Ear at -x, Right Ear at +x)
    x_L, y_L = -head_radius, 0.0
    x_R, y_R = head_radius, 0.0
    
    # Distance to each ear (Prevent zero-division scaling explosions)
    d_L = np.maximum(np.sqrt((x_s - x_L)**2 + (y_s - y_L)**2), 1e-10)
    d_R = np.maximum(np.sqrt((x_s - x_R)**2 + (y_s - y_R)**2), 1e-10)
    
    # ITD: Delays
    delay_L = d_L / c
    delay_R = d_R / c
    
    # Interpolate delayed signals (Doppler shifts naturally handled here)
    y_L_sig = np.interp(T - delay_L, T, y, left=0, right=0)
    y_R_sig = np.interp(T - delay_R, T, y, left=0, right=0)
    
    # ILD: Head Shadowing & Inverse Square Law Gain
    cos_alpha_L = -x_s / np.sqrt(x_s**2 + y_s**2 + 1e-10)
    cos_alpha_R = x_s / np.sqrt(x_s**2 + y_s**2 + 1e-10)
    
    shadow_L = 0.6 + 0.4 * cos_alpha_L
    shadow_R = 0.6 + 0.4 * cos_alpha_R
    
    gain_L = (1.0 / d_L) * shadow_L
    gain_R = (1.0 / d_R) * shadow_R
    
    y_L_sig *= gain_L
    y_R_sig *= gain_R
    
    # Normalize stereo matrix to avoid clipping while preserving panning ratio
    max_val = max(np.max(np.abs(y_L_sig)), np.max(np.abs(y_R_sig)))
    if max_val > 0:
        y_L_sig /= max_val
        y_R_sig /= max_val
        
    return np.vstack((y_L_sig, y_R_sig)).T # Shape [samples, 2] for Stereo WAV


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_granular_synthesis(y: np.ndarray, sr: int, grain_size_ms: float, stretch_factor: float, overlap: float = 0.5) -> np.ndarray:
    """Applies Time-Stretching using Quantum Acoustic Granular Synthesis."""
    grain_length = int(sr * (grain_size_ms / 1000.0))
    if grain_length <= 0: return y
    
    hop_length = int(grain_length * (1.0 - overlap))
    if hop_length <= 0: hop_length = 1
    
    out_hop_length = int(hop_length * stretch_factor)
    
    num_grains = 1 + (len(y) - grain_length) // hop_length
    if num_grains <= 0: return y
    
    out_len = int(num_grains * out_hop_length + grain_length)
    y_out = np.zeros(out_len)
    
    window = np.hanning(grain_length)
    
    for i in range(num_grains):
        in_start = i * hop_length
        in_end = in_start + grain_length
        grain = y[in_start:in_end] * window
        
        out_start = i * out_hop_length
        out_end = out_start + grain_length
        y_out[out_start:out_end] += grain
        
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out /= max_val
        
    return y_out


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_distortion(y: np.ndarray, dist_type: str, drive: float) -> np.ndarray:
    """Applies non-linear analog distortion, saturation, or wavefolding to the signal."""
    y_driven = y * drive
    
    if dist_type == "Soft Clipping (Vacuum Tube)":
        y_out = np.tanh(y_driven) / np.tanh(drive) if drive > 0 else y
    elif dist_type == "Hard Clipping (Transistor Fuzz)":
        y_out = np.clip(y_driven, -1.0, 1.0)
    elif dist_type == "Sine Wavefolding (Buchla Synth)":
        y_out = np.sin(y_driven * np.pi / 2)
    else: # Triangle Wavefolding
        y_out = (2 / np.pi) * np.arcsin(np.sin(y_driven * np.pi / 2))
        
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val
        
    return y_out


@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_cross_synthesis(y_mod: np.ndarray, sr: int, carrier_type: str, carrier_freq: float) -> np.ndarray:
    """Applies Spectral Cross-Synthesis (Vocoder) between a voice modulator and a generated carrier."""
    t = np.arange(len(y_mod)) / sr
    
    if carrier_type == "Sawtooth Wave":
        y_car = scipy.signal.sawtooth(2 * np.pi * carrier_freq * t)
    elif carrier_type == "Square Wave":
        y_car = scipy.signal.square(2 * np.pi * carrier_freq * t)
    elif carrier_type == "Synth Chord (Minor 7th)":
        f_r = carrier_freq
        f_m3 = carrier_freq * (2 ** (3/12))
        f_p5 = carrier_freq * (2 ** (7/12))
        f_m7 = carrier_freq * (2 ** (10/12))
        y_car = (scipy.signal.sawtooth(2 * np.pi * f_r * t) +
                 scipy.signal.sawtooth(2 * np.pi * f_m3 * t) +
                 scipy.signal.sawtooth(2 * np.pi * f_p5 * t) +
                 scipy.signal.sawtooth(2 * np.pi * f_m7 * t)) / 4.0
    else: # White Noise
        y_car = np.random.randn(len(y_mod))
        
    n_fft = 2048
    hop_length = 512
    if len(y_mod) < n_fft:
        return y_mod
        
    S_mod = librosa.stft(y_mod, n_fft=n_fft, hop_length=hop_length)
    S_car = librosa.stft(y_car, n_fft=n_fft, hop_length=hop_length)
    
    mag_mod = np.abs(S_mod)
    mag_car = np.abs(S_car)
    phase_car = np.angle(S_car)
    
    mag_out = mag_mod * (mag_car / (np.max(mag_car) + 1e-10))
    S_out = mag_out * np.exp(1j * phase_car)
    
    y_out = librosa.istft(S_out, hop_length=hop_length, length=len(y_mod))
    
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out /= max_val
        
    return y_out


@st.cache_data(hash_funcs=FAST_NP_HASH)
def generate_spectrogram_art(text: str, sr: int = 44100, duration: float = 3.0, min_freq: float = 1000.0, max_freq: float = 10000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converts a text string into an audio signal using Inverse STFT (ISTFT) so it appears on a spectrogram."""
    n_fft = 2048
    hop_length = 512
    
    num_frames = int((duration * sr) / hop_length)
    num_bins = n_fft // 2 + 1
    
    mag_spec = np.zeros((num_bins, num_frames))
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    min_bin = int(np.argmin(np.abs(freqs - min_freq)))
    max_bin = int(np.argmin(np.abs(freqs - max_freq)))
    
    if max_bin <= min_bin:
        max_bin = min_bin + 10
        
    box_height = max_bin - min_bin
    if box_height <= 0: box_height = 1
    box_width = num_frames
    
    if HAS_PIL:
        base_width = max(len(text) * 8, 20)
        base_height = 15
        small_img = Image.new('L', (base_width, base_height), color=0)
        draw = ImageDraw.Draw(small_img)
        draw.text((2, 2), text, fill=255)
        
        try:
            resample_filter = Image.Resampling.NEAREST
        except AttributeError:
            resample_filter = Image.NEAREST
            
        img = small_img.resize((box_width, box_height), resample_filter)
        img_arr = np.array(img, dtype=float) / 255.0
    else:
        img_arr = np.random.rand(box_height, box_width) * 0.5
        
    img_arr = np.flipud(img_arr)
    mag_spec[min_bin:max_bin, :] = img_arr * 50.0  
    
    random_phase = np.random.uniform(-np.pi, np.pi, mag_spec.shape)
    S_complex = mag_spec * np.exp(1j * random_phase)
    
    y_art = librosa.istft(S_complex, hop_length=hop_length)
    
    max_val = np.max(np.abs(y_art))
    if max_val > 0:
        y_art /= max_val
        
    t_axis = librosa.frames_to_time(np.arange(num_frames), sr=sr, hop_length=hop_length)
        
    return y_art, mag_spec, t_axis, freqs


@st.cache_data(hash_funcs=FAST_NP_HASH)
def generate_shepard_tone(cycle_dur: float, sr: int, f_min: float, num_octaves: int, direction: str, cycles: int = 4) -> np.ndarray:
    """Generates the Shepard-Risset Glissando (The continuous Shepard Tone Illusion)."""
    total_samples = int(cycle_dur * sr * cycles)
    t = np.arange(total_samples) / sr
    y = np.zeros(total_samples)
    nyquist = sr / 2.0
    
    for i in range(num_octaves):
        if direction == "Ascending":
            v_time = ((t / cycle_dur) + i) % num_octaves
        else:
            v_time = (num_octaves - ((t / cycle_dur) + i) % num_octaves)
            
        f_inst = f_min * (2 ** v_time)
        
        # Hard cap at Nyquist to prevent physical aliasing loopback
        valid_freq_mask = f_inst < nyquist
        
        phase = np.cumsum(2 * np.pi * f_inst / sr)
        
        center = num_octaves / 2.0
        sigma = num_octaves / 4.0
        amp = np.exp(-0.5 * ((v_time - center) / sigma) ** 2)
        
        # Mask out frequencies beyond Nyquist
        amp = np.where(valid_freq_mask, amp, 0.0)
        
        y += amp * np.sin(phase)
        
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y /= max_val
        
    return y


@st.cache_data(hash_funcs=FAST_NP_HASH)
def compute_mel_spec(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates the Mel-Spectrogram mimicking human ear frequency perception."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    ref_val = np.max(mel_spec) if np.max(mel_spec) > 0 else 1.0
    mel_db = librosa.power_to_db(mel_spec, ref=ref_val)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr)
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr/2.0)
    return times, mel_freqs, mel_db


# ==========================================
# PDF REPORT GENERATOR 
# ==========================================
def create_lab_report(file_name, sr, duration, num_samples, peak_amp, rms_mean, zcr_mean, rt60, dom_freq, thd):
    if FPDF is None:
        return None
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Advanced Audio Analyzer - Lab Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Audio File Information", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Filename: {file_name}", ln=True)
    pdf.cell(0, 8, f"Sample Rate: {sr} Hz", ln=True)
    pdf.cell(0, 8, f"Duration: {duration:.2f} seconds", ln=True)
    pdf.cell(0, 8, f"Total Samples: {num_samples:,}", ln=True)
    pdf.cell(0, 8, f"Peak Amplitude: {peak_amp:.4f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Time Domain Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Mean RMS Energy: {rms_mean:.4f}", ln=True)
    pdf.cell(0, 8, f"Mean Zero Crossing Rate: {zcr_mean:.4f}", ln=True)
    rt60_str = f"{rt60:.2f} seconds" if rt60 > 0 else "N/A (No clear decay tail found)"
    pdf.cell(0, 8, f"Estimated RT60 (Reverb Time): {rt60_str}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Frequency Domain Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Dominant Frequency: {dom_freq:.2f} Hz", ln=True)
    pdf.cell(0, 8, f"Total Harmonic Distortion (THD): {thd:.2f}%", ln=True)
    pdf.ln(15)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Generated by Advanced Audio Analyzer (v5.0)", align='C')
    
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
    st.markdown("### ⚙ System Info")
    st.info(
        "Uses **Librosa** for extraction, **SciPy** for DSP, **PyWavelets** for"
        " CWT, and **Plotly** for interactive visualization."
    )

  # --- Live Streaming Logic ---
  if audio_source == "Real-Time WebRTC":
      if not HAS_WEBRTC:
          st.error("⚠️ **WebRTC Missing:** Please install `streamlit-webrtc` and `av` to use live streaming features.")
          return
          
      st.title("🔴 Live Real-Time Audio Streaming")
      st.write("Speak into your microphone. The audio frames are streaming to the server in real-time.")
      
      class AudioViewer(AudioProcessorBase):
          def __init__(self):
              self.audio_queue = queue.Queue(maxsize=10)
              
          def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
              audio_data = frame.to_ndarray()
              if not self.audio_queue.full():
                  self.audio_queue.put(audio_data[0, :])
              return frame

      webrtc_ctx = webrtc_streamer(
          key="live-audio-analyzer",
          mode=WebRtcMode.SENDONLY,
          audio_processor_factory=AudioViewer,
          media_stream_constraints={"audio": True, "video": False},
      )
      
      # Non-blocking continuous refresh loop
      if webrtc_ctx and webrtc_ctx.state.playing:
          st.success("🎙️ Microphone is LIVE! (Streaming active)")
          st.markdown("### 🌊 Live Oscilloscope (Waveform)")
          
          plot_spot = st.empty()
          if webrtc_ctx.audio_processor:
              try:
                  audio_chunk = webrtc_ctx.audio_processor.audio_queue.get_nowait()
                  plot_chunk = audio_chunk[::5] 
                  
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
                  plot_spot.plotly_chart(fig, use_container_width=True)
              except queue.Empty:
                  pass
          
          time.sleep(0.1)
          st.rerun()
      
      return 
  
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
      "🧩 3D Fourier Series",
      "📡 Wave Modulation",
      "🔲 Chladni Resonance",
      "🎧 Active Noise Cancellation",
      "🏛 3D Room Acoustics",
      "📶 Information Theory",
      "🛰 Phased Array Beamforming",
      "🚀 Supersonic Shockwave", 
      "🌌 Uncertainty Principle",
      "👾 ADC & Quantization", 
      "🎧 3D Spatial Audio & 8D Panning", 
      "🤫 Psychoacoustic Masking",
      "🛸 Acoustic Levitation", 
      "⚛️ Granular Synthesis", 
      "🎸 Analog Saturation", 
      "🤖 Phase Vocoder", 
      "🕵️ Spectrogram Art", 
      "🌀 Shepard Tone Illusion",
      "🌌 Black Hole",
      "⚛️ Quantum Tunneling",
      "📊 Data Export",
  ])

  # TAB 1: Audio Metadata
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

  # TAB 2: Waveform & Hilbert Transform
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
      
      crest = peak_amp / np.maximum(1e-10, np.mean(rms))
      col_t3.metric("Crest Factor", f"{crest:.2f}")

      with st.spinner("Estimating RT60..."):
        rt60, r_times, r_db, slope, intercept, d_times, d_db = estimate_rt60(y, sr)

      if rt60 > 0:
        col_t4.metric("Est. RT60 (Reverb Time)", f"{rt60:.2f} s")

        st.markdown("#### Energy Decay Curve & RT60 Extrapolation")
        st.caption("Acoustic physics measures RT60 by tracking the logarithmic energy decay after a loud impulse.")

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
        st.info("Could not calculate RT60. This requires a sharp impulsive sound in a room with a clear decay tail.")

  # TAB 3: FFT Spectrum & Welch's Method
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

      dom_idx = int(np.argmax(mag)) if len(mag) > 0 else 0
      dom_freq = freqs[dom_idx] if len(freqs) > 0 else 0.0

      for i in range(2, 6):
        harmonic_freq = dom_freq * i
        if harmonic_freq > 0 and harmonic_freq <= max_freq:
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
      fund_mag = mag[dom_idx] if len(mag) > 0 else 0.0

      for i in range(2, 11):
        h_freq = dom_freq * i
        if len(freqs) == 0 or h_freq > freqs[-1]:
          break
        idx = int(np.argmin(np.abs(freqs - h_freq)))
        window = mag[max(0, idx - 3) : min(len(mag), idx + 4)]
        h_mag = np.max(window) if len(window) > 0 else 0
        harmonic_sq_sum += h_mag**2

      thd = ((np.sqrt(harmonic_sq_sum) / fund_mag) * 100.0) if fund_mag > 1e-10 else 0.0

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

  # TAB 4: Spectrogram & 3D Waterfall
  with tabs[3]:
    st.header("Short-Time Fourier Transform (STFT) & 3D Waterfall")

    c1, c2, c3, c4 = st.columns(4)
    n_fft = c1.selectbox("FFT Size (Resolution)", options=[512, 1024, 2048, 4096], index=2)
    hop_length = c2.selectbox("Hop Length", options=[256, 512, 1024, 2048], index=1)
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
            data=go.Heatmap(z=S_db_plot, x=t_axis, y=f_axis, colorscale="Inferno")
        )

        if y_scale == "Logarithmic":
          fig_stft.update_layout(yaxis=dict(type="log", range=[np.log10(20), np.log10(nyquist)]))

        fig_stft.update_layout(
            title="2D Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_dark",
        )
      else:
        fig_stft = go.Figure(
            data=[go.Surface(z=S_db_plot, x=t_axis, y=f_axis, colorscale="Inferno")]
        )

        scene_dict = dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis_title="Magnitude (dB)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        )

        if y_scale == "Logarithmic":
          scene_dict["yaxis"] = dict(type="log", title="Frequency (Hz)", range=[np.log10(20), np.log10(nyquist)])

        fig_stft.update_layout(
            title="3D Cumulative Spectral Decay (Waterfall)",
            scene=scene_dict,
            template="plotly_dark",
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

      st.plotly_chart(fig_stft, use_container_width=True)

  # TAB 5: Continuous Wavelet Transform (CWT)
  with tabs[4]:
    st.header("Continuous Wavelet Transform (CWT)")
    st.markdown("CWT is computationally intensive. Select a specific time slice to analyze below.")

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
            
            t_axis += cwt_start

            fig_cwt = go.Figure(
                data=go.Heatmap(z=power_matrix, x=t_axis, y=f_axis, colorscale="Viridis")
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

  # TAB 6: Pitch & Tempo
  with tabs[5]:
    st.header("Pitch & Rhythm Analysis")

    c_pitch, c_tempo = st.columns(2)

    with c_pitch:
      st.subheader("Pitch Estimation (YIN)")
      st.info("Calculates fundamental frequency ($f_0$) and translates to a Musical Note.")
      
      if st.button("Estimate Pitch"):
        with st.spinner("Calculating pitch..."):
          target_sr = min(sr, 11025)
          y_pitch_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

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
                  marker=dict(symbol="line-ns", size=30, color="#FFC107", line=dict(width=2)),
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

  # TAB 7: Advanced Audio Features
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
            x=t_features, y=spectral_rolloff, template="plotly_dark", color_discrete_sequence=["#E91E63"]
        )
        fig_roll.update_layout(xaxis_title="Time (s)", yaxis_title="Hz")
        st.plotly_chart(fig_roll, use_container_width=True)

      st.markdown("**Spectral Flatness (Wiener Entropy)**")
      fig_flat = px.line(
          x=t_features, y=spectral_flatness, template="plotly_dark", color_discrete_sequence=["#00BCD4"]
      )
      fig_flat.update_layout(xaxis_title="Time (s)", yaxis_title="Flatness Ratio (0 to 1)")
      st.plotly_chart(fig_flat, use_container_width=True)

    st.subheader("Mel-Frequency Cepstral Coefficients (MFCC)")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig_mfcc = go.Figure(
        data=go.Heatmap(z=mfccs, x=t_features, y=np.arange(1, 14), colorscale="Viridis")
    )
    fig_mfcc.update_layout(
        xaxis_title="Time (s)", yaxis_title="MFCC Coefficient", template="plotly_dark"
    )
    st.plotly_chart(fig_mfcc, use_container_width=True)

  # TAB 8: Chaos Theory
  with tabs[7]:
    st.header("Phase Space Trajectory (Attractor Reconstruction)")
    
    c_chaos1, c_chaos2 = st.columns([1, 2])

    with c_chaos1:
      st.markdown("#### Embedding Parameters")
      tau = st.slider("Time Delay (τ) in samples", min_value=1, max_value=500, value=25, step=1)
      plot_dim = st.radio("Plot Dimension", ["2D Phase Space", "3D Phase Space"])
      max_pts = st.slider("Samples to Plot", 1000, 20000, 5000, step=1000)

    with c_chaos2:
      with st.spinner("Reconstructing Attractor..."):
        y_plot = y[:max_pts]

        if plot_dim == "2D Phase Space":
          if len(y_plot) > tau:
              y_t = y_plot[:-tau]
              y_t_tau = y_plot[tau:]

              fig_phase = go.Figure(
                  go.Scattergl(
                      x=y_t, y=y_t_tau, mode="markers+lines",
                      marker=dict(size=2, color=np.arange(len(y_t)), colorscale="Viridis", showscale=False),
                      line=dict(color="rgba(255,255,255,0.2)", width=1),
                  )
              )
              fig_phase.update_layout(
                  title=f"2D Phase Space (Delay τ = {tau})",
                  xaxis_title="y(t)", yaxis_title="y(t + τ)",
                  template="plotly_dark", height=500, width=500,
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
                    x=y_t, y=y_t_tau1, z=y_t_tau2, mode="lines",
                    line=dict(color=np.arange(len(y_t)), colorscale="Plasma", width=3),
                )
            )

            fig_phase3d.update_layout(
                title="3D Phase Space Strange Attractor",
                scene=dict(xaxis_title="y(t)", yaxis_title="y(t + τ)", zaxis_title="y(t + 2τ)"),
                template="plotly_dark", height=600,
            )
            st.plotly_chart(fig_phase3d, use_container_width=True)

  # TAB 9: Kinematics & DSP
  with tabs[8]:
    st.header("Kinematics & Digital Signal Processing (DSP)")

    st.subheader("1. Thermodynamics & Medium Kinetics")
    st.markdown("The speed of sound ($c$) is not constant. It depends heavily on the medium's density, elasticity, and in gases, the absolute temperature.")
    
    col_therm1, col_therm2, col_therm3 = st.columns(3)
    medium = col_therm1.selectbox("Acoustic Medium", ["Air (Ideal Gas)", "Water (Liquid)", "Seawater", "Steel (Solid)", "Helium (Gas)"])
    
    if medium == "Air (Ideal Gas)":
        temp_c = col_therm2.slider("Temperature (°C)", -50.0, 100.0, 20.0, 0.5)
        c_speed = 331.3 * np.sqrt(1 + temp_c / 273.15)
        col_therm3.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
        st.caption(r"💡 Calculated using the thermodynamic formula for air: $c = 331.3 \sqrt{1 + \frac{T}{273.15}}$")
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
    else:
        c_speed = 972.0
        col_therm2.success(f"**Speed of Sound ($c$):** {c_speed:.2f} m/s")
        st.caption("💡 Helium is much less dense than air, so sound travels almost 3 times faster.")

    st.divider()

    st.subheader("2. Wave Kinematics: Doppler Effect Simulator")
    
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
            xaxis_title="Time (s)", yaxis_title="Pressure Amplitude",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig_dop, use_container_width=True)

    st.divider()

    st.subheader("3. IIR Butterworth Filters")

    filt_type = st.selectbox("Filter Type", ["Low-Pass", "High-Pass", "Band-Pass", "Band-Stop"])
    filt_order = st.slider("Filter Order", min_value=1, max_value=10, value=4)

    cutoff1, cutoff2 = 1000.0, 3000.0

    if filt_type in ["Low-Pass", "High-Pass"]:
      default_cutoff = min(1000.0, float(nyquist - 1.0))
      cutoff1 = st.slider("Cutoff Frequency (Hz)", 20.0, float(nyquist - 1.0), default_cutoff)
    else:
      default_high = min(3000.0, float(nyquist - 1.0))
      default_low = min(500.0, default_high - 1.0)
      c1, c2 = st.slider("Frequency Band (Hz)", 20.0, float(nyquist - 1.0), (default_low, default_high))
      cutoff1, cutoff2 = c1, c2

    if st.button("Apply Filter"):
      with st.spinner("Filtering audio..."):
        try:
          y_filt = apply_filter(y, sr, filt_type, cutoff1, filt_order, cutoff2)
          st.success("Filter applied successfully!")

          buffer = io.BytesIO()
          sf.write(buffer, y_filt, sr, format="WAV")
          st.audio(buffer.getvalue(), format="audio/wav")

          fig_comp = go.Figure()
          fig_comp.add_trace(go.Scatter(y=y[:1000], name="Original", opacity=0.5))
          fig_comp.add_trace(go.Scatter(y=y_filt[:1000], name="Filtered", opacity=0.8))
          fig_comp.update_layout(template="plotly_dark")
          st.plotly_chart(fig_comp, use_container_width=True)

        except ValueError as e:
          st.error(f"Filter Error: {e}")

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
                xaxis_title="Real", yaxis_title="Imaginary", 
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
    
    st.subheader("5. Echolocation & SONAR (Cross-Correlation)")
    
    c_delay, c_noise = st.columns(2)
    true_delay = c_delay.slider("Simulated Echo Delay (seconds)", 0.01, 1.0, 0.25, 0.01)
    noise_lvl = c_noise.slider("Simulation Noise Level", 0.0, 1.0, 0.1, 0.1)
    
    if st.button("Run Echolocation Radar"):
        with st.spinner("Correlating signals and calculating distance..."):
            lags, corr, est_delay = compute_cross_correlation(y, sr, true_delay, noise_lvl)
            est_distance = (est_delay * c_speed) / 2.0
            
            st.success(f"⏱️ **Detected Echo Delay:** {est_delay:.4f} seconds")
            st.info(rf"📏 **Calculated Object Distance:** {est_distance:.2f} meters (using $d = \frac{{t \times c}}{{2}}$)")
            
            if len(lags) > 0:
                fig_xcorr = go.Figure()
                ds_x = 5000
                lags_plot = downsample_array(lags, ds_x)
                corr_plot = downsample_array(corr, ds_x)
                
                fig_xcorr.add_trace(go.Scatter(x=lags_plot, y=corr_plot, mode='lines', line=dict(color='#FFC107', width=1), name="Cross-Correlation"))
                fig_xcorr.add_vline(x=est_delay, line=dict(color='red', width=2, dash='dash'), annotation_text="Detected Peak")
                fig_xcorr.update_layout(title="Radar Cross-Correlation vs Time Lag", xaxis_title="Time Lag (s)", yaxis_title="Correlation Amplitude", template="plotly_dark")
                st.plotly_chart(fig_xcorr, use_container_width=True)

  # TAB 10: Psychoacoustics & Human Hearing
  with tabs[9]:
    st.header("Psychoacoustics & Human Hearing")
    
    c_psyc1, c_psyc2 = st.columns(2)
    
    with c_psyc1:
      st.subheader("1. A-Weighted Power Spectrum")
      
      with st.spinner("Applying perceptual weighting..."):
        freqs_p, _, mag_db_p = compute_fft(y, sr)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_weights = librosa.A_weighting(freqs_p)
            
        perceived_db = mag_db_p + a_weights
        
        valid_idx = (freqs_p > 0) & (freqs_p <= (sr / 2.0))
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
      
      with st.spinner("Generating Mel-Spectrogram..."):
        mel_times, mel_freqs, mel_db = compute_mel_spec(y, sr)
        
        time_factor = max(1, len(mel_times) // 400)
        mel_db_plot = mel_db[:, ::time_factor]
        mel_times_plot = mel_times[::time_factor]
        
        fig_mel = go.Figure(data=go.Heatmap(
            z=mel_db_plot, x=mel_times_plot, y=mel_freqs, colorscale="Magma"
        ))
        fig_mel.update_layout(
            title="Perceptual Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Mel Frequency (Hz)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_mel, use_container_width=True)

  # TAB 11: Lissajous & Simple Harmonic Motion 
  with tabs[10]:
    st.header("Oscilloscope: Lissajous Figures & Damped Harmonic Motion")
    
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    freq_x = col_l1.slider("X-Axis Frequency ($f_x$) in Hz", 1.0, 1000.0, 440.0, 1.0)
    freq_y = col_l2.slider("Y-Axis Frequency ($f_y$) in Hz", 1.0, 1000.0, 440.0, 1.0)
    phase_delta = col_l3.slider("Phase Difference ($\delta$)", 0, 360, 90)
    damping = col_l4.slider("Damping Friction ($\gamma$)", 0.0, 2.0, 0.0, 0.1)

    with st.spinner("Generating Lissajous Curve..."):
        t_end = max(1.0/freq_x, 1.0/freq_y) * 20 
        t_lin = np.linspace(0, t_end, 5000)

        phase_rad = np.deg2rad(phase_delta)
        
        decay_time = np.linspace(0, 5, 5000)
        decay = np.exp(-damping * decay_time)

        x_sig = decay * np.sin(2 * np.pi * freq_x * t_lin + phase_rad)
        y_sig = decay * np.sin(2 * np.pi * freq_y * t_lin)

        fig_liss = go.Figure(go.Scattergl(
            x=x_sig, y=y_sig, mode='lines', line=dict(color='#E91E63', width=1.5)
        ))
        
        fig_liss.update_layout(
            title=f"Lissajous Curve (Ratio {freq_x}:{freq_y}) | Damping: {damping}",
            xaxis_title="Amplitude (X-Axis)", yaxis_title="Amplitude (Y-Axis)",
            template="plotly_dark", width=600, height=600,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(range=[-1.1, 1.1]),
        )
        
        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(fig_liss, use_container_width=True)
            
        with col_info:
            st.info("💡 **Physics Insight:**")
            st.markdown(
                r"""
                A Lissajous figure is produced by the intersection of two Simple Harmonic Motions (SHM) at right angles:
                
                $x(t) = A e^{-\gamma t} \sin(2\pi f_x t + \delta)$
                
                $y(t) = B e^{-\gamma t} \sin(2\pi f_y t)$
                """
            )
            
    st.divider()
    st.subheader("Wave Interference: Acoustic Beats Simulator")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    f1 = col_b1.slider("Frequency 1 ($f_1$) Hz", 200.0, 1000.0, 440.0, 1.0)
    f2 = col_b2.slider("Frequency 2 ($f_2$) Hz", 200.0, 1000.0, 444.0, 1.0)
    beat_duration = col_b3.slider("Simulation Duration (s)", 1.0, 5.0, 3.0, 0.5)
    
    if st.button("Simulate Acoustic Beats"):
        with st.spinner("Calculating wave superposition..."):
            sr_beat = 44100
            t_beat = np.linspace(0, beat_duration, int(sr_beat * beat_duration), endpoint=False)
            
            y1 = 0.5 * np.sin(2 * np.pi * f1 * t_beat)
            y2 = 0.5 * np.sin(2 * np.pi * f2 * t_beat)
            y_beat = y1 + y2
            
            buffer_beat = io.BytesIO()
            sf.write(buffer_beat, y_beat, sr_beat, format="WAV")
            st.audio(buffer_beat.getvalue(), format="audio/wav")
            
            beat_freq = abs(f1 - f2)
            st.success(f"**Beat Frequency ($f_{{beat}}$):** {beat_freq:.1f} Hz")
            
            plot_limit = min(len(t_beat), int(sr_beat * (4.0 / beat_freq if beat_freq > 0 else 0.1))) 
            
            fig_beat = go.Figure()
            fig_beat.add_trace(go.Scatter(
                x=t_beat[:plot_limit:5], y=y_beat[:plot_limit:5], 
                mode='lines', line=dict(color='#00BCD4', width=1)
            ))
            fig_beat.update_layout(
                title=f"Superposition Envelope (Interference of {f1} Hz & {f2} Hz)",
                xaxis_title="Time (s)", yaxis_title="Pressure Amplitude",
                template="plotly_dark", height=350, margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_beat, use_container_width=True)

  # TAB 12: 3D Fourier Series Deconstruction 
  with tabs[11]:
    st.header("3D Fourier Series Deconstruction")

    col_f1, col_f2, col_f3 = st.columns(3)
    complex_wave_type = col_f1.selectbox("Complex Wave Type", ["Square Wave", "Sawtooth Wave"])
    fund_freq = col_f2.slider("Fundamental Frequency (Hz)", 1.0, 50.0, 5.0, 1.0)
    num_harmonics = col_f3.slider("Number of Harmonics (N)", 1, 50, 10)

    if st.button("Deconstruct Wave to 3D Harmonics"):
        with st.spinner("Calculating 3D Fourier Harmonics..."):
            t_fourier = np.linspace(0, 2.0/fund_freq, 1000) 
            
            fig_fourier = go.Figure()
            summed_wave = np.zeros_like(t_fourier)

            for n in range(1, num_harmonics + 1):
                if complex_wave_type == "Square Wave":
                    if n % 2 == 0: continue 
                    amplitude = (4.0 / np.pi) * (1.0 / n)
                else: 
                    amplitude = (2.0 / np.pi) * ((-1.0)**(n+1) / n)

                harmonic_wave = amplitude * np.sin(2 * np.pi * (n * fund_freq) * t_fourier)
                summed_wave += harmonic_wave

                fig_fourier.add_trace(go.Scatter3d(
                    x=t_fourier, y=[n]*len(t_fourier), z=harmonic_wave,
                    mode='lines', name=f"Harmonic {n}",
                    line=dict(color=px.colors.sequential.Plasma[n % len(px.colors.sequential.Plasma)], width=3)
                ))

            fig_fourier.add_trace(go.Scatter3d(
                x=t_fourier, y=[0]*len(t_fourier), z=summed_wave,
                mode='lines', name="Resultant Wave",
                line=dict(color='#1DB954', width=6)
            ))

            fig_fourier.update_layout(
                title=f"3D Fourier Deconstruction ({complex_wave_type}, N={num_harmonics})",
                scene=dict(xaxis_title="Time (s)", yaxis_title="Harmonic Number (n)", zaxis_title="Amplitude", yaxis=dict(autorange="reversed")),
                template="plotly_dark", height=700, margin=dict(l=0, r=0, b=0, t=40), showlegend=False
            )
            st.plotly_chart(fig_fourier, use_container_width=True)

  # TAB 13: Wave Modulation Simulator 
  with tabs[12]:
    st.header("Wave Modulation Simulator (AM & FM)")

    col_m1, col_m2 = st.columns(2)
    mod_type = col_m1.radio("Modulation Type", ["Amplitude Modulation (AM)", "Frequency Modulation (FM)"])

    c_fm1, c_fm2, c_fm3 = st.columns(3)
    f_c = c_fm1.slider("Carrier Frequency ($f_c$) Hz", 100.0, 2000.0, 800.0, 50.0)
    f_m = c_fm2.slider("Message Frequency ($f_m$) Hz", 1.0, 100.0, 10.0, 1.0)

    if mod_type == "Amplitude Modulation (AM)":
        mod_index = c_fm3.slider("Modulation Index ($m$)", 0.0, 2.0, 0.5, 0.1)
        equation = r"y_{AM}(t) = [1 + m \cdot \sin(2\pi f_m t)] \cdot \sin(2\pi f_c t)"
    else:
        mod_index = c_fm3.slider("Modulation Index ($\beta$)", 0.0, 20.0, 5.0, 1.0)
        equation = r"y_{FM}(t) = \sin[2\pi f_c t + \beta \cdot \sin(2\pi f_m t)]"

    st.latex(equation)

    if st.button("Simulate Modulation"):
        with st.spinner(f"Calculating {mod_type}..."):
            sr_mod = 44100
            dur_mod = 0.5  
            t_mod = np.linspace(0, dur_mod, int(sr_mod * dur_mod), endpoint=False)

            msg_wave = np.sin(2 * np.pi * f_m * t_mod)
            carrier_wave = np.sin(2 * np.pi * f_c * t_mod)

            if mod_type == "Amplitude Modulation (AM)":
                mod_wave = (1 + mod_index * msg_wave) * carrier_wave
            else:
                mod_wave = np.sin(2 * np.pi * f_c * t_mod + mod_index * np.sin(2 * np.pi * f_m * t_mod))

            plot_limit = int(sr_mod * 0.1)
            t_plot = t_mod[:plot_limit]
            
            fig_mod = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                subplot_titles=("Message Signal", "Carrier Signal", f"Resultant {mod_type} Signal")
            )

            fig_mod.add_trace(go.Scatter(x=t_plot, y=msg_wave[:plot_limit], line=dict(color='#00BCD4', width=2), name="Message"), row=1, col=1)
            fig_mod.add_trace(go.Scatter(x=t_plot, y=carrier_wave[:plot_limit], line=dict(color='#FF9800', width=1), name="Carrier"), row=2, col=1)

            if mod_type == "Amplitude Modulation (AM)":
                env_upper = 1 + mod_index * msg_wave[:plot_limit]
                fig_mod.add_trace(go.Scatter(x=t_plot, y=mod_wave[:plot_limit], line=dict(color='#E91E63', width=1.5), name="AM Signal"), row=3, col=1)
                fig_mod.add_trace(go.Scatter(x=t_plot, y=env_upper, line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name="Envelope+"), row=3, col=1)
            else:
                fig_mod.add_trace(go.Scatter(x=t_plot, y=mod_wave[:plot_limit], line=dict(color='#E91E63', width=1.5), name="FM Signal"), row=3, col=1)

            fig_mod.update_layout(height=650, template="plotly_dark", title_text="Wave Modulation")
            fig_mod.update_xaxes(title_text="Time (s)", row=3, col=1)
            
            st.plotly_chart(fig_mod, use_container_width=True)

            dur_listen = 2.0
            t_listen = np.linspace(0, dur_listen, int(sr_mod * dur_listen), endpoint=False)
            if mod_type == "Amplitude Modulation (AM)":
                listen_wave = (1 + mod_index * np.sin(2 * np.pi * f_m * t_listen)) * np.sin(2 * np.pi * f_c * t_listen)
            else:
                listen_wave = np.sin(2 * np.pi * f_c * t_listen + mod_index * np.sin(2 * np.pi * f_m * t_listen))
                
            buffer_mod = io.BytesIO()
            sf.write(buffer_mod, listen_wave, sr_mod, format="WAV")
            st.audio(buffer_mod.getvalue(), format="audio/wav")

  # TAB 14: Chladni Plate Resonance 
  with tabs[13]:
    st.header("Chladni Plate Resonance (2D Standing Waves)")

    col_c1, col_c2, col_c3 = st.columns(3)
    m_mode = col_c1.slider("Mode (m)", 1, 15, 3)
    n_mode = col_c2.slider("Mode (n)", 1, 15, 4)
    sign = col_c3.radio("Superposition Sign", ["Positive (+)", "Negative (-)"])

    if st.button("Generate Chladni Pattern"):
        with st.spinner("Calculating 2D resonance pattern..."):
            x = np.linspace(0, 1, 400)
            y = np.linspace(0, 1, 400)
            X, Y = np.meshgrid(x, y)

            term1 = np.sin(m_mode * np.pi * X) * np.sin(n_mode * np.pi * Y)
            term2 = np.sin(n_mode * np.pi * X) * np.sin(m_mode * np.pi * Y)

            if sign == "Positive (+)":
                Z = term1 + term2
                eq_str = fr"z(x,y) = \sin({m_mode}\pi x)\sin({n_mode}\pi y) + \sin({n_mode}\pi x)\sin({m_mode}\pi y)"
            else:
                Z = term1 - term2
                eq_str = fr"z(x,y) = \sin({m_mode}\pi x)\sin({n_mode}\pi y) - \sin({n_mode}\pi x)\sin({m_mode}\pi y)"

            st.latex(eq_str)
            Z_abs = np.abs(Z)

            fig_chladni = go.Figure(data=go.Heatmap(
                z=-Z_abs, x=x, y=y, colorscale="Oranges", showscale=False
            ))

            fig_chladni.update_layout(
                title=f"Chladni Resonance Pattern (m={m_mode}, n={n_mode})",
                xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark", height=600, margin=dict(l=0, r=0, b=40, t=40)
            )
            st.plotly_chart(fig_chladni, use_container_width=True)

  # TAB 15: Active Noise Cancellation (ANC) 
  with tabs[14]:
    st.header("Active Noise Cancellation (ANC) Simulator")

    col_anc1, col_anc2, col_anc3 = st.columns(3)
    phase_shift = col_anc1.slider("Phase Shift (Degrees)", 0, 180, 180, 1)
    amp_match = col_anc2.slider("Anti-Noise Amplitude Match (%)", 0, 100, 100, 1)
    delay_ms = col_anc3.slider("DSP Latency Delay (ms)", 0.0, 5.0, 0.0, 0.1)

    if st.button("Simulate ANC on Real Audio"):
        with st.spinner("Generating Anti-Noise and calculating superposition..."):
            anti_noise = y * (amp_match / 100.0)
            
            if phase_shift == 180:
                anti_noise = -anti_noise
            elif phase_shift != 0:
                analytic_sig = scipy.signal.hilbert(anti_noise)
                anti_noise = np.real(analytic_sig * np.exp(1j * np.deg2rad(phase_shift)))

            if delay_ms > 0:
                delay_samples = int((delay_ms / 1000.0) * sr)
                anti_noise = np.pad(anti_noise, (delay_samples, 0), mode='constant')[:len(anti_noise)]

            resultant = y + anti_noise

            rms_orig = np.sqrt(np.mean(y**2))
            rms_res = np.sqrt(np.mean(resultant**2))
            if rms_res > 0 and rms_orig > 0:
                reduction_db = 20 * np.log10(rms_res / rms_orig)
            else:
                reduction_db = -100.0 if rms_res == 0 else 0.0

            if reduction_db < -3:
                st.success(f"📉 **Noise Reduction Achieved:** {abs(reduction_db):.1f} dB")
            elif reduction_db > 3:
                st.error(f"⚠️ **Noise Increased (Constructive Interference):** {reduction_db:.1f} dB")
            else:
                st.warning(f"⚖️ **Little to No Cancellation:** {reduction_db:.1f} dB")

            plot_samples = min(len(y), int(sr * 0.05)) 
            t_plot = np.linspace(0, plot_samples/sr, plot_samples)
            
            fig_anc = go.Figure()
            fig_anc.add_trace(go.Scatter(x=t_plot, y=y[:plot_samples], mode='lines', line=dict(color='#00BCD4', width=2), name="Original Audio"))
            fig_anc.add_trace(go.Scatter(x=t_plot, y=anti_noise[:plot_samples], mode='lines', line=dict(color='#E91E63', width=2, dash='dash'), name="Anti-Noise (Generated)"))
            fig_anc.add_trace(go.Scatter(x=t_plot, y=resultant[:plot_samples], mode='lines', line=dict(color='#1DB954', width=3), name="Resultant Output (User hears)"))
            
            fig_anc.update_layout(
                title="Real-Time Active Noise Cancellation Waveforms (50ms)",
                xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark", height=400
            )
            st.plotly_chart(fig_anc, use_container_width=True)

            buffer_anc = io.BytesIO()
            sf.write(buffer_anc, resultant, sr, format="WAV")
            st.audio(buffer_anc.getvalue(), format="audio/wav")

  # TAB 16: 3D Room Acoustics 
  with tabs[15]:
    st.header("3D Room Acoustics & Standing Waves (Room Modes)")
    st.latex(r"f_{p,q,r} = \frac{c}{2} \sqrt{\left(\frac{p}{L}\right)^2 + \left(\frac{q}{W}\right)^2 + \left(\frac{r}{H}\right)^2}")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    room_L = col_r1.slider("Room Length (m)", 2.0, 30.0, 5.0, 0.1)
    room_W = col_r2.slider("Room Width (m)", 2.0, 30.0, 4.0, 0.1)
    room_H = col_r3.slider("Room Height (m)", 2.0, 15.0, 3.0, 0.1)
    rt60_sim = col_r4.slider("Simulated RT60 Decay (s)", 0.1, 5.0, 1.2, 0.1)
    
    if st.button("Simulate Room Acoustics on Audio"):
        with st.spinner("Calculating modes and convoluting room impulse response..."):
            c_room = 343.0 
            
            modes = []
            for p in range(4):
                for q in range(4):
                    for r in range(4):
                        if p == 0 and q == 0 and r == 0: continue
                        freq = (c_room / 2.0) * np.sqrt((p/room_L)**2 + (q/room_W)**2 + (r/room_H)**2)
                        zeros = [p, q, r].count(0)
                        mode_type = "Axial Mode" if zeros == 2 else ("Tangential Mode" if zeros == 1 else "Oblique Mode")
                        modes.append({"p": p, "q": q, "r": r, "Frequency (Hz)": freq, "Type": mode_type})
            
            df_modes = pd.DataFrame(modes).sort_values("Frequency (Hz)").reset_index(drop=True)
            df_modes = df_modes[df_modes["Frequency (Hz)"] <= 300] 
            
            st.success(f"Calculated {len(df_modes)} resonant standing waves (Room Modes) below 300 Hz.")
            fig_modes = px.bar(df_modes, x="Frequency (Hz)", y="Type", color="Type", title="Room Modes (< 300 Hz)", orientation='h', template="plotly_dark", color_discrete_map={"Axial Mode": "#E91E63", "Tangential Mode": "#FF9800", "Oblique Mode": "#00BCD4"})
            st.plotly_chart(fig_modes, use_container_width=True)
            
            ir_length = int(rt60_sim * sr)
            t_ir = np.arange(ir_length) / sr
            decay_env = np.exp(-6.91 * t_ir / rt60_sim)
            noise = np.random.randn(ir_length)
            
            cutoff = max(500, min(8000, 15000 / (room_L * room_W * room_H / 100))) 
            b, a = scipy.signal.butter(2, cutoff / (sr/2), btype='low')
            colored_noise = scipy.signal.filtfilt(b, a, noise)
            
            rir = colored_noise * decay_env
            rir = rir / np.max(np.abs(rir)) 
            
            y_reverb = scipy.signal.fftconvolve(y, rir, mode='full')
            y_padded = np.pad(y, (0, len(y_reverb) - len(y))) 
            wet_mix = 0.35 
            y_final = (1 - wet_mix) * y_padded + wet_mix * y_reverb
            
            if np.max(np.abs(y_final)) > 0:
                y_final = y_final / np.max(np.abs(y_final)) 
            
            buffer_rev = io.BytesIO()
            sf.write(buffer_rev, y_final, sr, format="WAV")
            st.audio(buffer_rev.getvalue(), format="audio/wav")

  # TAB 17: Information Theory & SNR 
  with tabs[16]:
    st.header("Information Theory & Signal-to-Noise Ratio (SNR)")
    st.latex(r"C = B \log_2\left(1 + \frac{S}{N}\right)")

    col_snr1, col_snr2 = st.columns(2)
    channel_bw = col_snr1.slider("Channel Bandwidth (Hz)", 1000, int(nyquist), 3000, step=500)
    noise_level = col_snr2.slider("Added White Noise Level (%)", 0.0, 200.0, 20.0, step=5.0)

    if st.button("Calculate Channel Capacity & Apply Noise"):
        with st.spinner("Calculating..."):
            signal_power = np.mean(y**2)
            if signal_power == 0: signal_power = 1e-10 

            noise_amp = (noise_level / 100.0) * np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 0.01
            noise = np.random.randn(len(y)) * noise_amp
            noise_power = np.mean(noise**2)
            if noise_power == 0: noise_power = 1e-10

            y_noisy = y + noise
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)

            capacity_bps = channel_bw * np.log2(1 + snr_linear)
            capacity_kbps = capacity_bps / 1000.0

            st.success("Analysis Complete!")
            m1, m2, m3 = st.columns(3)
            m1.metric("Signal-to-Noise Ratio (SNR)", f"{snr_db:.2f} dB")
            m2.metric("Channel Bandwidth", f"{channel_bw:,} Hz")
            m3.metric("Max Channel Capacity (C)", f"{capacity_kbps:.2f} kbps")

            plot_limit = min(len(y), int(sr * 0.05)) 
            t_plot = np.linspace(0, plot_limit/sr, plot_limit)

            fig_snr = go.Figure()
            fig_snr.add_trace(go.Scatter(x=t_plot, y=y[:plot_limit], mode='lines', line=dict(color='#1DB954', width=2), name="Original Signal"))
            fig_snr.add_trace(go.Scatter(x=t_plot, y=y_noisy[:plot_limit], mode='lines', line=dict(color='rgba(233, 30, 99, 0.6)', width=1), name="Noisy Signal (Transmitted)"))
            fig_snr.update_layout(title=f"SNR: {snr_db:.1f} dB", template="plotly_dark", height=400)
            st.plotly_chart(fig_snr, use_container_width=True)

            y_noisy_norm = y_noisy / np.max(np.abs(y_noisy)) if np.max(np.abs(y_noisy)) > 0 else y_noisy
            buffer_noisy = io.BytesIO()
            sf.write(buffer_noisy, y_noisy_norm, sr, format="WAV")
            st.audio(buffer_noisy.getvalue(), format="audio/wav")

  # TAB 18: Phased Array Beamforming 
  with tabs[17]:
    st.header("Phased Array Beamforming Simulator")
    st.latex(r"AF(\theta) = \frac{1}{N} \left| \frac{\sin(N \psi / 2)}{\sin(\psi / 2)} \right| \quad \text{where} \quad \psi = 2\pi \frac{d}{\lambda} (\sin\theta - \sin\theta_0)")

    col_pa1, col_pa2, col_pa3 = st.columns(3)
    N_sources = col_pa1.slider("Number of Sources (N)", 2, 32, 8, 1)
    d_lambda = col_pa2.slider("Element Spacing ($d/\lambda$)", 0.1, 2.0, 0.5, 0.1)
    steer_angle = col_pa3.slider("Steering Angle ($\theta_0$)", -90, 90, 30, 1)

    if st.button("Simulate Beamforming"):
        with st.spinner("Calculating spatial wave interference..."):
            theta = np.linspace(-np.pi/2, np.pi/2, 1000)
            steer_rad = np.deg2rad(steer_angle)
            
            psi = 2 * np.pi * d_lambda * (np.sin(theta) - np.sin(steer_rad))
            AF = np.zeros_like(psi)
            
            for i, p in enumerate(psi):
                if p == 0:
                    AF[i] = 1.0
                else:
                    AF[i] = np.abs(np.sin(N_sources * p / 2) / (N_sources * np.sin(p / 2)))

            fig_polar = go.Figure(go.Scatterpolar(
                r=AF, theta=np.rad2deg(theta), mode='lines', fill='toself',
                fillcolor='rgba(0, 188, 212, 0.2)', line=dict(color='#00BCD4', width=2), name="Beam Pattern"
            ))
            fig_polar.update_layout(
                title="Far-Field Directivity (Polar Pattern)",
                polar=dict(sector=[-90, 90], angularaxis=dict(rotation=90, direction="clockwise"), radialaxis=dict(visible=False)),
                template="plotly_dark", height=400
            )
            
            x = np.linspace(-10, 10, 300)
            y = np.linspace(0, 20, 300)
            X, Y = np.meshgrid(x, y)
            
            Z = np.zeros_like(X, dtype=complex)
            for n in range(N_sources):
                x_n = (n - (N_sources - 1) / 2.0) * d_lambda
                y_n = 0.0
                
                r_n = np.sqrt((X - x_n)**2 + (Y - y_n)**2)
                r_n = np.where(r_n == 0, 1e-10, r_n) 
                
                phase_shift = -2 * np.pi * d_lambda * n * np.sin(steer_rad)
                Z += np.exp(1j * (2 * np.pi * r_n + phase_shift)) / np.sqrt(r_n)
            
            Z_real = np.real(Z)
            
            fig_heat = go.Figure(data=go.Heatmap(z=Z_real, x=x, y=y, colorscale="RdBu", zmid=0, showscale=False))
            fig_heat.update_layout(title="Near-Field Wavefront", xaxis_title="X (λ)", yaxis_title="Y (λ)", template="plotly_dark", height=500, yaxis=dict(scaleanchor="x", scaleratio=1))
            
            col_p1, col_p2 = st.columns([1, 1.2])
            with col_p1: st.plotly_chart(fig_polar, use_container_width=True)
            with col_p2: st.plotly_chart(fig_heat, use_container_width=True)

  # TAB 19: Supersonic Shockwave
  with tabs[18]:
      st.header("Supersonic Shockwave & Mach Cone (2D Sonic Boom)")
      col_mach1, col_mach2 = st.columns(2)
      mach_number = col_mach1.slider("Source Velocity (Mach Number, M)", 0.0, 3.0, 1.5, 0.1)
      num_waves = col_mach2.slider("Number of Wave Fronts", 10, 100, 40)

      if st.button("Simulate 2D Wave Propagation"):
          with st.spinner("Calculating spatial wavefronts..."):
              c_sound = 343.0  
              v_source = mach_number * c_sound
              
              t_current = 2.0  
              times = np.linspace(0, t_current, num_waves, endpoint=False)
              
              fig_mach = go.Figure()
              phi = np.linspace(0, 2*np.pi, 100)
              for t_emitted in times:
                  x_emit = v_source * t_emitted
                  r_wave = c_sound * (t_current - t_emitted)
                  x_circle = x_emit + r_wave * np.cos(phi)
                  y_circle = r_wave * np.sin(phi)
                  fig_mach.add_trace(go.Scatter(x=x_circle, y=y_circle, mode='lines', line=dict(color='rgba(0, 188, 212, 0.4)', width=1), showlegend=False, hoverinfo='skip'))
                  
              x_current = v_source * t_current
              fig_mach.add_trace(go.Scatter(x=[x_current], y=[0], mode='markers', marker=dict(color='#E91E63', size=12, symbol='triangle-right'), name='Moving Source'))
              
              if mach_number > 1.0:
                  mach_angle = np.arcsin(1.0 / mach_number)
                  cone_length = x_current * 1.2
                  x_line = [x_current, x_current - cone_length * np.cos(mach_angle)]
                  y_line_up = [0, cone_length * np.sin(mach_angle)]
                  y_line_down = [0, -cone_length * np.sin(mach_angle)]
                  
                  fig_mach.add_trace(go.Scatter(x=x_line, y=y_line_up, mode='lines', line=dict(color='#FF9800', width=3, dash='dash'), name='Mach Cone Envelope'))
                  fig_mach.add_trace(go.Scatter(x=x_line, y=y_line_down, mode='lines', line=dict(color='#FF9800', width=3, dash='dash'), showlegend=False))
                  st.error(f"💥 **Supersonic Flight! (Sonic Boom Created)** \n\nMach Angle ($\\theta$): **{np.rad2deg(mach_angle):.2f}°**")
              
              x_range_max = max(x_current * 1.2, c_sound * t_current * 1.2)
              fig_mach.update_layout(
                  title=f"2D Wave Propagation (Mach {mach_number:.1f})",
                  xaxis_title="Distance X (meters)", yaxis_title="Distance Y (meters)",
                  template="plotly_dark", yaxis=dict(scaleanchor="x", scaleratio=1),
                  xaxis=dict(range=[-c_sound * t_current * 1.1, x_range_max]), height=600
              )
              st.plotly_chart(fig_mach, use_container_width=True)
              st.latex(r"\sin(\theta) = \frac{c}{v} = \frac{1}{\text{Mach Number (M)}}")

  # TAB 20: Uncertainty Principle
  with tabs[19]:
    st.header("The Acoustic Uncertainty Principle (Gabor Limit)")
    st.latex(r"\Delta t \cdot \Delta f \ge \frac{1}{4\pi} \approx 0.0795")

    col_up1, col_up2 = st.columns(2)
    sigma_t_ms = col_up1.slider("Gaussian Time Width ($\sigma_t$) in ms", 0.5, 50.0, 10.0, 0.5)
    f_c = col_up2.slider("Center Frequency ($f_c$) Hz", 100.0, 2000.0, 440.0, 10.0)

    if st.button("Generate Gaussian Wave Packet"):
        with st.spinner("Calculating time-frequency bounds..."):
            sr_gabor = 44100
            dur_gabor = 0.5 
            t_gabor = np.linspace(-dur_gabor/2, dur_gabor/2, int(sr_gabor * dur_gabor), endpoint=False)
            
            sigma_t = sigma_t_ms / 1000.0
            envelope = np.exp(-(t_gabor**2) / (2 * sigma_t**2))
            wave_packet = envelope * np.cos(2 * np.pi * f_c * t_gabor)
            
            freqs = scipy.fft.rfftfreq(len(wave_packet), 1/sr_gabor)
            fft_mag = np.abs(scipy.fft.rfft(wave_packet))
            fft_mag = fft_mag / np.max(fft_mag)
            
            power_t = wave_packet**2
            power_t = power_t / np.sum(power_t)
            delta_t_num = np.sqrt(np.sum(t_gabor**2 * power_t))
            
            power_f = fft_mag**2
            power_f = power_f / np.sum(power_f)
            delta_f_num = np.sqrt(np.sum((freqs - f_c)**2 * power_f))
            
            uncertainty_product = delta_t_num * delta_f_num

            col_graph1, col_graph2 = st.columns(2)
            zoom_samples = int(sr_gabor * 0.1) 
            mid_idx = len(t_gabor) // 2
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=t_gabor[mid_idx-zoom_samples:mid_idx+zoom_samples]*1000, y=wave_packet[mid_idx-zoom_samples:mid_idx+zoom_samples], mode='lines', line=dict(color='#00BCD4', width=2), name='Wave Packet'))
            fig_time.update_layout(title="Time Domain ($\Delta t$)", template="plotly_dark", height=350)
            
            with col_graph1:
                st.plotly_chart(fig_time, use_container_width=True)
                st.info(f"⏱️ **Time Spread ($\Delta t$):** {delta_t_num*1000:.2f} ms")

            f_idx = np.where((freqs > f_c - 1000) & (freqs < f_c + 1000))[0]
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(x=freqs[f_idx], y=fft_mag[f_idx], mode='lines', line=dict(color='#FF9800', width=2), fill='tozeroy', name='Frequency Spectrum'))
            fig_freq.update_layout(title="Frequency Domain ($\Delta f$)", template="plotly_dark", height=350)
            
            with col_graph2:
                st.plotly_chart(fig_freq, use_container_width=True)
                st.info(f"📻 **Frequency Spread ($\Delta f$):** {delta_f_num:.2f} Hz")

            st.divider()
            st.success(f"**Mathematical Proof:** \n\n$\Delta t \cdot \Delta f$ = ({delta_t_num:.6f}) × ({delta_f_num:.2f}) = **{uncertainty_product:.5f}**")
            st.caption(r"✅ The product perfectly respects the theoretical lower bound of $1/4\pi \approx 0.0795$. The universe is stable!")

  # TAB 21: ADC & Quantization
  with tabs[20]:
    st.header("Analog-to-Digital Conversion (ADC)")

    col_adc1, col_adc2 = st.columns(2)
    target_sr = col_adc1.select_slider("Sampling Rate", options=[1000, 2000, 4000, 8000, 11025, 22050, 44100], value=8000)
    target_bits = col_adc2.select_slider("Bit Depth", options=[2, 3, 4, 8, 16], value=4)

    if st.button("Digitize & Crunch Audio"):
        with st.spinner("Applying Sampling & Quantization..."):
            factor = max(1, min(int(sr / target_sr), len(y)))
            y_sampled = y[::factor]
            y_zoh = np.repeat(y_sampled, factor)
            
            if len(y_zoh) > len(y):
                y_zoh = y_zoh[:len(y)]
            else:
                y_zoh = np.pad(y_zoh, (0, len(y) - len(y_zoh)), mode='edge')
            
            y_norm = y_zoh / np.max(np.abs(y_zoh)) if np.max(np.abs(y_zoh)) > 0 else y_zoh
            levels = max(2, 2 ** target_bits)
            
            y_quantized = np.round(y_norm * (levels / 2.0 - 1.0)) / (levels / 2.0 - 1.0)
            q_error = y_norm - y_quantized
            
            st.success(f"🎛️ **ADC Complete:** Converted to {target_sr} Hz / {target_bits}-bit audio.")
            
            col_play1, col_play2 = st.columns(2)
            with col_play1:
                buffer_q = io.BytesIO()
                sf.write(buffer_q, y_quantized, sr, format="WAV") 
                st.audio(buffer_q.getvalue(), format="audio/wav")
            with col_play2:
                buffer_err = io.BytesIO()
                sf.write(buffer_err, q_error, sr, format="WAV")
                st.audio(buffer_err.getvalue(), format="audio/wav")

            plot_dur = 0.01 
            plot_samples = int(sr * plot_dur)
            plot_samples = min(plot_samples, len(y))
            
            t_plot = np.linspace(0, plot_dur, plot_samples, endpoint=False)
            
            fig_adc = go.Figure()
            fig_adc.add_trace(go.Scatter(x=t_plot*1000, y=y_norm[:plot_samples], mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=2), name='Analog'))
            fig_adc.add_trace(go.Scatter(x=t_plot*1000, y=y_quantized[:plot_samples], mode='lines', line=dict(color='#E91E63', width=2, shape='hv'), name=f'Digital'))
            fig_adc.update_layout(xaxis_title="Time (ms)", template="plotly_dark", height=400)
            st.plotly_chart(fig_adc, use_container_width=True)

            freqs_orig, mag_orig, mag_db_orig = compute_fft(y_norm, sr)
            freqs_q, mag_q, mag_db_q = compute_fft(y_quantized, sr)
            
            valid_idx = freqs_orig <= (sr / 2.0)
            f_orig_p, m_orig_p = downsample_fft(freqs_orig[valid_idx], mag_db_orig[valid_idx], 5000)
            f_q_p, m_q_p = downsample_fft(freqs_q[valid_idx], mag_db_q[valid_idx], 5000)
            
            fig_fft_adc = go.Figure()
            fig_fft_adc.add_trace(go.Scatter(x=f_orig_p, y=m_orig_p, mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=1), name='Original'))
            fig_fft_adc.add_trace(go.Scatter(x=f_q_p, y=m_q_p, mode='lines', line=dict(color='#00BCD4', width=1.5), name='Quantized'))
            fig_fft_adc.add_vline(x=target_sr/2.0, line=dict(color='red', width=2, dash='dash'))
            fig_fft_adc.update_layout(xaxis_title="Frequency (Hz)", template="plotly_dark", height=400)
            st.plotly_chart(fig_fft_adc, use_container_width=True)

  # TAB 22: 3D Spatial Audio & 8D Panning
  with tabs[21]:
    st.header("3D Spatial Audio & Binaural Panning (8D Audio)")
    
    col_3d1, col_3d2 = st.columns(2)
    spatial_mode = col_3d1.radio("Spatial Mode", ["Static 3D Position", "8D Auto-Rotation (Revolving)"])
    source_dist = col_3d2.slider("Source Distance (meters)", 0.5, 5.0, 1.0, 0.1)
    
    if spatial_mode == "Static 3D Position":
        static_angle = st.slider("Sound Angle (Degrees)", 0, 360, 90)
        rot_hz = 0.0
    else:
        rot_hz = st.slider("Rotation Speed (Revolutions per sec)", 0.1, 2.0, 0.5, 0.1)
        static_angle = 0.0
        
    if st.button("Synthesize 3D Binaural Audio"):
        with st.spinner("Calculating ITD/ILD..."):
            stereo_3d = apply_3d_spatial_audio(y, sr, spatial_mode, static_angle, source_dist, rot_hz)
            
            buffer_3d = io.BytesIO()
            sf.write(buffer_3d, stereo_3d, sr, format="WAV")
            st.audio(buffer_3d.getvalue(), format="audio/wav")
            
            plot_samples = min(int(sr * 0.05), len(stereo_3d))
            t_plot = np.linspace(0, 0.05, plot_samples, endpoint=False)
            
            fig_3d_wave = go.Figure()
            fig_3d_wave.add_trace(go.Scatter(x=t_plot*1000, y=stereo_3d[:plot_samples, 0], mode='lines', line=dict(color='#00BCD4', width=2), name="Left Ear"))
            fig_3d_wave.add_trace(go.Scatter(x=t_plot*1000, y=stereo_3d[:plot_samples, 1], mode='lines', line=dict(color='#E91E63', width=2), name="Right Ear"))
            fig_3d_wave.update_layout(xaxis_title="Time (ms)", template="plotly_dark", height=400)
            st.plotly_chart(fig_3d_wave, use_container_width=True)

  # TAB 23: Psychoacoustic Masking
  with tabs[22]:
    st.header("Psychoacoustic Masking (MP3 Compression Physics)")

    col_mask1, col_mask2 = st.columns(2)
    with col_mask1:
        target_freq = st.slider("Target Frequency (Hz)", 500.0, 4000.0, 2000.0, step=100.0)
        target_amp_db = st.slider("Target Volume (dB)", -60.0, -10.0, -35.0, step=1.0)
    with col_mask2:
        masker_freq = st.slider("Masker Center Frequency (Hz)", 500.0, 4000.0, 1000.0, step=100.0)
        masker_amp_db = st.slider("Masker Volume (dB)", -40.0, 0.0, -5.0, step=1.0)

    if st.button("Generate Masking Test Audio"):
        with st.spinner("Synthesizing acoustic illusion..."):
            sr_mask = 44100
            dur_mask = 3.0
            t_mask = np.linspace(0, dur_mask, int(sr_mask * dur_mask), endpoint=False)
            
            amp_t_linear = 10 ** (target_amp_db / 20.0)
            y_target = amp_t_linear * np.sin(2 * np.pi * target_freq * t_mask)
            
            amp_m_linear = 10 ** (masker_amp_db / 20.0)
            white_noise = np.random.randn(len(t_mask))
            
            nyq = sr_mask / 2.0
            low = max(50.0, masker_freq - 150) / nyq
            high = min(nyq - 50.0, masker_freq + 150) / nyq
            b, a = scipy.signal.butter(4, [low, high], btype='bandpass')
            y_masker = scipy.signal.filtfilt(b, a, white_noise)
            
            if np.max(np.abs(y_masker)) > 0:
                y_masker = y_masker / np.max(np.abs(y_masker))
            y_masker = y_masker * amp_m_linear
            
            y_combined = y_target + y_masker
            
            max_comb = np.max(np.abs(y_combined))
            if max_comb > 1.0:
                y_combined = y_combined / max_comb
                
            buffer_mask = io.BytesIO()
            sf.write(buffer_mask, y_combined, sr_mask, format="WAV")
            st.audio(buffer_mask.getvalue(), format="audio/wav")
            
            freqs_c, mag_c, mag_db_c = compute_fft(y_combined, sr_mask)
            valid_idx = freqs_c <= 5000 
            f_plot = freqs_c[valid_idx]
            m_plot = mag_db_c[valid_idx]
            
            mask_threshold = np.full_like(f_plot, -100.0) 
            for i, f in enumerate(f_plot):
                if f == 0: continue
                ratio = f / masker_freq
                drop = 30 * np.log10(ratio) if ratio >= 1 else -50 * np.log10(ratio)
                mask_threshold[i] = masker_amp_db - drop - 10 
                
            fig_mask = go.Figure()
            fig_mask.add_trace(go.Scatter(x=f_plot, y=m_plot, mode='lines', line=dict(color='#00BCD4', width=1.5), name="Audio Spectrum"))
            fig_mask.add_vline(x=target_freq, line=dict(color='#1DB954', width=2, dash='dash'))
            fig_mask.add_trace(go.Scatter(x=f_plot, y=mask_threshold, mode='lines', fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.2)', line=dict(color='#E91E63', width=2), name="Masking Threshold"))

            fig_mask.update_layout(xaxis_title="Frequency (Hz)", template="plotly_dark", height=450, yaxis=dict(range=[-80, 5]))
            st.plotly_chart(fig_mask, use_container_width=True)

  # TAB 24: Acoustic Levitation
  with tabs[23]:
    st.header("Acoustic Levitation (Standing Wave Physics)")
    st.latex(r"F_{rad} = \frac{5\pi}{6} \frac{P_0^2}{\rho c^2} R^3 k \sin(2kz)")
    
    col_lev1, col_lev2, col_lev3 = st.columns(3)
    lev_freq = col_lev1.slider("Transducer Frequency (Hz)", 20000.0, 50000.0, 40000.0, 1000.0)
    lev_dist_cm = col_lev2.slider("Transducer Distance (cm)", 2.0, 10.0, 4.28, 0.01)
    spl_db = col_lev3.slider("Sound Pressure Level (SPL dB)", 140, 170, 160, 1)

    if st.button("Calculate Levitation Field"):
        with st.spinner("Calculating acoustic radiation pressure field..."):
            c_air = 343.0 
            wavelength = c_air / lev_freq 
            k = 2 * np.pi / wavelength 
            
            p_ref = 2e-5 
            p_0 = p_ref * (10 ** (spl_db / 20.0))
            max_radius = wavelength / 3.0
            
            z_m = np.linspace(0, lev_dist_cm / 100.0, 500)
            p_z = p_0 * np.cos(k * z_m)
            acoustic_potential = p_z**2
            
            node_indices = scipy.signal.argrelextrema(acoustic_potential, np.less)[0]
            node_z = z_m[node_indices]
            
            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(x=z_m*100, y=np.abs(p_z), mode='lines', line=dict(color='rgba(0, 188, 212, 0.6)', width=2), fill='tozeroy', name='Pressure Envelope'))
            fig_lev.add_trace(go.Scatter(x=node_z*100, y=np.zeros_like(node_z), mode='markers', marker=dict(color='#E91E63', size=15, symbol='diamond'), name='Levitation Trap'))
            fig_lev.update_layout(title="Acoustic Levitation Field", xaxis_title="Z (cm)", template="plotly_dark", height=450)
            st.plotly_chart(fig_lev, use_container_width=True)

  # TAB 25: Granular Synthesis
  with tabs[24]:
    st.header("Granular Synthesis (Time-Stretching)")
    
    col_gran1, col_gran2, col_gran3 = st.columns(3)
    grain_size_ms = col_gran1.slider("Grain Size (ms)", 10.0, 100.0, 30.0, 5.0)
    stretch_factor = col_gran2.slider("Time Stretch Factor", 0.5, 4.0, 2.0, 0.1)
    overlap = col_gran3.slider("Grain Overlap", 0.1, 0.9, 0.5, 0.1)

    if st.button("Synthesize Grains & Stretch Time"):
        with st.spinner("Slicing audio into quantum grains..."):
            y_granular = apply_granular_synthesis(y, sr, grain_size_ms, stretch_factor, overlap)
            
            buffer_gran = io.BytesIO()
            sf.write(buffer_gran, y_granular, sr, format="WAV")
            st.audio(buffer_gran.getvalue(), format="audio/wav")

            plot_limit = min(len(y), int(sr * 1.0)) 
            plot_limit_gran = min(len(y_granular), int(sr * 1.0 * stretch_factor))
            
            fig_gran = make_subplots(rows=2, cols=1, shared_xaxes=False)
            fig_gran.add_trace(go.Scatter(x=np.linspace(0, 1, plot_limit), y=downsample_array(y[:plot_limit], 3000), line=dict(color='#00BCD4', width=1)), row=1, col=1)
            fig_gran.add_trace(go.Scatter(x=np.linspace(0, stretch_factor, plot_limit_gran), y=downsample_array(y_granular[:plot_limit_gran], 3000), line=dict(color='#E91E63', width=1)), row=2, col=1)
            
            fig_gran.update_layout(template="plotly_dark", height=500, showlegend=False)
            st.plotly_chart(fig_gran, use_container_width=True)

  # TAB 26: Analog Saturation
  with tabs[25]:
    st.header("Non-Linear Wavefolding & Saturation (Analog Physics)")
    
    col_dist1, col_dist2 = st.columns(2)
    dist_type = col_dist1.selectbox("Analog Distortion Circuit Type", [
        "Soft Clipping (Vacuum Tube)", "Hard Clipping (Transistor Fuzz)", 
        "Sine Wavefolding (Buchla Synth)", "Triangle Wavefolding"
    ])
    drive = col_dist2.slider("Drive / Input Gain (Multiplier)", 1.0, 10.0, 3.0, 0.5)

    if st.button("Apply Analog Saturation Circuit"):
        with st.spinner(f"Routing audio through {dist_type} simulator..."):
            y_distorted = apply_distortion(y, dist_type, drive)
            
            buffer_dist = io.BytesIO()
            sf.write(buffer_dist, y_distorted, sr, format="WAV")
            st.audio(buffer_dist.getvalue(), format="audio/wav")

            x_sweep = np.linspace(-1.0, 1.0, 1000)
            y_curve = apply_distortion(x_sweep, dist_type, drive)
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=x_sweep, y=x_sweep, mode='lines', line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dash')))
            fig_curve.add_trace(go.Scatter(x=x_sweep, y=y_curve, mode='lines', line=dict(color='#FF9800', width=3)))
            fig_curve.update_layout(template="plotly_dark", height=400, width=400, yaxis=dict(scaleanchor="x", scaleratio=1))
            
            plot_samples = min(len(y), int(sr * 0.05)) 
            fig_wave = go.Figure()
            y_orig_norm = y[:plot_samples] / np.max(np.abs(y[:plot_samples])) if np.max(np.abs(y[:plot_samples])) > 0 else y[:plot_samples]
            fig_wave.add_trace(go.Scatter(y=y_orig_norm, mode='lines', line=dict(color='rgba(0, 188, 212, 0.4)', width=1.5)))
            fig_wave.add_trace(go.Scatter(y=y_distorted[:plot_samples], mode='lines', line=dict(color='#E91E63', width=2)))
            fig_wave.update_layout(template="plotly_dark", height=400)

            col_ui1, col_ui2 = st.columns([1, 2])
            with col_ui1: st.plotly_chart(fig_curve, use_container_width=True)
            with col_ui2: st.plotly_chart(fig_wave, use_container_width=True)

            freqs_orig, mag_orig, mag_db_orig = compute_fft(y, sr)
            freqs_dist, mag_dist, mag_db_dist = compute_fft(y_distorted, sr)
            
            valid_idx_o = (freqs_orig > 0) & (freqs_orig <= (sr / 2.0))
            valid_idx_d = (freqs_dist > 0) & (freqs_dist <= (sr / 2.0))
            
            f_orig_p, m_orig_p = downsample_fft(freqs_orig[valid_idx_o], mag_db_orig[valid_idx_o], 5000)
            f_dist_p, m_dist_p = downsample_fft(freqs_dist[valid_idx_d], mag_db_dist[valid_idx_d], 5000)
            
            fig_fft_dist = go.Figure()
            fig_fft_dist.add_trace(go.Scatter(x=f_orig_p, y=m_orig_p, mode='lines', line=dict(color='rgba(0, 188, 212, 0.4)', width=1.5), fill='tozeroy'))
            fig_fft_dist.add_trace(go.Scatter(x=f_dist_p, y=m_dist_p, mode='lines', line=dict(color='#FFC107', width=1.5)))
            fig_fft_dist.update_layout(template="plotly_dark", height=400, xaxis_type="log")
            st.plotly_chart(fig_fft_dist, use_container_width=True)

  # TAB 27: Phase Vocoder
  with tabs[26]:
    st.header("Phase Vocoder & Spectral Cross-Synthesis")
    st.latex(r"\text{S}_{out}(f, t) = \left|\text{S}_{voice}(f, t)\right| \times \left|\text{S}_{synth}(f, t)\right| \cdot e^{i \angle \text{S}_{synth}(f, t)}")
    
    col_voc1, col_voc2 = st.columns(2)
    carrier_type = col_voc1.selectbox("Carrier Synth Waveform", ["Sawtooth Wave", "Square Wave", "Synth Chord (Minor 7th)", "White Noise (Whisper)"])
    carrier_freq = col_voc2.slider("Carrier Fundamental Pitch (Hz)", 50.0, 500.0, 110.0, 10.0)

    if st.button("Synthesize Vocoder Effect", type="primary"):
        with st.spinner("Executing Phase Vocoder STFT Cross-Synthesis..."):
            y_vocoded = apply_cross_synthesis(y, sr, carrier_type, carrier_freq)
            
            buffer_voc = io.BytesIO()
            sf.write(buffer_voc, y_vocoded, sr, format="WAV")
            st.audio(buffer_voc.getvalue(), format="audio/wav")
            
            f_orig, p_orig = compute_welch_psd(y, sr, nperseg=2048)
            f_voc, p_voc = compute_welch_psd(y_vocoded, sr, nperseg=2048)
            
            if len(f_orig) > 0 and len(f_voc) > 0:
                valid_idx = f_orig <= 5000 
                f_p_o, p_p_o = downsample_fft(f_orig[valid_idx], p_orig[valid_idx], 3000)
                f_p_v, p_p_v = downsample_fft(f_voc[valid_idx], p_voc[valid_idx], 3000)
                
                fig_voc = go.Figure()
                fig_voc.add_trace(go.Scatter(x=f_p_o, y=p_p_o, mode='lines', line=dict(color='rgba(233, 30, 99, 0.4)', width=2), fill='tozeroy'))
                fig_voc.add_trace(go.Scatter(x=f_p_v, y=p_p_v, mode='lines', line=dict(color='#00BCD4', width=2)))
                fig_voc.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_voc, use_container_width=True)

  # TAB 28: Audio Steganography
  with tabs[27]:
    st.header("Audio Steganography (Spectrogram Art)")
    if not HAS_PIL:
        st.error("⚠️ **Pillow Library Missing:** Please run `pip install Pillow` to use the text-to-image engine.")
    else:
        col_steg1, col_steg2 = st.columns(2)
        secret_text = col_steg1.text_input("Secret Word (Max 10 chars)", value="HELLO", max_chars=10)
        steg_dur = col_steg2.slider("Audio Duration (seconds)", 1.0, 5.0, 2.0, 0.5)
        
        col_steg3, col_steg4 = st.columns(2)
        min_f = col_steg3.slider("Minimum Frequency (Hz)", 100.0, 5000.0, 1000.0, step=100.0)
        max_f = col_steg4.slider("Maximum Frequency (Hz)", max(100.0, min_f + 100.0), 20000.0, 10000.0, step=500.0)

        if st.button("Synthesize Hidden Audio", type="primary"):
            with st.spinner("Converting text pixels to complex frequency matrices..."):
                sr_steg = 44100
                y_art, mag_spec, t_axis, freqs_arr = generate_spectrogram_art(
                    text=secret_text.upper(), sr=sr_steg, duration=steg_dur, min_freq=min_f, max_freq=max_f
                )
                
                buffer_art = io.BytesIO()
                sf.write(buffer_art, y_art, sr_steg, format="WAV")
                st.audio(buffer_art.getvalue(), format="audio/wav")
                
                safe_name = "".join(c for c in secret_text if c.isalnum())
                st.download_button(label="⬇️ Download Secret Audio", data=buffer_art.getvalue(), file_name=f"secret_{safe_name}.wav", mime="audio/wav")
                
                freq_factor = max(1, mag_spec.shape[0] // 300)
                time_factor = max(1, mag_spec.shape[1] // 400)
                
                fig_steg = go.Figure(data=go.Heatmap(
                    z=mag_spec[::freq_factor, ::time_factor], 
                    x=t_axis[::time_factor], 
                    y=freqs_arr[::freq_factor], 
                    colorscale="Inferno"
                ))
                fig_steg.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_steg, use_container_width=True)

  # TAB 29: Shepard Tone Illusion 
  with tabs[28]:
    st.header("The Shepard Tone Illusion (Psycho-Acoustic Maze)")
    st.latex(r"f_i(t) = f_{base} \cdot 2^{t/T + i} \quad \text{with} \quad A_i(t) = e^{-\frac{(t/T + i - N/2)^2}{2\sigma^2}}")
    
    col_shep1, col_shep2, col_shep3 = st.columns(3)
    shep_direction = col_shep1.radio("Illusion Direction", ["Ascending (Rising infinitely)", "Descending (Falling infinitely)"])
    shep_octaves = col_shep2.slider("Number of Octaves (N)", 3, 10, 6)
    shep_speed = col_shep3.slider("Sweep Speed (seconds per cycle)", 1.0, 10.0, 4.0, 0.5)

    if st.button("Generate Shepard Illusion", type="primary"):
        with st.spinner("Calculating exponential sweeps..."):
            sr_shep = 44100
            y_shep = generate_shepard_tone(
                cycle_dur=shep_speed, sr=sr_shep, f_min=55.0, 
                num_octaves=shep_octaves, direction=shep_direction.split(" ")[0], cycles=3
            )
            
            buffer_shep = io.BytesIO()
            sf.write(buffer_shep, y_shep, sr_shep, format="WAV")
            st.audio(buffer_shep.getvalue(), format="audio/wav")
            
            S_shep = compute_stft(y_shep, n_fft=2048, hop_length=512)
            max_f_display = min(55.0 * (2 ** (shep_octaves + 1)), sr_shep/2)
            
            t_axis_shep = librosa.frames_to_time(np.arange(S_shep.shape[1]), sr=sr_shep, hop_length=512)
            f_axis_shep = librosa.fft_frequencies(sr=sr_shep, n_fft=2048)
            
            valid_idx = (f_axis_shep > 0) & (f_axis_shep <= max_f_display)
            
            freq_factor = max(1, len(f_axis_shep[valid_idx]) // 300)
            time_factor = max(1, len(t_axis_shep) // 400)
            
            fig_shep = go.Figure(data=go.Heatmap(
                z=S_shep[valid_idx][::freq_factor, ::time_factor], 
                x=t_axis_shep[::time_factor], 
                y=f_axis_shep[valid_idx][::freq_factor], 
                colorscale="Magma"
            ))
            fig_shep.update_layout(yaxis_type="log", template="plotly_dark", height=500)
            st.plotly_chart(fig_shep, use_container_width=True)

  # TAB: BLACK HOLE RELATIVISTIC LAB
  with tabs[-3]: 
      if HAS_BLACKHOLE:
          blackhole_lab.render_blackhole_tab(y, sr)
      else:
          st.warning("⚠️ `blackhole_lab.py` module is missing from the working directory.")

  # TAB: QUANTUM MECHANICS LAB
  with tabs[-2]: 
      if HAS_QUANTUM:
          quantum_lab.render_quantum_tab(y, sr)
      else:
          st.warning("⚠️ `quantum_lab.py` module is missing from the working directory.")

  # TAB: Data Export & PDF Report
  with tabs[-1]:
      st.header("Export Analysis Data & Reports")
      
      col_exp1, col_exp2 = st.columns(2)
      
      with col_exp1:
          st.subheader("📄 Automated PDF Lab Report")
          st.markdown("Generate a comprehensive PDF summary of the audio analysis.")
          
          if st.button("Generate PDF Report", type="primary"):
              with st.spinner("Compiling scientific report..."):
                  if FPDF is None:
                      st.error("FPDF library is not installed (`pip install fpdf2`).")
                  else:
                      freqs_exp, mag_exp, _ = compute_fft(y, sr)
                      dom_idx = int(np.argmax(mag_exp)) if len(mag_exp) > 0 else 0
                      dom_freq = freqs_exp[dom_idx] if len(freqs_exp) > 0 else 0.0
                      fund_mag = mag_exp[dom_idx] if len(mag_exp) > 0 else 0.0
                      
                      harmonic_sq_sum = 0.0
                      for i in range(2, 11):
                          h_freq = dom_freq * i
                          if len(freqs_exp) == 0 or h_freq > freqs_exp[-1]:
                              break
                          idx = int(np.argmin(np.abs(freqs_exp - h_freq)))
                          window = mag_exp[max(0, idx - 3) : min(len(mag_exp), idx + 4)]
                          h_mag = np.max(window) if len(window) > 0 else 0
                          harmonic_sq_sum += h_mag**2
                      thd = ((np.sqrt(harmonic_sq_sum)) / fund_mag) * 100 if fund_mag > 1e-10 else 0.0
                      
                      rt60_exp, _, _, _, _, _, _ = estimate_rt60(y, sr)
                      
                      rms_exp = librosa.feature.rms(y=y)[0]
                      zcr_exp = librosa.feature.zero_crossing_rate(y)[0]
                      
                      pdf_bytes = create_lab_report(
                          file_name=file_name, sr=sr, duration=duration,
                          num_samples=num_samples, peak_amp=peak_amp, 
                          rms_mean=np.mean(rms_exp), zcr_mean=np.mean(zcr_exp),
                          rt60=rt60_exp, dom_freq=dom_freq, thd=thd
                      )
                      
                      if pdf_bytes:
                          st.success("Report Generated Successfully!")
                          st.download_button(
                              label="📥 Download PDF Lab Report",
                              data=pdf_bytes,
                              file_name=f"Lab_Report_{file_name.split('.')[0] if '.' in file_name else 'Audio'}.pdf",
                              mime="application/pdf"
                          )

      with col_exp2:
          st.subheader("📊 Raw Data Export")
          
          freqs_exp, mag_exp, mag_db_exp = compute_fft(y, sr)
          freqs_plot, mag_exp_plot = downsample_fft(freqs_exp, mag_exp, 5000)
          _, mag_db_exp_plot = downsample_fft(freqs_exp, mag_db_exp, 5000)

          df_fft = pd.DataFrame({
              "Frequency_Hz": freqs_plot,
              "Magnitude": mag_exp_plot,
              "Magnitude_dB": mag_db_exp_plot,
          })
          st.download_button("Download FFT Data (CSV)", df_fft.to_csv(index=False).encode("utf-8"), "fft_data.csv", "text/csv")

          rms = librosa.feature.rms(y=y)[0]
          zcr = librosa.feature.zero_crossing_rate(y)[0]
          times = librosa.frames_to_time(range(len(rms)), sr=sr)
          df_feat = pd.DataFrame({"Time_s": times, "RMS_Energy": rms, "Zero_Crossing_Rate": zcr})
          st.download_button("Download Audio Features (CSV)", df_feat.to_csv(index=False).encode("utf-8"), "audio_features.csv", "text/csv")

          buffer_mono = io.BytesIO()
          sf.write(buffer_mono, y, sr, format="WAV")
          st.download_button(label="Download Mono WAV", data=buffer_mono.getvalue(), file_name="mono_converted.wav", mime="audio/wav")

if __name__ == "__main__":
    main()
