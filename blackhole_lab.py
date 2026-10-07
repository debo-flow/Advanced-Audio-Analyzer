import streamlit as st
import numpy as np
import scipy.signal
import soundfile as sf
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom hashing function for caching numpy arrays efficiently in Streamlit
def fast_hash_np(x: np.ndarray):
    if x.size == 0: return 0
    return hash((x.shape, x[0], np.sum(x[::100])))

FAST_NP_HASH = {np.ndarray: fast_hash_np}

# Downsampler to prevent Plotly from crashing with large datasets
def downsample_array(array: np.ndarray, max_points: int = 10000) -> np.ndarray:
    if len(array) <= max_points: return array
    factor = len(array) // max_points
    return array[::factor]

@st.cache_data(hash_funcs=FAST_NP_HASH)
def apply_gravitational_redshift(y: np.ndarray, r_ratio: float) -> tuple:
    """
    Applies Gravitational Time Dilation (Redshift) to the audio signal.
    This is simulated by physically resampling the audio wave, stretching it 
    and lowering its frequency based on the Schwarzschild metric.
    """
    # Prevent division by zero at the exact event horizon
    if r_ratio <= 1.001: 
        r_ratio = 1.001
    
    # Calculate the Time Dilation Factor Z (dt / dtau)
    actual_Z = 1.0 / np.sqrt(1.0 - 1.0/r_ratio)
    
    # Cap the stretch factor to 5.0x to prevent memory overflow in Streamlit
    Z = min(actual_Z, 5.0) 
    
    orig_indices = np.arange(len(y))
    new_length = int(len(y) * Z)
    new_indices = np.linspace(0, len(y) - 1, new_length)
    
    # Fast linear interpolation for physical time stretching
    y_redshifted = np.interp(new_indices, orig_indices, y)
    
    return y_redshifted, actual_Z

@st.cache_data(hash_funcs=FAST_NP_HASH)
def generate_gw_chirp(m1: float, m2: float, sr: int, duration: float = 2.0) -> tuple:
    """
    Generates the Gravitational Wave Strain (h) and Frequency (f_GW) 
    for a binary black hole inspiral using the Post-Newtonian (PN) approximation.
    """
    # Physical constants
    G = 6.674e-11
    c = 3e8
    M_sun = 1.989e30
    
    # Convert solar masses to kilograms
    M1 = m1 * M_sun
    M2 = m2 * M_sun
    M_tot = M1 + M2
    
    # Calculate Chirp Mass
    M_c = ((M1 * M2)**0.6) / (M_tot**0.2)
    
    # Time array (from -duration up to the merger at t=0)
    t = np.linspace(-duration, 0.0, int(sr * duration), endpoint=False)
    
    # Time until merger (tau)
    tau = np.maximum(-t, 0.0001) 
    
    # Calculate GW frequency evolution over time
    const1 = (5.0 / 256.0) ** (3.0/8.0)
    const2 = (G * M_c / c**3) ** (-5.0/8.0)
    
    f_gw = (1.0 / np.pi) * const1 * (tau ** (-3.0/8.0)) * const2
    
    # Cap frequency to prevent Nyquist aliasing
    f_gw = np.minimum(f_gw, sr / 2.1) 
    
    # Exact phase integration
    phase = np.cumsum(2 * np.pi * f_gw / sr)
    
    # Strain amplitude envelope (grows as frequency increases)
    amp = f_gw ** (2.0/3.0)
    if np.max(amp) > 0: 
        amp = amp / np.max(amp)
    
    # Apply a simple fade-out at the ringdown (merger climax)
    fade_len = int(sr * 0.05)
    if len(amp) > fade_len:
        amp[-fade_len:] *= np.linspace(1, 0, fade_len)
        
    # Final gravitational wave strain signal
    strain = amp * np.cos(phase)
    
    return t, strain, f_gw


def render_blackhole_tab(y: np.ndarray, sr: int):
    """
    Renders the Black Hole Astrophysics UI inside the main Streamlit app.
    Call this function inside your designated tab context.
    """
    st.header("🌌 Black Hole Astrophysics & General Relativity")
    st.markdown("Experience how extreme gravity warps space, time, and sound! Modulate your audio through a Black Hole's **Event Horizon** or collide two massive singularities to generate **Gravitational Waves**.")
    
    # Split UI into two columns for the two physics experiments
    col_bh1, col_bh2 = st.columns(2)
    
    # ---------------------------------------------------------
    # Experiment 1: Gravitational Time Dilation (Redshift)
    # ---------------------------------------------------------
    with col_bh1:
        st.subheader("1. Gravitational Time Dilation")
        st.markdown("As you approach a Black Hole, time slows down relative to a distant observer (Gravitational Redshift). This physically stretches your audio wave and drops its pitch!")
        st.latex(r"d\tau = dt \sqrt{1 - \frac{r_s}{r}} \implies Z = \frac{1}{\sqrt{1 - \frac{1}{r_{ratio}}}}")
        
        # User Input: Distance from event horizon
        r_ratio = st.slider("Distance from Event Horizon ($r/r_s$)", 1.001, 10.0, 3.0, 0.001, help="1.0 is the exact Event Horizon. Time freezes completely here!")
        
        if st.button("Apply Relativistic Redshift", type="primary"):
            with st.spinner("Warping Spacetime Curve..."):
                
                # Execute physics model
                y_red, Z_factor = apply_gravitational_redshift(y, r_ratio)
                
                st.success(f"🕰️ **Time Dilation Factor (Z):** Time is running **{Z_factor:.2f}x slower** here!")
                
                # Audio Playback Generation
                buffer_red = io.BytesIO()
                sf.write(buffer_red, y_red, sr, format="WAV")
                st.audio(buffer_red.getvalue(), format="audio/wav")
                
                # Plot Waveform Comparison
                plot_samples = min(int(sr * 0.05), len(y))
                plot_red_samples = int(plot_samples * Z_factor)
                
                t_orig = np.linspace(0, 0.05, plot_samples)
                t_red = np.linspace(0, 0.05 * Z_factor, plot_red_samples)
                
                fig_bh_time = go.Figure()
                fig_bh_time.add_trace(go.Scatter(x=t_orig*1000, y=y[:plot_samples], name="Distant Observer (Earth)", line=dict(color='#00BCD4', width=1.5)))
                fig_bh_time.add_trace(go.Scatter(x=t_red*1000, y=y_red[:plot_red_samples], name=f"Near Black Hole (r={r_ratio}r_s)", line=dict(color='#E91E63', width=2)))
                
                fig_bh_time.update_layout(
                    title="Spacetime Waveform Expansion", 
                    xaxis_title="Proper Time (ms)", 
                    yaxis_title="Amplitude", 
                    template="plotly_dark", 
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_bh_time, use_container_width=True)

    # ---------------------------------------------------------
    # Experiment 2: Binary Merger (Gravitational Waves)
    # ---------------------------------------------------------
    with col_bh2:
        st.subheader("2. Binary Merger Gravitational Waves")
        st.markdown("When two black holes orbit and merge, they squeeze and stretch the fabric of spacetime. This simulation generates the famous GW 'Chirp' and modulates your audio with gravitational strain!")
        st.latex(r"f_{GW}(t) \propto \left( t_{merger} - t \right)^{-3/8} \quad \text{and} \quad h(t) \propto f_{GW}^{2/3} \cos(\Phi(t))")
        
        # User Input: Black Hole Masses
        m1 = st.slider("Black Hole 1 Mass ($M_\odot$)", 5.0, 100.0, 35.0, 1.0)
        m2 = st.slider("Black Hole 2 Mass ($M_\odot$)", 5.0, 100.0, 30.0, 1.0)
        
        if st.button("Generate Spacetime Strain Simulation", type="primary"):
            with st.spinner("Solving Post-Newtonian Inspiral Equations..."):
                
                # Execute physics model
                t_gw, strain, f_gw = generate_gw_chirp(m1, m2, sr, duration=2.0)
                
                chirp_mass = ((m1*m2)**0.6) / ((m1+m2)**0.2)
                st.success(f"🌊 **Gravitational Wave Generated!** Chirp Mass: **{chirp_mass:.2f} $M_\odot$**")
                
                # Squeeze User Audio with Strain (Amplitude Modulation)
                min_len = min(len(y), len(strain))
                strain_sub = strain[-min_len:]
                y_mod = y[:min_len] * (1.0 + 0.8 * strain_sub)
                
                # 1. Pure Gravitational Wave Playback
                st.markdown("**A. Pure Gravitational Wave (The 'Chirp'):**")
                buffer_chirp = io.BytesIO()
                sf.write(buffer_chirp, strain / np.max(np.abs(strain)), sr, format="WAV")
                st.audio(buffer_chirp.getvalue(), format="audio/wav")
                
                # 2. Modulated User Audio Playback
                st.markdown("**B. Your Audio Strained by Gravity (Modulated):**")
                buffer_mod = io.BytesIO()
                sf.write(buffer_mod, y_mod / np.max(np.abs(y_mod)), sr, format="WAV")
                st.audio(buffer_mod.getvalue(), format="audio/wav")

                # Visualizing the Chirp Math
                fig_chirp = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=("GW Spacetime Strain Amplitude $h(t)$", "Frequency Evolution $f_{GW}$ (Hz)")
                )
                
                # Downsample for faster UI rendering
                t_plot = downsample_array(t_gw, 3000)
                strain_plot = downsample_array(strain, 3000)
                f_plot = downsample_array(f_gw, 3000)
                
                fig_chirp.add_trace(go.Scatter(x=t_plot, y=strain_plot, line=dict(color='#00BCD4')), row=1, col=1)
                fig_chirp.add_trace(go.Scatter(x=t_plot, y=f_plot, line=dict(color='#E91E63')), row=2, col=1)
                
                fig_chirp.update_layout(
                    template="plotly_dark", 
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                    height=400, showlegend=False,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                fig_chirp.update_xaxes(title_text="Time to Merger (s)", row=2, col=1)
                st.plotly_chart(fig_chirp, use_container_width=True)
