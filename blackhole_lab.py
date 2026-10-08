import numpy as np
import plotly.graph_objects as go
import scipy.signal
import streamlit as st
import soundfile as sf
import io

# ==========================================
# SAFE AUDIO BYTES HELPER
# ==========================================
def get_audio_bytes(data: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# BLACK HOLE PHYSICS ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def apply_blackhole_physics(y, sr, mass_solar, distance_km):
    """
    Simulates Acoustic Redshift and Gravitational Time Dilation 
    near a Schwarzschild Black Hole.
    """
    # Physics Constants
    G = 6.67430e-11         # Gravitational constant
    c = 299792458           # Speed of light
    M_sun = 1.989e30        # Mass of the sun
    
    # Calculate Schwarzschild Radius (Event Horizon)
    M = mass_solar * M_sun
    r_s = (2 * G * M) / (c**2)  # in meters
    
    # Current distance from the singularity
    r = r_s + (distance_km * 1000) 
    
    # Time dilation factor: sqrt(1 - rs/r)
    if r <= r_s:
        # At or inside the event horizon, time stops relative to outside observer
        dilation_factor = 0.001 
    else:
        dilation_factor = np.sqrt(1 - (r_s / r))
        
    # Acoustic Redshift: Frequencies drop and time stretches.
    # To simulate, we resample (stretch) the audio by 1/dilation_factor.
    # We cap the stretch factor to 20x to prevent RAM crashes.
    stretch_factor = max(1.0, min(1.0 / (dilation_factor + 1e-6), 20.0)) 
    
    new_length = int(len(y) * stretch_factor)
    y_dilated = scipy.signal.resample(y, new_length)
    
    # Apply Lowpass filter to simulate energy loss (extreme redshift masking high frequencies)
    nyquist = sr / 2.0
    cutoff = max(50.0, nyquist * dilation_factor)
    
    b, a = scipy.signal.butter(4, cutoff / nyquist, btype='low')
    y_redshifted = scipy.signal.filtfilt(b, a, y_dilated)
    
    # Normalize output to prevent clipping
    max_val = np.max(np.abs(y_redshifted))
    if max_val > 0:
        y_redshifted /= max_val
        
    return y_redshifted, r_s, dilation_factor

# ==========================================
# UI RENDER FUNCTION
# ==========================================
def render_blackhole_tab(y, sr):
    st.header("🌌 Black Hole Relativistic Lab (Event Horizon)")
    st.markdown("Experience **Gravitational Time Dilation** and **Acoustic Redshift** as your audio approaches a supermassive black hole. According to Einstein's General Relativity, as you get closer to the Event Horizon (Schwarzschild radius), spacetime curves severely. Time slows down relative to an outside observer, stretching your sound and dropping its pitch infinitely!")
    
    st.latex(r"\Delta t' = \Delta t \sqrt{1 - \frac{2GM}{rc^2}}")
    
    col1, col2 = st.columns(2)
    mass = col1.slider("Black Hole Mass (Solar Masses)", 1.0, 100.0, 10.0, 1.0)
    dist = col2.slider("Distance from Event Horizon (km)", 0.0, 100.0, 50.0, 1.0, help="At 0 km, you touch the Event Horizon.")
    
    if st.button("Simulate Spacetime Curvature", type="primary"):
        with st.spinner("Calculating tensor metrics and dilating time..."):
            
            # Run Physics Engine
            y_bh, r_s, dilation = apply_blackhole_physics(y, sr, mass, dist)
            
            # Metrics Feedback
            st.success(f"⚛️ **Spacetime Warped!** Time is ticking at **{dilation*100:.2f}%** of normal speed.")
            st.info(f"📏 **Event Horizon Radius ($r_s$):** {r_s / 1000:.2f} km")
            
            # Audio Playback
            st.audio(get_audio_bytes(y_bh, sr), format="audio/wav")
            
            # Visualization: Spacetime Gravity Well
            st.divider()
            st.subheader("Spacetime Curvature (Gravity Well)")
            
            # Create a 3D grid for the gravity well
            x = np.linspace(-3, 3, 100)
            y_grid = np.linspace(-3, 3, 100)
            X, Y = np.meshgrid(x, y_grid)
            
            R = np.sqrt(X**2 + Y**2) + 0.1
            Z = -1.0 / R  # Inverse relationship for the gravity well
            
            # Truncate Z to visually represent the event horizon "hole"
            Z[R < 0.5] = -2.0
            
            fig_bh = go.Figure(data=[go.Surface(
                z=Z, x=X, y=Y, 
                colorscale="Magma",
                showscale=False
            )])
            
            fig_bh.update_layout(
                title="3D Visualization of the Singularity",
                template="plotly_dark",
                height=600,
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                    zaxis=dict(range=[-2, 0], showticklabels=False, showgrid=False, zeroline=False, title="Spacetime Depth")
                )
            )
            st.plotly_chart(fig_bh, use_container_width=True)
