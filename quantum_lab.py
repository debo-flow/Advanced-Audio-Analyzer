import numpy as np
import plotly.graph_objects as go
import scipy.fft
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
# QUANTUM TUNNELING PHYSICS ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def apply_quantum_tunneling(y, sr, barrier_height, barrier_width, particle_mass):
    """
    Applies the Quantum Tunneling probability equation (Transmission Coefficient)
    to the audio frequencies, treating them as energy states of a wavepacket.
    """
    # 1. Transform audio to Frequency Domain (Energy States)
    fft_result = scipy.fft.rfft(y)
    freqs = scipy.fft.rfftfreq(len(y), 1/sr)
    
    # Physics Mapping: E (Energy) is mapped to Frequency
    E = np.abs(freqs) + 1e-6  # Avoid zero division
    V = barrier_height
    L = barrier_width
    m = particle_mass
    
    # 2. Schrödinger Transmission Probability Equation T(E)
    # T = e^(-2 * L * sqrt(2m * (V - E))) for E < V
    # T = 1 for E >= V (Classical passage)
    kappa = np.sqrt(2 * m * np.maximum(V - E, 0))
    T_prob = np.exp(-2 * L * kappa)
    T_prob[E >= V] = 1.0  # Frequencies with energy > barrier pass completely
    
    # 3. Apply Quantum Filter (Tunneling)
    tunneled_fft = fft_result * T_prob
    
    # 4. Inverse FFT to get the audio that made it to the other side
    y_tunneled = scipy.fft.irfft(tunneled_fft, n=len(y))
    
    # Normalize Output
    max_val = np.max(np.abs(y_tunneled))
    if max_val > 0:
        y_tunneled /= max_val
        
    return y_tunneled, T_prob, freqs

# ==========================================
# UI RENDER FUNCTION
# ==========================================
def render_quantum_tab(y, sr):
    st.header("Quantum Mechanics: Schrödinger Equation & Tunneling")
    st.markdown("In classical physics, if a sound wave (or particle) doesn't have enough energy, it bounces off a wall. But in Quantum Mechanics, the **Schrödinger Equation** reveals that waves have a probability of simply 'teleporting' through solid barriers. This is called **Quantum Tunneling**.")
    
    st.latex(r"T \approx e^{-2L \sqrt{\frac{2m(V - E)}{\hbar^2}}}")
    
    col1, col2, col3 = st.columns(3)
    barrier_height = col1.slider("Barrier Potential (V)", 100.0, 5000.0, 2000.0, 100.0, help="Energy required to cross classically.")
    barrier_width = col2.slider("Barrier Width (L) - Nano-scale", 0.001, 0.05, 0.01, 0.001)
    particle_mass = col3.slider("Particle Mass (m)", 0.1, 5.0, 1.0, 0.1)

    if st.button("Simulate Quantum Tunneling", type="primary"):
        with st.spinner("Solving Schrödinger's wave equation..."):
            
            # Run Physics Engine
            y_tunneled, T_prob, freqs = apply_quantum_tunneling(y, sr, barrier_height, barrier_width, particle_mass)
            
            st.success("⚛️ **Wavepacket Tunneled!** The audio you hear below mathematically 'leaked' through the solid barrier.")
            
            # Audio Playback
            st.audio(get_audio_bytes(y_tunneled, sr), format="audio/wav")
            
            # Visualization: The Potential Barrier & Tunneling Probability
            st.divider()
            st.subheader("Wavepacket Transmission Probability Graph")
            
            valid_idx = freqs <= 8000 # Plot up to 8kHz for visibility
            f_plot = freqs[valid_idx]
            T_plot = T_prob[valid_idx]
            
            fig_quantum = go.Figure()
            
            # Plot the solid barrier wall
            fig_quantum.add_vrect(
                x0=0, x1=barrier_height,
                fillcolor="rgba(233, 30, 99, 0.2)", opacity=0.5,
                line_width=2, line_color="#E91E63",
                annotation_text="Solid Barrier (V)", annotation_position="top left"
            )
            
            # Plot the transmission probability curve
            fig_quantum.add_trace(go.Scatter(
                x=f_plot, y=T_plot, 
                mode='lines', 
                line=dict(color='#00BCD4', width=2),
                fill='tozeroy', fillcolor='rgba(0, 188, 212, 0.3)',
                name="Transmission Probability T(E)"
            ))
            
            fig_quantum.update_layout(
                title="Quantum Tunneling Probability vs. Wave Energy (Frequency)",
                xaxis_title="Wave Energy / Frequency (E)",
                yaxis_title="Probability of Tunneling (T)",
                template="plotly_dark",
                height=450,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_quantum, use_container_width=True)
            
            st.info("💡 **Physics Insight:** Notice how frequencies (Energy $E$) that are LOWER than the pink barrier ($V$) still have a small probability of passing through (the blue curve doesn't instantly hit zero). This is physically impossible in the classical world, but happens billions of times a second in the sun and inside your computer's microchips!")
