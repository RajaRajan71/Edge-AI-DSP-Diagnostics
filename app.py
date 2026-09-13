# app.py
import streamlit as st
import matplotlib.pyplot as plt
from agent import app_agent

st.set_page_config(page_title="EdgeAgent-DSP Telemetry", layout="wide")

st.title("⚡ Autonomous Edge-AI Telemetry & Anomaly Diagnostics Agent")
st.markdown("---")

st.sidebar.header("🕹️ Hardware Control Panel")
inject_anomaly = st.sidebar.toggle("Inject Hardware Anomaly (320Hz Noise)", value=True)

# Run button or use session state to persist results
run_diagnostic = st.button("🚀 Run Real-Time Telemetry Diagnostic Pass", use_container_width=True)

if run_diagnostic or "agent_result" not in st.session_state:
    st.session_state.agent_result = app_agent.invoke({"inject_anomaly": inject_anomaly})

result = st.session_state.agent_result
t, sig = result["signal_data"]
features = result["dsp_features"]

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📈 Real-Time Sensor Telemetry Waveform")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t[:250], sig[:250], color="#d62728" if features["anomaly_detected"] else "#1f77b4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (Voltage/Vibration)")
    ax.set_title("Live Oscilloscope Capture (First 250ms)")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("📊 DSP FFT Spectral Extraction Metrics")
    st.json(features)
    
    st.subheader("🤖 LangGraph Agent Diagnostic Output")
    if features["anomaly_detected"]:
        st.error(result["diagnostic_result"])
    else:
        st.success(result["diagnostic_result"])