# dsp_engine.py
import numpy as np

def generate_sensor_signal(duration=1.0, sampling_rate=1000, anomaly=False):
    """Generates synthetic telemetry signal (vibration/voltage waveform)."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Base nominal signal (50 Hz primary + 120 Hz harmonic)
    base_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    
    if anomaly:
        # Inject high-frequency distortion (320 Hz) and increased thermal noise
        noise = np.random.normal(0, 0.8, t.shape)
        anomaly_signal = 1.5 * np.sin(2 * np.pi * 320 * t)
        return t, base_signal + anomaly_signal + noise
    
    noise = np.random.normal(0, 0.1, t.shape)
    return t, base_signal + noise

def process_dsp_signal(t, signal, sampling_rate=1000):
    """Performs FFT feature extraction to compute peak frequency and RMS power."""
    n = len(signal)
    fft_vals = np.abs(np.fft.fft(signal))[:n // 2]
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)[:n // 2]
    
    peak_freq = freqs[np.argmax(fft_vals)]
    rms_power = np.sqrt(np.mean(signal**2))
    
    # Anomaly threshold check based on spectral response
    is_anomaly = (rms_power > 1.2) or (peak_freq > 200)
    
    return {
        "peak_frequency_hz": round(float(peak_freq), 2),
        "rms_power": round(float(rms_power), 4),
        "anomaly_detected": bool(is_anomaly)  # Ensures clean JSON output in Streamlit
    }

if __name__ == "__main__":
    t, sig = generate_sensor_signal(anomaly=True)
    features = process_dsp_signal(t, sig)
    print("DSP Feature Extraction Result:", features)