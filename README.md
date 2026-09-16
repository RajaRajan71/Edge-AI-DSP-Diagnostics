# ⚡ Autonomous Edge-AI Telemetry & Anomaly Diagnostics Agent

An industrial-grade Edge-AI diagnostic system combining real-time Digital Signal Processing (DSP), a **LangGraph** decision agent, and a fine-tuned **Llama-3.2-3B (QLoRA)** model for real-time hardware anomaly detection and automated intervention.

---

## 🛠️ System Architecture

[ Hardware Telemetry ] ──> [ DSP Spectral Engine ] ──> [ LangGraph Agent ] ──> [ Fine-Tuned Llama-3.2-3B ] ──> [ Streamlit Dashboard ]

1. **Telemetry Simulation**: Generates synthetic voltage and vibration sensor signals with real-time anomaly injection capability.
2. **DSP Extraction Engine**: Runs Fast Fourier Transforms (FFT) using NumPy to convert raw time-domain signals into spectral features (Peak Frequency and RMS Power).
3. **LangGraph Agent Core**: Orchestrates automated decision workflows using a state machine (`capture_signal` $\rightarrow$ `process_dsp` $\rightarrow$ `agent_reasoning`).
4. **QLoRA Fine-Tuned LLM**: Powered by Llama-3.2-3B, fine-tuned down to a **0.15 training loss** on industrial error logs to provide structured root-cause analysis and operational mitigation steps.
5. **Interactive UI**: Real-time Streamlit monitoring dashboard displaying oscilloscope captures, raw JSON telemetry metrics, and actionable diagnostic logs.

---

## 📊 Key Features & Performance

* **Real-Time Feature Extraction**: Extracts dominant frequencies and RMS power metrics under 5ms.
* **Domain Adaptation**: Custom QLoRA fine-tuning yields low-latency, deterministic structured outputs (`DIAGNOSIS`, `ROOT CAUSE`, `ACTION`).
* **Interactive Control**: Toggle real-time $320\text{ Hz}$ high-frequency harmonic noise injection directly from the control panel.

---

## 📁 Repository Structure

```text
.
├── app.py                   # Main Streamlit web application
├── agent.py                 # LangGraph state machine agent logic
├── dsp_engine.py            # Signal simulation and FFT analysis engine
├── qlora_finetuning.ipynb   # Unsloth QLoRA fine-tuning notebook
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation

└── README.md                # Project documentation

##Quickstart (Local Run)
1. Clone the Repository

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name




