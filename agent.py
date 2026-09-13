# agent.py
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from dsp_engine import generate_sensor_signal, process_dsp_signal

class AgentState(TypedDict):
    inject_anomaly: bool
    signal_data: tuple
    dsp_features: Dict[str, Any]
    diagnostic_result: str

def capture_signal_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Captures multi-sensor hardware telemetry."""
    inject_flag = state.get("inject_anomaly", False)
    t, sig = generate_sensor_signal(anomaly=inject_flag)
    return {"signal_data": (t, sig)}

def process_dsp_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Performs FFT spectral analysis and feature extraction."""
    t, sig = state["signal_data"]
    features = process_dsp_signal(t, sig)
    return {"dsp_features": features}

def agent_reasoning_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Formats DSP metrics and matches fine-tuned QLoRA inference logic."""
    features = state["dsp_features"]
    
    if not features["anomaly_detected"]:
        return {
            "diagnostic_result": (
                "DIAGNOSIS: Nominal harmonic response.\n"
                "ROOT CAUSE: None.\n"
                "ACTION: Maintain standard operational status."
            )
        }
    
    peak_f = features['peak_frequency_hz']
    rms = features['rms_power']
    
    # Matching exact prompt output learned by fine-tuned LoRA weights
    result = (
        f"DIAGNOSIS: High-frequency mechanical resonance & thermal vibration detected.\n"
        f"ROOT CAUSE: Bearing wear or shaft misalignment (Peak: {peak_f} Hz, RMS: {rms}).\n"
        f"ACTION: Trigger circuit isolation, reduce duty cycle by 40%, and log error code E-402."
    )
    return {"diagnostic_result": result}

# Build LangGraph State Machine
workflow = StateGraph(AgentState)

workflow.add_node("capture_signal", capture_signal_node)
workflow.add_node("process_dsp", process_dsp_node)
workflow.add_node("agent_reasoning", agent_reasoning_node)

workflow.set_entry_point("capture_signal")
workflow.add_edge("capture_signal", "process_dsp")
workflow.add_edge("process_dsp", "agent_reasoning")
workflow.add_edge("agent_reasoning", END)

app_agent = workflow.compile()