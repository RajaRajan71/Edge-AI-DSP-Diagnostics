import json
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from unsloth import FastLanguageModel

# Define state structure
class AgentState(TypedDict):
    signal_data: list
    dsp_features: dict
    diagnostic_result: str

# 1. Initialize model and tokenizer from your local lora_model directory
print("Loading fine-tuned QLoRA model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Node 1: Capture Raw Signal (Placeholder or call dsp_engine)
def capture_signal_node(state: AgentState) -> Dict[str, Any]:
    # Simulating or retrieving raw input array
    return {"signal_data": state.get("signal_data", [])}

# Node 2: Process Signal DSP Telemetry
def process_dsp_node(state: AgentState) -> Dict[str, Any]:
    # In a real setup, import process_signal from dsp_engine.py
    # Simulating DSP extracted metrics:
    features = {
        "peak_frequency_hz": 320.0,
        "rms_power": 1.53,
        "anomaly_detected": True
    }
    return {"dsp_features": features}

# Node 3: LLM Reasoning Node using Fine-Tuned LoRA Adapter
def agent_reasoning_node(state: AgentState) -> Dict[str, Any]:
    features = state["dsp_features"]
    peak_f = features["peak_frequency_hz"]
    rms = features["rms_power"]
    anomaly = features.get("anomaly_detected", True)

    # Reconstruct the exact prompt layout used during QLoRA training
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Analyze telemetry DSP metrics and determine corrective system action.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Peak Frequency: {peak_f} Hz, RMS Power: {rms}, Anomaly Detected: {anomaly}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Tokenize input and run inference on GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    decoded_text = tokenizer.batch_decode(outputs)[0]

    # Extract assistant's output by stripping the prompt tags
    if "<|start_header_id|>assistant<|end_header_id|>\n" in decoded_text:
        result = decoded_text.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1]
        result = result.replace("<|eot_id|>", "").strip()
    else:
        result = decoded_text

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

# Execution block
if __name__ == "__main__":
    initial_state = {"signal_data": [0.1, 0.5, 1.2, 0.8]}
    output = app_agent.invoke(initial_state)
    
    print("\n--- Final Fine-Tuned AI Diagnostic Output ---")
    print(output["diagnostic_result"])
