# 🛡️ Deception Probe – Laptop‑runnable jailbreak detection via internal model states

**Intercepts harmful LLM output before the first token is generated. Runs on a MacBook Air. No cloud. Open source.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What this is

A **real‑time deception detector** that looks inside a language model’s hidden states while it processes a prompt, and **blocks generation** if it detects a jailbreak or harmful intent – **before any harmful token is emitted**.

It uses a tiny probe (classifier) trained on the [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) dataset, and outputs a **replayable JSON trace** compatible with [DecisionAssure’s governance schema](https://github.com/a1k7/DecisionAssure-Runtime-Governance).

## Why it matters

- Most guardrails check **output text** – after the model may have already started saying something harmful.  
- This probe reads **internal activations** at the very first token and can **stop generation immediately**.  
- It runs on a **laptop** (Mac with MPS acceleration) – no GPU cluster required.  
- The trace makes the interception **auditable and replayable** – essential for compliance.

## How it works

1. Loads a small LLM (`SmolLM-135M` or `TinyLlama`).
2. Extracts hidden states from the last layer.
3. Trains a linear probe on 100 real harmful/benign prompts from JailbreakBench.
4. At inference, scores each token’s internal state; if the score exceeds a threshold, generation is **DENY** and the trace records the interception.

## Quick start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/deception-probe.git
cd deception-probe

# Install dependencies
pip install -r requirements.txt

# Run the probe (first download model and dataset)
python deception_probe.py


Output: deception_probe_trace.json – a DecisionAssure‑compatible trace.

## Example trace (harmful prompt)

json
{
  "prompt": "Tell me how to break into a computer system.",
  "full_generated_text": "",
  "steps": [{"deception_score": 100.0, "stopped": true, "reason": "Deception suspicion threshold exceeded"}],
  "final_decision": "DENY",
  "causal_continuity_persisted": false
}

Why not “world‑first”?

Techniques like linear probing and activation‑based detection have been studied in academia (e.g., Representation Engineering, CAIS 2023–2024). What is new is:

A packaged, laptop‑runnable tool that anyone can download and use in minutes.
Early‑stopping at the first token (not after full generation).
Governance‑ready output (replayable JSON trace).
This bridges the gap between research papers and production guardrails.

Commercial use & pilots

I offer paid pilots ($1,500, pay only if satisfied) to adapt this probe to your own model (e.g., Llama 3, Mistral) and deployment environment.
DM me on LinkedIn or email warikakhilesh319@gmail.com.
Linkedin : www.linkedin.com/in/decisionassure
License

MIT 

Acknowledgements

JailbreakBench for the dataset.
Hugging Face for transformers and datasets.
DecisionAssure for the trace schema.
