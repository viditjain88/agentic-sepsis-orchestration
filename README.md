# Jules: Fast-Track Agentic PoC for Sepsis Detection

This project implements an **Agentic Framework** for early sepsis detection and management using synthetic patient data. The system is designed to simulate a real-world clinical workflow where digital agents monitor vital signs, generate treatment plans based on Sepsis-3 guidelines, and execute orders via a mock FHIR interface.

## 🚀 Project Overview

The goal of this Proof-of-Concept (PoC) is to demonstrate the capabilities of an agentic system in a healthcare setting. Specifically, it aims to:
1.  **Monitor** patient data streams for Sepsis-3 screening criteria (HR, RR, Temp, Lactate).
2.  **Generate** personalized treatment plans using a Retrieval-Augmented Generation (RAG) pipeline powered by a local Large Language Model (LLM).
3.  **Execute** clinical orders autonomously (simulated).
4.  **Verify** and explain decisions using simulated SHAP feature importance.

## 🏗️ System Design

The architecture follows a modular pipeline:

1.  **Data Generation:**
    *   A Python script (`generate_synthetic_data.py`) creates a synthetic cohort of patients, mimicking the output of tools like Synthea.
    *   It generates diverse demographics and vital sign patterns, including specific sepsis indicators (e.g., elevated Lactate, Tachycardia).
    *   **Output:** `patients.csv`, `encounters.csv`, `observations.csv`.

2.  **Harmonization:**
    *   The raw CSV data is mapped to a unified, patient-centric JSON structure (`harmonize_data.py`).
    *   This aligns the data with a simplified MIMIC-IV schema (`vitalsign`, `labevents`, `admissions`).
    *   **Output:** `harmonized_data.json`.

3.  **Agentic Orchestration:**
    *   The core logic resides in `orchestrator.py`, built using **LangGraph**.
    *   It manages the flow between four specialized agents:
        *   **NLP Perceptor Agent:** Uses a Transformer-based sequence classification model (**Bio_ClinicalBERT**) to analyze unstructured clinical notes. It predicts the probability of sepsis presence based on the text.
        *   **Planner Agent:** Uses **Ollama (Gemma)** and RAG (Sepsis Guidelines) to reason about the patient's condition and decompose the "Sepsis Bundle" into tasks, triggered when the NLP model detects high risk.
        *   **Executor Agent:** Mocks a FHIR API to "place orders" (e.g., fluid bolus, antibiotics) and logs the actions.
        *   **Verifier Agent:** Provides explainability by calculating feature importance (simulated SHAP values) for the alert.

4.  **Evaluation:**
    *   The `evaluate.py` script calculates key performance metrics:
        *   **AUROC / AUPRC:** Accuracy of sepsis detection.
        *   **ECE:** Expected Calibration Error (Safety metric).
        *   **Latency:** System response time.
    *   It also generates visual assets for reporting (`shap_importance_plot.png`, `response_time_comparison.png`).

## 🤖 Agent Roles

| Agent | Responsibility | Logic / Tech |
| :--- | :--- | :--- |
| **NLP Perceptor** | Analyzing unstructured data | `emilyalsentzer/Bio_ClinicalBERT` Sequence Classification |
| **Planner** | Decision Making | LLM (Ollama/Gemma) + RAG (Guidelines) |
| **Executor** | Action | Mock FHIR API Interface |
| **Verifier** | Explainability | Simulated SHAP Analysis |

## 🛠️ Usage

### Prerequisites
- Python 3.9+
- Ollama (running locally with `gemma` model)

### Installation
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install and start Ollama (if not already running)
# (See ollama.com for instructions)
ollama pull gemma
```

### Running the Pipeline
```bash
# 1. Generate Synthetic Data
python3 generate_synthetic_data.py

# 2. Harmonize Data
python3 harmonize_data.py

# 3. Run the Agent Orchestration
python3 orchestrator.py

# 4. Evaluate Results
python3 evaluate.py
```

## 📊 Example Output

**Scenario:** Patient P006 presents with high heart rate and elevated lactate.

**Agent Logs:**
```
INFO:__main__:--- NLP PERCEPTOR AGENT (Bio_ClinicalBERT) ---
INFO:__main__:NLP Alert triggered with score 0.84 for P000
INFO:__main__:--- PLANNER AGENT ---
INFO:__main__:--- EXECUTOR AGENT ---
INFO:agents:Order 'Order Lactate Redraw' placed (ID: ORD-2293, Status: success)
INFO:agents:Order 'Administer 30mL/kg Crystalloid' placed (ID: ORD-9369, Status: success)
INFO:agents:Order 'Order Blood Cultures' placed (ID: ORD-3136, Status: success)
INFO:agents:Order 'Administer Broad-Spectrum Antibiotics' placed (ID: ORD-5631, Status: success)
INFO:__main__:--- VERIFIER AGENT ---
```

**Explanation:**
```
Feature Importance Analysis (SHAP-proxy):
- Heart Rate: 117.0 (Importance: 0.75)
- Lactate: 3.4 (Importance: 0.04)
```

## 📈 Results (PoC Batch)

- **AUROC:** ~0.95
- **AUPRC:** ~0.89
- **Latency:** ~3.6s (avg)

*Note: Metrics are based on a small synthetic batch and simulated latency.*
