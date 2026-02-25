# Agentic Sepsis Orchestration — AI-Driven Early Detection & Clinical Decision Support

An **AI-driven agentic framework** for early sepsis detection and automated clinical order generation using synthetic patient data, NLP-based entity recognition, and advanced Large Language Models (LLMs).

---

## 🚀 Project Overview

This Proof-of-Concept (PoC) demonstrates an agentic AI system for healthcare that:

1. **Monitors** patient data streams using clinical NLP (LOINC entity recognition + Sepsis-3 threshold pattern matching)
2. **Generates** personalized treatment plans via a Retrieval-Augmented Generation (RAG) pipeline powered by a local LLM (Ollama/Gemma)
3. **Executes** clinical orders autonomously via a simulated FHIR interface
4. **Explains** decisions using SHAP-proxy feature importance scoring

---

## 🏗️ System Architecture

```
Synthetic Data → Harmonization → [ Perceptor → Planner → Executor → Verifier ] → Evaluation
                                         ↑ LangGraph Orchestration ↑
```

### Pipeline Steps

| Step | Script | Output |
|------|--------|--------|
| 1. Data Generation | `generate_synthetic_data.py` | `patients.csv`, `encounters.csv`, `observations.csv` |
| 2. Harmonization | `harmonize_data.py` | `harmonized_data.json` |
| 3. Agent Orchestration | `orchestrator.py` | `orchestration_results.json` |
| 4. Evaluation | `evaluate.py` | `evaluation_metrics.csv`, plots |

---

## 🤖 Agent Roles

| Agent | Responsibility | Technique |
|-------|---------------|-----------|
| **Perceptor** | Sepsis screening | LOINC entity recognition + Sepsis-3 threshold pattern matching |
| **Planner** | Treatment planning | Ollama (Gemma) LLM + RAG over sepsis guidelines |
| **Executor** | Order placement | Mock FHIR API interface |
| **Verifier** | Explainability | SHAP-proxy feature importance (deviation from clinical baseline) |

### Clinical NLP — Perceptor Agent

The Perceptor applies two clinical NLP techniques:

1. **LOINC-coded entity recognition** — maps structured observation codes to clinical concepts:
   - `8867-4` → Heart Rate (tachycardia marker)
   - `9279-1` → Respiratory Rate (tachypnea marker)
   - `8310-5` → Body Temperature (hyperthermia marker)
   - `32693-4` → Lactate (hyperlactatemia marker)

2. **Threshold-based pattern matching** (Sepsis-3 criteria):
   - HR > 90 bpm → +1 risk point
   - RR ≥ 22 breaths/min → +1 risk point
   - Temp > 38.0°C → +1 risk point
   - Lactate > 2.0 mmol/L → +2 risk points (strong indicator)
   - Alert triggered if **risk score ≥ 2**

---

## 🛠️ Usage

### Prerequisites
- Python 3.9+
- Ollama running locally with the `gemma` model

### Installation
```bash
pip install -r requirements.txt
ollama pull gemma
```

### Running the Pipeline
```bash
# 1. Generate synthetic data (200 patients)
python3 generate_synthetic_data.py

# 2. Harmonize to unified JSON
python3 harmonize_data.py

# 3. Run agent orchestration
python3 orchestrator.py

# 4. Evaluate and generate plots
python3 evaluate.py
```

---

## 📊 Results (200-Patient Cohort)

| Metric | Value |
|--------|-------|
| **Patients** | 200 |
| **Encounters** | 603 |
| **Alerts Triggered** | 299 (49.6%) |
| **AUROC** | 0.9886 |
| **AUPRC** | 0.9761 |
| **ECE** (calibration) | 0.0481 |
| **Avg Latency — Alert path** (LLM) | 3.86s |
| **Avg Latency — Non-alert path** | 0.10s |

> *Latency reflects simulated LLM inference. All other metrics computed from actual pipeline output.*

### Evaluation Dashboard
![Evaluation Dashboard](output/evaluation_dashboard.png)

### Precision-Recall Curve
![Precision-Recall Curve](output/precision_recall_curve.png)

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `output/patients.csv` | 200 synthetic patients |
| `output/encounters.csv` | 603 clinical encounters |
| `output/observations.csv` | LOINC-coded vital sign observations |
| `output/harmonized_data.json` | Unified patient-event JSON (MIMIC-IV schema) |
| `output/orchestration_results.json` | Full per-encounter agent pipeline results |
| `output/evaluation_metrics.csv` | AUROC, AUPRC, ECE, latency |
| `output/evaluation_dashboard.png` | ROC curve + SHAP importance + latency distribution |
| `output/precision_recall_curve.png` | Precision-Recall curve |

---

## ⚠️ Limitations

- Synthetic data only — not validated on real EHR data
- Planner LLM (Ollama/Gemma) requires local installation; offline runs use mocked orders
- SHAP values are proxies (deviation-based), not true SHAP from a trained ML model
- Latency figures are simulated for the LLM inference component

---

## 🔭 Future Work

- Validate on real-world EHR data (MIMIC-IV)
- Integrate full SOFA score computation
- Extend NLP pipeline to unstructured clinical notes (discharge summaries, nursing notes)
- Replace SHAP-proxy with true SHAP from a trained gradient-boosted model
- Clinical validation for responsible deployment in acute care settings
