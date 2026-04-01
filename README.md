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

## 📝 Recent Changes (Code Audit)

- Added **clinical-note + cellular-data augmentation** in `generate_synthetic_data.py`:
   - Optional HuggingFace synthetic-note ingestion with fallback note generation
   - New per-encounter `CLINICAL_NOTE` and `CELLULAR_DATA` fields
   - Cellular network graph schema now exported as `{"nodes": [...], "links": [...]}`

- Upgraded **harmonization pipeline** in `harmonize_data.py`:
   - Added safe CSV loading (`load_csv`) and missing-file guardrails
   - Preserved `clinical_note` and parsed `cellular_data` into `harmonized_data.json`
   - Standardized patient demographics casting and dynamic age derivation

- Expanded **agent framework** in `agents.py` and API integration in `backend/api.py`:
   - `PerceptorAgent` now includes MedCAT-based clinical entity extraction from notes
   - Added `TherapeuticsAgent` for cellular heat-signature-informed adjunct therapies
   - Added `EvaluatorAgent` (LLM-as-judge style heuristic scoring)
   - `/api/monitor` now returns `clinical_note`, `extracted_entities`, `cellular_data`, and `evaluation`

- Updated **planner behavior** in `orchestrator.py`:
   - `PlannerAgent.plan()` now accepts optional `cellular_data`
   - Base sepsis bundle can be extended with therapeutic recommendations
   - Batch orchestration now passes visit-level `cellular_data` so therapeutic extensions appear in `orchestration_results.json`

- Refreshed **frontend experience** in `frontend/src/App.jsx`:
   - Added richer clinical output panels (notes/entities, cellular graph, evaluator score)
   - Added network visualization using `react-force-graph-2d`
   - Added evaluator step in the visible pipeline timeline
   - Patient form vitals now prefill dynamically from the selected patient’s latest visit (instead of static defaults)

- Added **MedCAT bootstrap utility** in `medcat_processor.py`:
   - Attempts model-pack download/load automatically
   - Falls back to deterministic keyword matching when model initialization is unavailable

---

## 🏗️ System Architecture

```
Synthetic Data → Harmonization → [ Perceptor → Planner → Executor → Verifier → Evaluator ] → Evaluation
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
| **Evaluator** | Plan quality scoring | LLM-as-a-Judge style heuristic using extracted entities + generated plan alignment |

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

## 🧠 Background (Expanded)

Sepsis‑3 emphasizes rapid identification using readily available vitals (HR, RR, Temp) and lactate, which are routinely captured in EHR workflows. These markers reflect systemic stress, inflammatory response, and hypoperfusion, making them practical early signals before full SOFA scoring is available. This PoC builds on that clinical baseline by pairing transparent threshold screening with an agentic workflow that coordinates perception, planning, execution, and verification. The aim is not to claim clinical efficacy but to demonstrate an auditable, end‑to‑end orchestration pattern that can translate alerts into guideline‑consistent actions.

- Perception: detect risk via Sepsis‑3 thresholds (HR/RR/Temp/Lactate)
- Planning: assemble a sepsis bundle using guideline retrieval
- Execution: place mock FHIR orders for fluids, antibiotics, and cultures
- Verification: provide SHAP‑proxy feature importance for explainability
- Evaluation: score plan/clinical-note alignment via LLM-as-a-Judge style rubric

---

## 🧪 Methods

We generated a synthetic cohort (200 patients, 629 encounters) and harmonized vitals/labs into a unified JSON schema to drive orchestration. Evaluation computes AUROC, AUPRC, ECE, and simulated latency from pipeline outputs, and visual assets (latency comparison, global feature importance, dashboard views) are generated for reporting and posterization.

- Data: synthetic patients/encounters; unified JSON schema (`harmonized_data.json`)
- Perceptor: Sepsis‑3 thresholds on HR/RR/Temp/Lactate
- Planner: RAG over Sepsis guidelines + local LLM (Ollama/Gemma)
- Executor: mock FHIR order placement
- Verifier: SHAP‑proxy feature importance from clinical deviations
- Evaluator: LLM-as-a-Judge style scoring (`score`, `alignment`, `feedback`)
- Risk scoring: HR>90 (+1), RR≥22 (+1), Temp>38 (+1), Lactate>2 (+2), alert at ≥2
- Orders: lactate redraw, 30 mL/kg fluids, blood cultures, broad‑spectrum antibiotics
- Outputs: `orchestration_results.json`, `evaluation_metrics.csv`, plots

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
| **Encounters** | 629 |
| **Alerts Triggered** | 304 (48.3%) |
| **AUROC** | 0.9894 |
| **AUPRC** | 0.9769 |
| **ECE** (calibration) | 0.0528 |
| **Avg Latency — All encounters** | 1.92s |
| **Avg Latency — Alert path** (LLM) | 3.86s |
| **Avg Latency — Non-alert path** | 0.10s |

Results indicate strong discrimination on the synthetic cohort, with alert rates aligned to the generator’s sepsis prevalence. Alerts were associated with higher average HR, RR, Temp, and Lactate values compared to non‑alert encounters, reflecting clinically consistent signal patterns.

Alert rates and model performance remain stable across the cohort, while simulated latency reflects the added overhead of LLM‑driven planning. The system consistently produced guideline‑aligned orders (fluids, antibiotics, lactate redraw, cultures), supporting end‑to‑end orchestration feasibility.

Key observation summary is shown below, contrasting mean vitals for alert vs non‑alert encounters.

Results breakdown: AUROC and AUPRC remain high; calibration (ECE) is acceptable for a PoC; and latency remains within a few seconds for alert cases.
Order execution is consistent across alert encounters, and explanations emphasize HR/RR with supporting contributions from temperature and lactate.
System outputs provide a complete audit trail (alerts, plans, execution logs, explanations) for each visit.

> *Latency reflects simulated LLM inference. All other metrics computed from actual pipeline output.*

### Alert Breakdown (Simulated NLP Score)

| Group | Count | Percent | Avg NLP Sepsis Score (simulated) |
|---|---|---|---|
| Alerts | 304 | 48.3% | 0.82 |
| No Alerts | 325 | 51.7% | 0.19 |
| Total | 629 | 100.0% | — |

*Note: NLP score averages are illustrative placeholders for poster presentation.*

### Evaluation Dashboard
![Evaluation Dashboard](output/evaluation_dashboard.png)

### Precision-Recall Curve
![Precision-Recall Curve](output/precision_recall_curve.png)

### Observation Summary
![Observation Summary](output/observation_summary.png)

### Combined Summary Table
![Combined Summary Table](output/combined_summary_table.png)

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `output/patients.csv` | 200 synthetic patients |
| `output/encounters.csv` | 629 clinical encounters |
| `output/observations.csv` | LOINC-coded vital sign observations |
| `output/harmonized_data.json` | Unified patient-event JSON (MIMIC-IV schema) |
| `output/orchestration_results.json` | Full per-encounter agent pipeline results |
| `output/evaluation_metrics.csv` | AUROC, AUPRC, ECE, latency |
| `output/evaluation_dashboard.png` | ROC curve + SHAP importance + latency distribution |
| `output/precision_recall_curve.png` | Precision-Recall curve |

---

## 📚 References

- Singer M, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis‑3).* JAMA. 2016.
- Johnson AEW, et al. *MIMIC‑IV (Medical Information Mart for Intensive Care).* PhysioNet. 2020.
- Lundberg SM, Lee S‑I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS. 2017.
- LangGraph Documentation: https://langgraph.readthedocs.io/
- Ollama Documentation: https://ollama.com/
- HL7 FHIR Standard: https://www.hl7.org/fhir/

## ⚠️ Limitations

- Synthetic data only — not validated on real EHR data
- Planner LLM (Ollama/Gemma) requires local installation; offline runs use mocked orders
- SHAP values are proxies (deviation-based), not true SHAP from a trained ML model
- Latency figures are simulated for the LLM inference component

---

## 🔭 Future Work

Near-term improvements focus on realism, robustness, and clinical utility while preserving transparency.

- Validate on real-world EHR data (e.g., MIMIC-IV) with careful cohort selection
- Integrate full SOFA/qSOFA scoring and comorbidity context
- Extend NLP to unstructured notes (discharge summaries, nursing notes)
- Replace SHAP-proxy with true SHAP using a calibrated ML model
- Add clinician-in-the-loop review and alert escalation policies
- Prospectively evaluate latency and usability in simulated workflow studies
