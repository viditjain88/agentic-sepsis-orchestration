# Test Plan — Agentic Sepsis Orchestration

This document provides a scenario-based test checklist for the latest pipeline architecture (MedCAT + NLP model + LangGraph orchestration + LLM Judge + frontend/backend).

## Scope

- Python data generation + harmonization pipeline
- MedCAT setup and CUI extraction
- NLP sepsis scoring (`Bio_ClinicalBERT` stub integration)
- LangGraph orchestration flow (`Perceptor → Planner → Evaluator → Executor → Verifier`)
- Evaluation metrics generation
- Backend API endpoints and frontend rendering paths
- Regression and failure handling

## Environment Prerequisites

- Python environment activated
- Dependencies installed from `requirements.txt`
- Ollama installed (optional but recommended)
- `gemma` model pulled for real planner/evaluator runs
- Ports `8000` and `5173` available

```bash
cd /Users/viditjain/Desktop/Github/agentic-sepsis-orchestration
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Environment & Dependency Tests

### TC-ENV-01: Python dependencies install
- **Goal:** Ensure all required packages install.
- **Steps:**
```bash
pip install -r requirements.txt
```
- **Expected:** Installation completes without unresolved package conflicts.

### TC-ENV-02: Ollama availability (optional)
- **Goal:** Confirm local LLM runtime is reachable.
- **Steps:**
```bash
ollama pull gemma
curl -s http://localhost:11434/api/tags | head
```
- **Expected:** Tag list returns successfully.

### TC-ENV-03: SpaCy model for MedCAT setup
- **Goal:** Ensure required model is installed.
- **Steps:**
```bash
python -m spacy download en_core_web_md
```
- **Expected:** Model download/installation completes.

---

## 2) MedCAT Setup & Extraction Tests

### TC-MEDCAT-01: Build MedCAT PoC models
- **Goal:** Generate local CDB/vocab artifacts.
- **Steps:**
```bash
python medcat_setup.py
ls -la output/medcat_models
```
- **Expected:** `output/medcat_models/cdb` and `output/medcat_models/vocab` exist.

### TC-MEDCAT-02: MedCAT library import sanity
- **Goal:** Validate medcat dependency load.
- **Steps:**
```bash
python test_medcat_api.py
```
- **Expected:** Prints medcat version and import checks (e.g., `Vocab ok`, `CDB ok`).

### TC-MEDCAT-03: Clinical entities extraction path
- **Goal:** Validate CUI extraction path from note text.
- **Steps:**
```bash
python - <<'PY'
from agents import MedCATPipeline
pipe = MedCATPipeline()
text = "Patient with severe sepsis and hypotension progressing to septic shock."
print(pipe.get_entities(text))
PY
```
- **Expected:** Non-empty list with CUI-like outputs when setup/models are valid.

### TC-MEDCAT-04 (Negative): Missing model files
- **Goal:** Validate graceful behavior if MedCAT model artifacts are missing.
- **Steps:** Temporarily rename `output/medcat_models` and run TC-MEDCAT-03.
- **Expected:** No hard crash loop; logs indicate loading failure.

---

## 3) NLP Model Scoring Tests

### TC-NLP-01: Positive note scoring
- **Goal:** Confirm score generated for septic-like note.
- **Steps:**
```bash
python - <<'PY'
from nlp_model_stub import FineTunedClinicalBERT
m = FineTunedClinicalBERT()
print("positive", m.predict_sepsis_probability("Severe sepsis with elevated lactate and hypotension."))
PY
```
- **Expected:** Probability in `[0.0, 1.0]`.

### TC-NLP-02: Negative note scoring
- **Goal:** Confirm score generated for non-septic note.
- **Steps:**
```bash
python - <<'PY'
from nlp_model_stub import FineTunedClinicalBERT
m = FineTunedClinicalBERT()
print("negative", m.predict_sepsis_probability("Patient recovering well with stable vitals and no infection signs."))
PY
```
- **Expected:** Probability in `[0.0, 1.0]`.

### TC-NLP-03: Long note truncation
- **Goal:** Ensure >512-token notes do not crash.
- **Steps:** Create long string and run predictor.
- **Expected:** Successful inference with truncation.

---

## 4) Data Pipeline Tests

### TC-DATA-01: Synthetic generation outputs
- **Goal:** Validate expected CSV outputs.
- **Steps:**
```bash
python generate_synthetic_data.py
ls -la output | grep -E 'patients.csv|encounters.csv|observations.csv|notes.csv'
```
- **Expected:** All files present including `notes.csv`.

### TC-DATA-02: Harmonization output
- **Goal:** Validate unified JSON produced.
- **Steps:**
```bash
python harmonize_data.py
python - <<'PY'
import json
rows = json.load(open('output/harmonized_data.json'))
print('patients', len(rows))
print('sample_keys', rows[0].keys())
PY
```
- **Expected:** Non-empty JSON with `subject_id`, `demographics`, `visits`.

### TC-DATA-03: Notes-to-encounters consistency
- **Goal:** Ensure notes rows align to encounter rows.
- **Steps:**
```bash
python - <<'PY'
import pandas as pd
enc = pd.read_csv('output/encounters.csv')
notes = pd.read_csv('output/notes.csv')
print('encounters', len(enc), 'notes', len(notes))
PY
```
- **Expected:** Counts should be equal or clearly justified by generation logic.

---

## 5) Orchestration Flow Tests

### TC-ORCH-01: Full orchestrator run
- **Goal:** Validate full graph execution.
- **Steps:**
```bash
python orchestrator.py
```
- **Expected:** Completes and writes `output/orchestration_results.json`.

### TC-ORCH-02: Output schema check
- **Goal:** Verify expected fields in results.
- **Steps:**
```bash
python - <<'PY'
import json
rows = json.load(open('output/orchestration_results.json'))
print('rows', len(rows))
print('keys', sorted(rows[0].keys()))
PY
```
- **Expected:** Includes fields such as `alert_triggered`, `plan`, `execution_result`, `explanation`; newer flow may include `nlp_sepsis_score`, `evaluation_result`, `extracted_entities`.

### TC-ORCH-03: Alert path behavior
- **Goal:** Ensure plans generated when alerts trigger.
- **Steps:**
```bash
python - <<'PY'
import json
rows = json.load(open('output/orchestration_results.json'))
alerts = [r for r in rows if r.get('alert_triggered')]
print('alerts', len(alerts))
print('sample_plan', alerts[0].get('plan') if alerts else [])
PY
```
- **Expected:** Alert rows contain non-empty plans.

### TC-ORCH-04: Therapeutics extension presence
- **Goal:** Confirm plan extension beyond base bundle where applicable.
- **Steps:**
```bash
python - <<'PY'
import json
rows = json.load(open('output/orchestration_results.json'))
alerts = [r for r in rows if r.get('alert_triggered')]
extended = [r for r in alerts if len(r.get('plan', [])) > 4]
print('extended_plans', len(extended), 'of', len(alerts))
PY
```
- **Expected:** Extended plans present for at least some alert cases.

### TC-ORCH-05 (Negative): Ollama unavailable fallback
- **Goal:** Validate fallback execution.
- **Steps:** Stop Ollama, rerun orchestrator.
- **Expected:** Run still completes using fallback/mock path.

---

## 6) Evaluator (LLM Judge) Tests

### TC-JUDGE-01: Evaluator sanity test
- **Goal:** Validate evaluator API contract.
- **Steps:**
```bash
python test_llm_judge.py
```
- **Expected:** Returns evaluation summary string without exception.

### TC-JUDGE-02: Empty plan behavior
- **Goal:** Ensure evaluator handles missing plan gracefully.
- **Steps:** Call evaluator with `[]`.
- **Expected:** Returns "No plan generated to evaluate." or equivalent.

### TC-JUDGE-03: Empty CUI behavior
- **Goal:** Ensure evaluator handles no extracted entities.
- **Steps:** Call evaluator with plan and `[]` entities.
- **Expected:** Returns message indicating no relevant CUI evidence.

---

## 7) Evaluation Metrics Tests

### TC-MET-01: Standard evaluation run
- **Goal:** Validate metric generation.
- **Steps:**
```bash
python evaluate.py
cat output/evaluation_metrics.csv
```
- **Expected:** CSV contains AUROC/AUPRC/ECE/latency columns and non-empty values.

### TC-MET-02: `nlp_sepsis_score` fallback behavior
- **Goal:** Confirm evaluator uses NLP score when present and threshold score otherwise.
- **Steps:** Run evaluate against mixed-format results.
- **Expected:** No crash; metric values produced.

---

## 8) Backend API Tests

### TC-API-01: Start backend
- **Steps:**
```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```
- **Expected:** Startup complete on port `8000`.

### TC-API-02: Patients list endpoint
- **Steps:**
```bash
curl -s http://127.0.0.1:8000/api/patients | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))"
```
- **Expected:** Non-zero patient count.

### TC-API-03: Patient detail endpoint
- **Steps:**
```bash
curl -s http://127.0.0.1:8000/api/patients/P000 | head
```
- **Expected:** JSON includes demographics and visits.

### TC-API-04: Monitor endpoint payload contract
- **Steps:**
```bash
curl -s -X POST http://127.0.0.1:8000/api/monitor \
  -H 'Content-Type: application/json' \
  -d '{"subject_id":"P000","visit_id":"EP000_000","hr":110,"rr":24,"temp":39.0,"lactate":3.0}' | head
```
- **Expected:** JSON includes keys such as `alert_triggered`, `plan`, `evaluation`, `cellular_data`, `extracted_entities`, `clinical_note`.

---

## 9) Frontend Tests

### TC-FE-01: Start frontend
- **Steps:**
```bash
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```
- **Expected:** Vite serves at `http://127.0.0.1:5173/`.

### TC-FE-02: Patient selection prefill
- **Goal:** Verify selected patient updates vitals from latest visit.
- **Expected:** HR/RR/Temp/Lactate fields change across different patients (not static defaults).

### TC-FE-03: Alert visualization path
- **Goal:** Validate full result display on alert.
- **Expected:** Shows treatment plan, evaluator panel, SHAP section, and note/entities panel.

### TC-FE-04: Cellular network rendering
- **Goal:** Validate graph and heat bars render when `cellular_data` exists.
- **Expected:** Force graph visible; heat bars show sorted node heat percentages.

### TC-FE-05: Non-alert UI behavior
- **Goal:** Validate no-plan path.
- **Expected:** Appropriate no-alert visual state and no execution/evaluator errors.

---

## 10) Regression / Reliability Tests

### TC-REG-01: End-to-end pipeline command
```bash
python generate_synthetic_data.py && \
python harmonize_data.py && \
python orchestrator.py && \
python evaluate.py
```
- **Expected:** Completes without fatal errors.

### TC-REG-02: Re-run consistency
- **Goal:** Run full pipeline 3 times and compare output schema stability.
- **Expected:** No schema regressions; metrics generated each run.

### TC-REG-03: Port collision handling
- **Goal:** Validate start/stop script handling.
- **Steps:** Kill and restart using port cleanup commands.
- **Expected:** Services restart reliably.

---

## Quick Smoke Checklist

- [ ] Dependencies installed
- [ ] MedCAT models generated
- [ ] Synthetic data + notes generated
- [ ] Harmonized JSON generated
- [ ] Orchestration results generated
- [ ] Evaluation metrics generated
- [ ] Backend `/api/patients` returns data
- [ ] Frontend loads at `127.0.0.1:5173`
- [ ] Patient vitals prefill dynamic
- [ ] Evaluator output visible in API/UI
- [ ] Cellular graph visible for alert visits

---

## Notes

- This plan intentionally covers both **happy-path** and **failure-path** behavior.
- If runtime is constrained, prioritize the **Quick Smoke Checklist** then run deeper scenarios incrementally.
