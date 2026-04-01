from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import sys

# Add parent dir to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import PerceptorAgent, ExecutorAgent, VerifierAgent, EvaluatorAgent
from orchestrator import PlannerAgent

app = FastAPI(title="Sepsis Orchestration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VitalsInput(BaseModel):
    subject_id: str
    visit_id: str
    hr: float
    rr: float
    temp: float
    lactate: float

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "harmonized_data.json")

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

@app.get("/api/patients")
def get_patients():
    data = load_data()
    # Return just the basics for the list
    patients = []
    for p in data:
        patients.append({
            "subject_id": p.get("subject_id"),
            "demographics": p.get("demographics")
        })
    return patients

@app.get("/api/patients/{subject_id}")
def get_patient(subject_id: str):
    data = load_data()
    for p in data:
        if p.get("subject_id") == subject_id:
            return p
    raise HTTPException(status_code=404, detail="Patient not found")

@app.post("/api/monitor")
def run_agents(vitals: VitalsInput):
    perceptor = PerceptorAgent()
    planner = PlannerAgent()
    executor = ExecutorAgent()
    verifier = VerifierAgent()

    # Format data for PerceptorAgent
    events = [
        {'itemid': '8867-4', 'valuenum': vitals.hr},
        {'itemid': '9279-1', 'valuenum': vitals.rr},
        {'itemid': '8310-5', 'valuenum': vitals.temp},
        {'itemid': '32693-4', 'valuenum': vitals.lactate},
    ]

    patient_data = {
        'subject_id': vitals.subject_id,
        'visits': [{
            'hadm_id': vitals.visit_id,
            'admittime': "Now",
            'events': events
        }]
    }

    # Fetch clinical note and cellular data from harmonized_data to pass to the pipeline
    data = load_data()
    clinical_note = ""
    cellular_data = None
    for p in data:
        if p.get("subject_id") == vitals.subject_id:
            for v in p.get("visits", []):
                if v.get("hadm_id") == vitals.visit_id:
                    clinical_note = v.get("clinical_note", "")
                    cellular_data = v.get("cellular_data")
                    break
            break

    patient_data['visits'][0]['clinical_note'] = clinical_note
    patient_data['visits'][0]['cellular_data'] = cellular_data

    alerts = perceptor.monitor(patient_data)
    alert_triggered = len(alerts) > 0

    plan = []
    execution_result = []
    explanation = ""
    shap_importance = {}
    evaluation = {}

    clinical_data = {
        'HR': vitals.hr,
        'RR': vitals.rr,
        'Temp': vitals.temp,
        'Lactate': vitals.lactate
    }

    if alert_triggered:
        plan = planner.plan(clinical_data, cellular_data=cellular_data)
        execution_result = executor.execute_orders(plan, vitals.visit_id)
        alert_obj = alerts[0]
        explanation, shap_importance = verifier.explain(alert_obj)
        
        evaluator = EvaluatorAgent()
        evaluation = evaluator.evaluate(alert_obj, plan)

    return {
        "alert_triggered": alert_triggered,
        "alerts": alerts,
        "plan": plan,
        "execution_result": execution_result,
        "explanation": explanation,
        "shap_importance": shap_importance,
        "clinical_data": clinical_data,
        "evaluation": evaluation,
        "cellular_data": cellular_data,
        "extracted_entities": alerts[0].get("extracted_entities", []) if alert_triggered else [],
        "clinical_note": clinical_note
    }
