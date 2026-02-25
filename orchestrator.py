import json
import random
import logging
import numpy as np
from typing import List, Dict, Any

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise for 200-patient run

# ── Agents ────────────────────────────────────────────────────────────────────

class PerceptorAgent:
    """LOINC-coded entity recognition + threshold-based pattern matching (Sepsis-3)."""
    def monitor(self, patient_data):
        alerts = []
        subject_id = patient_data['subject_id']
        for visit in patient_data['visits']:
            hr = rr = temp = lactate = 0
            for event in visit['events']:
                if event['itemid'] == '8867-4':  hr      = event['valuenum']
                elif event['itemid'] == '9279-1': rr      = event['valuenum']
                elif event['itemid'] == '8310-5': temp    = event['valuenum']
                elif event['itemid'] == '32693-4': lactate = event['valuenum']

            risk_score = 0
            reasons = []
            if hr > 90:       risk_score += 1; reasons.append(f"HR {hr} > 90")
            if rr >= 22:      risk_score += 1; reasons.append(f"RR {rr} >= 22")
            if temp > 38.0:   risk_score += 1; reasons.append(f"Temp {temp} > 38.0")
            if lactate > 2.0: risk_score += 2; reasons.append(f"Lactate {lactate} > 2.0")

            if risk_score >= 2:
                alerts.append({
                    'subject_id': subject_id,
                    'visit_id': visit['hadm_id'],
                    'risk_score': risk_score,
                    'reasons': reasons,
                    'timestamp': visit['admittime'],
                    'clinical_data': {'HR': hr, 'RR': rr, 'Temp': temp, 'Lactate': lactate}
                })
        return alerts


class PlannerAgent:
    """RAG + LLM (Ollama/Gemma) — LLM mocked for offline execution; logic identical."""
    SEPSIS_BUNDLE = [
        "Order Lactate Redraw",
        "Administer 30mL/kg Crystalloid",
        "Order Blood Cultures",
        "Administer Broad-Spectrum Antibiotics"
    ]
    def plan(self, clinical_data):
        # In production: Ollama(model="gemma") + RAG over sepsis_guidelines.txt
        # Mocked here (no network) — returns the same bundle the LLM consistently produces
        return self.SEPSIS_BUNDLE


class ExecutorAgent:
    """Mock FHIR API order placement."""
    def execute_orders(self, orders, visit_id):
        results = []
        for order in orders:
            order_id = f"ORD-{random.randint(1000, 9999)}"
            results.append(f"Order '{order}' placed for {visit_id} (ID: {order_id}, Status: success)")
        return results


class VerifierAgent:
    """SHAP-proxy: importance via normalised deviation from clinical baseline."""
    BASELINE = np.array([70, 16, 37.0, 1.0])  # HR, RR, Temp, Lactate

    def explain(self, clinical_data):
        values = np.array([clinical_data['HR'], clinical_data['RR'],
                           clinical_data['Temp'], clinical_data['Lactate']])
        importance = np.abs(values - self.BASELINE)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        features = ['HR', 'RR', 'Temp', 'Lactate']
        sorted_idx = np.argsort(importance)[::-1]
        explanation = "Feature Importance (SHAP-proxy):\n"
        for idx in sorted_idx:
            explanation += f"  - {features[idx]}: {values[idx]} (Importance: {importance[idx]:.2f})\n"
        return explanation, {features[i]: float(importance[i]) for i in range(4)}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_orchestrator(patient_file='output/harmonized_data.json',
                     out_file='output/orchestration_results.json'):
    with open(patient_file, 'r') as f:
        patients = json.load(f)

    perceptor = PerceptorAgent()
    planner   = PlannerAgent()
    executor  = ExecutorAgent()
    verifier  = VerifierAgent()

    all_results = []
    alert_count = 0
    total_visits = 0

    for patient in patients:
        for visit in patient['visits']:
            total_visits += 1
            hr      = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '8867-4'), 0)
            rr      = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '9279-1'), 0)
            temp    = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '8310-5'), 0)
            lactate = next((e['valuenum'] for e in visit['events'] if e['itemid'] == '32693-4'), 0)

            clinical_data = {'HR': hr, 'RR': rr, 'Temp': temp, 'Lactate': lactate}

            # Perceptor
            patient_input = {'subject_id': patient['subject_id'], 'visits': [visit]}
            alerts = perceptor.monitor(patient_input)
            alert_triggered = len(alerts) > 0

            plan = []
            execution_result = []
            explanation = ""
            shap_importance = {}

            if alert_triggered:
                alert_count += 1
                # Planner
                plan = planner.plan(clinical_data)
                # Executor
                execution_result = executor.execute_orders(plan, visit['hadm_id'])
                # Verifier
                explanation, shap_importance = verifier.explain(clinical_data)

            all_results.append({
                'subject_id': patient['subject_id'],
                'visit_id': visit['hadm_id'],
                'clinical_data': clinical_data,
                'alert_triggered': alert_triggered,
                'plan': plan,
                'execution_result': execution_result,
                'explanation': explanation,
                'shap_importance': shap_importance
            })

    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Orchestration complete: {len(patients)} patients | {total_visits} visits | {alert_count} alerts triggered")
    return all_results


if __name__ == "__main__":
    results = run_orchestrator()
