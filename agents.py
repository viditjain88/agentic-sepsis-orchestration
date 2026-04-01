from typing import List, Dict, Any
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from medcat_processor import MedCATProcessor

class PerceptorAgent:
    """
    Clinical NLP — three techniques:
    1. LOINC-coded entity recognition: maps observation codes to named clinical concepts
    2. Threshold-based pattern matching: screens entities against Sepsis-3 criteria
    3. MedCAT Entity Extraction: Extracts entities from unstructured clinical notes
    """

    # LOINC code → clinical entity mapping
    LOINC_MAP = {
        '8867-4':  'Heart Rate',       # tachycardia marker
        '9279-1':  'Respiratory Rate', # tachypnea marker
        '8310-5':  'Temperature',      # hyperthermia marker
        '32693-4': 'Lactate',          # hyperlactatemia marker
    }

    def __init__(self):
        self.medcat = MedCATProcessor()

    def monitor(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = []
        subject_id = patient_data['subject_id']

        for visit in patient_data['visits']:
            visit_id = visit['hadm_id']

            # ── Step 1: LOINC entity recognition ─────────────────
            entities = {v: 0 for v in self.LOINC_MAP.values()}
            for event in visit['events']:
                code = event['itemid']
                if code in self.LOINC_MAP:
                    entities[self.LOINC_MAP[code]] = event['valuenum']

            hr      = entities['Heart Rate']
            rr      = entities['Respiratory Rate']
            temp    = entities['Temperature']
            lactate = entities['Lactate']

            # ── Step 2: Sepsis-3 threshold pattern matching ───────
            risk_score = 0
            reasons = []

            if hr > 90:
                risk_score += 1
                reasons.append(f"Heart Rate {hr} > 90 bpm")
            if rr >= 22:
                risk_score += 1
                reasons.append(f"Resp Rate {rr} >= 22 breaths/min")
            if temp > 38.0:
                risk_score += 1
                reasons.append(f"Temp {temp} > 38.0°C")
            if lactate > 2.0:
                risk_score += 2  # strong indicator — double weight
                reasons.append(f"Lactate {lactate} > 2.0 mmol/L")

            # ── Step 3: MedCAT Extraction from clinical notes ─────
            note = visit.get('clinical_note', '')
            extracted_entities = self.medcat.get_entities(note)
            
            # Boost risk score if sepsis/infection entities are found in the notes
            note_mentions_sepsis = any(e['name'] == 'Sepsis' for e in extracted_entities)
            if note_mentions_sepsis:
                risk_score += 1
                reasons.append(f"Clinical Note mentions Sepsis/Infection markers")

            if risk_score >= 2:
                alert = {
                    'subject_id': subject_id,
                    'visit_id': visit_id,
                    'risk_score': risk_score,
                    'reasons': reasons,
                    'timestamp': visit['admittime'],
                    'clinical_data': {
                        'HR': hr, 'RR': rr, 'Temp': temp, 'Lactate': lactate
                    },
                    'extracted_entities': extracted_entities
                }
                alerts.append(alert)
                logger.info(f"Sepsis Alert for {subject_id} @ {visit['admittime']}: {reasons}")

        return alerts


class ExecutorAgent:
    """Simulates FHIR API order placement (mock)."""

    def execute_orders(self, orders: List[str], visit_id: str) -> List[str]:
        results = []
        for order in orders:
            order_id = f"ORD-{random.randint(1000, 9999)}"
            result = f"Order '{order}' placed for {visit_id} (ID: {order_id}, Status: success)"
            results.append(result)
            logger.info(result)
        return results


class VerifierAgent:
    """
    SHAP-proxy explainability:
    Feature importance estimated as normalised deviation from clinical baseline.
    Baseline: HR=70 bpm, RR=16 br/min, Temp=37.0°C, Lactate=1.0 mmol/L
    """

    FEATURES  = ['HR', 'RR', 'Temp', 'Lactate']
    BASELINE  = np.array([70.0, 16.0, 37.0, 1.0])

    def explain(self, alert: Dict[str, Any]):
        data   = alert['clinical_data']
        values = np.array([data['HR'], data['RR'], data['Temp'], data['Lactate']])

        importance = np.abs(values - self.BASELINE)
        if importance.sum() > 0:
            importance = importance / importance.sum()

        sorted_idx  = np.argsort(importance)[::-1]
        explanation = "Feature Importance Analysis (SHAP-proxy):\n"
        for idx in sorted_idx:
            explanation += (f"- {self.FEATURES[idx]}: {values[idx]} "
                            f"(Importance: {importance[idx]:.2f})\n")

        importance_dict = {self.FEATURES[i]: float(importance[i])
                           for i in range(len(self.FEATURES))}
        return explanation, importance_dict

class EvaluatorAgent:
    """
    LLM as a Judge: Evaluates the pipeline outputs (extracted entities, treatment plan, etc.) 
    against the raw clinical notes and provided inputs.
    """
    def evaluate(self, alert: Dict[str, Any], plan: List[str]) -> Dict[str, Any]:
        # Mocking the LLM Judge evaluation process
        # In a real scenario, this would send a prompt to an LLM like Gemma or GPT-4
        # containing the clinical note, extracted entities, and the planner's output.
        
        extracted_entities = alert.get('extracted_entities', [])
        note = alert.get('clinical_note', '')
        
        evaluation_score = 0
        feedback = []
        
        # Simple heuristic to mock LLM scoring:
        # Check if plan contains 'Lactate' if 'Lactate' entity was found, etc.
        entity_names = [e['name'] for e in extracted_entities]
        
        if 'Sepsis' in entity_names and any('Broad-Spectrum Antibiotics' in p for p in plan):
            evaluation_score += 40
            feedback.append("Excellent alignment: Antibiotics ordered for suspected Sepsis.")
        
        if 'Lactate' in entity_names and any('Lactate Redraw' in p for p in plan):
            evaluation_score += 20
            feedback.append("Good alignment: Lactate redraw ordered corresponding to Lactate mention.")
            
        if not extracted_entities:
            evaluation_score += 50 # Baseline if no entities found
            feedback.append("Neutral: No specific entities extracted from notes to evaluate against.")
        else:
            evaluation_score += 40 # Base score for having a standard plan
            feedback.append("Plan follows standard Sepsis-3 bundle appropriately.")
            
        final_score = min(100, evaluation_score)
        
        return {
            "score": final_score,
            "feedback": " ".join(feedback),
            "alignment": "High" if final_score >= 80 else "Medium" if final_score >= 50 else "Low"
        }

class TherapeuticsAgent:
    """
    Analyzes cellular data (genes, proteins, signaling pathways) and heat signatures
    to predict the best combination of therapies to correct cellular dysfunction.
    """
    def predict_therapies(self, cellular_data: Dict[str, Any]) -> List[str]:
        if not cellular_data or 'nodes' not in cellular_data:
            return ["Standard Sepsis Bundle (No cellular data provided)"]
            
        nodes = cellular_data.get('nodes', [])
        
        # Analyze heat signatures to identify highly expressed or "hot" targets
        hot_genes = [n['id'] for n in nodes if n.get('type') == 'gene' and n.get('heat', 0) > 0.6]
        hot_proteins = [n['id'] for n in nodes if n.get('type') == 'protein' and n.get('heat', 0) > 0.6]
        hot_pathways = [n['id'] for n in nodes if n.get('type') == 'pathway' and n.get('heat', 0) > 0.6]
        
        therapies = []
        
        # Map targets to specific therapeutic combinations
        if 'TNF-alpha' in hot_proteins or 'IL-6' in hot_proteins:
            therapies.append("Administer targeted anti-cytokine therapy (e.g., Tocilizumab) to reduce inflammation.")
            
        if 'Apoptosis' in hot_pathways:
            therapies.append("Administer apoptosis inhibitors to prevent excessive cell death.")
            
        if 'MAPK' in hot_pathways or 'PI3K-AKT' in hot_pathways:
            therapies.append("Consider kinase inhibitors to stabilize cellular signaling pathways.")
            
        if 'VEGFA' in hot_genes or 'VEGF' in hot_proteins:
            therapies.append("Administer VEGF inhibitors to modulate angiogenesis.")
            
        if not therapies:
            therapies.append("Cellular heat signatures are stable. Continue standard supportive care.")
            
        return therapies
