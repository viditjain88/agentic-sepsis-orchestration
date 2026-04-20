from typing import List, Dict, Any
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from medcat.vocab import Vocab
    from medcat.cdb import CDB
    from medcat.cat import CAT
    from medcat.config import Config
    MEDCAT_AVAILABLE = True
except ImportError:
    MEDCAT_AVAILABLE = False

from medcat_processor import MedCATProcessor

class MedCATPipeline:
    """MedCAT entity extraction pipeline to identify Sepsis CUI codes."""

    def __init__(self):
        if MEDCAT_AVAILABLE:
            try:
                self.vocab = Vocab.load('output/medcat_models/vocab')
                self.cdb = CDB.load('output/medcat_models/cdb')
                self.config = Config()
                self.config.general.spacy_model = 'en_core_web_md'
                self.cat = CAT(self.cdb, vocab=self.vocab, config=self.config)
            except Exception as e:
                logger.error(f"Failed to load real MedCAT models. Did you run medcat_setup.py?: {e}")
                self.cat = None
        else:
            logger.error("MedCAT libraries not found.")
            self.cat = None

    def get_entities(self, text: str) -> List[Dict[str, str]]:
        if self.cat:
            try:
                entities = self.cat.get_entities(text)
                results = []
                for ent in entities['entities'].values():
                    # get pretty name if available, otherwise fallback
                    cui_name = ent.get('pretty_name', ent.get('cui', 'Unknown'))
                    results.append({
                        "source_value": ent['source_value'],
                        "cui": ent['cui'],
                        "cui_name": cui_name
                    })
                return results
            except Exception as e:
                logger.error(f"MedCAT extraction failed: {e}")
        return []

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

class EvaluatorAgent:
    """Evaluates the generated treatment plan against the extracted CUI codes."""
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, plan: List[str], cui_entities: List[Dict[str, str]]) -> str:
        logger.info("--- EVALUATOR AGENT (LLM JUDGE) ---")
        if not plan:
            return "No plan generated to evaluate."

        if not cui_entities:
            return "No relevant CUI conditions found. Plan seems unprompted by explicit textual evidence."

        conditions = ", ".join([f"{e['cui_name']} ({e['cui']})" for e in cui_entities])
        plan_str = "\n".join([f"- {p}" for p in plan])

        prompt = f"""
        You are a medical evaluator agent (LLM Judge). Your task is to validate if a proposed treatment plan appropriately addresses the patient's conditions.

        Identified Conditions (from MedCAT NLP):
        {conditions}

        Proposed Treatment Plan:
        {plan_str}

        Please analyze the plan. Does it adequately address the identified conditions? Are there any missing standard-of-care steps for these conditions? Provide a brief evaluation summary.
        """

        try:
            response = self.llm.invoke(prompt)
            # Depending on the LLM interface, response might be a string or an object
            return str(response)
        except Exception as e:
            logger.error(f"LLM Evaluation failed: {e}")
            return "Evaluation failed due to LLM error."


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
