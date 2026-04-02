from typing import List, Dict, Any
import random
import logging
import shap
import pandas as pd
import numpy as np

try:
    from medcat.vocab import Vocab
    from medcat.cdb import CDB
    from medcat.cat import CAT
    from medcat.config import Config
    MEDCAT_AVAILABLE = True
except ImportError:
    MEDCAT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Monitors patient data for sepsis indicators."""

    def __init__(self):
        pass

    def monitor(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = []
        subject_id = patient_data['subject_id']

        for visit in patient_data['visits']:
            visit_id = visit['hadm_id']
            # Simple threshold check based on Sepsis-3 (simplified for PoC)
            # HR > 90, RR > 22, Lactate > 2.0, Temp > 38.0

            hr = 0
            rr = 0
            temp = 0
            lactate = 0

            for event in visit['events']:
                if event['itemid'] == '8867-4': # HR
                    hr = event['valuenum']
                elif event['itemid'] == '9279-1': # RR
                    rr = event['valuenum']
                elif event['itemid'] == '8310-5': # Temp
                    temp = event['valuenum']
                elif event['itemid'] == '32693-4': # Lactate
                    lactate = event['valuenum']

            # Screening Logic
            risk_score = 0
            reasons = []

            if hr > 90:
                risk_score += 1
                reasons.append(f"Heart Rate {hr} > 90")
            if rr >= 22:
                risk_score += 1
                reasons.append(f"Resp Rate {rr} >= 22")
            if temp > 38.0:
                risk_score += 1
                reasons.append(f"Temp {temp} > 38.0")
            if lactate > 2.0:
                risk_score += 2 # Strong indicator
                reasons.append(f"Lactate {lactate} > 2.0")

            if risk_score >= 2:
                alert = {
                    'subject_id': subject_id,
                    'visit_id': visit_id,
                    'risk_score': risk_score,
                    'reasons': reasons,
                    'timestamp': visit['admittime'],
                    'clinical_data': {
                        'HR': hr,
                        'RR': rr,
                        'Temp': temp,
                        'Lactate': lactate
                    }
                }
                alerts.append(alert)
                logger.info(f"Sepsis Alert for {subject_id} at {visit['admittime']}: {reasons}")

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
    """Mock FHIR API to execute orders."""

    def execute_orders(self, orders: List[str], visit_id: str) -> List[str]:
        results = []
        for order in orders:
            # Simulate FHIR transaction
            order_id = f"ORD-{random.randint(1000, 9999)}"
            status = "success"
            # In a real system, this would POST to a FHIR server
            result = f"Order '{order}' placed for visit {visit_id} (ID: {order_id}, Status: {status})"
            results.append(result)
            logger.info(result)
        return results

class VerifierAgent:
    """Explains the alert using SHAP values (simulated for PoC)."""

    def explain(self, alert: Dict[str, Any]) -> str:
        # For a true SHAP explanation, we'd need a trained ML model.
        # Here we simulate the feature importance based on the rule-based logic we used.

        data = alert['clinical_data']
        # Feature names and their values
        features = list(data.keys())
        values = np.array(list(data.values()))

        # Simple heuristic for 'importance' based on deviation from normal
        # Normal: HR=70, RR=16, Temp=37, Lactate=1.0
        baseline = np.array([70, 16, 37.0, 1.0])
        # Calculate deviation (importance)
        importance = np.abs(values - baseline)

        # Normalize importance
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)

        # Create a text explanation
        explanation = "Feature Importance Analysis (SHAP-proxy):\n"
        sorted_indices = np.argsort(importance)[::-1]

        for idx in sorted_indices:
            feat_name = features[idx]
            imp_val = importance[idx]
            val = values[idx]
            explanation += f"- {feat_name}: {val} (Importance: {imp_val:.2f})\n"

        return explanation
