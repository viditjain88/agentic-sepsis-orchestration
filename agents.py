from typing import List, Dict, Any
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerceptorAgent:
    """
    Clinical NLP — two techniques:
    1. LOINC-coded entity recognition: maps observation codes to named clinical concepts
    2. Threshold-based pattern matching: screens entities against Sepsis-3 criteria
    """

    # LOINC code → clinical entity mapping
    LOINC_MAP = {
        '8867-4':  'Heart Rate',       # tachycardia marker
        '9279-1':  'Respiratory Rate', # tachypnea marker
        '8310-5':  'Temperature',      # hyperthermia marker
        '32693-4': 'Lactate',          # hyperlactatemia marker
    }

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

            if risk_score >= 2:
                alert = {
                    'subject_id': subject_id,
                    'visit_id': visit_id,
                    'risk_score': risk_score,
                    'reasons': reasons,
                    'timestamp': visit['admittime'],
                    'clinical_data': {
                        'HR': hr, 'RR': rr, 'Temp': temp, 'Lactate': lactate
                    }
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
