import os
import zipfile
import urllib.request
import logging
try:
    from medcat.cat import CAT
except ImportError:
    CAT = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedCATProcessor:
    def __init__(self):
        self.model_url = "https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip"
        self.model_zip_path = "medmen_wstatus_2021_oct.zip"
        self.model_dir = "medmen_wstatus_2021_oct"
        
        self.cat = None
        self._initialize_model()

    def _initialize_model(self):
        if CAT is None:
            logger.warning("`medcat` package not installed. Falling back to keyword-based extraction.")
            self.cat = None
            return

        try:
            if not os.path.exists(self.model_dir) or not os.listdir(self.model_dir):
                logger.info(f"Downloading MedCAT model pack from {self.model_url}...")
                urllib.request.urlretrieve(self.model_url, self.model_zip_path)
                logger.info("Extracting MedCAT model pack...")
                with zipfile.ZipFile(self.model_zip_path, 'r') as zip_ref:
                    zip_ref.extractall()
                if os.path.exists(self.model_zip_path):
                    os.remove(self.model_zip_path)
                    
            logger.info(f"Loading MedCAT model from {self.model_dir}...")
            self.cat = CAT.load_model_pack(self.model_dir)
            logger.info("MedCAT model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize MedCAT model: {e}")
            logger.warning("Falling back to a mock processor (string matching).")
            self.cat = None

    def get_entities(self, text):
        if not text:
            return []
            
        if self.cat is not None:
            try:
                entities = self.cat.get_entities(text)
                formatted_entities = []
                for entity_id, entity_info in entities['entities'].items():
                    name = entity_info.get('pretty_name', entity_info.get('cui', 'Unknown'))
                    type_val = 'Unknown'
                    if entity_info.get('type_ids'):
                        type_val = entity_info['type_ids'][0]
                        
                    formatted_entities.append({
                        'cui': entity_info.get('cui'),
                        'name': name,
                        'source_value': entity_info.get('source_value'),
                        'type': type_val,
                        'confidence': entity_info.get('acc', 0.0)
                    })
                return formatted_entities
            except Exception as e:
                logger.error(f"Error during MedCAT extraction: {e}")
                return []
        else:
            # Fallback mock logic
            keywords = {
                'Heart Rate': ['heart rate', 'tachycardia', 'pulse', 'bpm', 'hr'],
                'Respiratory Rate': ['respiratory rate', 'tachypnea', 'breathing', 'rr', 'breaths/min'],
                'Temperature': ['temperature', 'fever', 'hyperthermia', 'temp', 'celcius', 'febrile'],
                'Lactate': ['lactate', 'lactic acid', 'hyperlactatemia'],
                'Sepsis': ['sepsis', 'septic shock', 'bacteremia', 'infection', 'ards'],
                'Blood Pressure': ['blood pressure', 'hypotension', 'bp', 'mmhg']
            }
            text_lower = text.lower()
            found_entities = []
            for entity_type, kws in keywords.items():
                for kw in kws:
                    if kw in text_lower:
                        found_entities.append({
                            'cui': f'C_{entity_type.upper().replace(" ", "_")}',
                            'name': entity_type,
                            'source_value': kw,
                            'type': 'Finding',
                            'confidence': 0.95
                        })
                        break
            return found_entities
