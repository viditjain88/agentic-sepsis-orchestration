import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import urllib.request
import zipfile
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

def download_clinical_notes_dataset():
    logger.info("Attempting to load clinical notes dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset('starmpcc/Asclepius-Synthetic-Clinical-Notes', split='train')
        logger.info(f"Loaded {len(ds)} clinical notes.")
        return ds
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}. Will generate fallback notes.")
        return None

def generate_cellular_data():
    genes = ['BRCA1', 'TP53', 'TNF', 'IL6', 'EGFR', 'MYC', 'VEGFA', 'PTEN', 'PIK3CA', 'BRAF']
    proteins = ['p53', 'TNF-alpha', 'IL-6', 'EGFR', 'c-Myc', 'VEGF', 'PTEN', 'PI3K', 'B-Raf', 'AKT']
    pathways = ['Apoptosis', 'Inflammation', 'Cell Cycle', 'Angiogenesis', 'MAPK', 'PI3K-AKT']

    n_genes = random.randint(3, 7)
    selected_genes = random.sample(genes, n_genes)
    selected_proteins = random.sample(proteins, n_genes)
    selected_pathways = random.sample(pathways, random.randint(2, 4))
    
    nodes = []
    edges = []
    
    for g in selected_genes:
        expr = random.uniform(-3.0, 3.0)
        nodes.append({'id': g, 'type': 'gene', 'expression': round(expr, 2), 'heat': abs(expr) / 3.0})
        
    for p in selected_proteins:
        expr = random.uniform(-3.0, 3.0)
        nodes.append({'id': p, 'type': 'protein', 'expression': round(expr, 2), 'heat': abs(expr) / 3.0})
        
    for p in selected_pathways:
        nodes.append({'id': p, 'type': 'pathway', 'heat': random.uniform(0.1, 1.0)})
        
    # Generate random edges between genes, proteins, pathways
    all_ids = [n['id'] for n in nodes]
    for _ in range(random.randint(n_genes, n_genes*3)):
        source = random.choice(all_ids)
        target = random.choice(all_ids)
        if source != target:
            edges.append({'source': source, 'target': target, 'weight': round(random.uniform(-1.0, 1.0), 2)})
            
    return {'nodes': nodes, 'links': edges}

def generate_synthetic_data(num_patients=200, output_dir='output'):
    ds = download_clinical_notes_dataset()
    
    patients = []
    for i in range(num_patients):
        patient_id = f"P{i:03d}"
        gender = random.choice(['M', 'F'])
        birthdate = datetime.now() - timedelta(days=random.randint(20*365, 80*365))
        patients.append({
            'Id': patient_id,
            'BIRTHDATE': birthdate.strftime('%Y-%m-%d'),
            'DEATHDATE': None,
            'GENDER': gender,
            'RACE': random.choice(['white', 'black', 'asian', 'hispanic']),
            'ETHNICITY': random.choice(['nonhispanic', 'hispanic']),
            'CITY': 'Boston', 'STATE': 'MA',
        })
    df_patients = pd.DataFrame(patients)

    encounters = []
    for p in patients:
        num_encounters = random.randint(1, 5)
        for j in range(num_encounters):
            encounter_id = f"E{p['Id']}_{j:03d}"
            start_time = datetime.now() - timedelta(days=random.randint(1, 365))
            stop_time = start_time + timedelta(hours=random.randint(1, 48))
            
            note_text = ""
            if ds is not None:
                note_idx = random.randint(0, len(ds)-1)
                note_text = ds[note_idx]['note']
            else:
                note_text = f"Patient {p['Id']} admitted for observation. Vitals monitored."
                
            encounters.append({
                'Id': encounter_id,
                'START': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'STOP': stop_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'PATIENT': p['Id'],
                'ENCOUNTERCLASS': 'inpatient',
                'CLINICAL_NOTE': note_text,
                'CELLULAR_DATA': json.dumps(generate_cellular_data())
            })
    df_encounters = pd.DataFrame(encounters)

    observations = []
    for enc in encounters:
        is_septic = random.choice([True, False])
        hr_val = random.randint(95, 130) if is_septic else random.randint(55, 90)
        observations.append({'DATE': enc['START'], 'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'], 'CODE': '8867-4',
            'DESCRIPTION': 'Heart rate', 'VALUE': hr_val, 'UNITS': 'beats/min', 'TYPE': 'numeric'})
        rr_val = random.randint(22, 32) if is_septic else random.randint(10, 20)
        observations.append({'DATE': enc['START'], 'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'], 'CODE': '9279-1',
            'DESCRIPTION': 'Respiratory rate', 'VALUE': rr_val, 'UNITS': 'breaths/min', 'TYPE': 'numeric'})
        temp_val = random.uniform(38.1, 40.5) if is_septic else random.uniform(36.2, 37.5)
        observations.append({'DATE': enc['START'], 'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'], 'CODE': '8310-5',
            'DESCRIPTION': 'Body temperature', 'VALUE': round(temp_val, 1), 'UNITS': 'Cel', 'TYPE': 'numeric'})
        if random.random() > 0.3:
            lactate_val = random.uniform(2.5, 6.0) if is_septic else random.uniform(0.4, 1.8)
            observations.append({'DATE': enc['START'], 'PATIENT': enc['PATIENT'],
                'ENCOUNTER': enc['Id'], 'CODE': '32693-4',
                'DESCRIPTION': 'Lactate', 'VALUE': round(lactate_val, 1), 'UNITS': 'mmol/L', 'TYPE': 'numeric'})
    df_observations = pd.DataFrame(observations)

    os.makedirs(output_dir, exist_ok=True)
    df_patients.to_csv(f'{output_dir}/patients.csv', index=False)
    df_encounters.to_csv(f'{output_dir}/encounters.csv', index=False)
    df_observations.to_csv(f'{output_dir}/observations.csv', index=False)
    print(f"Generated {num_patients} patients, {len(encounters)} encounters, {len(observations)} observations → {output_dir}/")
    return len(encounters)

if __name__ == "__main__":
    generate_synthetic_data(200)
