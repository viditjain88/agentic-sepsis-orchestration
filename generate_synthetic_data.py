import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

def generate_synthetic_data(num_patients=200, output_dir='output'):
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
            encounters.append({
                'Id': encounter_id,
                'START': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'STOP': stop_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'PATIENT': p['Id'],
                'ENCOUNTERCLASS': 'inpatient',
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
