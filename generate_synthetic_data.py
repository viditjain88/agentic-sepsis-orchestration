import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_patients=10, output_dir='output'):
    """Generates synthetic patient data mimicking Synthea output."""

    # 1. Patients Table
    patients = []
    for i in range(num_patients):
        patient_id = f"P{i:03d}"
        gender = random.choice(['M', 'F'])
        birthdate = datetime.now() - timedelta(days=random.randint(20*365, 80*365))
        patients.append({
            'Id': patient_id,
            'BIRTHDATE': birthdate.strftime('%Y-%m-%d'),
            'DEATHDATE': None,
            'SSN': f"999-{random.randint(10,99)}-{random.randint(1000,9999)}",
            'DRIVERS': f"S{random.randint(10000000,99999999)}",
            'PASSPORT': f"N{random.randint(10000000,99999999)}",
            'PREFIX': 'Mr.' if gender == 'M' else 'Ms.',
            'FIRST': f"First{i}",
            'LAST': f"Last{i}",
            'SUFFIX': None,
            'MAIDEN': None,
            'MARITAL': random.choice(['M', 'S']),
            'RACE': random.choice(['white', 'black', 'asian', 'hispanic']),
            'ETHNICITY': random.choice(['nonhispanic', 'hispanic']),
            'GENDER': gender,
            'BIRTHPLACE': 'Boston',
            'ADDRESS': '123 Main St',
            'CITY': 'Boston',
            'STATE': 'MA',
            'COUNTY': 'Suffolk',
            'ZIP': '02115',
            'LAT': 42.3601,
            'LON': -71.0589,
            'HEALTHCARE_EXPENSES': random.uniform(1000, 50000),
            'HEALTHCARE_COVERAGE': random.uniform(500, 10000)
        })
    df_patients = pd.DataFrame(patients)

    # 2. Encounters Table
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
                'ORGANIZATION': 'General Hospital',
                'PROVIDER': 'Dr. Smith',
                'PAYER': 'Blue Cross',
                'ENCOUNTERCLASS': 'inpatient',
                'CODE': '185345009',
                'DESCRIPTION': 'Encounter for symptom',
                'BASE_ENCOUNTER_COST': random.uniform(100, 5000),
                'TOTAL_CLAIM_COST': random.uniform(100, 5000),
                'PAYER_COVERAGE': random.uniform(50, 4000),
                'REASONCODE': None,
                'REASONDESCRIPTION': None
            })
    df_encounters = pd.DataFrame(encounters)

    # 3. Observations Table (Vitals & Labs)
    observations = []
    for enc in encounters:
        # Generate sepsis-indicative values for some patients
        is_septic = random.choice([True, False])

        # Heart Rate
        hr_val = random.randint(95, 120) if is_septic else random.randint(60, 90)
        observations.append({
            'DATE': enc['START'],
            'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'],
            'CODE': '8867-4',
            'DESCRIPTION': 'Heart rate',
            'VALUE': hr_val,
            'UNITS': 'beats/min',
            'TYPE': 'numeric'
        })

        # Respiratory Rate
        rr_val = random.randint(22, 30) if is_septic else random.randint(12, 20)
        observations.append({
            'DATE': enc['START'],
            'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'],
            'CODE': '9279-1',
            'DESCRIPTION': 'Respiratory rate',
            'VALUE': rr_val,
            'UNITS': 'breaths/min',
            'TYPE': 'numeric'
        })

        # Temperature
        temp_val = random.uniform(38.0, 40.0) if is_septic else random.uniform(36.5, 37.5)
        observations.append({
            'DATE': enc['START'],
            'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'],
            'CODE': '8310-5',
            'DESCRIPTION': 'Body temperature',
            'VALUE': round(temp_val, 1),
            'UNITS': 'Cel',
            'TYPE': 'numeric'
        })

        # Lactate (Lab)
        if random.random() > 0.5: # Not all encounters have labs immediately
            lactate_val = random.uniform(2.5, 5.0) if is_septic else random.uniform(0.5, 1.5)
            observations.append({
                'DATE': enc['START'],
                'PATIENT': enc['PATIENT'],
                'ENCOUNTER': enc['Id'],
                'CODE': '32693-4',
                'DESCRIPTION': 'Lactate [Moles/volume] in Blood',
                'VALUE': round(lactate_val, 1),
                'UNITS': 'mmol/L',
                'TYPE': 'numeric'
            })

    df_observations = pd.DataFrame(observations)

    # 4. Clinical Notes Table (For NLP Model)
    notes = []

    sepsis_keywords = [
        "Patient presents with severe sepsis.",
        "Lactate is elevated.",
        "Hypotensive requiring vasopressors.",
        "Blood cultures positive for gram negative rods.",
        "ICU admission for septic shock.",
        "Diagnosed with sepsis (ICD-9 995.92)."
    ]

    normal_keywords = [
        "Patient admitted for elective procedure.",
        "Vital signs are stable.",
        "Recovering well, progressing to regular diet.",
        "No signs of infection.",
        "Discharged home in stable condition.",
        "Routine physical exam normal."
    ]

    for enc in encounters:
        # Determine if this encounter is 'septic' based on the vitals generated above
        enc_obs = [o for o in observations if o['ENCOUNTER'] == enc['Id']]
        hr_obs = next((o for o in enc_obs if o['CODE'] == '8867-4'), None)
        temp_obs = next((o for o in enc_obs if o['CODE'] == '8310-5'), None)

        is_septic = False
        if hr_obs and hr_obs['VALUE'] > 95 and temp_obs and temp_obs['VALUE'] > 38.0:
            is_septic = True

        note_length = random.randint(3, 8)
        if is_septic:
            note_parts = random.choices(sepsis_keywords, k=2) + random.choices(normal_keywords, k=note_length-2)
        else:
            note_parts = random.choices(normal_keywords, k=note_length)

        random.shuffle(note_parts)
        note_text = " ".join(note_parts)

        notes.append({
            'DATE': enc['START'],
            'PATIENT': enc['PATIENT'],
            'ENCOUNTER': enc['Id'],
            'TEXT': note_text
        })

    df_notes = pd.DataFrame(notes)

    # Save to CSV
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_patients.to_csv(f'{output_dir}/patients.csv', index=False)
    df_encounters.to_csv(f'{output_dir}/encounters.csv', index=False)
    df_observations.to_csv(f'{output_dir}/observations.csv', index=False)
    df_notes.to_csv(f'{output_dir}/notes.csv', index=False)
    print(f"Generated {num_patients} patients, {len(encounters)} encounters, {len(observations)} observations, and {len(notes)} notes in {output_dir}/")

if __name__ == "__main__":
    generate_synthetic_data()
