import pandas as pd
import json
import os

def harmonize_data(input_dir='output', output_file='output/harmonized_data.json'):
    patients   = pd.read_csv(f'{input_dir}/patients.csv')
    encounters = pd.read_csv(f'{input_dir}/encounters.csv')
    observations = pd.read_csv(f'{input_dir}/observations.csv')

    harmonized_data = []
    for _, patient in patients.iterrows():
        p_id = patient['Id']
        p_data = {
            'subject_id': p_id,
            'demographics': {
                'age': 2026 - int(str(patient['BIRTHDATE'])[:4]),
                'gender': patient['GENDER'],
                'race': patient['RACE']
            },
            'visits': []
        }
        p_encounters = encounters[encounters['PATIENT'] == p_id]
        for _, enc in p_encounters.iterrows():
            enc_id = enc['Id']
            visit_data = {
                'hadm_id': enc_id,
                'admittime': enc['START'],
                'dischtime': enc['STOP'],
                'events': []
            }
            enc_obs = observations[observations['ENCOUNTER'] == enc_id]
            for _, obs in enc_obs.iterrows():
                visit_data['events'].append({
                    'charttime': obs['DATE'],
                    'itemid': obs['CODE'],
                    'label': obs['DESCRIPTION'],
                    'value': obs['VALUE'],
                    'valuenum': obs['VALUE'],
                    'valueuom': obs['UNITS']
                })
            p_data['visits'].append(visit_data)
        harmonized_data.append(p_data)

    with open(output_file, 'w') as f:
        json.dump(harmonized_data, f, indent=2)
    print(f"Harmonized {len(harmonized_data)} patients → {output_file}")

if __name__ == "__main__":
    harmonize_data()
