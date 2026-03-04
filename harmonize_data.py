import pandas as pd
import json
import os

def harmonize_data(input_dir='output', output_file='output/harmonized_data.json'):
    """Maps synthetic CSVs to a unified patient-event JSON structure."""

    # Load CSVs
    try:
        patients = pd.read_csv(f'{input_dir}/patients.csv')
        encounters = pd.read_csv(f'{input_dir}/encounters.csv')
        observations = pd.read_csv(f'{input_dir}/observations.csv')
        notes = pd.read_csv(f'{input_dir}/notes.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    harmonized_data = []

    for _, patient in patients.iterrows():
        p_id = patient['Id']
        p_data = {
            'subject_id': p_id,
            'demographics': {
                'age': 2025 - int(patient['BIRTHDATE'][:4]), # Approx age
                'gender': patient['GENDER'],
                'race': patient['RACE']
            },
            'visits': []
        }

        # Get encounters for this patient
        p_encounters = encounters[encounters['PATIENT'] == p_id]

        for _, enc in p_encounters.iterrows():
            enc_id = enc['Id']
            visit_data = {
                'hadm_id': enc_id, # Using Encounter ID as admission ID
                'admittime': enc['START'],
                'dischtime': enc['STOP'],
                'events': []
            }

            # Get observations for this encounter
            enc_obs = observations[observations['ENCOUNTER'] == enc_id]

            for _, obs in enc_obs.iterrows():
                event = {
                    'charttime': obs['DATE'],
                    'itemid': obs['CODE'],
                    'label': obs['DESCRIPTION'],
                    'value': obs['VALUE'],
                    'valuenum': obs['VALUE'], # Assuming numeric for now
                    'valueuom': obs['UNITS']
                }
                visit_data['events'].append(event)

            # Get clinical note for this encounter
            enc_notes = notes[notes['ENCOUNTER'] == enc_id]
            visit_notes = []
            for _, note in enc_notes.iterrows():
                visit_notes.append({
                    'charttime': note['DATE'],
                    'text': note['TEXT']
                })
            visit_data['clinical_notes'] = visit_notes

            p_data['visits'].append(visit_data)

        harmonized_data.append(p_data)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(harmonized_data, f, indent=2)

    print(f"Harmonized data for {len(harmonized_data)} patients saved to {output_file}")

if __name__ == "__main__":
    harmonize_data()
