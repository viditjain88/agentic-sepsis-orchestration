import pandas as pd
import json
import os

def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def harmonize(input_dir='output', output_file='output/harmonized_data.json'):
    df_patients = load_csv(f'{input_dir}/patients.csv')
    df_encounters = load_csv(f'{input_dir}/encounters.csv')
    df_observations = load_csv(f'{input_dir}/observations.csv')
    
    if df_patients.empty or df_encounters.empty or df_observations.empty:
        print("Missing required CSV files.")
        return
        
    unified_patients = []
    
    for _, p_row in df_patients.iterrows():
        pid = p_row['Id']
        p_encs = df_encounters[df_encounters['PATIENT'] == pid]
        
        visits = []
        for _, e_row in p_encs.iterrows():
            eid = e_row['Id']
            e_obs = df_observations[df_observations['ENCOUNTER'] == eid]
            
            events = []
            for _, o_row in e_obs.iterrows():
                events.append({
                    'itemid': str(o_row['CODE']),
                    'valuenum': float(o_row['VALUE']) if not pd.isna(o_row['VALUE']) else None,
                    'charttime': str(o_row['DATE'])
                })
                
            visit_data = {
                'hadm_id': str(eid),
                'admittime': str(e_row['START']),
                'dischtime': str(e_row['STOP']),
                'events': events
            }
            
            if 'CLINICAL_NOTE' in e_row and not pd.isna(e_row['CLINICAL_NOTE']):
                visit_data['clinical_note'] = str(e_row['CLINICAL_NOTE'])
            if 'CELLULAR_DATA' in e_row and not pd.isna(e_row['CELLULAR_DATA']):
                try:
                    visit_data['cellular_data'] = json.loads(e_row['CELLULAR_DATA'])
                except json.JSONDecodeError:
                    visit_data['cellular_data'] = None
            
            visits.append(visit_data)
            
        unified_patients.append({
            'subject_id': str(pid),
            'demographics': {
                'gender': str(p_row['GENDER']),
                'race': str(p_row['RACE']),
                'age': (pd.to_datetime('today') - pd.to_datetime(p_row['BIRTHDATE'])).days // 365
            },
            'visits': visits
        })
        
    with open(output_file, 'w') as f:
        json.dump(unified_patients, f, indent=2)
        
    print(f"Harmonized {len(unified_patients)} patients into {output_file}")

if __name__ == '__main__':
    harmonize()
