import os
import pandas as pd
import json
from tqdm import tqdm

from utils import stringify_visit_codes, stringify_visit_labs, stringify_visit_meds, stringify_visit_meta

""" This program reads in MIMIC-IV hospital files and converts them to text. More documentation to follow..."""


def convert_admissions(patient_folder, output_folder_path, output='admissions_text.json'):
    patient_hosp_adm_file_url = os.path.join(patient_folder, 'Hosp', 'admissions.csv')    


    if not os.path.exists(patient_hosp_adm_file_url):
        print(f"Admissions file not found for patient folder {patient_folder}. Skipping...")
        return
    
    admissions = pd.read_csv(patient_hosp_adm_file_url)
    res = {}

    for _, r in tqdm(admissions.iterrows()):
        res[r['hadm_id']] = stringify_visit_meta(
            subject_id=r.get('subject_id'),
            hadm_id=r.get('hadm_id'),
            admittime=r.get('admittime'),
            dischtime=r.get('dischtime'),
            deathtime=r.get('deathtime'),
            admission_type=r.get('admission_type'),
            admission_location=r.get('admission_location'),
            discharge_location=r.get('discharge_location'),
            insurance=r.get('insurance'),
            language=r.get('language'),
            marital_status=r.get('marital_status'),
            race=r.get('race')
        )
	
    patient_hosp_adm_output_file_url = os.path.join(output_folder_path, output)
    with open(patient_hosp_adm_output_file_url, "w") as f:
        json.dump(dict(res), f)

    return patient_hosp_adm_output_file_url


def convert_codes(patient_folder, output_folder_path, output="codes_text.json"):
    if not os.path.exists(os.path.join(patient_folder, 'Hosp', 'diagnoses_icd.csv')):
        print(f"Diagnoses ICD file not found for patient folder {patient_folder}. Skipping...")
        return


    codes = pd.read_csv(os.path.join(patient_folder, 'Hosp', 'diagnoses_icd.csv'))
    code_key = pd.read_csv(os.path.join(mimic_eye_folder, "spreadsheets" , 'Hosp', 'd_icd_diagnoses.csv')).set_index(['icd_code', 'icd_version'])

    # def _stringify(tdf):
    #     this_codes = [dict(
    #         code_type=r['icd_version'],
    #         code_value=r['icd_code'],
    #         code_text=code_key.loc[r['icd_code'], int(r['icd_version'])].long_title
    #     ) for _, r in tdf.iterrows()]
    #     this_string = stringify_visit_codes(this_codes)
    #     return this_string
    
    def _stringify(tdf):
        this_codes = []
        for _, r in tdf.iterrows():
            icd_code = r['icd_code']
            icd_version = int(r['icd_version'])
            if (icd_code, icd_version) in code_key.index:
                code_text = code_key.loc[(icd_code, icd_version)].long_title
                this_codes.append(dict(
                    code_type=r['icd_version'],
                    code_value=r['icd_code'],
                    code_text=code_text
                ))
            else:
                print(f"ICD code {icd_code} version {icd_version} not found in the key. Skipping...")
        this_string = stringify_visit_codes(this_codes)
        return this_string

    res = codes.groupby('hadm_id').apply(_stringify)
    with open(os.path.join(output_folder_path, output), "w") as f:
        json.dump(dict(res), f)
    return os.path.join(output_folder_path, output)


def convert_labs(patient_folder, output_folder_path, output="labs_text.json"):
    labs_key = pd.read_csv(os.path.join(mimic_eye_folder,'spreadsheets', 'Hosp', 'd_labitems.csv')).set_index(['itemid'])

    def _dataprep(r):
        return dict(
            valueuom=r.get('valueuom'),
            lab_value=r.get('value'),
            lab_name=labs_key.loc[r['itemid']].label,
            flag=r.get('flag'),
            datetime=r.get('storetime'))

    print('processing in chunks...')
    if not os.path.exists(os.path.join(patient_folder, 'Hosp', 'labevents.csv')):
        print(f"Lab events file not found for patient folder {patient_folder}. Skipping...")
        return
    labs_chunked = pd.read_csv(os.path.join(patient_folder, 'Hosp', 'labevents.csv'), chunksize=50000)
    data_to_convert = {}
    for labs in tqdm(labs_chunked):
        labs = labs[~labs.hadm_id.isna() & ~labs['value'].isna()]
        for _, r in labs.iterrows():
            if r['hadm_id'] not in data_to_convert:
                data_to_convert[r['hadm_id']] = []
            data_to_convert[r['hadm_id']].append(_dataprep(r))
    print('prepped.')

    print('stringifying...')
    res = {k: stringify_visit_labs(v) for k, v in tqdm(data_to_convert.items())}
    print('writing...')
    with open(os.path.join(output_folder_path, output), "w") as f:
        json.dump(dict(res), f)
    print('done.')
    return os.path.join(output_folder_path, output)


def convert_meds(patient_folder, output_folder_path, output="meds_text.json"):
    print('loading data...')
    if not os.path.exists(os.path.join(patient_folder, 'Hosp', 'prescriptions.csv')):
        print(f"Prescriptions file not found for patient folder {patient_folder}. Skipping...")
        return
    meds = pd.read_csv(os.path.join(patient_folder, 'Hosp', 'prescriptions.csv'))
    meds = meds[~meds.hadm_id.isna()]

    def _stringify(tdf):
        this_meds = [dict(
            drug=r.get('drug'),
            route=r.get('route'),
            starttime=r.get('starttime'),
            endtime=r.get('endtime')
        ) for _, r in tdf.iterrows()]
        this_string = stringify_visit_meds(this_meds)
        return this_string
    
    

    print('processing...')
    res = meds.groupby('hadm_id').apply(_stringify)
    with open(os.path.join(output_folder_path, output), "w") as f:
        json.dump(dict(res), f)
    print('done.')
    return os.path.join(output_folder_path, output)


if __name__ == '__main__':
    mimic_eye_folder = os.path.join("/", "scratch", "sampath.ki", "data", "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0", "mimic-eye")

    output_folder_path = os.path.join("/", "home", "sampath.ki", "MEME", "output", "gen_reports")
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for patient_folder in tqdm(os.listdir(mimic_eye_folder)):
        if os.path.isdir(os.path.join(mimic_eye_folder, patient_folder)):
            if not os.path.exists(os.path.join(output_folder_path, patient_folder)):
                os.makedirs(os.path.join(output_folder_path, patient_folder))

            print(f"Processing patient {patient_folder}...")
            print("Running codes")
            codes_p = convert_codes(os.path.join(mimic_eye_folder, patient_folder), os.path.join(output_folder_path, patient_folder))
            print("Codes complete.")
            print("Running meds")
            meds_p = convert_meds(os.path.join(mimic_eye_folder, patient_folder), os.path.join(output_folder_path, patient_folder))
            print("Meds complete.")
            print("Running labs")
            labs_p = convert_labs(os.path.join(mimic_eye_folder, patient_folder), os.path.join(output_folder_path, patient_folder))
            print("Labs complete.")
            print("Running admissions.")
            adm_p = convert_admissions(os.path.join(mimic_eye_folder, patient_folder), os.path.join(output_folder_path, patient_folder))
            print("Done.")
