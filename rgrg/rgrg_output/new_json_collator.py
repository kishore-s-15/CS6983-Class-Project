import os
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    BASE_DIR = Path().cwd()
    print(f"Current working directory: {str(BASE_DIR)}")

    GEN_REPORTS_DIR = BASE_DIR / "gen_reports"
    gen_reports_glob = glob(str(GEN_REPORTS_DIR / "gen_*"))

    print(f"There are a total of {len(gen_reports_glob)} reports.")
    
    # CXR_META_URL = os.path.join("/", "scratch", "sampath.ki", "data", "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0", "mimic-eye", "spreadsheets", "cxr_meta_with_stay_id_only.csv")
    CXR_META_URL = os.path.join("/", "scratch", "sampath.ki", "data", "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0", "mimic-eye", "spreadsheets", "cxr_meta.csv")
    cxr_meta_df = pd.read_csv(CXR_META_URL)
    
    print(f"Shape of CXR Meta File = {cxr_meta_df.shape}")

    TEXT_REPR_JSON_URL = BASE_DIR.parent.parent / "MEME" / "output" / "new_text_repr.json"
    with open(TEXT_REPR_JSON_URL, "r") as f:
        text_repr = json.load(f)
    
    cnt = 0   
    null_stayid, no_association_text_repr = 0, 0 
    for gen_report in tqdm(gen_reports_glob, desc="Getting the X-ray text"):
        # print(f"Gen Report: {gen_report}")
        dicom_id = gen_report.split("gen_")[2].split(".txt")[0]
        # print(f"Dicom ID: {dicom_id}")

        with open(gen_report, "r") as report_file:
            xray_text = report_file.read()
            xray_text = xray_text.split("Generated report:")[1]
            xray_text = " ".join(list(filter(lambda x: "=" not in x, xray_text.split("\n"))))

            stay_id = cxr_meta_df[cxr_meta_df["dicom_id"] == dicom_id]["stay_id"]
            
            if pd.isnull(stay_id.iloc[0]):
                null_stayid += 1
                continue 

            stay_id = str(int(stay_id.iloc[0]))
            if stay_id not in text_repr:
                no_association_text_repr += 1
                continue

            cnt += 1

            text_repr[stay_id]["xray_report"] = xray_text

    with open("new_text_repr_with_xray.json", "w") as file:
        json.dump(text_repr, file)
    
    print(f"There are a total of {null_stayid} Xray's with Null Stay IDs and {no_association_text_repr} no associations")
    print(f"{cnt} X-ray information matches with the text_repr data")


