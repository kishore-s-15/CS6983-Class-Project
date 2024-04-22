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
    
    print("Shape of CXR Meta File = {cxr_meta_df.shape}")\

    TEXT_REPR_JSON_URL = ba
   
    xray_reports_for_stayids = {}
    for gen_report in tqdm(gen_reports_glob, desc="Getting the X-ray text"):
        print(f"Gen Report: {gen_report}")
        dicom_id = gen_report.split("gen_")[2].split(".txt")[0]
        print(f"Dicom ID: {dicom_id}")

        with open(gen_report, "r") as report_file:
            xray_text = report_file.read()
            xray_text = xray_text.split("Generated report:")[1]
            # print("Orig:")
            # print(xray_text)
            xray_text = " ".join(list(filter(lambda x: "=" not in x, xray_text.split("\n"))))
            # print(xray_text)

            stay_id = cxr_meta_df[cxr_meta_df["dicom_id"] == dicom_id]["stay_id"]
            # print(stay_id)
            if stay_id.shape == 0:
                continue
            
            stay_id = int(stay_id.iloc[0])
            xray_reports_for_stayids[stay_id] = xray_text
            print(xray_reports_for_stayids)
            break
        
             
