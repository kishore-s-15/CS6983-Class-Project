# Importing the required libraries

import os
from glob import glob
from pathlib import Path
from collections import defaultdict

import spacy
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.full_model.train_full_model import get_tokenizer
from src.full_model.generate_reports_for_images import get_model
from src.full_model.generate_reports_for_images import get_image_tensor
from src.full_model.generate_reports_for_images import get_report_for_image
from src.full_model.generate_reports_for_images import write_generated_reports_to_txt

# Data URLs

BASE_DIR = os.path.join("/", "scratch", "sampath.ki")
DATA_DIR = os.path.join(
    BASE_DIR, "data",
    "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0",
    "mimic-eye"
)

print(f"DATA_DIR = {DATA_DIR}")

# Number of patient folders

PATIENTS_DIR = os.path.join(DATA_DIR, "patient_*")
patient_folders = glob(PATIENTS_DIR)

print(f"len(patient_folders) = {len(patient_folders)}")

# Number of X-ray reports

CXR_DICOM_DIR = os.path.join(DATA_DIR, "patient_*", "CXR-DICOM")

REFERENCE_REPORT_PATHS = glob(os.path.join(CXR_DICOM_DIR, "*.txt"))

print(f"len(REFERENCE_REPORT_PATHS) = {len(REFERENCE_REPORT_PATHS)}")


IMAGE_PATHS = []

for ref_report_path in tqdm(REFERENCE_REPORT_PATHS):
    patient_id = ref_report_path.split("/")[-3].split("_")[-1]
    xray_id = ref_report_path.split("/")[-1].split(".")[0]

    meta_file_url = os.path.join(DATA_DIR, f"patient_{patient_id}", "CXR-JPG", "cxr_meta.csv")
    
    df = pd.read_csv(meta_file_url)
    studied_xrays = set(df["dicom_id"].values)

    study_xray_images = glob(os.path.join(DATA_DIR, f"patient_{patient_id}", "CXR-JPG", xray_id, "*.jpg"))

    for xray_image in study_xray_images:
        image_id = xray_image.split("/")[-1].split(".")[0]
        
        if image_id in studied_xrays:
            IMAGE_PATHS.append(xray_image)
    
print(f"len(IMAGE_PATHS) = {len(IMAGE_PATHS)}")


cnt = set()

patient_ids = []

for img_path in IMAGE_PATHS:
    study_id = img_path.split("/")[-2]

    if study_id in cnt:
        patient_id = img_path.split("/")[-4].split("_")[1]
        patient_ids.append(patient_id)
    
    cnt.add(study_id)

print(f"len(cnt) = {len(cnt)}")

# Value counts of number of reports per patient

xray_report_counts = defaultdict(int)

for ref_file_path in REFERENCE_REPORT_PATHS:
    patient_id = ref_file_path.split("/")[-3].split("_")[-1]
    xray_report_counts[patient_id] += 1
    
values, counts = np.unique(list(xray_report_counts.values()), return_counts=True)

print("list(zip(list(values), list(counts)))")
print(list(zip(list(values), list(counts))))

checkpoint_path = "full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt"
model = get_model(checkpoint_path)
print("Model instantiated.")

bert_score = evaluate.load("bertscore")
sentence_tokenizer = spacy.load("en_core_web_trf")
tokenizer = get_tokenizer()

import torch

from src.full_model.generate_reports_for_images import convert_generated_sentences_to_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERTSCORE_SIMILARITY_THRESHOLD = 0.9
IMAGE_INPUT_SIZE = 512
MAX_NUM_TOKENS_GENERATE = 300
NUM_BEAMS = 4
mean = 0.471  # see get_transforms in src/dataset/compute_mean_std_dataset.py
std = 0.302

def get_report_for_images(model, images_tensor, tokenizer, bert_score, sentence_tokenizer):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.generate(
            images_tensor.to(device, non_blocking=True),
            max_length=MAX_NUM_TOKENS_GENERATE,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    beam_search_output, _, _, _ = output

    generated_sents_for_selected_regions = tokenizer.batch_decode(
        beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )  # list[str]

    generated_report = convert_generated_sentences_to_report(
        generated_sents_for_selected_regions, bert_score, sentence_tokenizer
    )  # str

    return generated_report


import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_image_tensor(image_path):
    # cv2.imread by default loads an image with 3 channels
    # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape (3056, 2544)

    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    transform = val_test_transforms(image=image)
    image_transformed = transform["image"]  # shape (1, 512, 512)
    image_transformed_batch = image_transformed.unsqueeze(0)  # shape (1, 1, 512, 512)

    return image_transformed_batch


import shutil


if not os.path.exists("orig_reports"):
    os.makedirs("orig_reports")

if not os.path.exists("gen_reports"):
    os.makedirs("gen_reports")


for image_path in tqdm(IMAGE_PATHS):
    generated_reports = []
    
    xray_id = image_path.split("/")[-1].split(".")[0]
    patient_id = image_path.split("/")[-4].split("_")[1]
    generated_reports_txt_path = os.path.join("gen_reports", f"gen_{xray_id}.txt")

    subject_id = image_path.split("/")[-2]
    original_reports_txt_path = os.path.join("orig_reports", f"orig_{xray_id}.txt")
    report_path = os.path.join(DATA_DIR, f"patient_{patient_id}", "CXR-DICOM", f"{subject_id}.txt")

    shutil.copy(report_path, original_reports_txt_path)
    
    image_tensor = get_image_tensor(image_path)  # shape (1, 1, 512, 512)
    generated_report = get_report_for_images(model, image_tensor, tokenizer, bert_score, sentence_tokenizer)
    generated_reports.append(generated_report)
    
    write_generated_reports_to_txt([image_path], generated_reports, generated_reports_txt_path)
    
    
def read_txt_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        content = file.read()

    sents = content.split("\n")
    sents = list(map(lambda x: x.strip(), sents))

    return sents


def preprocess_reference(sents: list[str]) -> str:
    def isHeader(sent: str) -> bool:
        if sent == "FINAL REPORT":
            return True

    preprocessed_sents = []

    for sent in sents:
        if sent != '' and not isHeader(sent):
            if ':' in sent:
                sent = sent.split(":")[-1].strip()
                
            preprocessed_sents.append(sent)

    return " ".join(preprocessed_sents)


def preprocess_prediction(sents: list[str]) -> str:
    def isHeader(sent: str) -> bool:
        if "Image path:" in sent or '=' in sent:
            return True

    preprocessed_sents = []

    for sent in sents:
        if sent != '' and not isHeader(sent):
            if "Generated report:" in sent:
                sent = sent.split(":")[-1].strip()
                
            preprocessed_sents.append(sent)

    return " ".join(preprocessed_sents)


def get_reference(file_path: str) -> str:
    reference = read_txt_file(file_path)
    reference = preprocess_reference(reference)

    return reference


def get_prediction(file_path: str) -> str:
    prediction = read_txt_file(file_path)
    prediction = preprocess_prediction(prediction)

    return prediction

xray_ids = []

for file_path in glob(os.path.join(Path().cwd(), "gen_reports", "gen_*.txt")):
    xray_ids.append(
        file_path.split("/")[-1].split(".")[0].split("_")[1]
    )

print(f"len(xray_ids) = {len(xray_ids)}")

references = []
predictions = []

for id in xray_ids:
    ref_file_path = os.path.join("orig_reports", f"orig_{id}.txt")
    gen_file_path = os.path.join("gen_reports", f"gen_{id}.txt")

    reference = get_reference(ref_file_path)
    prediction = get_prediction(gen_file_path)

    references.append(reference)
    predictions.append(prediction)
    
rouge = evaluate.load("rouge")

results = rouge.compute(predictions=predictions,
                        references=references)

print(f"ROUGE: {results}")


bleu = evaluate.load("bleu")

results = bleu.compute(predictions=predictions,
                        references=references)

print(f"BLEU Scores: {results}")

