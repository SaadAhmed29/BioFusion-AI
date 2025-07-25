from pathlib import Path
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from joblib import load

BASE_DIR = Path(__file__).resolve().parent  # Now points to core/
MODELS_DIR = BASE_DIR / "models"  # Correct path: BioFusion-AI/core/models/

scaler_mamm = load(MODELS_DIR / "scaler_mamm.pkl")
scaler_ultra = load(MODELS_DIR / "scaler_ultra.pkl")
pca_mamm = load(MODELS_DIR / "pca_mamm.pkl")
pca_ultra = load(MODELS_DIR / "pca_ultra.pkl")
classifier_model = load(MODELS_DIR / "GB_classifier_model.pkl")



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mobilenet_model(weights_path):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model.features  # use only feature extractor


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mamm_model = load_mobilenet_model(str(MODELS_DIR / "mammogram_model.pth"))
ultra_model = load_mobilenet_model(str(MODELS_DIR / "ultrasound_model.pth"))

#FEATURE EXTRACTION FUNCTIONS

def extract_image_features(model, images):
    with torch.no_grad():
        images = images.to(DEVICE)
        feats = model(images)
        feats = feats.mean([2, 3])  # GAP
    return feats.cpu().numpy()


def getFeatures(mamm,ult,mam_model,ult_model,s_mamm,s_ultra,pc_mamm,pc_ultra,struct_vector):
  mamm_feat = extract_image_features(mam_model, mamm)
  ultra_feat = extract_image_features(ult_model, ult)

  mamm_feat = s_mamm.transform(mamm_feat)
  ultra_feat = s_ultra.transform(ultra_feat)

  mamm_feats = pc_mamm.transform(mamm_feat)
  ultra_feats = pc_ultra.transform(ultra_feat)

  merged_features = np.concatenate([mamm_feats, ultra_feats, struct_vector], axis=1)

  return merged_features


# Load trained one-hot encoder and scaler
ohe = load(MODELS_DIR / "onehot_encod.pkl")
scaler_sub = load(MODELS_DIR / "numeric_scaler.pkl")

#FUNCTION TO PROCESS THE 4 ADDITIONAL FEATURES
def process_structured_input(metadata_dict, onehot, scaler_sb):
    # Define columns to extract in correct order
    categorical_cols = [
        'breast_density', 'mass shape', 'mass margins',
        'family_history', 'hormone_therapy', 'previous_biopsy',
        'breastfeeding', 'breast_pain', 'brca_mutation_status'
    ]

    numeric_cols = ['subtlety', 'age', 'bmi']

    # Prepare DataFrames
    df_cat = pd.DataFrame([{key: metadata_dict[key] for key in categorical_cols}])
    df_num = np.array([[metadata_dict[key] for key in numeric_cols]])

    # Transform
    encoded_cat = onehot.transform(df_cat)
    scaled_num = scaler_sb.transform(df_num)

    # Combine
    structured_vector = np.concatenate([encoded_cat, scaled_num], axis=1).squeeze()

    return structured_vector

sample = {
    "breast_density": 3,
    "mass shape": "ROUND",
    "mass margins": "CIRCUMSCRIBED",
    "family_history": "Yes",
    "hormone_therapy": "No",
    "previous_biopsy": "Yes",
    "breastfeeding": "No",
    "breast_pain": "Yes",
    "brca_mutation_status": "Negative",
    "subtlety": 4,
    "age": 55,
    "bmi": 27.6
}

vector = process_structured_input(sample,ohe,scaler_sub)
print(vector.shape)



#Take inputs
#Convert those in dictionary
#pass through process_structured_input function to get its feature vector
#Apply img_transform function on both images
#Pass in getFeatures() to get the final merged vector
#Pass that merged vcector to the already loaded classifier for prediction