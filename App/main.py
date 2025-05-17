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

# Load Scalers
scaler_mamm = load(r'BioFusion-AI\models\scaler_mamm.pkl')
scaler_ultra = load(r'BioFusion-AI\models\scaler_ultra.pkl')

# Load PCA Transformers
pca_mamm = load(r'BioFusion-AI\models\pca_mamm.pkl')
pca_ultra = load(r'BioFusion-AI\models\pca_ultra.pkl')

# Load Final Classifier
classifier_model = load(r'BioFusion-AI\models\final_classifier.pkl')


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

mamm_model = load_mobilenet_model(r"BioFusion-AI\models\mammogram_model.pth")
ultra_model = load_mobilenet_model(r"BioFusion-AI\models\ultrasound_model.pth")


#FEATURE EXTRACTION FUNCTIONS

def extract_image_features(model, images):
    with torch.no_grad():
        images = images.to(DEVICE)
        feats = model(images)
        feats = feats.mean([2, 3])  # GAP
    return feats.cpu().numpy()


def getFeatures(mamm,ult,mam_model,ult_model,s_mamm,s_ultra,pc_mamm,pc_ultra):
  mamm_feat = extract_image_features(mam_model, mamm)
  ultra_feat = extract_image_features(ult_model, ult)

  mamm_feat = s_mamm.transform(mamm_feat)
  ultra_feat = s_ultra.transform(ultra_feat)

  mamm_feats = pc_mamm.transform(mamm_feats)
  ultra_feats = pc_ultra.transform(ultra_feats)

  merged_features = np.concatenate([mamm_feats, ultra_feats], axis=1)

  return merged_features


#Take input mammogram and ultrasound
#Apply img_transform function on both images
#Pass in getFeatures() to get the final merged vector
#Pass that merged vcector to the already loaded classifier for prediction