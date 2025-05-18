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
classifier_model = load(r'BioFusion-AI\models\final_classifier_model.pkl')


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


def getFeatures(mamm,ult,mam_model,ult_model,s_mamm,s_ultra,pc_mamm,pc_ultra,struct_vector):
  mamm_feat = extract_image_features(mam_model, mamm)
  ultra_feat = extract_image_features(ult_model, ult)

  mamm_feat = s_mamm.transform(mamm_feat)
  ultra_feat = s_ultra.transform(ultra_feat)

  mamm_feats = pc_mamm.transform(mamm_feats)
  ultra_feats = pc_ultra.transform(ultra_feats)

  merged_features = np.concatenate([mamm_feats, ultra_feats, struct_vector], axis=1)

  return merged_features


# Load trained one-hot encoder and scaler
ohe = load(r"BioFusion-AI\models\onehot_encoder.pkl")       # Fitted on training data
scaler_sub = load(r"BioFusion-AI\models\scaler_subtlety.pkl")  # Fitted StandardScaler


#FUNCTION TO PROCESS THE 4 ADDITIONAL FEATURES
def process_structured_input(metadata_dict, onehot, scaler):
    # Create DataFrame for encoding
    df_cat = pd.DataFrame([{
        "breast_density": metadata_dict["breast_density"],
        "mass shape": metadata_dict["mass shape"],
        "mass margins": metadata_dict["mass margins"]
    }])

    # One-hot encode
    encoded_cat = onehot.transform(df_cat)

    # Scale subtlety
    subtlety_val = np.array([[metadata_dict["subtlety"]]])  
    subtlety_scaled = scaler.transform(subtlety_val)        

    # Combine
    structured_vector = np.concatenate([encoded_cat, subtlety_scaled], axis=1).squeeze()

    return structured_vector

new_metadata = {
    "breast_density": 3,
    "mass shape": "ARCHITECTURAL_DISTORTION",
    "mass margins": "ILL_DEFINED",
    "subtlety": 3
}

vector = process_structured_input(new_metadata,ohe,scaler_sub)
print(vector.shape)



#Take input mammogram and ultrasound
#Take additional 4 inputs through dropdowns
#Convert those in dictionary
#pass through process_structured_input function to get its feature vector
#Apply img_transform function on both images
#Pass in getFeatures() to get the final merged vector
#Pass that merged vcector to the already loaded classifier for prediction