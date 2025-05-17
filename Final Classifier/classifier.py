import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
SEED = 42
torch.manual_seed(SEED)

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

mamm_model = load_mobilenet_model("/content/drive/MyDrive/mammogram_model.pth")
ultra_model = load_mobilenet_model("/content/drive/MyDrive/ultrasound_model.pth")

from transformers import AutoTokenizer, AutoModel

# Load the ClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# Function to extract features using ClinicalBERT
def extract_bert_features(text,model,tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().numpy()


class MultiModalDataset(Dataset):
    def __init__(self, mamm_dir, ultra_dir, report_df, transform=None):
        self.mamm_dir = mamm_dir
        self.ultra_dir = ultra_dir
        self.reports = report_df['Report'].tolist()
        self.labels = [0]*400 + [1]*400
        self.transform = transform
        self.mamm_imgs = sorted(os.listdir(mamm_dir))
        self.ultra_imgs = sorted(os.listdir(ultra_dir))

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        # Load images
        mamm_img = Image.open(os.path.join(self.mamm_dir, self.mamm_imgs[idx])).convert("RGB")
        ultra_img = Image.open(os.path.join(self.ultra_dir, self.ultra_imgs[idx])).convert("RGB")

        if self.transform:
            mamm_img = self.transform(mamm_img)
            ultra_img = self.transform(ultra_img)

        report = self.reports[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return mamm_img, ultra_img, report, label


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# LOAD DATASETS

report_df = pd.read_csv("/content/drive/MyDrive/Diagnosis reports.csv", encoding="ISO-8859-1")
reports = report_df["Report"].astype(str).tolist()
dataset = MultiModalDataset("/content/drive/MyDrive/mammograms", "/content/drive/MyDrive/ultrasounds", report_df, transform=img_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# IMAGE FEATURE EXTRACTION

def extract_image_features(model, images):
    with torch.no_grad():
        images = images.to(DEVICE)
        feats = model(images)
        feats = feats.mean([2, 3])  # GAP
    return feats.cpu().numpy()

all_mamm_feats = []
all_ultra_feats = []
all_text_feats = []
all_labels = []

# FEATURE EXTRACTION LOOP

for mamm_imgs, ultra_imgs, report_texts, labels in tqdm(loader, desc="Extracting features"):
    mamm_feat = extract_image_features(mamm_model, mamm_imgs)
    ultra_feat = extract_image_features(ultra_model, ultra_imgs)

    # Ensure report_texts are strings (and not the 'index' column)
    text_feat = np.array([extract_bert_features(str(txt),bert_model,bert_tokenizer) for txt in report_texts])

    all_mamm_feats.extend(mamm_feat)
    all_ultra_feats.extend(ultra_feat)
    all_text_feats.extend(text_feat)
    all_labels.extend(labels)


print(len(all_mamm_feats))
print(len(all_mamm_feats[0]))


# CONVERTING ARRAYS INTO COMBINED MATRIX

all_mamm_feats = np.vstack(all_mamm_feats)
all_ultra_feats = np.vstack(all_ultra_feats)
all_text_feats = np.vstack(all_text_feats)
all_labels = np.hstack(all_labels)