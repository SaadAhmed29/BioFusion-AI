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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from joblib import dump


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


class MultiModalDataset(Dataset):
    def __init__(self, mamm_dir, ultra_dir, transform=None):
        self.mamm_dir = mamm_dir
        self.ultra_dir = ultra_dir
        self.labels = [0]*400 + [1]*400
        self.transform = transform
        self.mamm_imgs = sorted(os.listdir(mamm_dir))
        self.ultra_imgs = sorted(os.listdir(ultra_dir))

    def __len__(self):
        return len(self.mamm_imgs)

    def __getitem__(self, idx):
        # Load images
        mamm_img = Image.open(os.path.join(self.mamm_dir, self.mamm_imgs[idx])).convert("RGB")
        ultra_img = Image.open(os.path.join(self.ultra_dir, self.ultra_imgs[idx])).convert("RGB")

        if self.transform:
            mamm_img = self.transform(mamm_img)
            ultra_img = self.transform(ultra_img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return mamm_img, ultra_img, label


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# LOAD DATASETS

dataset = MultiModalDataset("/content/drive/MyDrive/mammograms", "/content/drive/MyDrive/ultrasounds", transform=img_transform)
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
all_labels = []

# FEATURE EXTRACTION LOOP

for mamm_imgs, ultra_imgs, labels in tqdm(loader, desc="Extracting features"):
    mamm_feat = extract_image_features(mamm_model, mamm_imgs)
    ultra_feat = extract_image_features(ultra_model, ultra_imgs)

    all_mamm_feats.extend(mamm_feat)
    all_ultra_feats.extend(ultra_feat)
    all_labels.extend(labels)


print(len(all_mamm_feats))
print(len(all_mamm_feats[0]))


# CONVERTING ARRAYS INTO COMBINED MATRIX

all_mamm_feats = np.vstack(all_mamm_feats)
all_ultra_feats = np.vstack(all_ultra_feats)
all_labels = np.hstack(all_labels)


from sklearn.model_selection import train_test_split
import numpy as np

# Labels
y_all = np.array(all_labels)

# Get indices only
indices = np.arange(len(y_all))

train_idx, test_idx, _, _ = train_test_split(indices, y_all, test_size=0.2, stratify=y_all, random_state=42)

# Apply to individual modalities
X_mamm_train = all_mamm_feats[train_idx]
X_mamm_test  = all_mamm_feats[test_idx]

X_ultra_train = all_ultra_feats[train_idx]
X_ultra_test  = all_ultra_feats[test_idx]

y_train = y_all[train_idx]
y_test  = y_all[test_idx]


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Mammogram
scaler_mamm = StandardScaler().fit(X_mamm_train)
X_mamm_train = scaler_mamm.transform(X_mamm_train)
X_mamm_test  = scaler_mamm.transform(X_mamm_test)

pca_mamm = PCA(n_components=256).fit(X_mamm_train)
X_mamm_train = pca_mamm.transform(X_mamm_train)
X_mamm_test  = pca_mamm.transform(X_mamm_test)

# Ultrasound
scaler_ultra = StandardScaler().fit(X_ultra_train)
X_ultra_train = scaler_ultra.transform(X_ultra_train)
X_ultra_test  = scaler_ultra.transform(X_ultra_test)

pca_ultra = PCA(n_components=256).fit(X_ultra_train)
X_ultra_train = pca_ultra.transform(X_ultra_train)
X_ultra_test  = pca_ultra.transform(X_ultra_test)


X_train_final = np.concatenate([X_mamm_train, X_ultra_train], axis=1)
X_test_final  = np.concatenate([X_mamm_test,  X_ultra_test],  axis=1)

print(X_train_final.shape)
print(X_test_final.shape)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(max_iter=1000)
clf.fit(X_train_final, y_train)

y_pred = clf.predict(X_test_final)

accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)

print("EVALUATION METRICS:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

# Save Scalers
dump(scaler_mamm, 'scaler_mamm.pkl')
dump(scaler_ultra, 'scaler_ultra.pkl')

# Save PCA Transformers
dump(pca_mamm, 'pca_mamm.pkl')
dump(pca_ultra, 'pca_ultra.pkl')

# Save Final Classifier Model
dump(clf, 'final_classifier.pkl')
