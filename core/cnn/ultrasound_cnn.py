# -*- coding: utf-8 -*-
"""BioFusion Ultrasound.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13L31o03jij1vfeEz2p9iAawEWhtMEyTe
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset path
data_path = "/content/drive/MyDrive/Ultrasound Dataset"

# Load dataset
full_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)

# Split into train and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply validation transforms
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Modify classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop with early stopping
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 3
trigger_times = 0
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    # Validation
    model.eval()
    val_loss = 0
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            y_probs.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_ultrasound_model.pth")
        print(f"Epoch {epoch+1}: Val loss improved. Model saved.")
    else:
        trigger_times += 1
        print(f"Epoch {epoch+1}: No improvement. Patience {trigger_times}/{patience}")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

    print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}")

# Evaluation
y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print(f"\nFinal Evaluation Metrics on Ultrasound Dataset:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

from google.colab import files
files.download("best_ultrasound_model.pth")

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Import confusion_matrix

# Assuming y_true and y_pred are defined from your previous code
cm = confusion_matrix(y_true, y_pred)

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix for Ultrasound Dataset")
plt.show()

# prompt: #line plot of loss

# Plot training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()