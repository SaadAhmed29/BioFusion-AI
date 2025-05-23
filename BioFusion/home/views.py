from django.shortcuts import render
from .forms import PredictionForm
from PIL import Image
import torch
from torchvision import transforms

from App.main import (
    getFeatures, process_structured_input,
    scaler_mamm, scaler_ultra, pca_mamm, pca_ultra,
    classifier_model, ohe, scaler_sub,
    mamm_model, ultra_model, img_transform
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            # Process images
            mamm_img = Image.open(form.cleaned_data['mammogram']).convert('RGB')
            ultra_img = Image.open(form.cleaned_data['ultrasound']).convert('RGB')
            mamm_tensor = img_transform(mamm_img).unsqueeze(0)
            ultra_tensor = img_transform(ultra_img).unsqueeze(0)

            # Process metadata
            metadata = {
                'breast_density': int(form.cleaned_data['breast_density']),
                'mass shape': form.cleaned_data['mass_shape'],
                'mass margins': form.cleaned_data['mass_margins'],
                'family_history': form.cleaned_data['family_history'],
                'hormone_therapy': form.cleaned_data['hormone_therapy'],
                'previous_biopsy': form.cleaned_data['previous_biopsy'],
                'breastfeeding': form.cleaned_data['breastfeeding'],
                'breast_pain': form.cleaned_data['breast_pain'],
                'brca_mutation_status': form.cleaned_data['brca_mutation_status'],
                'subtlety': int(form.cleaned_data['subtlety']),
                'age': form.cleaned_data['age'],
                'bmi': form.cleaned_data['bmi']
            }

            struct_vector = process_structured_input(metadata, ohe, scaler_sub).reshape(1, -1)

            # Get prediction
            merged_features = getFeatures(
                mamm_tensor, ultra_tensor,
                mamm_model, ultra_model,
                scaler_mamm, scaler_ultra,
                pca_mamm, pca_ultra,
                struct_vector
            )
            prediction = classifier_model.predict_proba(merged_features)[0][1]
            result = f"Predicted Probability: {prediction:.2f}"

            return render(request, 'result.html', {'result': result})
    else:
        form = PredictionForm()

    return render(request, 'index.html', {'form': form})
