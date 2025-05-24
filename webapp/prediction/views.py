from django.shortcuts import render
from .forms import MedicalDiagnosisForm
from PIL import Image
import numpy as np
from core.services import ( 
    process_structured_input,
    getFeatures,
    mamm_model,
    ultra_model,
    scaler_mamm,
    scaler_ultra,
    pca_mamm,
    pca_ultra,
    classifier_model,
    img_transform,
    ohe,
    scaler_sub
)

def diagnosis_view(request):
    if request.method == 'POST':
        form = MedicalDiagnosisForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # 1. Process structured data
                structured_data = {
                    'breast_density': form.cleaned_data['breast_density'],
                    'mass shape': form.cleaned_data['mass_shape'],
                    'mass margins': form.cleaned_data['mass_margins'],
                    'family_history': form.cleaned_data['family_history'],
                    'hormone_therapy': form.cleaned_data['hormone_therapy'],
                    'previous_biopsy': form.cleaned_data['previous_biopsy'],
                    'breastfeeding': form.cleaned_data['breastfeeding'],
                    'brca_mutation_status': form.cleaned_data['brca_mutation_status'],
                    'breast_pain': form.cleaned_data['breast_pain'],
                    'subtlety': form.cleaned_data['subtlety'],
                    'age': form.cleaned_data['age'],
                    'bmi': form.cleaned_data['bmi']
                }
                
                struct_vector = process_structured_input(
                    structured_data,
                    ohe,  
                    scaler_sub
                )

                # 2. Process images
                mamm_img = None
                ultra_img = None
                
                if 'mammogram' in request.FILES:
                    mamm_img = img_transform(
                        Image.open(request.FILES['mammogram']).convert('RGB')
                    ).unsqueeze(0)
                
                if 'ultrasound' in request.FILES:
                    ultra_img = img_transform(
                        Image.open(request.FILES['ultrasound']).convert('RGB')
                    ).unsqueeze(0)

                # 3. Get features and predict
                features = getFeatures(
                    mamm_img,
                    ultra_img,
                    mamm_model,
                    ultra_model,
                    scaler_mamm,
                    scaler_ultra,
                    pca_mamm,
                    pca_ultra,
                    struct_vector.reshape(1, -1)
                )
                
                
                proba = classifier_model.predict_proba(features)[0][1]  # Probability of malignancy

                # Determine diagnosis and corresponding probability display
                if proba > 0.5:
                    diagnosis = 'Malignant'
                    probability_display = float(proba * 100)
                    probability_type = 'of malignancy'
                else:
                    diagnosis = 'Benign'
                    probability_display = float((1 - proba) * 100)  # Convert to benign probability
                    probability_type = 'of being benign'

                # Prepare results
                # Prepare results
                results = {
                    'probability': probability_display,
                    'probability_type': probability_type,
                    'diagnosis': diagnosis,
                    'confidence': 'High' if proba > 0.7 or proba < 0.3 else 'Medium',
                    'input_data': {k: v for k, v in structured_data.items()}  # Clean dict: no need to filter images here
                }

                
                print(f"\n=== Prediction Results ===")
                print(f"Diagnosis: {diagnosis}")
                print(f"Probability: {probability_display:.2f}% {probability_type}")
                print(f"Confidence: {results['confidence']}")
                print(f"Raw Malignancy Probability: {proba:.4f}")

                return render(request, 'prediction/results.html', {'results': results})
            
            except Exception as e:
                print(f"Prediction Error: {str(e)}")
                return render(request, 'prediction/results.html', {
                    'results': {
                        'error': f"Analysis failed: {str(e)}",
                        'diagnosis': 'Error',
                        'probability': None
                    }
                })
        
        # Form is invalid
        return render(request, 'prediction/form.html', {'form': form})

    form = MedicalDiagnosisForm()
    return render(request, 'prediction/form.html', {'form': form})
    

