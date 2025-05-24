# core/predict.py
from PIL import Image
from .services import (
    img_transform,
    mamm_model,
    ultra_model,
    scaler_mamm,
    scaler_ultra, 
    pca_mamm,
    pca_ultra,
    classifier_model,
    process_structured_input,
    getFeatures
)

def make_prediction(form_data, mammogram_file=None, ultrasound_file=None):
    try:
        # 1. Process structured data
        struct_vector = process_structured_input(form_data)
        
        # 2. Process images if provided
        mamm_tensor = None
        ultra_tensor = None
        
        if mammogram_file:
            mamm_img = Image.open(mammogram_file).convert('RGB')
            mamm_tensor = img_transform(mamm_img).unsqueeze(0)  # Add batch dimension
            
        if ultrasound_file:
            ultra_img = Image.open(ultrasound_file).convert('RGB')
            ultra_tensor = img_transform(ultra_img).unsqueeze(0)
        
        # 3. Get combined features
        features = getFeatures(
            mamm_tensor, 
            ultra_tensor,
            mamm_model,
            ultra_model,
            scaler_mamm,
            scaler_ultra,
            pca_mamm,
            pca_ultra,
            struct_vector.reshape(1, -1)  # Ensure 2D array
        )
        
        proba = classifier_model.predict_proba(features)[0]
        malignant_prob = proba[1]  # Assuming class 1 is malignant
        benign_prob = proba[0]     # Class 0 is benign
        
        if malignant_prob > 0.5:
            return {
                'diagnosis': 'Malignant',
                'probability': float(malignant_prob * 100)
            }
        else:
            return {
                'diagnosis': 'Benign', 
                'probability': float(benign_prob * 100)
            }
        
    except Exception as e:
        return {
            'error': str(e),
            'diagnosis': 'Error',
            'probability': None
        }