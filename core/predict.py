# core/predict.py
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
    """
    Main prediction pipeline
    Args:
        form_data: Dictionary from Django form
        mammogram_file: UploadedFile object
        ultrasound_file: UploadedFile object
    Returns:
        Dictionary of results
    """
    # 1. Process structured data
    struct_vector = process_structured_input(form_data)
    
    # 2. Process images if provided
    mamm_tensor = img_transform(Image.open(mammogram_file).convert('RGB')) if mammogram_file else None
    ultra_tensor = img_transform(Image.open(ultrasound_file).convert('RGB')) if ultrasound_file else None
    
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
        struct_vector.reshape(1, -1)
    )
    
    # 4. Make prediction
    proba = classifier_model.predict_proba(features)[0][1]
    
    return {
        'probability': float(proba * 100),
        'diagnosis': 'Malignant' if proba > 0.5 else 'Benign',
        'confidence': 'High' if proba > 0.7 or proba < 0.3 else 'Medium'
    }