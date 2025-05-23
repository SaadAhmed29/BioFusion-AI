# prediction/views.py
from django.shortcuts import render
from .forms import MedicalDiagnosisForm
from core.predict import make_prediction
import os
from django.conf import settings

def handle_upload(file):
    """Save uploaded file temporarily"""
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    path = os.path.join(settings.MEDIA_ROOT, file.name)
    with open(path, 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    return path

# prediction/views.py
def diagnosis_view(request):
    print(f"\n{'='*50}\nNew Request: {request.method}\n{'='*50}")
    
    if request.method == 'POST':
        print("\nPOST DATA:", request.POST)
        print("FILES:", request.FILES.keys())
        
        form = MedicalDiagnosisForm(request.POST, request.FILES)
        print("\nFORM VALID:", form.is_valid())
        print("FORM ERRORS:", form.errors)
        
        if form.is_valid():
            print("\nPROCESSING FORM...")
            try:
                # Add temporary test results
                test_results = {
                    'probability': 82.5,
                    'diagnosis': 'Malignant',
                    'confidence': 'High',
                    'debug': True
                }
                return render(request, 'prediction/results.html', {'results': test_results})
            except Exception as e:
                print("ERROR:", str(e))
                return render(request, 'prediction/form.html', {'form': form, 'error': str(e)})
        
        return render(request, 'prediction/form.html', {'form': form})
    
    # GET request
    form = MedicalDiagnosisForm()
    return render(request, 'prediction/form.html', {'form': form})