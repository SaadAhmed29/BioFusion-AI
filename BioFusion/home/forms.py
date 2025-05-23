from django import forms

class PredictionForm(forms.Form):
    mammogram = forms.ImageField(required=True)
    ultrasound = forms.ImageField(required=True)
    breast_density = forms.ChoiceField(choices=[(str(i), str(i)) for i in range(1, 5)])
    mass_shape = forms.ChoiceField(choices=[('ROUND', 'ROUND'), ('OVAL', 'OVAL'), ('IRREGULAR', 'IRREGULAR')])  # etc.
    mass_margins = forms.ChoiceField(choices=[('CIRCUMSCRIBED', 'CIRCUMSCRIBED'), ('ILL_DEFINED', 'ILL_DEFINED')])
    subtlety = forms.ChoiceField(choices=[(str(i), str(i)) for i in range(6)])
    family_history = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    hormone_therapy = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    previous_biopsy = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    breastfeeding = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    breast_pain = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    brca_mutation_status = forms.ChoiceField(choices=[('Positive', 'Positive'), ('Negative', 'Negative'), ('Unknown', 'Unknown')])
    age = forms.IntegerField(min_value=0)
    bmi = forms.FloatField(min_value=0)
