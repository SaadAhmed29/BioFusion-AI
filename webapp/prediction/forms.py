from django import forms

class MedicalDiagnosisForm(forms.Form):
    # Image upload fields at the top
    mammogram = forms.ImageField(
        label="Mammogram Image",
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )
    ultrasound = forms.ImageField(
        label="Ultrasound Image",
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )

    # Numerical inputs
    age = forms.IntegerField(
        label="Age",
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    bmi = forms.FloatField(
        label="BMI",
        min_value=10.0,
        max_value=60.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': "0.1"
        })
    )

    # Dropdown fields
    breast_density = forms.ChoiceField(
        label="Breast Density",
        choices=[
            ('1', '1'),
            ('2', '2'),
            ('3', '3'),
            ('4', '4')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    mass_shape = forms.ChoiceField(
        label="Mass Shape",
        choices=[
            ('ARCHITECTURAL_DISTORTION', 'Architectural Distortion'),
            ('ASYMMETRIC_BREAST_TISSUE', 'Asymmetric Breast Tissue'),
            ('FOCAL_ASYMMETRIC_DENSITY', 'Focal Asymmetric Density'),
            ('IRREGULAR', 'Irregular'),
            ('IRREGULAR-ARCHITECTURAL_DISTORTION', 'Irregular-Architectural Distortion'),
            ('IRREGULAR-FOCAL_ASYMMETRIC_DENSITY', 'Irregular-Focal Asymmetric Density'),
            ('LOBULATED', 'Lobulated'),
            ('LOBULATED-ARCHITECTURAL_DISTORTION', 'Lobulated-Architectural Distortion'),
            ('LOBULATED-IRREGULAR', 'Lobulated-Irregular'),
            ('LOBULATED-OVAL', 'Lobulated-Oval'),
            ('N/A', 'N/A'),
            ('OVAL', 'Oval'),
            ('ROUND', 'Round'),
            ('ROUND-OVAL', 'Round-Oval')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    mass_margins = forms.ChoiceField(
        label="Mass Margins",
        choices=[
            ('CIRCUMSCRIBED', 'Circumscribed'),
            ('CIRCUMSCRIBED-ILL_DEFINED', 'Circumscribed-Ill Defined'),
            ('CIRCUMSCRIBED-MICROLOBULATED', 'Circumscribed-Microlobulated'),
            ('CIRCUMSCRIBED-OBSCURED', 'Circumscribed-Obscured'),
            ('ILL_DEFINED', 'Ill Defined'),
            ('ILL_DEFINED-SPICULATED', 'Ill Defined-Spiculated'),
            ('MICROLOBULATED', 'Microlobulated'),
            ('MICROLOBULATED-ILL_DEFINED', 'Microlobulated-Ill Defined'),
            ('MICROLOBULATED-ILL_DEFINED-SPICULATED', 'Microlobulated-Ill Defined-Spiculated'),
            ('MICROLOBULATED-SPICULATED', 'Microlobulated-Spiculated'),
            ('N/A', 'N/A'),
            ('OBSCURED', 'Obscured'),
            ('OBSCURED-ILL_DEFINED', 'Obscured-Ill Defined'),
            ('OBSCURED-ILL_DEFINED-SPICULATED', 'Obscured-Ill Defined-Spiculated'),
            ('OBSCURED-SPICULATED', 'Obscured-Spiculated'),
            ('SPICULATED', 'Spiculated')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    subtlety = forms.ChoiceField(
        label="Subtlety",
        choices=[(str(i), str(i)) for i in range(6)],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    # Radio button fields (Yes/No)
    family_history = forms.ChoiceField(
        label="Family History of Breast Cancer",
        choices=[('yes', 'Yes'), ('no', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='no'
    )

    hormone_therapy = forms.ChoiceField(
        label="Hormone Therapy",
        choices=[('yes', 'Yes'), ('no', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='no'
    )

    previous_biopsy = forms.ChoiceField(
        label="Previous Biopsy",
        choices=[('yes', 'Yes'), ('no', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='no'
    )

    breastfeeding = forms.ChoiceField(
        label="Breastfeeding",
        choices=[('yes', 'Yes'), ('no', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='no'
    )

    brca_mutation_status = forms.ChoiceField(
        label="BRCA Mutation Status",
        choices=[
            ('negative', 'Negative'),
            ('positive', 'Positive'),
            ('unknown', 'Unknown')
        ],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='unknown'
    )

    breast_pain = forms.ChoiceField(
        label="Breast Pain",
        choices=[('yes', 'Yes'), ('no', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='no'
    )