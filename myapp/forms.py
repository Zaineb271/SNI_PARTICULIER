# forms.py
from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()


class ColumnSelectionForm(forms.Form):
    column = forms.ChoiceField(choices=[])
    
    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        selected_column = kwargs.pop('selected_column', None)  # Colonne sélectionnée par défaut
        super().__init__(*args, **kwargs)
        
        self.fields['column'].choices = [('', '--choisir une colonne--')] + [(col, col) for col in columns]
        
        if selected_column:
            self.fields['column'].initial = selected_column
        
    
    
   


class BivariateColumnSelectionForm(forms.Form):
    col1 = forms.ChoiceField(choices=[])
    col2 = forms.ChoiceField(choices=[])
    
    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)
        self.fields['col1'].choices = [(col, col) for col in columns]
        self.fields['col2'].choices = [(col, col) for col in columns]


class TargetSelectionForm(forms.Form):
    column = forms.ChoiceField(choices=[('target1', 'Target 1'), ('target2', 'Target 2')],
                               widget=forms.Select(attrs={'class': 'form-control'}))

class FeatureSelectionForm(forms.Form):
    columns = forms.MultipleChoiceField(choices=[('feature1', 'Feature 1'), ('feature2', 'Feature 2')],
                                        widget=forms.SelectMultiple(attrs={'class': 'form-control'}))

class TargetSelectionForm(ColumnSelectionForm):
    target = forms.ChoiceField(label='Select Target Variable', choices=[])
    
class MultipleColumnSelectionForm(forms.Form):
    columns = forms.MultipleChoiceField(
        choices=[],
        required=True,
        widget=forms.CheckboxSelectMultiple,
        label="Feature Variables"
    )

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super(MultipleColumnSelectionForm, self).__init__(*args, **kwargs)
        self.fields['columns'].choices = [(col, col) for col in columns]


class DataTreatmentForm(forms.Form):
    threshold = forms.FloatField(label='Threshold', required=False, initial=0.5)
    method = forms.ChoiceField(label='Method', choices=[('iqr', 'IQR')], required=False)
    replace_with = forms.ChoiceField(label='Replace Outliers With', choices=[('none', 'None'), ('median', 'Median'), ('mean', 'Mean')], required=False)
    action = forms.ChoiceField(label='Action', choices=[('remove_missing', 'Remove Columns with Missing Data'), ('impute_missing', 'Impute Missing Values'), ('treat_outliers', 'Treat Outliers')])

class DataTreatmentForm(forms.Form):
    ACTION_CHOICES = [
        ('remove_missing', 'Remove Columns with Missing Data'),
        ('impute_missing', 'Impute Missing Values'),
        ('treat_outliers', 'Treat Outliers'),
        ('remove_column', 'Remove Column')
    ]
    action = forms.ChoiceField(choices=ACTION_CHOICES, required=True)
    threshold = forms.FloatField(required=False)
    method = forms.ChoiceField(choices=[('mean', 'Mean'), ('median', 'Median'), ('mode', 'Mode')], required=False)
    replace_with = forms.CharField(required=False)