"""
Formularios para la aplicación de Red Neuronal RBF
"""
from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator


class RBFDataUploadForm(forms.Form):
    """
    Formulario para cargar archivos de datos
    """
    data_file = forms.FileField(
        label='Archivo de Datos',
        help_text='Selecciona un archivo CSV, XLSX, JSON o TXT',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.xlsx,.json,.txt'
        })
    )
    
    def clean_data_file(self):
        file = self.cleaned_data.get('data_file')
        if file:
            # Verificar extensión
            allowed_extensions = ['.csv', '.xlsx', '.json', '.txt']
            file_extension = file.name.split('.')[-1].lower()
            if f'.{file_extension}' not in allowed_extensions:
                raise forms.ValidationError(
                    'Formato de archivo no soportado. Use CSV, XLSX, JSON o TXT.'
                )
            
            # Verificar tamaño (máximo 10MB)
            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError(
                    'El archivo es demasiado grande. Máximo 10MB.'
                )
        
        return file


class RBFConfigForm(forms.Form):
    """
    Formulario para configurar los parámetros de entrenamiento RBF
    """
    nombre_entrenamiento = forms.CharField(
        label='Nombre del Entrenamiento',
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ej: Entrenamiento RBF Dataset 1'
        })
    )

    num_centros = forms.IntegerField(
        label='Número de Centros Radiales',
        initial=3,
        validators=[MinValueValidator(2), MaxValueValidator(50)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '2',
            'max': '50'
        }),
        help_text='Número de neuronas ocultas (centros radiales). Valores típicos: 2-10.'
    )

    porcentaje_entrenamiento = forms.FloatField(
        label='Porcentaje de Entrenamiento (%)',
        initial=70.0,
        validators=[MinValueValidator(50.0), MaxValueValidator(90.0)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '5',
            'min': '50',
            'max': '90'
        }),
        help_text='Porcentaje de datos para entrenamiento (resto para prueba).'
    )

    error_aproximacion = forms.FloatField(
        label='Error de Aproximación Óptimo',
        initial=0.1,
        validators=[MaxValueValidator(1.0)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01',
            'min': '0.01',
            'max': '1.0'
        }),
        help_text='Error objetivo para verificar convergencia.'
    )
    
    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)

        # Crear campos dinámicos para seleccionar columnas
        if columns:
            # Campo para columnas de entrada
            self.fields['input_columns'] = forms.MultipleChoiceField(
                label='Columnas de Entrada (X)',
                choices=[(col, col) for col in columns],
                widget=forms.CheckboxSelectMultiple(attrs={
                    'class': 'form-check-input'
                }),
                help_text='Selecciona las columnas que serán las características de entrada.'
            )

            # Campo para columnas de salida
            self.fields['output_columns'] = forms.MultipleChoiceField(
                label='Columnas de Salida (Y)',
                choices=[(col, col) for col in columns],
                widget=forms.CheckboxSelectMultiple(attrs={
                    'class': 'form-check-input'
                }),
                help_text='Selecciona la columna que será la variable objetivo (solo una).'
            )
    
    def clean(self):
        cleaned_data = super().clean()
        input_columns = cleaned_data.get('input_columns', [])
        output_columns = cleaned_data.get('output_columns', [])

        # Validaciones
        if not input_columns:
            raise forms.ValidationError('Debes seleccionar al menos una columna de entrada.')

        if not output_columns:
            raise forms.ValidationError('Debes seleccionar al menos una columna de salida.')

        if len(output_columns) > 1:
            raise forms.ValidationError('RBF solo puede manejar una variable de salida a la vez.')

        # Verificar que no haya solapamiento entre entradas y salidas
        if set(input_columns) & set(output_columns):
            raise forms.ValidationError('Las columnas de entrada y salida no pueden ser las mismas.')

        return cleaned_data


class RBFPredictionForm(forms.Form):
    """
    Formulario dinámico para hacer predicciones con la red RBF entrenada
    """
    def __init__(self, input_columns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Crear campos dinámicos para cada columna de entrada
        for col in input_columns:
            self.fields[col] = forms.FloatField(
                label=f'{col}',
                required=True,
                widget=forms.NumberInput(attrs={
                    'class': 'form-control',
                    'step': 'any',
                    'placeholder': f'Ingresa el valor para {col}'
                }),
                help_text=f'Valor numérico para la característica {col}'
            )
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Validar que todos los campos estén completos
        for field_name, value in cleaned_data.items():
            if value is None:
                raise forms.ValidationError(f'El campo {field_name} es requerido.')
        
        return cleaned_data

