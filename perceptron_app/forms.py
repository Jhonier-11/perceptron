"""
Formularios para la aplicación del perceptrón simple
"""

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator


class DataUploadForm(forms.Form):
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
    
    def limpiar_archivo_datos(self):
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


class TrainingConfigForm(forms.Form):
    """
    Formulario para configurar los parámetros de entrenamiento
    """
    training_name = forms.CharField(
        label='Nombre del Entrenamiento',
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ej: Entrenamiento AND Gate'
        })
    )
    
    learning_rate = forms.FloatField(
        label='Tasa de Aprendizaje (η)',
        initial=0.1,
        validators=[MinValueValidator(0.001), MaxValueValidator(1.0)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.001',
            'min': '0.001',
            'max': '1.0'
        }),
        help_text='Valor entre 0.001 y 1.0. Valores más altos = aprendizaje más rápido pero menos estable.'
    )
    
    epochs = forms.IntegerField(
        label='Número de Iteraciones',
        initial=100,
        validators=[MinValueValidator(1), MaxValueValidator(10000)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '10000'
        }),
        help_text='Número máximo de iteraciones de entrenamiento.'
    )
    
    max_error = forms.FloatField(
        label='Error Máximo Permitido',
        initial=0.1,
        validators=[MaxValueValidator(0.1)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.001',
            'min': '0',
            'max': '0.1'
        }),
        help_text='El entrenamiento se detendrá cuando el error sea menor o igual a este valor (máximo 0.1).'
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
                help_text='Selecciona la columna que será la variable objetivo.'
            )
    
    def limpiar(self):
        cleaned_data = super().limpiar()
        input_columns = cleaned_data.get('input_columns', [])
        output_columns = cleaned_data.get('output_columns', [])
        
        # Validaciones
        if not input_columns:
            raise forms.ValidationError('Debes seleccionar al menos una columna de entrada.')
        
        if not output_columns:
            raise forms.ValidationError('Debes seleccionar al menos una columna de salida.')
        
        if len(output_columns) > 1:
            raise forms.ValidationError('El perceptrón simple solo puede manejar una variable de salida.')
        
        # Verificar que no haya solapamiento entre entradas y salidas
        if set(input_columns) & set(output_columns):
            raise forms.ValidationError('Las columnas de entrada y salida no pueden ser las mismas.')
        
        return cleaned_data


class PredictionForm(forms.Form):
    """
    Formulario para hacer predicciones
    """
    def __init__(self, *args, **kwargs):
        input_columns = kwargs.pop('input_columns', [])
        super().__init__(*args, **kwargs)
        
        # Crear campos dinámicos para cada columna de entrada
        for col in input_columns:
            self.fields[col] = forms.FloatField(
                label=f'Valor de {col}',
                widget=forms.NumberInput(attrs={
                    'class': 'form-control',
                    'step': 'any',
                    'placeholder': f'Ingresa el valor para {col}'
                }),
                help_text=f'Valor numérico para la característica {col}'
            )
    
    def limpiar(self):
        cleaned_data = super().limpiar()
        
        # Validar que todos los campos estén completos
        for field_name, value in cleaned_data.items():
            if value is None:
                raise forms.ValidationError(f'El campo {field_name} es requerido.')
        
        return cleaned_data


class AdvancedConfigForm(forms.Form):
    """
    Formulario para configuración avanzada del perceptrón
    """
    weight_initialization = forms.ChoiceField(
        label='Inicialización de Pesos',
        choices=[
            ('random', 'Aleatoria (-0.5 a 0.5)'),
            ('zeros', 'Ceros'),
            ('ones', 'Unos'),
            ('xavier', 'Xavier/Glorot'),
        ],
        initial='random',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    activation_function = forms.ChoiceField(
        label='Función de Activación',
        choices=[
            ('step', 'Escalón (Step)'),
            ('sigmoid', 'Sigmoide'),
            ('tanh', 'Tangente Hiperbólica'),
        ],
        initial='step',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    early_stopping = forms.BooleanField(
        label='Parada Temprana',
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Detener el entrenamiento cuando no hay errores (convergencia)'
    )
    
    validation_split = forms.FloatField(
        label='División de Validación',
        initial=0.2,
        validators=[MinValueValidator(0.0), MaxValueValidator(0.5)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.0',
            'max': '0.5'
        }),
        help_text='Porcentaje de datos para validación (0.0 a 0.5)',
        required=False
    )
