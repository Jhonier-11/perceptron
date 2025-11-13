"""
Formularios para la aplicación de Predicción del Rendimiento Académico
"""

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from .models import Estudiante


class CargaEstudiantesForm(forms.Form):
    """
    Formulario para cargar archivos de datos de estudiantes
    """
    archivo = forms.FileField(
        label='Archivo de Estudiantes',
        help_text='Selecciona un archivo CSV con los datos de estudiantes',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.xlsx,.json'
        })
    )
    
    def clean_archivo(self):
        archivo = self.cleaned_data.get('archivo')
        if archivo:
            # Verificar extensión
            extensiones_permitidas = ['.csv', '.xlsx', '.json']
            extension = archivo.name.split('.')[-1].lower()
            if f'.{extension}' not in extensiones_permitidas:
                raise forms.ValidationError(
                    'Formato de archivo no soportado. Use CSV, XLSX o JSON.'
                )
            
            # Verificar tamaño (máximo 50MB)
            if archivo.size > 50 * 1024 * 1024:
                raise forms.ValidationError(
                    'El archivo es demasiado grande. Máximo 50MB.'
                )
        
        return archivo


class EstudianteForm(forms.ModelForm):
    """
    Formulario para crear/editar un estudiante
    """
    class Meta:
        model = Estudiante
        fields = [
            'identificacion', 'nombre', 'apellido', 'edad', 'sexo', 'direccion',
            'tamano_familia', 'estado_padres', 'educacion_madre', 'educacion_padre',
            'trabajo_madre', 'trabajo_padre', 'tiempo_viaje', 'tiempo_estudio',
            'fallos_previos', 'apoyo_escuela', 'apoyo_familia', 'clases_pagadas',
            'actividades_extra', 'guarderia', 'quiere_superior', 'internet',
            'relacion_romantica', 'relacion_familiar', 'tiempo_libre', 'salidas',
            'alcohol_semana', 'alcohol_fin_semana', 'salud', 'ausencias',
            'calificacion_g1', 'calificacion_g2', 'calificacion_g3'
        ]
        widgets = {
            'identificacion': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'placeholder': 'Ej: EST0001'
            }),
            'nombre': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'apellido': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'edad': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '10', 'max': '30'
            }),
            'sexo': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'direccion': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'tamano_familia': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'estado_padres': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'educacion_madre': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'educacion_padre': forms.Select(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'trabajo_madre': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'trabajo_padre': forms.TextInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white'
            }),
            'tiempo_viaje': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '4'
            }),
            'tiempo_estudio': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '4'
            }),
            'fallos_previos': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '0'
            }),
            'apoyo_escuela': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'apoyo_familia': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'clases_pagadas': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'actividades_extra': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'guarderia': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'quiere_superior': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'internet': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'relacion_romantica': forms.CheckboxInput(attrs={
                'class': 'w-4 h-4 text-primary-600 bg-white border-accent-300 rounded focus:ring-primary-500 focus:ring-2'
            }),
            'relacion_familiar': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '5'
            }),
            'tiempo_libre': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '5'
            }),
            'salidas': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '5'
            }),
            'alcohol_semana': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '5'
            }),
            'alcohol_fin_semana': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '5'
            }),
            'salud': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '1', 'max': '5'
            }),
            'ausencias': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '0'
            }),
            'calificacion_g1': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '0', 'max': '20', 'step': '0.1'
            }),
            'calificacion_g2': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '0', 'max': '20', 'step': '0.1'
            }),
            'calificacion_g3': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-2 border border-accent-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-accent-800 bg-white',
                'min': '0', 'max': '20', 'step': '0.1'
            }),
        }


class ConfiguracionEntrenamientoMLPForm(forms.Form):
    """
    Formulario para configurar el entrenamiento del MLP
    """
    nombre = forms.CharField(
        label='Nombre del Entrenamiento',
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ej: MLP Predicción G3 - 32 neuronas'
        }),
        help_text='Nombre descriptivo para identificar este entrenamiento'
    )
    
    # Tipo de implementación
    tipo_implementacion = forms.ChoiceField(
        label='Tipo de Implementación',
        choices=[
            ('numpy', 'Implementación desde cero (NumPy)'),
            ('tensorflow', 'TensorFlow/Keras'),
        ],
        initial='numpy',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'onchange': 'mostrarInfoImplementacion(this.value)'
        }),
        help_text='Selecciona la implementación a usar para entrenar el modelo'
    )
    
    # Arquitectura
    num_capas_ocultas = forms.IntegerField(
        label='Número de Capas Ocultas',
        initial=1,
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '5'
        }),
        help_text='Número de capas ocultas (1-5)'
    )
    
    neuronas_capa_1 = forms.IntegerField(
        label='Neuronas Capa Oculta 1',
        initial=32,
        min_value=1,
        max_value=256,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '256'
        }),
        help_text='Número de neuronas en la primera capa oculta'
    )
    
    neuronas_capa_2 = forms.IntegerField(
        label='Neuronas Capa Oculta 2',
        required=False,
        min_value=1,
        max_value=256,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '256'
        }),
        help_text='Número de neuronas en la segunda capa oculta (opcional)'
    )
    
    neuronas_capa_3 = forms.IntegerField(
        label='Neuronas Capa Oculta 3',
        required=False,
        min_value=1,
        max_value=256,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '256'
        }),
        help_text='Número de neuronas en la tercera capa oculta (opcional)'
    )
    
    funcion_activacion = forms.ChoiceField(
        label='Función de Activación',
        choices=[
            ('relu', 'ReLU (Recomendado)'),
            ('sigmoid', 'Sigmoid'),
            ('tanh', 'Tanh'),
        ],
        initial='relu',
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Función de activación para las capas ocultas'
    )
    
    # Parámetros de entrenamiento
    tasa_aprendizaje = forms.FloatField(
        label='Tasa de Aprendizaje',
        initial=0.01,
        min_value=0.0001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.0001',
            'min': '0.0001',
            'max': '1.0'
        }),
        help_text='Tasa de aprendizaje (0.0001 - 1.0). Valores más bajos = aprendizaje más estable.'
    )
    
    iteraciones = forms.IntegerField(
        label='Número de Iteraciones',
        initial=1000,
        min_value=1,
        max_value=10000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '10000'
        }),
        help_text='Número máximo de iteraciones de entrenamiento'
    )
    
    tamanio_batch = forms.IntegerField(
        label='Tamaño de Batch',
        initial=32,
        min_value=1,
        max_value=256,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '256'
        }),
        help_text='Número de muestras por batch (1-256)'
    )
    
    porcentaje_entrenamiento = forms.FloatField(
        label='Porcentaje de Entrenamiento',
        initial=70.0,
        min_value=50.0,
        max_value=90.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1.0',
            'min': '50.0',
            'max': '90.0'
        }),
        help_text='Porcentaje de datos para entrenamiento (50-90%). El resto se usa para validación.'
    )
    
    # Selección de características
    columna_salida = forms.ChoiceField(
        label='Variable Objetivo',
        choices=[
            ('G1', 'G1 (Primer Período)'),
            ('G2', 'G2 (Segundo Período)'),
            ('G3', 'G3 (Tercer Período)'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Variable objetivo a predecir'
    )
    
    def __init__(self, *args, **kwargs):
        columnas_disponibles = kwargs.pop('columnas_disponibles', [])
        super().__init__(*args, **kwargs)
        
        # Campo para seleccionar columnas de entrada
        if columnas_disponibles:
            # Filtrar columnas G1, G2, G3 de las opciones de entrada
            columnas_entrada = [col for col in columnas_disponibles if col not in ['G1', 'G2', 'G3']]
            
            self.fields['columnas_entrada'] = forms.MultipleChoiceField(
                label='Características de Entrada',
                choices=[(col, col) for col in columnas_entrada],
                widget=forms.CheckboxSelectMultiple(attrs={
                    'class': 'form-check-input'
                }),
                help_text='Selecciona las características que se usarán para la predicción'
            )
    
    def clean(self):
        cleaned_data = super().clean()
        num_capas_ocultas = cleaned_data.get('num_capas_ocultas', 1)
        neuronas_capa_1 = cleaned_data.get('neuronas_capa_1')
        neuronas_capa_2 = cleaned_data.get('neuronas_capa_2')
        neuronas_capa_3 = cleaned_data.get('neuronas_capa_3')
        columnas_entrada = cleaned_data.get('columnas_entrada', [])
        
        # Validar número de capas vs neuronas especificadas
        if num_capas_ocultas >= 2 and not neuronas_capa_2:
            raise forms.ValidationError(
                'Debes especificar el número de neuronas para la capa oculta 2 si tienes 2 o más capas.'
            )
        
        if num_capas_ocultas >= 3 and not neuronas_capa_3:
            raise forms.ValidationError(
                'Debes especificar el número de neuronas para la capa oculta 3 si tienes 3 o más capas.'
            )
        
        # Validar que se hayan seleccionado columnas de entrada
        if not columnas_entrada:
            raise forms.ValidationError(
                'Debes seleccionar al menos una característica de entrada.'
            )
        
        # Validar que la columna de salida no esté en las columnas de entrada
        columna_salida = cleaned_data.get('columna_salida')
        if columna_salida in columnas_entrada:
            raise forms.ValidationError(
                f'La variable objetivo ({columna_salida}) no puede estar en las características de entrada.'
            )
        
        return cleaned_data
    
    def get_neuronas_por_capa(self):
        """Retorna la lista de neuronas por capa según la configuración"""
        cleaned_data = self.cleaned_data
        num_capas = cleaned_data.get('num_capas_ocultas', 1)
        neuronas = [cleaned_data.get('neuronas_capa_1')]
        
        if num_capas >= 2 and cleaned_data.get('neuronas_capa_2'):
            neuronas.append(cleaned_data.get('neuronas_capa_2'))
        
        if num_capas >= 3 and cleaned_data.get('neuronas_capa_3'):
            neuronas.append(cleaned_data.get('neuronas_capa_3'))
        
        return neuronas


class PrediccionForm(forms.Form):
    """
    Formulario para hacer predicciones individuales
    """
    estudiante = forms.ModelChoiceField(
        label='Estudiante',
        queryset=Estudiante.objects.all(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Selecciona el estudiante para hacer la predicción'
    )
    
    entrenamiento = forms.ModelChoiceField(
        label='Modelo de Entrenamiento',
        queryset=None,  # Se configurará en el __init__
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Selecciona el modelo entrenado a usar para la predicción'
    )
    
    def __init__(self, *args, **kwargs):
        columna_salida = kwargs.pop('columna_salida', None)
        super().__init__(*args, **kwargs)
        
        # Filtrar entrenamientos por columna de salida si se especifica
        from .models import EntrenamientoMLP
        if columna_salida:
            self.fields['entrenamiento'].queryset = EntrenamientoMLP.objects.filter(
                columna_salida=columna_salida
            )
        else:
            self.fields['entrenamiento'].queryset = EntrenamientoMLP.objects.all()

