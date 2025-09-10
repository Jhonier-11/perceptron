# Documentación Técnica - Perceptrón Simple

## 🏗️ Arquitectura del Sistema

### Estructura del Proyecto
```
perceptron/
├── percep/                    # Configuración principal de Django
│   ├── settings.py           # Configuración de la aplicación
│   ├── urls.py              # URLs principales
│   └── ...
├── perceptron_app/           # Aplicación principal
│   ├── models.py            # Modelos de base de datos
│   ├── views.py             # Vistas (controladores)
│   ├── forms.py             # Formularios de Django
│   ├── perceptron.py        # Implementación del algoritmo
│   ├── urls.py              # URLs de la aplicación
│   └── admin.py             # Configuración del admin
├── templates/                # Plantillas HTML
├── static/                   # Archivos estáticos (CSS, JS, imágenes)
└── media/                    # Archivos subidos por usuarios
```

## 🧠 Implementación del Algoritmo

### Clase PerceptronSimple

La implementación del perceptrón se encuentra en `perceptron_app/perceptron.py`:

```python
class PerceptronSimple:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.training_errors = []
        self.weight_evolution = []
```

### Algoritmo de Entrenamiento

1. **Inicialización**: Los pesos se inicializan aleatoriamente entre -0.5 y 0.5
2. **Predicción**: Para cada muestra, calcula `y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)`
3. **Error**: Calcula la diferencia entre predicción y valor real
4. **Actualización**: Si hay error, actualiza pesos: `w = w + η × error × x`
5. **Repetición**: Repite hasta converger o alcanzar el máximo de épocas

### Función de Activación

```python
def _step_function(self, x):
    return 1 if x >= 0 else 0
```

## 🗄️ Modelos de Base de Datos

### PerceptronTraining
Almacena los resultados de cada entrenamiento:
- `name`: Nombre del entrenamiento
- `learning_rate`: Tasa de aprendizaje utilizada
- `epochs`: Número de épocas configuradas
- `input_columns`: Columnas de entrada seleccionadas
- `output_columns`: Columnas de salida seleccionadas
- `final_weights`: Pesos finales del perceptrón
- `final_bias`: Sesgo final
- `accuracy`: Precisión alcanzada
- `training_errors`: Lista de errores por época
- `weight_evolution`: Evolución de pesos durante el entrenamiento

### Prediction
Almacena las predicciones individuales:
- `training`: Referencia al entrenamiento utilizado
- `input_values`: Valores de entrada utilizados
- `predicted_output`: Salida predicha por el perceptrón
- `created_at`: Fecha y hora de la predicción

## 🎨 Interfaz de Usuario

### Tecnologías Frontend
- **Bootstrap 5**: Framework CSS para diseño responsivo
- **Font Awesome**: Iconos
- **Chart.js**: Gráficos interactivos (preparado para futuras mejoras)
- **JavaScript Vanilla**: Funcionalidades interactivas

### Plantillas HTML
- `base.html`: Plantilla base con navbar y footer
- `home.html`: Página principal con dashboard
- `upload_data.html`: Carga de archivos de datos
- `configure_training.html`: Configuración de parámetros
- `train_perceptron.html`: Interfaz de entrenamiento
- `training_results.html`: Visualización de resultados
- `make_prediction.html`: Interfaz de predicciones
- `training_history.html`: Historial de entrenamientos

## 🔧 Formularios de Django

### DataUploadForm
- Validación de tipos de archivo (CSV, XLSX, JSON, TXT)
- Validación de tamaño máximo (10MB)
- Vista previa de datos cargados

### TrainingConfigForm
- Campos dinámicos para selección de columnas
- Validación de parámetros de entrenamiento
- Validación de solapamiento entre entradas y salidas

### PredictionForm
- Campos dinámicos según las columnas de entrada
- Validación de valores numéricos
- Interfaz intuitiva para entrada de datos

## 📊 Visualizaciones

### Gráfico de Evolución del Error
- Muestra el número de errores por época
- Permite identificar la convergencia del algoritmo
- Generado con Matplotlib y convertido a base64

### Gráfico de Evolución de Pesos
- Muestra cómo cambian los pesos durante el entrenamiento
- Incluye el sesgo (bias)
- Ayuda a entender el proceso de aprendizaje

## 🔒 Seguridad

### Validaciones
- Validación de archivos subidos
- Sanitización de datos de entrada
- Validación de tipos de datos
- Límites en parámetros de entrenamiento

### CSRF Protection
- Tokens CSRF en todos los formularios
- Protección contra ataques CSRF

## 🚀 Rendimiento

### Optimizaciones
- Uso de NumPy para operaciones vectorizadas
- Almacenamiento eficiente de datos de entrenamiento
- Caché de resultados de entrenamiento en sesión

### Limitaciones
- El perceptrón simple solo maneja problemas linealmente separables
- No es adecuado para problemas complejos como XOR
- Una sola variable de salida

## 🧪 Testing

### Script de Pruebas
El archivo `test_perceptron.py` incluye pruebas para:
- Compuerta lógica AND (debe converger)
- Compuerta lógica OR (debe converger)
- Compuerta lógica XOR (debe fallar - no es linealmente separable)
- Función de predicción individual

### Cobertura de Pruebas
- ✅ Algoritmo de entrenamiento
- ✅ Función de predicción
- ✅ Convergencia en problemas lineales
- ✅ Fallo en problemas no lineales

## 📈 Escalabilidad

### Posibles Mejoras
1. **Perceptrón Multicapa**: Implementar MLP para problemas no lineales
2. **Múltiples Salidas**: Extender para clasificación multiclase
3. **Validación Cruzada**: Implementar k-fold cross-validation
4. **Métricas Adicionales**: Precisión, Recall, F1-Score
5. **Exportación de Modelos**: Guardar modelos en formato pickle
6. **API REST**: Crear endpoints para integración externa

### Limitaciones Actuales
- Solo clasificación binaria
- Problemas linealmente separables únicamente
- Sin validación de datos de prueba
- Interfaz síncrona (no hay entrenamiento asíncrono)

## 🔧 Configuración de Desarrollo

### Variables de Entorno
```python
DEBUG = True
SECRET_KEY = 'django-insecure-...'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### Archivos Estáticos
- CSS personalizado en `static/css/style.css`
- JavaScript en `static/js/main.js`
- Ejemplos de datos en `static/examples/`

## 📚 Referencias Técnicas

### Algoritmo del Perceptrón
- Rosenblatt, F. (1958). "The perceptron: a probabilistic model for information storage and organization in the brain"
- Minsky, M. & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"

### Implementación
- NumPy Documentation: https://numpy.org/doc/
- Django Documentation: https://docs.djangoproject.com/
- Matplotlib Documentation: https://matplotlib.org/stable/

## 🐛 Solución de Problemas

### Errores Comunes
1. **Error de importación de NumPy**: `pip install numpy`
2. **Error de migraciones**: `python manage.py makemigrations && python manage.py migrate`
3. **Error de archivos estáticos**: Verificar configuración de STATIC_URL
4. **Error de permisos**: Verificar permisos de escritura en directorio media/

### Logs de Debug
- Los logs de entrenamiento se muestran en la consola
- Errores de Django se registran en la consola del servidor
- Validaciones de formularios se muestran en la interfaz

## 📋 Checklist de Despliegue

- [ ] Configurar DEBUG = False
- [ ] Configurar ALLOWED_HOSTS
- [ ] Configurar base de datos de producción
- [ ] Configurar archivos estáticos
- [ ] Configurar HTTPS
- [ ] Configurar logging
- [ ] Configurar backup de base de datos
- [ ] Configurar monitoreo de errores
