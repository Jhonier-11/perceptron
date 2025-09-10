# Explicación de la Aplicación de Perceptrón Simple
## Una Aplicación Web para Entrenamiento y Predicción con Redes Neuronales

---

## Introducción

Buenos días, estimados compañeros. Hoy les voy a presentar una aplicación web desarrollada en Django que implementa un **Perceptrón Simple**, una de las redes neuronales más fundamentales en el campo de la inteligencia artificial. Esta aplicación permite cargar datos, entrenar el modelo y realizar predicciones de manera interactiva a través de una interfaz web moderna.

---

## Arquitectura General de la Aplicación

### Estructura del Proyecto

Nuestra aplicación está organizada siguiendo el patrón **MVC (Modelo-Vista-Controlador)** de Django:

- **Modelos** (`perceptron_app/models.py`): Definen la estructura de la base de datos
- **Vistas** (`perceptron_app/views.py`): Manejan la lógica de negocio y las peticiones HTTP
- **Templates** (`templates/perceptron_app/`): Contienen la interfaz de usuario
- **Algoritmo** (`perceptron_app/perceptron.py`): Implementa la lógica del perceptrón
- **Formularios** (`perceptron_app/forms.py`): Gestionan la validación de datos de entrada

---

## El Corazón del Sistema: El Algoritmo del Perceptrón

### Ubicación: `perceptron_app/perceptron.py`

El archivo `perceptron.py` contiene la clase `PerceptronSimple`, que es el núcleo de nuestra aplicación. Esta clase implementa el algoritmo de aprendizaje del perceptrón de manera completamente en español.

#### Componentes Principales:

**1. Inicialización del Perceptrón**
```python
def __init__(self, tasa_aprendizaje: float = 0.1, max_iteraciones: int = 100, error_maximo: float = 0.1):
```
- **Ubicación**: Líneas 15-30 en `perceptron.py`
- **Función**: Configura los parámetros iniciales del perceptrón
- **Parámetros**:
  - `tasa_aprendizaje`: Controla qué tan rápido aprende el modelo (eta)
  - `max_iteraciones`: Número máximo de épocas de entrenamiento
  - `error_maximo`: Umbral de error para detener el entrenamiento

**2. Función de Activación**
```python
def _funcion_escalon(self, x: float) -> int:
```
- **Ubicación**: Líneas 32-42 en `perceptron.py`
- **Función**: Implementa la función escalón (step function)
- **Comportamiento**: Retorna 1 si x ≥ 0, 0 en caso contrario

**3. Inicialización de Pesos**
```python
def _inicializar_pesos(self, num_caracteristicas: int) -> None:
```
- **Ubicación**: Líneas 44-53 en `perceptron.py`
- **Función**: Inicializa los pesos y el sesgo aleatoriamente
- **Rango**: Valores entre -1 y 1 (modificado para mejor convergencia)

---

## El Proceso de Entrenamiento: Paso a Paso

### Ubicación: `perceptron_app/perceptron.py` - Método `entrenar()`

El entrenamiento del perceptrón es un proceso iterativo que se ejecuta en las **líneas 91-189** del archivo `perceptron.py`. Permítanme explicarles cómo funciona:

#### Fase 1: Preparación de Datos
```python
# Líneas 102-114
X = np.array(X, dtype=float)  # Características de entrada
y = np.array(y, dtype=float)  # Etiquetas objetivo
num_muestras, num_caracteristicas = X.shape
```

#### Fase 2: Bucle Principal de Entrenamiento
```python
# Líneas 125-170
for iteracion in range(self.max_iteraciones):
    errores_iteracion = 0
    
    for i in range(num_muestras):
        prediccion = self._predecir_individual(X[i])
        error = y[i] - prediccion
        
        if error != 0:
            # Regla del perceptrón
            self.pesos += self.tasa_aprendizaje * error * X[i]
            self.sesgo += self.tasa_aprendizaje * error
            errores_iteracion += 1
```

#### Fase 3: Criterios de Parada
El entrenamiento se detiene cuando:
- Se alcanza el número máximo de iteraciones
- La tasa de error es menor o igual al error máximo permitido
- Se logra convergencia total (0 errores)

---

## Integración con la Interfaz Web

### Flujo de Datos en la Aplicación

#### 1. Carga de Datos
**Ubicación**: `perceptron_app/views.py` - Función `cargar_datos()`
- **Líneas**: 106-202
- **Función**: Procesa archivos CSV, Excel, JSON o TXT
- **Características**:
  - Detección automática de separadores
  - Validación de tipos de datos
  - Análisis de columnas numéricas vs no numéricas

#### 2. Configuración del Entrenamiento
**Ubicación**: `perceptron_app/views.py` - Función `configurar_entrenamiento()`
- **Líneas**: 205-242
- **Función**: Permite al usuario seleccionar columnas de entrada y salida
- **Validación**: Solo permite columnas numéricas

#### 3. Ejecución del Entrenamiento
**Ubicación**: `perceptron_app/views.py` - Función `entrenar_perceptron()`
- **Líneas**: 245-387
- **Proceso**:
  1. Crea instancia del perceptrón con parámetros del usuario
  2. Llama al método `entrenar()` del perceptrón
  3. Guarda resultados en la base de datos
  4. Genera gráficos de evolución

#### 4. Visualización de Resultados
**Ubicación**: `perceptron_app/views.py` - Función `resultados_entrenamiento()`
- **Líneas**: 390-410
- **Función**: Muestra métricas, gráficos y permite hacer predicciones

---

## Base de Datos: Almacenamiento de Resultados

### Ubicación: `perceptron_app/models.py`

#### Modelo `PerceptronTraining` (Líneas 7-35)
```python
class PerceptronTraining(models.Model):
    nombre = models.CharField(max_length=100)
    tasa_aprendizaje = models.FloatField()
    iteraciones = models.IntegerField()
    error_maximo = models.FloatField()
    columnas_entrada = models.JSONField()
    columnas_salida = models.JSONField()
    pesos_finales = models.JSONField()
    sesgo_final = models.FloatField()
    precision = models.FloatField()
    errores_entrenamiento = models.JSONField()
    evolucion_pesos = models.JSONField()
```

#### Modelo `Prediction` (Líneas 37-50)
```python
class Prediction(models.Model):
    entrenamiento = models.ForeignKey(PerceptronTraining)
    valores_entrada = models.JSONField()
    salida_predicha = models.FloatField()
    fecha_prediccion = models.DateTimeField()
```

---

## Interfaz de Usuario: Templates

### Estructura de Templates

#### 1. Página Principal
**Ubicación**: `templates/perceptron_app/home.html`
- Dashboard con estadísticas
- Enlaces a funcionalidades principales
- Lista de entrenamientos recientes

#### 2. Carga de Datos
**Ubicación**: `templates/perceptron_app/upload_data.html`
- Interfaz drag-and-drop para archivos
- Validación en tiempo real
- Vista previa de datos cargados

#### 3. Configuración de Entrenamiento
**Ubicación**: `templates/perceptron_app/configure_training.html`
- Selección de columnas de entrada y salida
- Configuración de parámetros
- Validación de datos numéricos

#### 4. Resultados del Entrenamiento
**Ubicación**: `templates/perceptron_app/training_results.html`
- Gráficos de evolución de errores y pesos
- Métricas de rendimiento
- Opciones para hacer predicciones

---

## Características Técnicas Avanzadas

### 1. Detección Automática de Separadores
**Ubicación**: `perceptron_app/views.py` - Función `detectar_separador_csv_y_leer()`
- **Líneas**: 23-87
- **Función**: Detecta automáticamente separadores en archivos CSV
- **Separadores soportados**: `,`, `;`, `|`, `\t`, ` `, `:`, `~`

### 2. Generación de Gráficos
**Ubicación**: `perceptron_app/perceptron.py`
- **Gráfico de Errores**: `crear_grafico_errores()` (líneas 216-241)
- **Gráfico de Pesos**: `crear_grafico_pesos()` (líneas 243-281)
- **Tecnología**: Matplotlib con salida en Base64

### 3. Validación de Datos
**Ubicación**: `perceptron_app/forms.py`
- **Formulario de Carga**: `DataUploadForm` (líneas 7-35)
- **Formulario de Configuración**: `TrainingConfigForm` (líneas 37-85)
- **Formulario de Predicción**: `PredictionForm` (líneas 87-130)

---

## Flujo Completo de la Aplicación

### Paso 1: Carga de Datos
1. Usuario accede a `/cargar-datos/`
2. Selecciona archivo con datos numéricos
3. Sistema detecta separador automáticamente
4. Valida que todas las columnas sean numéricas

### Paso 2: Configuración
1. Usuario accede a `/configurar-entrenamiento/`
2. Selecciona columnas de entrada y salida
3. Configura parámetros de entrenamiento
4. Sistema valida la configuración

### Paso 3: Entrenamiento
1. Usuario accede a `/entrenar-perceptron/`
2. Sistema crea instancia del perceptrón
3. Ejecuta algoritmo de entrenamiento
4. Guarda resultados en base de datos
5. Genera gráficos de evolución

### Paso 4: Resultados y Predicciones
1. Usuario accede a `/resultados-entrenamiento/`
2. Visualiza métricas y gráficos
3. Puede hacer nuevas predicciones
4. Accede al historial de entrenamientos

---

## Ventajas de la Implementación

### 1. **Completamente en Español**
- Todas las variables, funciones y mensajes están en español
- Facilita el aprendizaje y mantenimiento
- Ideal para entornos educativos

### 2. **Interfaz Intuitiva**
- Diseño moderno con Tailwind CSS
- Validación en tiempo real
- Feedback visual inmediato

### 3. **Robustez**
- Manejo de errores comprehensivo
- Validación de tipos de datos
- Detección automática de formatos

### 4. **Escalabilidad**
- Arquitectura modular
- Fácil extensión para otros algoritmos
- Base de datos normalizada

---

## Conclusiones

Esta aplicación demuestra cómo se puede implementar un algoritmo de machine learning complejo en una interfaz web accesible y educativa. La combinación de:

- **Algoritmo robusto** (perceptrón simple)
- **Interfaz moderna** (Django + Tailwind CSS)
- **Validación completa** (formularios y lógica de negocio)
- **Visualización clara** (gráficos de evolución)

Hace que el aprendizaje de redes neuronales sea más accesible y comprensible para estudiantes y profesionales.

La aplicación está lista para ser utilizada en entornos educativos, demostraciones técnicas o como base para proyectos más complejos de inteligencia artificial.

---

*Desarrollado con Django, Python, NumPy, Matplotlib y Tailwind CSS*
