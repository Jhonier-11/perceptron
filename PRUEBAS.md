# Guía de Pruebas: Predicción del Rendimiento Académico con MLP

## 1. Sistema de Alertas Implementado

### 1.1 Funcionalidades Implementadas

El sistema de alertas automáticas está implementado y genera alertas basadas en:

1. **Predicción Baja (< 10 puntos)**: Alerta cuando un estudiante tiene una predicción baja
2. **Alto Riesgo de Reprobación (< 8 puntos)**: Alerta crítica cuando hay riesgo alto de reprobación
3. **Ausencias Elevadas (> 10)**: Alerta cuando un estudiante tiene muchas ausencias
4. **Fallos Previos (> 2)**: Alerta cuando un estudiante tiene historial de fallos previos
5. **Discrepancia Grande**: Alerta cuando hay una diferencia significativa entre predicción y realidad (> 3 puntos)
6. **Bajo Tiempo de Estudio (< 2)**: Alerta cuando el tiempo de estudio es bajo
7. **Promedio Bajo**: Alerta cuando el promedio de calificaciones es bajo (< 10)

### 1.2 Cómo Usar el Sistema de Alertas

#### Generar Alertas Manualmente

```bash
# Activar el entorno virtual
source D:/blute/Anaconda/Scripts/activate ia

# Generar alertas para todos los estudiantes
python manage.py generar_alertas

# Generar alertas y limpiar alertas antiguas
python manage.py generar_alertas --limpiar --dias 90
```

#### Generar Alertas desde la Interfaz Web

1. Ir a la vista de Docentes: `/docentes/`
2. Las alertas se generan automáticamente cuando no hay alertas recientes
3. También se generan automáticamente después de hacer una predicción

#### Generar Alertas Programáticamente

```python
from prediccion_academica.alertas import generar_alertas_estudiante, generar_alertas_todos_estudiantes
from prediccion_academica.models import Estudiante, PrediccionRendimiento

# Generar alertas para un estudiante específico
estudiante = Estudiante.objects.get(id=1)
prediccion = PrediccionRendimiento.objects.filter(estudiante=estudiante).first()
alertas = generar_alertas_estudiante(estudiante, prediccion)

# Generar alertas para todos los estudiantes
total_alertas = generar_alertas_todos_estudiantes()
```

## 2. Pruebas con Datos Reales

### 2.1 Cargar Datos desde CSV

1. Ir a la vista de Estudiantes: `/estudiantes/`
2. Hacer clic en "Cargar Estudiantes"
3. Seleccionar el archivo `student-por.csv`
4. Hacer clic en "Cargar"

### 2.2 Entrenar el Modelo MLP

1. Ir a la vista de Entrenamiento: `/entrenar/`
2. Configurar los parámetros del modelo:
   - Nombre del entrenamiento
   - Número de capas ocultas
   - Neuronas por capa
   - Función de activación (ReLU, Sigmoid, Tanh)
   - Tasa de aprendizaje
   - Número de iteraciones
   - Tamaño de batch
   - Porcentaje para entrenamiento
   - Columnas de entrada (características)
   - Columna de salida (G1, G2, o G3)
3. Hacer clic en "Entrenar Modelo"
4. Esperar a que termine el entrenamiento
5. Ver los resultados del entrenamiento

### 2.3 Probar Predicciones con Datos Reales

#### Usando el Comando de Django

```bash
# Activar el entorno virtual
source D:/blute/Anaconda/Scripts/activate ia

# Probar predicciones para todos los estudiantes (límite: 10)
python manage.py probar_predicciones --limite 10

# Probar predicciones para un estudiante específico
python manage.py probar_predicciones --estudiante-id 1

# Probar predicciones usando un entrenamiento específico
python manage.py probar_predicciones --entrenamiento-id 1 --limite 20
```

#### Usando la Interfaz Web

1. Ir a la vista de Estudiantes: `/estudiantes/`
2. Seleccionar un estudiante
3. Hacer clic en "Hacer Predicción"
4. Seleccionar el modelo de entrenamiento
5. Hacer clic en "Predecir"
6. Ver la predicción y las alertas generadas

### 2.4 Validar Predicciones

El comando `probar_predicciones` muestra estadísticas detalladas:

- Total de predicciones
- Predicciones correctas (error ≤ 2.0)
- Precisión porcentual
- Error promedio
- Error mediano
- Error máximo
- Error mínimo

### 2.5 Ver Alertas Generadas

1. Ir a la vista de Docentes: `/docentes/`
2. Ver las alertas recientes en la sección de alertas
3. Hacer clic en una alerta para marcarla como vista
4. Filtrar alertas por prioridad, tipo, o estudiante

## 3. Flujo Completo de Prueba

### Paso 1: Cargar Datos

```bash
# 1. Ir a la interfaz web: http://localhost:8000/estudiantes/cargar/
# 2. Cargar el archivo student-por.csv
# 3. Verificar que los estudiantes se hayan cargado correctamente
```

### Paso 2: Entrenar el Modelo

```bash
# 1. Ir a la interfaz web: http://localhost:8000/entrenar/
# 2. Configurar el modelo:
#    - Nombre: "MLP Predicción G3 - 32 neuronas"
#    - Capas ocultas: 1
#    - Neuronas: 32
#    - Función: ReLU
#    - Tasa de aprendizaje: 0.01
#    - Iteraciones: 1000
#    - Batch: 32
#    - Porcentaje entrenamiento: 80%
#    - Columnas de entrada: Todas las disponibles
#    - Columna de salida: G3
# 3. Entrenar el modelo
# 4. Ver los resultados del entrenamiento
```

### Paso 3: Probar Predicciones

```bash
# Usar el comando de Django para probar predicciones
python manage.py probar_predicciones --limite 20

# Verificar que las predicciones sean razonables
# Verificar que las alertas se generen correctamente
```

### Paso 4: Validar Resultados

```bash
# 1. Ir a la interfaz web: http://localhost:8000/predicciones/
# 2. Ver todas las predicciones realizadas
# 3. Comparar predicciones con calificaciones reales
# 4. Verificar que los errores sean aceptables
```

### Paso 5: Verificar Alertas

```bash
# 1. Ir a la interfaz web: http://localhost:8000/docentes/
# 2. Ver las alertas generadas
# 3. Verificar que las alertas sean relevantes
# 4. Marcar alertas como vistas
```

## 4. Métricas de Validación

### 4.1 Métricas del Modelo

- **R² (Coeficiente de Determinación)**: Debe ser > 0.5 para un modelo aceptable
- **MAE (Error Absoluto Medio)**: Debe ser < 2.0 para predicciones precisas
- **RMSE (Raíz del Error Cuadrático Medio)**: Debe ser < 3.0 para predicciones precisas

### 4.2 Métricas de Predicción

- **Precisión**: Porcentaje de predicciones con error ≤ 2.0
- **Error Promedio**: Error promedio entre predicciones y valores reales
- **Error Mediano**: Error mediano entre predicciones y valores reales

### 4.3 Validación de Alertas

- **Relevancia**: Las alertas deben ser relevantes para el estudiante
- **Precisión**: Las alertas deben estar basadas en datos reales
- **Timeliness**: Las alertas deben generarse en tiempo oportuno

## 5. Troubleshooting

### Problema: No se generan alertas

**Solución**: 
1. Verificar que haya estudiantes cargados
2. Verificar que haya predicciones realizadas
3. Ejecutar manualmente: `python manage.py generar_alertas`

### Problema: Las predicciones son muy inexactas

**Solución**:
1. Verificar que los datos estén preprocesados correctamente
2. Ajustar los parámetros del modelo (tasa de aprendizaje, iteraciones)
3. Probar con diferentes funciones de activación
4. Aumentar el número de neuronas o capas ocultas

### Problema: El entrenamiento es muy lento

**Solución**:
1. Reducir el número de iteraciones
2. Reducir el tamaño del batch
3. Reducir el número de neuronas
4. Reducir el número de características de entrada

## 6. Notas Finales

- El sistema de alertas está completamente implementado y funcional
- Las pruebas con datos reales se pueden realizar usando el comando `probar_predicciones`
- Las alertas se generan automáticamente después de hacer predicciones
- El sistema está listo para usar en producción

