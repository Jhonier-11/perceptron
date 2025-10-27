# Perceptrón Simple - Aplicación Web

Una implementación completa del algoritmo de perceptrón simple desde cero, construida con Django y NumPy. Esta aplicación web permite cargar datos, entrenar perceptrones, visualizar el proceso de aprendizaje y hacer predicciones.

## 🚀 Características

### ✅ Funcionalidades Principales
- **Carga de Datos**: Soporte para archivos CSV, XLSX, JSON y TXT
- **Entrenamiento Interactivo**: Configuración de parámetros y visualización paso a paso
- **Implementación Pura**: Algoritmo implementado desde cero usando solo NumPy
- **Visualizaciones**: Gráficos de evolución del error y pesos durante el entrenamiento
- **Predicciones**: Interfaz para probar el perceptrón entrenado con nuevos datos
- **Historial**: Almacenamiento de todos los entrenamientos realizados
- **Exportación**: Descarga de pesos entrenados en formato JSON

### 🧠 Algoritmo del Perceptrón
- **Función de Activación**: Escalón (Step function)
- **Regla de Aprendizaje**: Regla del perceptrón clásica
- **Inicialización**: Pesos aleatorios entre -0.5 y 0.5
- **Convergencia**: Parada automática cuando no hay errores

## 📋 Requisitos

- Python 3.8+
- Django 5.2+
- NumPy
- Pandas
- Matplotlib
- OpenPyXL

## 🛠️ Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   git clone <url-del-repositorio>
   cd perceptron
   ```

2. **Instalar dependencias**
   ```bash
   pip install numpy pandas matplotlib openpyxl
   ```

3. **Configurar la base de datos**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Crear superusuario (opcional)**
   ```bash
   python manage.py createsuperuser
   ```

5. **Ejecutar el servidor**
   ```bash
   python manage.py runserver
   ```

6. **Abrir en el navegador**
   ```
   http://127.0.0.1:8000
   ```

## 📖 Uso de la Aplicación

### 1. Cargar Datos
- Ve a "Cargar Datos" en el menú
- Selecciona un archivo CSV, XLSX, JSON o TXT
- El sistema mostrará una vista previa de los datos

### 2. Configurar Entrenamiento
- Selecciona las columnas de entrada (X) y salida (Y)
- Ajusta la tasa de aprendizaje (recomendado: 0.1)
- Establece el número de épocas (recomendado: 100-1000)

### 3. Entrenar el Perceptrón
- Haz clic en "Iniciar Entrenamiento"
- Observa el progreso en tiempo real
- El sistema mostrará la evolución de pesos y errores

### 4. Ver Resultados
- Revisa la precisión final del perceptrón
- Examina los gráficos de evolución
- Descarga los pesos entrenados si lo deseas

### 5. Hacer Predicciones
- Ve a "Predecir" en el menú
- Ingresa nuevos valores de entrada
- Obtén la predicción del perceptrón entrenado

## 📊 Ejemplo: Compuerta Lógica AND

La aplicación incluye un ejemplo de la compuerta lógica AND:
source D:/blute/Anaconda/Scripts/activate ia
| x1 | x2 | y (AND) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 0       |
| 1  | 0  | 0       |
| 1  | 1  | 1       |

### Pasos para probar:
1. Crea un archivo CSV con los datos de la tabla
2. Carga el archivo en la aplicación
3. Configura: x1 y x2 como entradas, y como salida
4. Entrena con tasa de aprendizaje 0.5 y 100 épocas
5. El perceptrón debería converger rápidamente y alcanzar 100% de precisión

## 🔧 Estructura del Proyecto

```
perceptron/
├── manage.py
├── percep/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── perceptron_app/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── perceptron.py      # Implementación del perceptrón
│   ├── urls.py
│   └── views.py
├── templates/
│   ├── base.html
│   └── perceptron_app/
│       ├── home.html
│       ├── upload_data.html
│       ├── configure_training.html
│       ├── train_perceptron.html
│       ├── training_results.html
│       ├── make_prediction.html
│       └── training_history.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── examples/
│       └── ejemplo_and.csv
└── README.md
```

## 🧮 Algoritmo del Perceptrón Simple

### Fórmula de Predicción
```
y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Donde:
- `f()` es la función escalón: f(x) = 1 si x ≥ 0, 0 en caso contrario
- `wᵢ` son los pesos de las características
- `xᵢ` son los valores de entrada
- `b` es el sesgo (bias)

### Regla de Actualización
```
wᵢ = wᵢ + η × error × xᵢ
b = b + η × error
```

Donde:
- `η` (eta) es la tasa de aprendizaje
- `error = valor_real - predicción`

## 🎯 Limitaciones del Perceptrón Simple

- **Solo problemas linealmente separables**: No puede resolver XOR sin capas adicionales
- **Una sola salida**: Solo maneja clasificación binaria
- **Función de activación simple**: Solo escalón, no derivable

## 🔍 Solución de Problemas

### Error de importación de NumPy
```bash
pip install numpy
```

### Error de migraciones
```bash
python manage.py makemigrations perceptron_app
python manage.py migrate
```

### Error de archivos estáticos
```bash
python manage.py collectstatic
```

## 📚 Referencias

- [Perceptrón - Wikipedia](https://es.wikipedia.org/wiki/Perceptrón)
- [Django Documentation](https://docs.djangoproject.com/)
- [NumPy Documentation](https://numpy.org/doc/)

## 👨‍💻 Autor

Desarrollado como proyecto educativo para demostrar la implementación del perceptrón simple desde cero.

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.
