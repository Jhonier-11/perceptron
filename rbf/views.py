"""
Vistas para la aplicación de Red Neuronal RBF
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import json
import pandas as pd
import numpy as np
import os
import uuid
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import io
import base64

from .models import RBFTraining, RBFPrediction
from .rbf_engine import RBFNet, dividir_entrenamiento_prueba, normalizar_datos, desnormalizar_datos
from .rbf_engine import calcular_eg, calcular_mae, calcular_rmse, verificar_convergencia, calcular_metricas
from .rbf_engine import preprocesar_datos
from .forms import RBFDataUploadForm, RBFConfigForm, RBFPredictionForm


def convertir_a_tipos_nativos(obj):
    """
    Convierte objetos de NumPy y pandas a tipos nativos de Python para serialización JSON
    
    Args:
        obj: Objeto a convertir
        
    Returns:
        Objeto convertido a tipos nativos de Python
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_tipos_nativos(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_tipos_nativos(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convertir_a_tipos_nativos(item) for item in obj)
    else:
        return obj


def detectar_separador_csv_y_leer(archivo):
    """
    Detecta automáticamente el separador de un archivo CSV y lo lee
    """
    # Lista de separadores comunes a probar
    separadores = [',', ';', '|', '\t', ' ', ':', '~']
    
    # Leer las primeras líneas para detectar el separador
    archivo.seek(0)
    try:
        # Intentar leer como texto directamente
        sample = archivo.read(1024)
        if isinstance(sample, bytes):
            sample = sample.decode('utf-8', errors='ignore')
    except:
        # Si falla, intentar con diferentes encodings
        archivo.seek(0)
        try:
            sample = archivo.read(1024).decode('latin-1', errors='ignore')
        except:
            sample = archivo.read(1024).decode('utf-8', errors='ignore')
    
    archivo.seek(0)
    
    # Contar ocurrencias de cada separador
    separador_counts = {}
    for sep in separadores:
        separador_counts[sep] = sample.count(sep)
    
    # Encontrar el separador más común
    mejor_separador = max(separador_counts, key=separador_counts.get)
    
    # Si no hay separadores detectados, usar coma por defecto
    if separador_counts[mejor_separador] == 0:
        mejor_separador = ','
    
    try:
        # Intentar leer con el separador detectado
        df = pd.read_csv(archivo, sep=mejor_separador)
        
        # Verificar que el DataFrame tiene al menos 2 columnas
        if df.shape[1] < 2:
            # Si solo tiene 1 columna, probar con otros separadores
            for sep in separadores:
                if sep != mejor_separador:
                    archivo.seek(0)
                    try:
                        df_test = pd.read_csv(archivo, sep=sep)
                        if df_test.shape[1] > df.shape[1]:
                            df = df_test
                            mejor_separador = sep
                    except:
                        continue
        
        return df
        
    except Exception as e:
        # Si falla, intentar con coma por defecto
        archivo.seek(0)
        try:
            return pd.read_csv(archivo, sep=',')
        except:
            # Si todo falla, intentar con tabulador
            archivo.seek(0)
            return pd.read_csv(archivo, sep='\t')


def generar_graficos_rbf(y_train_real, y_train_pred, y_test_real, y_test_pred, 
                         metricas_train, metricas_test, error_objetivo,
                         num_entradas=None, num_centros=None):
    """
    Genera los gráficos requeridos para RBF y retorna Base64
    
    Args:
        y_train_real, y_train_pred, y_test_real, y_test_pred: Datos para gráficos
        metricas_train, metricas_test: Métricas calculadas
        error_objetivo: Error objetivo para comparación
        num_entradas: Número de características de entrada (opcional)
        num_centros: Número de centros radiales (opcional)
    
    Returns:
        dict: Diccionario con las imágenes en Base64
    """
    graficos = {}
    
    # 1. Gráfico de dispersión: Yd vs Yr (entrenamiento y prueba)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Entrenamiento
    ax1.scatter(y_train_real, y_train_pred, alpha=0.6, s=50, c='blue')
    min_val = min(min(y_train_real), min(y_train_pred))
    max_val = max(max(y_train_real), max(y_train_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Diagonal ideal')
    ax1.set_xlabel('Salida Deseada (Yd)', fontsize=11)
    ax1.set_ylabel('Salida Obtenida (Yr)', fontsize=11)
    ax1.set_title('Entrenamiento: Yd vs Yr', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Prueba
    ax2.scatter(y_test_real, y_test_pred, alpha=0.6, s=50, c='green')
    min_val = min(min(y_test_real), min(y_test_pred))
    max_val = max(max(y_test_real), max(y_test_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Diagonal ideal')
    ax2.set_xlabel('Salida Deseada (Yd)', fontsize=11)
    ax2.set_ylabel('Salida Obtenida (Yr)', fontsize=11)
    ax2.set_title('Prueba: Yd vs Yr', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Convertir a base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    graficos['dispersion'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 2. Gráfico de Error General vs error óptimo
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conjuntos = ['Entrenamiento', 'Prueba']
    eg_values = [metricas_train['EG'], metricas_test['EG']]
    colors = ['blue', 'green']
    
    bars = ax.bar(conjuntos, eg_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Línea de error objetivo
    ax.axhline(y=error_objetivo, color='red', linestyle='--', linewidth=2, 
               label=f'Error objetivo ({error_objetivo})')
    
    # Agregar valores en las barras
    for bar, val in zip(bars, eg_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Error General (EG)', fontsize=12)
    ax.set_title('Error General vs Error Objetivo', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Convertir a base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    graficos['error'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 3. Gráfico combinado de todas las métricas
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metricas_nombres = ['EG', 'MAE', 'RMSE']
    train_values = [metricas_train['EG'], metricas_train['MAE'], metricas_train['RMSE']]
    test_values = [metricas_test['EG'], metricas_test['MAE'], metricas_test['RMSE']]
    
    x = np.arange(len(metricas_nombres))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Entrenamiento', alpha=0.7, color='blue')
    bars2 = ax.bar(x + width/2, test_values, width, label='Prueba', alpha=0.7, color='green')
    
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparación de Métricas: Entrenamiento vs Prueba', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nombres)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Convertir a base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    graficos['metricas'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 4. Visualización de la estructura de la red neuronal RBF
    if num_entradas is not None and num_centros is not None:
        grafico_red = _generar_visualizacion_red_rbf(num_entradas, num_centros)
        if grafico_red:
            graficos['red_neuronal'] = grafico_red
    
    return graficos


def _generar_visualizacion_red_rbf(num_entradas, num_centros):
    """
    Genera una visualización gráfica de la estructura de la red neuronal RBF
    
    Args:
        num_entradas: Número de características de entrada
        num_centros: Número de centros radiales
        
    Returns:
        str: Imagen en Base64 o None si hay error
    """
    try:
        from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Definir posiciones de las capas
        # Capa de entrada (izquierda)
        entrada_x = 1
        entrada_y_inicio = 2
        entrada_espacio = 6 / max(num_entradas, 1)  # Espacio entre nodos de entrada
        
        # Capa de centros radiales (centro)
        centros_x = 5
        centros_y_inicio = 2
        centros_espacio = 6 / max(num_centros, 1)  # Espacio entre centros
        
        # Capa de salida (derecha)
        salida_x = 9
        salida_y = 5
        
        # Dibujar nodos de entrada
        entrada_posiciones = []
        for i in range(num_entradas):
            y_pos = entrada_y_inicio + i * entrada_espacio
            entrada_posiciones.append((entrada_x, y_pos))
            circle = Circle((entrada_x, y_pos), 0.3, color='#4A90E2', ec='black', lw=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(entrada_x, y_pos, f'X{i+1}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white', zorder=4)
        
        # Etiqueta de capa de entrada
        ax.text(entrada_x, entrada_y_inicio - 1, 'Entradas', ha='center', va='top',
               fontsize=11, fontweight='bold', color='#2C3E50')
        
        # Dibujar nodos de centros radiales
        centros_posiciones = []
        for i in range(num_centros):
            y_pos = centros_y_inicio + i * centros_espacio
            centros_posiciones.append((centros_x, y_pos))
            circle = Circle((centros_x, y_pos), 0.4, color='#E74C3C', ec='black', lw=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(centros_x, y_pos, f'R{i+1}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white', zorder=4)
        
        # Etiqueta de capa de centros
        ax.text(centros_x, centros_y_inicio - 1, 'Centros\nRadiales', ha='center', va='top',
               fontsize=11, fontweight='bold', color='#2C3E50')
        
        # Dibujar nodo de salida
        circle_salida = Circle((salida_x, salida_y), 0.35, color='#27AE60', ec='black', lw=2, zorder=3)
        ax.add_patch(circle_salida)
        ax.text(salida_x, salida_y, 'Y', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', zorder=4)
        ax.text(salida_x, salida_y - 1, 'Salida', ha='center', va='top',
               fontsize=11, fontweight='bold', color='#2C3E50')
        
        # Dibujar conexiones: Entradas -> Centros
        for entrada_pos in entrada_posiciones:
            for centro_pos in centros_posiciones:
                arrow = FancyArrowPatch(
                    (entrada_pos[0] + 0.3, entrada_pos[1]),
                    (centro_pos[0] - 0.4, centro_pos[1]),
                    arrowstyle='->', mutation_scale=15,
                    color='#7F8C8D', alpha=0.4, linewidth=0.8, zorder=1
                )
                ax.add_patch(arrow)
        
        # Dibujar conexiones: Centros -> Salida
        for centro_pos in centros_posiciones:
            arrow = FancyArrowPatch(
                (centro_pos[0] + 0.4, centro_pos[1]),
                (salida_x - 0.35, salida_y),
                arrowstyle='->', mutation_scale=20,
                color='#8E44AD', alpha=0.6, linewidth=1.2, zorder=2
            )
            ax.add_patch(arrow)
        
        # Agregar título
        ax.text(5, 9.5, 'Estructura de la Red Neuronal RBF', ha='center', va='top',
               fontsize=16, fontweight='bold', color='#2C3E50')
        
        # Agregar leyenda
        leyenda_elementos = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90E2', 
                      markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='Entradas'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
                      markersize=14, markeredgecolor='black', markeredgewidth=1.5, label='Centros Radiales'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', 
                      markersize=13, markeredgecolor='black', markeredgewidth=2, label='Salida'),
            plt.Line2D([0], [0], color='#7F8C8D', alpha=0.4, linewidth=2, label='Conexiones Entrada→Centro'),
            plt.Line2D([0], [0], color='#8E44AD', alpha=0.6, linewidth=2, label='Conexiones Centro→Salida')
        ]
        ax.legend(handles=leyenda_elementos, loc='upper right', fontsize=9, framealpha=0.9)
        
        # Agregar información adicional
        info_text = f'Número de Entradas: {num_entradas}\nNúmero de Centros: {num_centros}'
        ax.text(0.5, 0.95, info_text, transform=ax.transAxes, ha='left', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        grafico_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return grafico_base64
        
    except Exception as e:
        print(f"Error al generar visualización de red: {e}")
        return None


def inicio_rbf(request):
    """
    Vista principal que muestra el dashboard RBF
    """
    # Obtener los últimos entrenamientos
    recent_trainings = RBFTraining.objects.all()[:5]
    
    context = {
        'recent_trainings': recent_trainings,
        'total_trainings': RBFTraining.objects.count(),
    }
    
    return render(request, 'rbf/inicio.html', context)


def cargar_datos_rbf(request):
    """
    Vista para cargar archivos de datos desde el directorio de ejemplos o upload manual
    """
    # Obtener lista de archivos de ejemplo
    examples_dir = os.path.join(settings.BASE_DIR, 'static', 'examples', 'dt')
    archivos_ejemplo = []
    
    if os.path.exists(examples_dir):
        for filename in os.listdir(examples_dir):
            if filename.endswith(('.csv', '.json', '.xlsx', '.txt')):
                archivos_ejemplo.append(filename)
    
    # Limpiar sesión si se solicita
    if request.GET.get('clear') == 'true':
        if 'uploaded_data_rbf' in request.session:
            del request.session['uploaded_data_rbf']
        if 'rbf_training_config' in request.session:
            del request.session['rbf_training_config']
        if 'file_path_rbf' in request.session:
            try:
                file_path = request.session['file_path_rbf']
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error al eliminar archivo: {e}")
            del request.session['file_path_rbf']
        messages.info(request, 'Sesión limpiada. Puedes cargar un nuevo archivo.')
        return redirect('rbf:cargar_datos_rbf')
    
    if request.method == 'POST':
        form = RBFDataUploadForm(request.POST, request.FILES)
        
        # Llenar las opciones del select con archivos de ejemplo
        form.fields['archivo_ejemplo'].choices = [('', '-- Selecciona un archivo --')] + [(f, f) for f in archivos_ejemplo]
        
        if form.is_valid():
            archivo_ejemplo = form.cleaned_data.get('archivo_ejemplo')
            
            # Si se seleccionó un archivo de ejemplo
            if archivo_ejemplo:
                file_path = os.path.join(examples_dir, archivo_ejemplo)
                file_extension = archivo_ejemplo.split('.')[-1].lower()
                nombre_archivo = archivo_ejemplo
            else:
                # Procesar el archivo subido
                file = request.FILES['data_file']
                file_extension = file.name.split('.')[-1].lower()
                nombre_archivo = file.name
                file_path = None
            
            # Procesar el archivo
            if archivo_ejemplo:
                # Leer desde el directorio de ejemplos
                if file_extension == 'csv':
                    df = pd.read_csv(file_path)
                elif file_extension == 'json':
                    df = pd.read_json(file_path)
                elif file_extension == 'txt':
                    df = pd.read_csv(file_path)
                else:
                    messages.error(request, 'Formato de archivo no soportado.')
                    form.fields['archivo_ejemplo'].choices = [('', '-- Selecciona un archivo --')] + [(f, f) for f in archivos_ejemplo]
                    return render(request, 'rbf/cargar_datos.html', {'form': form, 'archivos_ejemplo': archivos_ejemplo})
            else:
                # Procesar como antes
                file = request.FILES['data_file']
                file_extension = file.name.split('.')[-1].lower()
            
            try:
                # Validar mínimo de patrones
                if len(df) < 10:
                    messages.error(request, 'El dataset debe tener al menos 10 patrones.')
                    form.fields['archivo_ejemplo'].choices = [('', '-- Selecciona un archivo --')] + [(f, f) for f in archivos_ejemplo]
                    return render(request, 'rbf/cargar_datos.html', {'form': form, 'archivos_ejemplo': archivos_ejemplo})
                
                # Si es archivo de ejemplo, copiar a uploads
                if archivo_ejemplo:
                    unique_filename = f"{uuid.uuid4()}_{nombre_archivo}"
                    destination_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)
                    
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    
                    # Copiar desde examples a uploads
                    import shutil
                    shutil.copy2(file_path, destination_path)
                    file_path_sesion = destination_path
                else:
                    # Guardar el archivo físicamente
                    file.seek(0)
                    unique_filename = f"{uuid.uuid4()}_{nombre_archivo}"
                    destination_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)
                    
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    
                    with open(destination_path, 'wb+') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)
                    
                    file_path_sesion = destination_path
                
                # Guardar información en la sesión
                uploaded_data = {
                    'filename': nombre_archivo,
                    'columns': df.columns.tolist(),
                    'shape': list(df.shape),
                }
                
                uploaded_data = convertir_a_tipos_nativos(uploaded_data)
                request.session['uploaded_data_rbf'] = uploaded_data
                request.session['file_path_rbf'] = file_path_sesion
                
                messages.success(request, f'Archivo cargado exitosamente. {df.shape[0]} filas, {df.shape[1]} columnas.')
                return redirect('rbf:configurar_rbf')
                
            except Exception as e:
                messages.error(request, f'Error al procesar el archivo: {str(e)}')
                form.fields['archivo_ejemplo'].choices = [('', '-- Selecciona un archivo --')] + [(f, f) for f in archivos_ejemplo]
                return render(request, 'rbf/cargar_datos.html', {'form': form, 'archivos_ejemplo': archivos_ejemplo})
    else:
        form = RBFDataUploadForm()
        # Llenar las opciones del select con archivos de ejemplo
        form.fields['archivo_ejemplo'].choices = [('', '-- Selecciona un archivo --')] + [(f, f) for f in archivos_ejemplo]
    
    uploaded_data = request.session.get('uploaded_data_rbf', None)
    
    context = {
        'form': form,
        'uploaded_data': uploaded_data,
        'archivos_ejemplo': archivos_ejemplo
    }
    
    return render(request, 'rbf/cargar_datos.html', context)


def configurar_rbf(request):
    """
    Vista para configurar los parámetros de entrenamiento RBF
    """
    if 'uploaded_data_rbf' not in request.session:
        messages.warning(request, 'Primero debes cargar un archivo de datos.')
        return redirect('rbf:cargar_datos_rbf')
    
    uploaded_data = request.session['uploaded_data_rbf']
    columns = uploaded_data['columns']
    
    if request.method == 'POST':
        form = RBFConfigForm(request.POST, columns=columns)
        if form.is_valid():
            # Guardar configuración en la sesión
            request.session['rbf_training_config'] = {
                'nombre_entrenamiento': form.cleaned_data['nombre_entrenamiento'],
                'num_centros': form.cleaned_data['num_centros'],
                'porcentaje_entrenamiento': form.cleaned_data['porcentaje_entrenamiento'] / 100.0,  # Convertir a fracción
                'error_aproximacion': form.cleaned_data['error_aproximacion'],
                'columnas_entrada': form.cleaned_data['input_columns'],
                'columnas_salida': form.cleaned_data['output_columns']
            }
            
            messages.success(request, 'Configuración guardada. Listo para entrenar.')
            return redirect('rbf:entrenar_rbf')
    else:
        form = RBFConfigForm(columns=columns)
    
    context = {
        'form': form,
        'uploaded_data': uploaded_data,
        'columns': columns
    }
    
    return render(request, 'rbf/configurar.html', context)


def entrenar_rbf(request):
    """
    Vista para entrenar la red RBF
    """
    if 'uploaded_data_rbf' not in request.session or 'rbf_training_config' not in request.session:
        messages.warning(request, 'Debes cargar datos y configurar el entrenamiento primero.')
        return redirect('rbf:cargar_datos_rbf')
    
    uploaded_data = request.session['uploaded_data_rbf']
    training_config = request.session['rbf_training_config']
    
    if request.method == 'POST':
        try:
            # Cargar datos desde el archivo guardado
            file_path = request.session.get('file_path_rbf')
            
            if not file_path or not os.path.exists(file_path):
                messages.error(request, 'No se encontró el archivo de datos.')
                return redirect('rbf:cargar_datos_rbf')
            
            # Leer datos
            file_extension = uploaded_data['filename'].split('.')[-1].lower()
            
            try:
                if file_extension == 'csv' or file_extension == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        df = detectar_separador_csv_y_leer(f)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(file_path)
                elif file_extension == 'json':
                    df = pd.read_json(file_path)
                    if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
                        first_col = df.iloc[:, 0]
                        if isinstance(first_col.iloc[0], dict):
                            df = pd.json_normalize(df.iloc[:, 0].tolist())
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    df = detectar_separador_csv_y_leer(f)
            
            # Preparar datos
            columnas_entrada = training_config['columnas_entrada']
            columnas_salida = training_config['columnas_salida']
            
            # Aplicar preprocesamiento automático (ahora normaliza por máximo)
            info_preprocesamiento, df_procesado, df_comparacion, df_antes_normalizacion = preprocesar_datos(df, columnas_entrada, columnas_salida)
            
            # Extraer X e y (ya están normalizados)
            X = df_procesado[columnas_entrada].values
            y = df_procesado[columnas_salida].values.flatten()
            
            # Convertir a float
            X = X.astype(float)
            y = y.astype(float)
            
            # Dividir en entrenamiento y prueba
            porcentaje = training_config['porcentaje_entrenamiento']
            X_train, X_test, y_train, y_test = dividir_entrenamiento_prueba(X, y, porcentaje)
            
            # Los datos ya están normalizados por máximo en el preprocesamiento
            # No necesitamos normalizar nuevamente aquí
            X_train_norm = X_train
            X_test_norm = X_test
            mean_train = None
            std_train = None
            
            # Guardar información de normalización para desnormalización posterior
            info_preprocesamiento['normalizado'] = True
            
            # Crear y entrenar la red RBF
            num_centros = training_config['num_centros']
            error_objetivo = training_config['error_aproximacion']
            
            rbf = RBFNet(num_centros=num_centros, error_aproximacion=error_objetivo)
            
            try:
                resultado_entrenamiento = rbf.fit(X_train_norm, y_train)
            except ValueError as e:
                messages.error(request, f'Error durante el entrenamiento: {str(e)}')
                return redirect('rbf:configurar_rbf')
            
            # Realizar predicciones
            y_train_pred = rbf.predict(X_train_norm)
            y_test_pred = rbf.predict(X_test_norm)
            
            # Calcular métricas adicionales para el conjunto de prueba
            metricas_test = {
                'EG': calcular_eg(y_test, y_test_pred),
                'MAE': calcular_mae(y_test, y_test_pred),
                'RMSE': calcular_rmse(y_test, y_test_pred)
            }
            
            # Las métricas de entrenamiento ya están en resultado_entrenamiento['paso_7_metricas']
            metricas_train = resultado_entrenamiento['paso_7_metricas']
            
            # Verificar convergencia
            convergencia = verificar_convergencia(metricas_train['EG'], error_objetivo)
            
            # Generar gráficos
            num_entradas = X_train_norm.shape[1]  # Número de características de entrada
            graficos = generar_graficos_rbf(
                y_train, y_train_pred, y_test, y_test_pred,
                metricas_train, metricas_test, error_objetivo,
                num_entradas=num_entradas, num_centros=num_centros
            )
            
            # Guardar en base de datos (usar máximos de normalización por máximo)
            estadisticas_norm = {
                'maximos_columnas': info_preprocesamiento.get('maximos_columnas', {}),
                'tipo_normalizacion': 'max'
            }
            
            # Preparar datos para mostrar: dataset original y dataset normalizado (vista previa)
            datos_original = None
            datos_normalizado = None
            
            # Dataset original (solo columnas relevantes, solo primeras 20 filas para visualización)
            if not df_comparacion.empty:
                columnas_mostrar = columnas_entrada + columnas_salida
                df_original_mostrar = df_comparacion[columnas_mostrar].head(20)
                datos_original = df_original_mostrar.to_dict('records')
            
            # Dataset normalizado (solo columnas relevantes, solo primeras 20 filas para visualización)
            if not df_procesado.empty:
                columnas_mostrar = columnas_entrada + columnas_salida
                df_normalizado_mostrar = df_procesado[columnas_mostrar].head(20)
                datos_normalizado = df_normalizado_mostrar.to_dict('records')
            
            # Datos antes de normalización pero después de codificación (para referencia)
            datos_antes_norm = None
            if not df_antes_normalizacion.empty:
                columnas_mostrar = columnas_entrada + columnas_salida
                df_antes_norm_mostrar = df_antes_normalizacion[columnas_mostrar].head(20)
                datos_antes_norm = df_antes_norm_mostrar.to_dict('records')
            
            # Guardar dataset normalizado COMPLETO (todas las filas) para exportación
            dataset_normalizado_completo = None
            if not df_procesado.empty:
                columnas_mostrar = columnas_entrada + columnas_salida
                df_normalizado_completo = df_procesado[columnas_mostrar]
                dataset_normalizado_completo = df_normalizado_completo.to_dict('records')
            
            # Preparar procesamiento interno completo para guardar
            procesamiento_interno_completo = {
                'num_patrones': resultado_entrenamiento['num_patrones'],
                'num_caracteristicas': resultado_entrenamiento['num_caracteristicas'],
                'num_centros': resultado_entrenamiento['num_centros'],
                'paso_1_inicializacion': convertir_a_tipos_nativos(resultado_entrenamiento['paso_1_inicializacion']),
                'paso_2_distancias': convertir_a_tipos_nativos(resultado_entrenamiento['paso_2_distancias']),
                'paso_3_activaciones': convertir_a_tipos_nativos(resultado_entrenamiento['paso_3_activaciones']),
                'paso_4_matriz_interpolacion': convertir_a_tipos_nativos(resultado_entrenamiento['paso_4_matriz_interpolacion']),
                'paso_5_calculo_pesos': convertir_a_tipos_nativos(resultado_entrenamiento['paso_5_calculo_pesos']),
                'paso_6_predicciones': convertir_a_tipos_nativos(resultado_entrenamiento['paso_6_predicciones']),
                'paso_7_metricas': resultado_entrenamiento['paso_7_metricas']
            }
            
            training_record = RBFTraining.objects.create(
                nombre=training_config['nombre_entrenamiento'],
                num_centros=num_centros,
                porcentaje_entrenamiento=porcentaje * 100,
                error_aproximacion=error_objetivo,
                columnas_entrada=columnas_entrada,
                columnas_salida=columnas_salida,
                centros_radiales=resultado_entrenamiento['centros'],
                pesos_finales=resultado_entrenamiento['pesos'],
                umbral=resultado_entrenamiento['umbral'],
                metricas_entrenamiento=metricas_train,
                metricas_prueba=metricas_test,
                convergencia=convergencia,
                estadisticas_normalizacion=estadisticas_norm,
                # Guardar dataset normalizado completo
                dataset_normalizado=dataset_normalizado_completo,
                # Guardar procesamiento interno completo
                procesamiento_interno=procesamiento_interno_completo,
                # Guardar valores reales y predichos para regenerar gráficos
                y_train_real=y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
                y_train_pred=y_train_pred.tolist() if hasattr(y_train_pred, 'tolist') else list(y_train_pred),
                y_test_real=y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                y_test_pred=y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else list(y_test_pred)
            )
            
            # Guardar gráficos y resultados en sesión
            request.session['rbf_results'] = {
                'training_id': training_record.id,
                'dispersion_plot': graficos['dispersion'],
                'error_plot': graficos['error'],
                'metricas_plot': graficos['metricas'],
                'red_neuronal_plot': graficos.get('red_neuronal', None),
                'metricas_train': metricas_train,
                'metricas_test': metricas_test,
                'convergencia': convergencia,
                'info_preprocesamiento': convertir_a_tipos_nativos(info_preprocesamiento),
                'datos_original': datos_original,
                'datos_normalizado': datos_normalizado,
                'datos_antes_norm': datos_antes_norm,
                'columnas_entrada': columnas_entrada,
                'columnas_salida': columnas_salida,
                # Datos internos del procesamiento matemático
                'procesamiento_interno': {
                    'num_patrones': resultado_entrenamiento['num_patrones'],
                    'num_caracteristicas': resultado_entrenamiento['num_caracteristicas'],
                    'num_centros': resultado_entrenamiento['num_centros'],
                    'paso_1_inicializacion': resultado_entrenamiento['paso_1_inicializacion'],
                    'paso_2_distancias': resultado_entrenamiento['paso_2_distancias'],
                    'paso_3_activaciones': resultado_entrenamiento['paso_3_activaciones'],
                    'paso_4_matriz_interpolacion': resultado_entrenamiento['paso_4_matriz_interpolacion'],
                    'paso_5_calculo_pesos': resultado_entrenamiento['paso_5_calculo_pesos'],
                    'paso_6_predicciones': resultado_entrenamiento['paso_6_predicciones'],
                    'paso_7_metricas': resultado_entrenamiento['paso_7_metricas']
                }
            }
            
            # Limpiar archivo temporal
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error al eliminar archivo temporal: {e}")
            
            if 'file_path_rbf' in request.session:
                del request.session['file_path_rbf']
            
            messages.success(request, '¡Entrenamiento completado exitosamente!')
            return redirect('rbf:resultados_rbf')
            
        except Exception as e:
            messages.error(request, f'Error durante el entrenamiento: {str(e)}')
            import traceback
            print(traceback.format_exc())
    
    context = {
        'uploaded_data': uploaded_data,
        'training_config': training_config
    }
    
    return render(request, 'rbf/entrenar.html', context)


def resultados_rbf(request):
    """
    Vista para mostrar los resultados del entrenamiento
    """
    if 'rbf_results' not in request.session:
        messages.warning(request, 'No hay resultados de entrenamiento disponibles.')
        return redirect('rbf:inicio_rbf')
    
    rbf_results = request.session['rbf_results']
    training_record = get_object_or_404(RBFTraining, id=rbf_results['training_id'])
    
    context = {
        'training_record': training_record,
        'dispersion_plot': rbf_results['dispersion_plot'],
        'error_plot': rbf_results['error_plot'],
        'metricas_plot': rbf_results['metricas_plot'],
        'red_neuronal_plot': rbf_results.get('red_neuronal_plot', None),
        'metricas_train': rbf_results['metricas_train'],
        'metricas_test': rbf_results['metricas_test'],
        'convergencia': rbf_results['convergencia'],
        'info_preprocesamiento': rbf_results.get('info_preprocesamiento', {}),
        'datos_original': rbf_results.get('datos_original', []),
        'datos_normalizado': rbf_results.get('datos_normalizado', []),
        'datos_antes_norm': rbf_results.get('datos_antes_norm', []),
        'columnas_entrada': rbf_results.get('columnas_entrada', []),
        'columnas_salida': rbf_results.get('columnas_salida', []),
        'procesamiento_interno': rbf_results.get('procesamiento_interno', {})
    }
    
    return render(request, 'rbf/resultados.html', context)


def historial_rbf(request):
    """
    Vista para mostrar el historial de entrenamientos RBF
    """
    entrenamientos = RBFTraining.objects.all().order_by('-fecha_creacion')
    
    context = {
        'trainings': entrenamientos
    }
    
    return render(request, 'rbf/historial.html', context)


def detalles_rbf(request, training_id):
    """
    Vista para mostrar detalles completos de un entrenamiento específico
    """
    training = get_object_or_404(RBFTraining, id=training_id)
    
    try:
        # Reconstruir la red RBF con los parámetros guardados
        rbf = RBFNet(num_centros=training.num_centros, error_aproximacion=training.error_aproximacion)
        
        # Restaurar centros y pesos
        rbf.centros = np.array(training.centros_radiales)
        rbf.pesos = np.array(training.pesos_finales)
        rbf.W0 = training.umbral
        rbf.W1_n = np.array(training.pesos_finales[1:])
        rbf.entrenado = True
        
        # Verificar si tenemos datos reales guardados
        if (training.y_train_real and training.y_train_pred and 
            training.y_test_real and training.y_test_pred):
            
            # Usar datos reales guardados
            y_train_real = np.array(training.y_train_real)
            y_train_pred = np.array(training.y_train_pred)
            y_test_real = np.array(training.y_test_real)
            y_test_pred = np.array(training.y_test_pred)
            
            # Usar métricas reales guardadas
            metricas_train = training.metricas_entrenamiento
            metricas_test = training.metricas_prueba
            
            # Generar todas las gráficas usando datos reales
            num_entradas = len(training.columnas_entrada)
            num_centros = training.num_centros
            graficos = generar_graficos_rbf(
                y_train_real, y_train_pred, y_test_real, y_test_pred,
                metricas_train, metricas_test, training.error_aproximacion,
                num_entradas=num_entradas, num_centros=num_centros
            )
            
            # Regenerar el procesamiento interno usando datos sintéticos (solo para visualización interna)
            # Esto es solo para mostrar el procesamiento matemático
            num_caracteristicas = len(training.columnas_entrada)
            num_patrones = len(y_train_real) + len(y_test_real)
            
            if training.estadisticas_normalizacion:
                mean_vals = np.array(training.estadisticas_normalizacion['mean'])
                std_vals = np.array(training.estadisticas_normalizacion['std'])
                X_sintetico = np.random.normal(0, 1, (min(num_patrones, 50), num_caracteristicas))
                X_sintetico = X_sintetico * std_vals + mean_vals
            else:
                X_sintetico = np.random.uniform(-5, 5, (min(num_patrones, 50), num_caracteristicas))
            
            y_sintetico = rbf.predict(X_sintetico)
            procesamiento_interno = _regenerar_procesamiento_interno(rbf, X_sintetico, y_sintetico)
            
        else:
            # Si no hay datos guardados, generar datos sintéticos (para entrenamientos antiguos)
            num_patrones = 50
            num_caracteristicas = len(training.columnas_entrada)
            
            if training.estadisticas_normalizacion:
                mean_vals = np.array(training.estadisticas_normalizacion['mean'])
                std_vals = np.array(training.estadisticas_normalizacion['std'])
                X_sintetico = np.random.normal(0, 1, (num_patrones, num_caracteristicas))
                X_sintetico = X_sintetico * std_vals + mean_vals
            else:
                X_sintetico = np.random.uniform(-5, 5, (num_patrones, num_caracteristicas))
            
            y_sintetico = rbf.predict(X_sintetico)
            procesamiento_interno = _regenerar_procesamiento_interno(rbf, X_sintetico, y_sintetico)
            
            split_idx = int(len(y_sintetico) * 0.7)
            y_train_sint = y_sintetico[:split_idx]
            y_test_sint = y_sintetico[split_idx:]
            
            np.random.seed(42)
            ruido_train = np.random.normal(0, np.std(y_train_sint) * 0.1, len(y_train_sint))
            ruido_test = np.random.normal(0, np.std(y_test_sint) * 0.1, len(y_test_sint))
            y_train_pred_sint = y_train_sint + ruido_train
            y_test_pred_sint = y_test_sint + ruido_test
            
            metricas_train_sint = calcular_metricas(y_train_sint, y_train_pred_sint)
            metricas_test_sint = calcular_metricas(y_test_sint, y_test_pred_sint)
            
            num_entradas = len(training.columnas_entrada)
            num_centros = training.num_centros
            graficos = generar_graficos_rbf(
                y_train_sint, y_train_pred_sint, y_test_sint, y_test_pred_sint,
                metricas_train_sint, metricas_test_sint, training.error_aproximacion,
                num_entradas=num_entradas, num_centros=num_centros
            )
        
    except Exception as e:
        # Si hay error, mostrar información básica sin procesamiento interno
        procesamiento_interno = None
        graficos = None
        print(f"Error al regenerar procesamiento interno: {e}")
    
    context = {
        'training': training,
        'procesamiento_interno': procesamiento_interno,
        'graficos': graficos
    }
    
    return render(request, 'rbf/detalles.html', context)


def descargar_modelo_rbf(request, training_id):
    """
    Vista para descargar el modelo entrenado completo, incluyendo:
    - Dataset normalizado completo
    - Resultados matemáticos (centros radiales, pesos, etc.)
    - Procesamiento interno detallado
    - Todas las métricas y configuraciones
    """
    training = get_object_or_404(RBFTraining, id=training_id)
    
    # Crear estructura completa del modelo para exportación
    datos_modelo = {
        'info_general': {
            'nombre_entrenamiento': training.nombre,
            'fecha_creacion': training.fecha_creacion.isoformat(),
            'fecha_creacion_formato': training.fecha_creacion.strftime('%d/%m/%Y %H:%M:%S'),
            'version': '1.0',
            'tipo_modelo': 'Red Neuronal de Función de Base Radial (RBF)'
        },
        
        'configuracion_entrenamiento': {
            'num_centros_radiales': training.num_centros,
            'porcentaje_entrenamiento': training.porcentaje_entrenamiento,
            'error_aproximacion_objetivo': training.error_aproximacion,
            'columnas_entrada': training.columnas_entrada,
            'columnas_salida': training.columnas_salida,
            'num_columnas_entrada': len(training.columnas_entrada),
            'num_columnas_salida': len(training.columnas_salida)
        },
        
        'resultados_matematicos': {
            'centros_radiales': convertir_a_tipos_nativos(training.centros_radiales),
            'pesos_finales': convertir_a_tipos_nativos(training.pesos_finales),
            'umbral': float(training.umbral),
            'nota': 'Los centros radiales (Rj) representan las posiciones en el espacio de características. Los pesos (W) y el umbral (W0) son los parámetros ajustados durante el entrenamiento.'
        },
        
        'metricas_evaluacion': {
            'entrenamiento': {
                'error_general_eg': training.metricas_entrenamiento.get('EG', None),
                'error_absoluto_medio_mae': training.metricas_entrenamiento.get('MAE', None),
                'raiz_error_cuadratico_medio_rmse': training.metricas_entrenamiento.get('RMSE', None)
            },
            'prueba': {
                'error_general_eg': training.metricas_prueba.get('EG', None),
                'error_absoluto_medio_mae': training.metricas_prueba.get('MAE', None),
                'raiz_error_cuadratico_medio_rmse': training.metricas_prueba.get('RMSE', None)
            },
            'convergencia': training.convergencia,
            'nota': 'EG = Error General, MAE = Error Absoluto Medio, RMSE = Raíz del Error Cuadrático Medio'
        },
        
        'datos_entrenamiento': {
            'y_train_real': training.y_train_real,
            'y_train_predicho': training.y_train_pred,
            'y_test_real': training.y_test_real,
            'y_test_predicho': training.y_test_pred,
            'num_patrones_entrenamiento': len(training.y_train_real) if training.y_train_real else 0,
            'num_patrones_prueba': len(training.y_test_real) if training.y_test_real else 0,
            'total_patrones': (len(training.y_train_real) if training.y_train_real else 0) + 
                             (len(training.y_test_real) if training.y_test_real else 0)
        },
        
        'normalizacion': {
            'estadisticas': training.estadisticas_normalizacion,
            'tipo_normalizacion': training.estadisticas_normalizacion.get('tipo_normalizacion', 'max') if training.estadisticas_normalizacion else 'max',
            'maximos_columnas': training.estadisticas_normalizacion.get('maximos_columnas', {}) if training.estadisticas_normalizacion else {},
            'nota': 'Los valores máximos se utilizan para desnormalizar las predicciones. Cada valor normalizado se multiplica por su máximo correspondiente.'
        },
        
        'dataset_normalizado': {
            'descripcion': 'Dataset completo normalizado utilizado para el entrenamiento (todos los valores divididos por el máximo de su columna)',
            'num_filas': len(training.dataset_normalizado) if training.dataset_normalizado else 0,
            'columnas': training.columnas_entrada + training.columnas_salida if training.columnas_entrada else [],
            'datos': training.dataset_normalizado if training.dataset_normalizado else [],
            'nota': 'Dataset normalizado completo con todas las filas. Si está vacío, el entrenamiento fue realizado antes de implementar esta característica.'
        },
        
        'procesamiento_matematico_interno': {
            'descripcion': 'Procesamiento matemático detallado del entrenamiento, incluyendo todos los pasos intermedios',
            'disponible': training.procesamiento_interno is not None and training.procesamiento_interno != {},
            'detalles': training.procesamiento_interno if training.procesamiento_interno else {},
            'nota': 'Procesamiento matemático completo del entrenamiento. Si está vacío, el entrenamiento fue realizado antes de implementar esta característica.'
        },
        
        'notas_adicionales': {
            'formulas_matematicas': {
                'inicializacion_centros': 'r_j ~ U[min(x_i), max(x_i)]',
                'calculo_distancias': 'D_{pj} = sqrt(sum_{i=1}^{m} (x_{pi} - r_{ji})^2)',
                'funcion_activacion': 'FA(D_{pj}) = D_{pj}^2 · ln(D_{pj})',
                'matriz_interpolacion': 'A = [1, FA(D_1), FA(D_2), ..., FA(D_k)]',
                'calculo_pesos': 'W = (A^T A)^{-1} A^T Y',
                'prediccion': 'y_pred = A · W'
            },
            'metricas_formulas': {
                'EG': 'EG = (1/N) sum_{i=1}^{N} |Y_d - Y_r|',
                'MAE': 'MAE = (1/N) sum_{i=1}^{N} |Y_d - Y_r|',
                'RMSE': 'RMSE = sqrt((1/N) sum_{i=1}^{N} (Y_d - Y_r)^2)'
            }
        }
    }
    
    # Limpiar nombre del archivo de caracteres especiales
    nombre_archivo_limpio = "".join(c for c in training.nombre if c.isalnum() or c in (' ', '-', '_')).rstrip()
    nombre_archivo_limpio = nombre_archivo_limpio.replace(' ', '_').replace('-', '_')
    
    response = HttpResponse(
        json.dumps(datos_modelo, indent=2, ensure_ascii=False),
        content_type='application/json; charset=utf-8'
    )
    response['Content-Disposition'] = f'attachment; filename="rbf_modelo_completo_{training.id}_{nombre_archivo_limpio}.json"'
    
    return response


def predecir_rbf(request, training_id):
    """
    Vista para hacer predicciones con la red RBF entrenada
    """
    training = get_object_or_404(RBFTraining, id=training_id)
    
    if request.method == 'POST':
        form = RBFPredictionForm(training.columnas_entrada, request.POST)
        if form.is_valid():
            try:
                # Obtener valores de entrada
                valores_entrada = []
                for col in training.columnas_entrada:
                    valores_entrada.append(form.cleaned_data[col])
                
                # Crear instancia de RBF con los parámetros guardados
                rbf = RBFNet(num_centros=training.num_centros)
                
                # Establecer centros y pesos desde la base de datos
                rbf.centros = np.array(training.centros_radiales)
                rbf.pesos = np.array(training.pesos_finales)
                rbf.W0 = training.umbral
                rbf.W1_n = rbf.pesos[1:]
                rbf.entrenado = True
                
                # Convertir a array numpy
                X_input = np.array([valores_entrada], dtype=float)
                
                # Si hay estadísticas de normalización, normalizar los datos
                if training.estadisticas_normalizacion:
                    stats = training.estadisticas_normalizacion
                    if isinstance(stats, dict):
                        # Normalización por máximo (nueva forma)
                        if 'maximos_columnas' in stats and 'tipo_normalizacion' in stats and stats['tipo_normalizacion'] == 'max':
                            maximos = stats['maximos_columnas']
                            # Normalizar cada columna de entrada dividiendo por su máximo
                            for i, col in enumerate(training.columnas_entrada):
                                if col in maximos and maximos[col] != 0:
                                    X_input[0, i] = X_input[0, i] / maximos[col]
                        # Normalización por mean/std (forma antigua, para compatibilidad)
                        elif 'mean' in stats and 'std' in stats:
                            mean = np.array(stats['mean'])
                            std = np.array(stats['std'])
                            X_input = (X_input - mean) / std
                
                # Realizar predicción
                prediccion = rbf.predict(X_input)[0]
                
                # Guardar la predicción en la base de datos
                RBFPrediction.objects.create(
                    entrenamiento=training,
                    valores_entrada=form.cleaned_data,
                    salida_predicha=float(prediccion)
                )
                
                messages.success(request, f'Predicción realizada: {prediccion:.6f}')
                
                context = {
                    'form': form,
                    'training': training,
                    'prediccion': float(prediccion),
                    'valores_entrada': {col: form.cleaned_data[col] for col in training.columnas_entrada}
                }
                return render(request, 'rbf/predecir.html', context)
                
            except Exception as e:
                messages.error(request, f'Error al realizar la predicción: {str(e)}')
    else:
        form = RBFPredictionForm(training.columnas_entrada)
    
    context = {
        'form': form,
        'training': training
    }
    
    return render(request, 'rbf/predecir.html', context)


def historial_predicciones_rbf(request, training_id):
    """
    Vista para mostrar el historial de predicciones de un entrenamiento
    """
    training = get_object_or_404(RBFTraining, id=training_id)
    predicciones = training.predicciones.all().order_by('-fecha_prediccion')
    
    context = {
        'training': training,
        'predicciones': predicciones
    }
    
    return render(request, 'rbf/historial_predicciones.html', context)


def eliminar_rbf(request, training_id):
    """
    Vista para eliminar un entrenamiento
    """
    if request.method == 'POST':
        try:
            training = get_object_or_404(RBFTraining, id=training_id)
            training_name = training.nombre
            training.delete()
            
            messages.success(request, f'Entrenamiento "{training_name}" eliminado exitosamente.')
        except Exception as e:
            messages.error(request, f'Error al eliminar el entrenamiento: {str(e)}')
    
    return redirect('rbf:historial_rbf')


def _regenerar_procesamiento_interno(rbf, X, y):
    """
    Función auxiliar para regenerar el procesamiento interno de la red RBF
    usando datos sintéticos para demostración
    """
    num_patrones, num_caracteristicas = X.shape
    
    # PASO 1: Información de inicialización (usar centros existentes)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    # PASO 2: Calcular distancias
    distancias = rbf._calcular_distancias(X, rbf.centros)
    
    # PASO 3: Calcular activaciones
    activaciones = rbf._calcular_activaciones(distancias)
    
    # PASO 4: Construir matriz de interpolación
    A = rbf._construir_matriz_interpolacion(activaciones)
    
    # PASO 5: Usar pesos existentes
    W = rbf.pesos
    
    # PASO 6: Calcular predicciones
    y_pred = A @ W
    
    # PASO 7: Calcular métricas
    metricas = calcular_metricas(y, y_pred)
    
    return {
        'num_patrones': num_patrones,
        'num_caracteristicas': num_caracteristicas,
        'num_centros': rbf.num_centros,
        
        'paso_1_inicializacion': {
            'min_vals': min_vals.tolist(),
            'max_vals': max_vals.tolist(),
            'rangos': (max_vals - min_vals).tolist(),
            'centros_inicializados': rbf.centros.tolist(),
            'formula': f'rⱼ ~ U[min(xᵢ), max(xᵢ)] para j=1,...,{rbf.num_centros} e i=1,...,n',
            'formula_latex': f'$r_j \\sim U[\\min(x_i), \\max(x_i)]$ para $j=1,...,{rbf.num_centros}$ e $i=1,...,n$'
        },
        
        'paso_2_distancias': {
            'matriz_distancias': distancias.tolist(),
            'dimensiones': f'{distancias.shape[0]} patrones x {distancias.shape[1]} centros',
            'formula': 'D_{pj} = √(∑ᵢ₌₁ᵐ (x_{pi} - r_{ji})²)',
            'formula_latex': '$D_{pj} = \\sqrt{\\sum_{i=1}^{m} (x_{pi} - r_{ji})^2}$',
            'ejemplo_primer_patron': distancias[0].tolist() if distancias.shape[0] > 0 else []
        },
        
        'paso_3_activaciones': {
            'matriz_activaciones': activaciones.tolist(),
            'dimensiones': f'{activaciones.shape[0]} patrones x {activaciones.shape[1]} centros',
            'formula': 'FA(D_{pj}) = D_{pj}² · ln(D_{pj})',
            'formula_latex': '$FA(D_{pj}) = D_{pj}^2 \\cdot \\ln(D_{pj})$',
            'ejemplo_primer_patron': activaciones[0].tolist() if activaciones.shape[0] > 0 else [],
            'valores_min': float(np.min(activaciones)),
            'valores_max': float(np.max(activaciones)),
            'valores_promedio': float(np.mean(activaciones))
        },
        
        'paso_4_matriz_interpolacion': {
            'matriz_A': A.tolist(),
            'dimensiones': f'{A.shape[0]} patrones x {A.shape[1]} columnas (1 umbral + {rbf.num_centros} centros)',
            'formula': 'A = [1, FA(D₁), FA(D₂), ..., FA(Dₖ)]',
            'formula_latex': '$A = [1, FA(D_1), FA(D_2), \\ldots, FA(D_k)]$',
            'primera_columna_umbral': A[:, 0].tolist(),
            'ejemplo_primer_patron': A[0].tolist() if A.shape[0] > 0 else []
        },
        
        'paso_5_calculo_pesos': {
            'vector_pesos': W.tolist(),
            'umbral_W0': float(rbf.W0),
            'pesos_centros': rbf.W1_n.tolist(),
            'formula': 'W = (AᵀA)⁻¹AᵀY',
            'formula_latex': '$W = (A^T A)^{-1} A^T Y$',
            'metodo': 'Mínimos cuadrados',
            'residuos': [0.0],  # No disponible para datos sintéticos
            'rank_matriz': A.shape[1],  # Estimación
            'valores_singulares': []
        },
        
        'paso_6_predicciones': {
            'y_real': y.tolist(),
            'y_predicho': y_pred.tolist(),
            'formula': 'ŷ = A · W',
            'formula_latex': '$\\hat{y} = A \\cdot W$',
            'diferencias': (y - y_pred).tolist(),
            'diferencias_absolutas': np.abs(y - y_pred).tolist()
        },
        
        'paso_7_metricas': metricas
    }


def _generar_graficos_detalles(y_real, y_pred, metricas):
    """
    Función auxiliar para generar gráficos básicos para la vista de detalles
    """
    try:
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Gráfico de dispersión simple
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_real, y_pred, alpha=0.6, s=50, c='blue')
        
        min_val = min(min(y_real), min(y_pred))
        max_val = max(max(y_real), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Diagonal ideal')
        
        ax.set_xlabel('Salida Real', fontsize=11)
        ax.set_ylabel('Salida Predicha', fontsize=11)
        ax.set_title('Dispersión: Y Real vs Y Predicho', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        grafico_dispersion = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'dispersion': grafico_dispersion
        }
        
    except Exception as e:
        print(f"Error al generar gráficos: {e}")
        return None