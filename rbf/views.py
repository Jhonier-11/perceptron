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
                         metricas_train, metricas_test, error_objetivo):
    """
    Genera los tres gráficos requeridos para RBF y retorna Base64
    
    Returns:
        dict: Diccionario con las tres imágenes en Base64
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
    
    return graficos


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
            
            # Aplicar preprocesamiento automático
            info_preprocesamiento, df_procesado, df_comparacion = preprocesar_datos(df, columnas_entrada, columnas_salida)
            
            # Convertir a numérico si es necesario
            if info_preprocesamiento['columnas_codificadas']:
                for item in info_preprocesamiento['columnas_codificadas']:
                    col = item['columna']
                    if col in df_procesado.columns:
                        df_procesado[col] = pd.to_numeric(df_procesado[col], errors='coerce')
            
            # Extraer X e y
            X = df_procesado[columnas_entrada].values
            y = df_procesado[columnas_salida].values.flatten()
            
            # Convertir a float
            X = X.astype(float)
            y = y.astype(float)
            
            # Dividir en entrenamiento y prueba
            porcentaje = training_config['porcentaje_entrenamiento']
            X_train, X_test, y_train, y_test = dividir_entrenamiento_prueba(X, y, porcentaje)
            
            # Decidir si normalizar basado en el análisis de preprocesamiento
            X_train_norm = X_train
            X_test_norm = X_test
            mean_train = None
            std_train = None
            
            if info_preprocesamiento['necesita_normalizacion']:
                # Normalizar datos (solo X, no y)
                X_train_norm, mean_train, std_train = normalizar_datos(X_train)
                X_test_norm, _, _ = normalizar_datos(X_test)
                info_preprocesamiento['normalizado'] = True
            else:
                info_preprocesamiento['normalizado'] = False
            
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
            graficos = generar_graficos_rbf(
                y_train, y_train_pred, y_test, y_test_pred,
                metricas_train, metricas_test, error_objetivo
            )
            
            # Guardar en base de datos
            estadisticas_norm = {
                'mean': mean_train.tolist() if hasattr(mean_train, 'tolist') else mean_train,
                'std': std_train.tolist() if hasattr(std_train, 'tolist') else std_train
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
                estadisticas_normalizacion=estadisticas_norm
            )
            
            # Preparar datos de comparación para la tabla
            datos_comparacion = None
            if not df_comparacion.empty:
                # Convertir a diccionario con solo las columnas relevantes
                columnas_mostrar = columnas_entrada + columnas_salida
                # Agregar columnas transformadas si existen
                for item in info_preprocesamiento.get('datos_comparacion', []):
                    if item['columna_transformada'] in df_comparacion.columns:
                        columnas_mostrar.append(item['columna_transformada'])
                
                df_mostrar = df_comparacion[columnas_mostrar]
                # Convertir a diccionario
                datos_comparacion = df_mostrar.head(20).to_dict('records')  # Solo primeras 20 filas
            
            # Guardar gráficos y resultados en sesión
            request.session['rbf_results'] = {
                'training_id': training_record.id,
                'dispersion_plot': graficos['dispersion'],
                'error_plot': graficos['error'],
                'metricas_plot': graficos['metricas'],
                'metricas_train': metricas_train,
                'metricas_test': metricas_test,
                'convergencia': convergencia,
                'info_preprocesamiento': convertir_a_tipos_nativos(info_preprocesamiento),
                'datos_comparacion': datos_comparacion,
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
        'metricas_train': rbf_results['metricas_train'],
        'metricas_test': rbf_results['metricas_test'],
        'convergencia': rbf_results['convergencia'],
        'info_preprocesamiento': rbf_results.get('info_preprocesamiento', {}),
        'datos_comparacion': rbf_results.get('datos_comparacion', []),
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
        
        # Simular datos de entrada para regenerar el procesamiento interno
        # Usar datos sintéticos basados en las estadísticas guardadas
        num_patrones = 50  # Número de patrones para la demostración
        num_caracteristicas = len(training.columnas_entrada)
        
        # Generar datos sintéticos basados en las estadísticas de normalización
        if training.estadisticas_normalizacion:
            mean_vals = np.array(training.estadisticas_normalizacion['mean'])
            std_vals = np.array(training.estadisticas_normalizacion['std'])
            
            # Generar datos normalizados y luego desnormalizar
            X_sintetico = np.random.normal(0, 1, (num_patrones, num_caracteristicas))
            X_sintetico = X_sintetico * std_vals + mean_vals
        else:
            # Si no hay normalización, generar datos en rango [-5, 5]
            X_sintetico = np.random.uniform(-5, 5, (num_patrones, num_caracteristicas))
        
        # Generar salidas sintéticas usando la red reconstruida
        y_sintetico = rbf.predict(X_sintetico)
        
        # Regenerar el procesamiento interno usando los datos sintéticos
        procesamiento_interno = _regenerar_procesamiento_interno(rbf, X_sintetico, y_sintetico)
        
        # Generar gráficos básicos para la demostración
        graficos = _generar_graficos_detalles(y_sintetico, y_sintetico, procesamiento_interno['paso_7_metricas'])
        
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
    Vista para descargar el modelo entrenado
    """
    training = get_object_or_404(RBFTraining, id=training_id)
    
    # Crear archivo JSON con los datos del modelo
    datos_modelo = {
        'nombre_entrenamiento': training.nombre,
        'fecha_creacion': training.fecha_creacion.isoformat(),
        'num_centros': training.num_centros,
        'columnas_entrada': training.columnas_entrada,
        'columnas_salida': training.columnas_salida,
        'centros_radiales': convertir_a_tipos_nativos(training.centros_radiales),
        'pesos': convertir_a_tipos_nativos(training.pesos_finales),
        'umbral': float(training.umbral),
        'metricas_entrenamiento': training.metricas_entrenamiento,
        'metricas_prueba': training.metricas_prueba,
        'convergencia': training.convergencia,
        'estadisticas_normalizacion': training.estadisticas_normalizacion
    }
    
    response = HttpResponse(
        json.dumps(datos_modelo, indent=2),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="rbf_model_{training.id}.json"'
    
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
                    if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
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
            'formula': f'R_j = random_uniform([min(X), max(X)]) para j=1,...,{rbf.num_centros}'
        },
        
        'paso_2_distancias': {
            'matriz_distancias': distancias.tolist(),
            'dimensiones': f'{distancias.shape[0]} patrones x {distancias.shape[1]} centros',
            'formula': 'D_pj = sqrt(sum((X_p - R_j)^2))',
            'ejemplo_primer_patron': distancias[0].tolist() if distancias.shape[0] > 0 else []
        },
        
        'paso_3_activaciones': {
            'matriz_activaciones': activaciones.tolist(),
            'dimensiones': f'{activaciones.shape[0]} patrones x {activaciones.shape[1]} centros',
            'formula': 'FA(D_pj) = (D_pj)^2 * ln(D_pj)',
            'ejemplo_primer_patron': activaciones[0].tolist() if activaciones.shape[0] > 0 else [],
            'valores_min': float(np.min(activaciones)),
            'valores_max': float(np.max(activaciones)),
            'valores_promedio': float(np.mean(activaciones))
        },
        
        'paso_4_matriz_interpolacion': {
            'matriz_A': A.tolist(),
            'dimensiones': f'{A.shape[0]} patrones x {A.shape[1]} columnas (1 umbral + {rbf.num_centros} centros)',
            'formula': 'A = [1, FA(D_1), FA(D_2), ..., FA(D_k)]',
            'primera_columna_umbral': A[:, 0].tolist(),
            'ejemplo_primer_patron': A[0].tolist() if A.shape[0] > 0 else []
        },
        
        'paso_5_calculo_pesos': {
            'vector_pesos': W.tolist(),
            'umbral_W0': float(rbf.W0),
            'pesos_centros': rbf.W1_n.tolist(),
            'formula': 'W = (A^T A)^(-1) A^T Y',
            'metodo': 'Mínimos cuadrados (pesos restaurados)',
            'residuos': [0.0],  # No disponible para datos sintéticos
            'rank_matriz': A.shape[1],  # Estimación
            'valores_singulares': []
        },
        
        'paso_6_predicciones': {
            'y_real': y.tolist(),
            'y_predicho': y_pred.tolist(),
            'formula': 'Y_pred = A * W',
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