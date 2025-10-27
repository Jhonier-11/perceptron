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
from .rbf_engine import calcular_eg, calcular_mae, calcular_rmse, verificar_convergencia
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
    Vista para cargar archivos de datos
    """
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
        if form.is_valid():
            # Procesar el archivo
            file = request.FILES['data_file']
            file_extension = file.name.split('.')[-1].lower()
            
            try:
                # Leer el archivo según su extensión
                if file_extension == 'csv':
                    df = detectar_separador_csv_y_leer(file)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(file)
                elif file_extension == 'json':
                    try:
                        df = pd.read_json(file)
                        # Normalizar si es necesario
                        if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
                            first_col = df.iloc[:, 0]
                            if isinstance(first_col.iloc[0], dict):
                                df = pd.json_normalize(df.iloc[:, 0].tolist())
                    except Exception as e:
                        messages.error(request, f'Error al leer el archivo JSON: {str(e)}')
                        return render(request, 'rbf/cargar_datos.html', {'form': form})
                elif file_extension == 'txt':
                    df = detectar_separador_csv_y_leer(file)
                else:
                    messages.error(request, 'Formato de archivo no soportado.')
                    return render(request, 'rbf/cargar_datos.html', {'form': form})
                
                # Validar mínimo de patrones
                if len(df) < 10:
                    messages.error(request, 'El dataset debe tener al menos 10 patrones.')
                    return render(request, 'rbf/cargar_datos.html', {'form': form})
                
                # Guardar el archivo físicamente
                file.seek(0)
                unique_filename = f"{uuid.uuid4()}_{file.name}"
                file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)
                
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                
                # Guardar información en la sesión
                uploaded_data = {
                    'filename': file.name,
                    'columns': df.columns.tolist(),
                    'shape': list(df.shape),
                }
                
                uploaded_data = convertir_a_tipos_nativos(uploaded_data)
                request.session['uploaded_data_rbf'] = uploaded_data
                request.session['file_path_rbf'] = file_path
                
                messages.success(request, f'Archivo cargado exitosamente. {df.shape[0]} filas, {df.shape[1]} columnas.')
                return redirect('rbf:configurar_rbf')
                
            except Exception as e:
                messages.error(request, f'Error al procesar el archivo: {str(e)}')
    else:
        form = RBFDataUploadForm()
    
    uploaded_data = request.session.get('uploaded_data_rbf', None)
    
    context = {
        'form': form,
        'uploaded_data': uploaded_data
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
            
            # Calcular métricas
            metricas_train = {
                'EG': calcular_eg(y_train, y_train_pred),
                'MAE': calcular_mae(y_train, y_train_pred),
                'RMSE': calcular_rmse(y_train, y_train_pred)
            }
            
            metricas_test = {
                'EG': calcular_eg(y_test, y_test_pred),
                'MAE': calcular_mae(y_test, y_test_pred),
                'RMSE': calcular_rmse(y_test, y_test_pred)
            }
            
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
                'columnas_salida': columnas_salida
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
        'columnas_salida': rbf_results.get('columnas_salida', [])
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
    
    # Regenerar gráficos si es necesario
    # (Para ahora, guardamos información básica)
    
    context = {
        'training': training
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