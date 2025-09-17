"""
Vistas para la aplicación del perceptrón simple
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
from .models import PerceptronTraining, Prediction
from .perceptron import PerceptronSimple
from .forms import DataUploadForm, TrainingConfigForm, PredictionForm


def generar_tabla_html_vista_previa(df, max_rows=20):
    """
    Genera una tabla HTML bien estructurada para la vista previa de datos
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        max_rows (int): Número máximo de filas a mostrar
    
    Returns:
        str: HTML de la tabla con estructura estándar
    """
    # Limitar el número de filas
    preview_df = df.head(max_rows)
    
    # Crear la estructura HTML de la tabla
    html = """
    <div class="table-responsive" style="max-height: 400px; overflow-y: auto; overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px;">
        <table class="tabla-vista-previa" style="width: 100%; border-collapse: collapse; margin: 0; font-size: 0.875rem;">
            <thead style="background-color: #374151; color: white; position: sticky; top: 0; z-index: 10;">
                <tr>
    """
    
    # Agregar encabezados
    for col in preview_df.columns:
        html += f'<th style="padding: 12px 8px; text-align: center; border: 1px solid #4b5563; font-weight: 600;">{col}</th>'
    
    html += """
                </tr>
            </thead>
            <tbody>
    """
    
    # Agregar filas de datos
    for index, row in preview_df.iterrows():
        # Alternar colores de fila
        bg_color = '#f9fafb' if index % 2 == 0 else 'white'
        html += f'<tr style="background-color: {bg_color};" onmouseover="this.style.backgroundColor=\'#f3f4f6\'" onmouseout="this.style.backgroundColor=\'{bg_color}\'">'
        
        for value in row:
            # Formatear valores según su tipo
            if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                # Valores numéricos: alineados a la derecha, con formato
                formatted_value = f'{value:.4f}' if isinstance(value, float) else str(value)
                html += f'<td style="padding: 8px; border: 1px solid #e5e7eb; text-align: right; font-family: monospace; white-space: nowrap; max-width: 120px; overflow: hidden; text-overflow: ellipsis;">{formatted_value}</td>'
            else:
                # Valores de texto: centrados
                html += f'<td style="padding: 8px; border: 1px solid #e5e7eb; text-align: center; white-space: nowrap; max-width: 120px; overflow: hidden; text-overflow: ellipsis;">{value}</td>'
        
        html += '</tr>'
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


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


def inicio(request):
    """
    Vista principal que muestra el dashboard
    """
    # Obtener los últimos entrenamientos
    recent_trainings = PerceptronTraining.objects.all()[:5]
    
    context = {
        'recent_trainings': recent_trainings,
        'total_trainings': PerceptronTraining.objects.count(),
        'total_predictions': Prediction.objects.count(),
    }
    
    return render(request, 'perceptron_app/home.html', context)


def cargar_datos(request):
    """
    Vista para cargar archivos de datos
    """
    # Limpiar sesión si se solicita
    if request.GET.get('clear') == 'true':
        if 'uploaded_data' in request.session:
            del request.session['uploaded_data']
        if 'training_config' in request.session:
            del request.session['training_config']
        if 'file_path' in request.session:
            try:
                file_path = request.session['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error al eliminar archivo: {e}")
            del request.session['file_path']
        messages.info(request, 'Sesión limpiada. Puedes cargar un nuevo archivo.')
        return redirect('perceptron_app:cargar_datos')
    
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Procesar el archivo
            file = request.FILES['data_file']
            file_extension = file.name.split('.')[-1].lower()
            
            try:
                # Leer el archivo según su extensión
                if file_extension == 'csv':
                    # Detectar automáticamente el separador del CSV
                    df = detectar_separador_csv_y_leer(file)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(file)
                elif file_extension == 'json':
                    df = pd.read_json(file)
                elif file_extension == 'txt':
                    # Detectar automáticamente el separador del archivo de texto
                    df = detectar_separador_csv_y_leer(file)
                else:
                    messages.error(request, 'Formato de archivo no soportado.')
                    return render(request, 'perceptron_app/upload_data.html', {'form': form})
                
                # Guardar el archivo físicamente con nombre único
                file.seek(0)  # Resetear el puntero del archivo
                unique_filename = f"{uuid.uuid4()}_{file.name}"
                file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)
                
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Guardar archivo
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                
                # Analizar tipos de datos de las columnas
                column_info = []
                for col in df.columns:
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    column_info.append({
                        'name': col,
                        'is_numeric': is_numeric,
                        'sample_values': df[col].head(3).tolist()
                    })
                
                # Crear vista previa mejorada con más filas y estadísticas
                preview_rows = min(20, len(df))  # Mostrar hasta 20 filas
                preview_df = df.head(preview_rows)

                # Agregar estadísticas básicas
                stats_info = []
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats = {
                            'name': col,
                            'count': df[col].count(),
                            'mean': round(df[col].mean(), 3) if not df[col].isna().all() else None,
                            'std': round(df[col].std(), 3) if not df[col].isna().all() else None,
                            'min': round(df[col].min(), 3) if not df[col].isna().all() else None,
                            'max': round(df[col].max(), 3) if not df[col].isna().all() else None,
                            'missing': df[col].isna().sum()
                        }
                    else:
                        stats = {
                            'name': col,
                            'count': df[col].count(),
                            'unique': df[col].nunique(),
                            'missing': df[col].isna().sum(),
                            'most_common': df[col].mode().iloc[0] if not df[col].isna().all() else None
                        }
                    stats_info.append(stats)

                # Crear vista previa estructurada para el template
                preview_rows = min(20, len(df))
                preview_df = df.head(preview_rows)
                
                # Convertir los datos a estructura que Django pueda iterar
                preview_data = []
                print(f"DEBUG: preview_df shape: {preview_df.shape}")
                print(f"DEBUG: preview_df columns: {preview_df.columns.tolist()}")
                
                for index, row in preview_df.iterrows():
                    row_data = []
                    for col in preview_df.columns:
                        value = row[col]
                        # Determinar el tipo y formatear el valor
                        if pd.api.types.is_numeric_dtype(type(value)) and not pd.isna(value):
                            formatted_value = f'{value:.4f}' if isinstance(value, float) else str(value)
                            row_data.append({
                                'value': formatted_value,
                                'is_numeric': True,
                                'raw_value': value
                            })
                        else:
                            row_data.append({
                                'value': str(value) if not pd.isna(value) else 'N/A',
                                'is_numeric': False,
                                'raw_value': value
                            })
                    preview_data.append(row_data)
                    
                print(f"DEBUG: preview_data length: {len(preview_data)}")
                if preview_data:
                    print(f"DEBUG: first row length: {len(preview_data[0])}")
                    print(f"DEBUG: first row sample: {preview_data[0][:3] if len(preview_data[0]) > 0 else 'empty'}")

                # Guardar información del archivo en la sesión (sin los datos completos)
                request.session['uploaded_data'] = {
                    'filename': file.name,
                    'columns': df.columns.tolist(),
                    'shape': df.shape,
                    'preview_data': preview_data,
                    'column_info': column_info,
                    'stats_info': stats_info,
                    'preview_rows': preview_rows
                }
                
                # Guardar la ruta del archivo
                request.session['file_path'] = file_path
                
                
                messages.success(request, f'Archivo cargado exitosamente. {df.shape[0]} filas, {df.shape[1]} columnas.')
                return redirect('perceptron_app:configurar_entrenamiento')
                
            except Exception as e:
                messages.error(request, f'Error al procesar el archivo: {str(e)}')
    else:
        form = DataUploadForm()
    
    # Verificar si hay datos cargados en la sesión
    uploaded_data = request.session.get('uploaded_data', None)
    
    context = {
        'form': form,
        'uploaded_data': uploaded_data
    }
    
    return render(request, 'perceptron_app/upload_data.html', context)


def configurar_entrenamiento(request):
    """
    Vista para configurar los parámetros de entrenamiento
    """
    if 'uploaded_data' not in request.session:
        messages.warning(request, 'Primero debes cargar un archivo de datos.')
        return redirect('perceptron_app:cargar_datos')
    
    # No limpiar el archivo aquí, se necesita para el entrenamiento
    
    uploaded_data = request.session['uploaded_data']
    columns = uploaded_data['columns']
    
    if request.method == 'POST':
        form = TrainingConfigForm(request.POST, request.FILES, columns=columns)
        if form.is_valid():
            # Procesar archivo de pesos si se seleccionó esa opción
            pesos_iniciales = None
            sesgo_inicial = None

            if form.cleaned_data['weight_initialization'] == 'file':
                weights_file = form.cleaned_data['weights_file']
                try:
                    # Leer y parsear el archivo JSON
                    weights_data = json.load(weights_file)
                    pesos_iniciales = weights_data.get('weights', [])
                    sesgo_inicial = weights_data.get('bias', 0.0)

                    # Validar estructura del archivo
                    if not isinstance(pesos_iniciales, list):
                        raise ValueError("Los pesos deben ser una lista de números.")

                    if not all(isinstance(w, (int, float)) for w in pesos_iniciales):
                        raise ValueError("Todos los pesos deben ser números.")

                    messages.success(request, f'Pesos cargados exitosamente: {len(pesos_iniciales)} pesos, sesgo: {sesgo_inicial}')

                except json.JSONDecodeError:
                    messages.error(request, 'El archivo de pesos no tiene un formato JSON válido.')
                    return render(request, 'perceptron_app/configure_training.html', {'form': form, 'uploaded_data': uploaded_data, 'columns': columns})
                except Exception as e:
                    messages.error(request, f'Error al procesar el archivo de pesos: {str(e)}')
                    return render(request, 'perceptron_app/configure_training.html', {'form': form, 'uploaded_data': uploaded_data, 'columns': columns})

            # Guardar configuración en la sesión
            request.session['training_config'] = {
                'tasa_aprendizaje': form.cleaned_data['learning_rate'],
                'iteraciones': form.cleaned_data['epochs'],
                'error_maximo': form.cleaned_data['max_error'],
                'columnas_entrada': form.cleaned_data['input_columns'],
                'columnas_salida': form.cleaned_data['output_columns'],
                'nombre_entrenamiento': form.cleaned_data['training_name'],
                'weight_initialization': form.cleaned_data['weight_initialization'],
                'pesos_iniciales': pesos_iniciales,
                'sesgo_inicial': sesgo_inicial
            }

            messages.success(request, 'Configuración guardada. Listo para entrenar.')
            return redirect('perceptron_app:entrenar_perceptron')
    else:
        form = TrainingConfigForm(columns=columns)
    
    context = {
        'form': form,
        'uploaded_data': uploaded_data,
        'columns': columns
    }
    
    return render(request, 'perceptron_app/configure_training.html', context)


def entrenar_perceptron(request):
    """
    Vista para entrenar el perceptrón
    """
    if 'uploaded_data' not in request.session or 'training_config' not in request.session:
        messages.warning(request, 'Debes cargar datos y configurar el entrenamiento primero.')
        return redirect('perceptron_app:cargar_datos')
    
    uploaded_data = request.session['uploaded_data']
    training_config = request.session['training_config']
    
    if request.method == 'POST':
        try:
            # Cargar los datos desde el archivo guardado
            file_path = request.session.get('file_path')
            
            if not file_path:
                messages.error(request, 'No se encontró la ruta del archivo en la sesión.')
                return redirect('perceptron_app:cargar_datos')
                
            if not os.path.exists(file_path):
                messages.error(request, f'El archivo no existe en la ruta: {file_path}')
                return redirect('perceptron_app:cargar_datos')
            
            # Leer datos desde el archivo
            file_extension = uploaded_data['filename'].split('.')[-1].lower()
            if file_extension == 'csv':
                # Usar detección automática de separador
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        df = detectar_separador_csv_y_leer(f)
                except UnicodeDecodeError:
                    # Si falla con UTF-8, intentar con latin-1
                    with open(file_path, 'r', encoding='latin-1') as f:
                        df = detectar_separador_csv_y_leer(f)
            elif file_extension == 'xlsx':
                df = pd.read_excel(file_path)
            elif file_extension == 'json':
                df = pd.read_json(file_path)
            elif file_extension == 'txt':
                # Usar detección automática de separador
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        df = detectar_separador_csv_y_leer(f)
                except UnicodeDecodeError:
                    # Si falla con UTF-8, intentar con latin-1
                    with open(file_path, 'r', encoding='latin-1') as f:
                        df = detectar_separador_csv_y_leer(f)
            
            
            # Preparar datos de entrenamiento
            # Verificar que las columnas seleccionadas contengan solo datos numéricos
            columnas_entrada = training_config['columnas_entrada']
            columnas_salida = training_config['columnas_salida']
            
            print(f"DEBUG - Configuración de entrenamiento:")
            print(f"  columnas_entrada: {columnas_entrada}")
            print(f"  columnas_salida: {columnas_salida}")
            print(f"  tipo de columnas_entrada: {type(columnas_entrada)}")
            print(f"  longitud de columnas_entrada: {len(columnas_entrada) if columnas_entrada else 'None'}")
            
            # Verificar columnas de entrada
            for col in columnas_entrada:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    messages.error(request, f'La columna "{col}" contiene datos no numéricos. El perceptrón solo puede trabajar con datos numéricos.')
                    return redirect('perceptron_app:configurar_entrenamiento')
            
            # Verificar columnas de salida
            for col in columnas_salida:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    messages.error(request, f'La columna "{col}" contiene datos no numéricos. El perceptrón solo puede trabajar con datos numéricos.')
                    return redirect('perceptron_app:configurar_entrenamiento')
            
            # Convertir a arrays numéricos
            X = df[columnas_entrada].values.astype(float)
            y = df[columnas_salida].values.flatten().astype(float)
            
            # Debug: verificar dimensiones de los datos
            print(f"Columnas de entrada seleccionadas: {columnas_entrada}")
            print(f"Columnas de salida seleccionadas: {columnas_salida}")
            print(f"Shape de X (entrada): {X.shape}")
            print(f"Shape de y (salida): {y.shape}")
            print(f"Primera fila de X: {X[0] if len(X) > 0 else 'X vacío'}")
            
            # Crear y entrenar el perceptrón
            print("Creando nueva instancia del perceptrón...")

            # Preparar pesos iniciales si se cargaron desde archivo
            pesos_iniciales = training_config.get('pesos_iniciales')
            sesgo_inicial = training_config.get('sesgo_inicial')

            perceptron = PerceptronSimple(
                tasa_aprendizaje=training_config['tasa_aprendizaje'],
                max_iteraciones=training_config['iteraciones'],
                error_maximo=training_config['error_maximo'],
                pesos_iniciales=pesos_iniciales,
                sesgo_inicial=sesgo_inicial
            )
            
            print(f"Perceptrón creado - pesos: {perceptron.pesos}")
            print(f"Perceptrón creado - longitud de pesos: {len(perceptron.pesos) if perceptron.pesos is not None else 'None'}")
            
            # Entrenar (los pesos se inicializarán automáticamente en la función entrenar)
            training_results = perceptron.entrenar(X, y)
            
            # Los datos ya vienen convertidos desde el perceptrón
            pesos_finales = training_results['pesos_finales']
            sesgo_final = training_results['sesgo_final']
            errores_entrenamiento = training_results['errores_entrenamiento']
            evolucion_pesos = training_results['evolucion_pesos']
            
            # Guardar en la base de datos
            training_record = PerceptronTraining.objects.create(
                nombre=training_config['nombre_entrenamiento'],
                tasa_aprendizaje=training_config['tasa_aprendizaje'],
                iteraciones=training_config['iteraciones'],
                error_maximo=training_config['error_maximo'],
                columnas_entrada=training_config['columnas_entrada'],
                columnas_salida=training_config['columnas_salida'],
                pesos_finales=pesos_finales,
                sesgo_final=sesgo_final,
                precision=training_results['precision'],
                errores_entrenamiento=errores_entrenamiento,
                evolucion_pesos=evolucion_pesos,
                archivo_datos=None  # No guardamos el archivo original en la base de datos
            )
            
            # Generar gráficos
            error_plot = perceptron.crear_grafico_errores()
            weights_plot = perceptron.crear_grafico_pesos()
            
            # Guardar gráficos en la sesión
            request.session['training_results'] = {
                'training_id': training_record.id,
                'error_plot': error_plot,
                'weights_plot': weights_plot,
                'summary': perceptron.obtener_resumen_entrenamiento(),
                'converged': training_results['converged'],
                'iteraciones_utilizadas': training_results['iteraciones_utilizadas']
            }
            
            # Limpiar archivo temporal después del entrenamiento
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error al eliminar archivo temporal: {e}")
            
            # Limpiar datos de la sesión
            if 'file_path' in request.session:
                del request.session['file_path']
            
            messages.success(request, '¡Entrenamiento completado exitosamente!')
            return redirect('perceptron_app:resultados_entrenamiento')
            
        except Exception as e:
            messages.error(request, f'Error durante el entrenamiento: {str(e)}')
    
    context = {
        'uploaded_data': uploaded_data,
        'training_config': training_config
    }
    
    return render(request, 'perceptron_app/train_perceptron.html', context)


def resultados_entrenamiento(request):
    """
    Vista para mostrar los resultados del entrenamiento
    """
    if 'training_results' not in request.session:
        messages.warning(request, 'No hay resultados de entrenamiento disponibles.')
        return redirect('perceptron_app:inicio')
    
    training_results = request.session['training_results']
    training_record = get_object_or_404(PerceptronTraining, id=training_results['training_id'])
    
    context = {
        'training_record': training_record,
        'error_plot': training_results['error_plot'],
        'weights_plot': training_results['weights_plot'],
        'summary': training_results['summary'],
        'converged': training_results['converged'],
        'epochs_used': training_results['iteraciones_utilizadas']
    }
    
    return render(request, 'perceptron_app/training_results.html', context)


def hacer_prediccion(request, training_id=None):
    """
    Vista para hacer predicciones con el perceptrón entrenado
    """
    # Si se proporciona un training_id específico, usarlo; si no, usar el de la sesión
    if training_id:
        training_record = get_object_or_404(PerceptronTraining, id=training_id)
    elif 'training_results' in request.session:
        training_results = request.session['training_results']
        training_record = get_object_or_404(PerceptronTraining, id=training_results['training_id'])
    else:
        messages.warning(request, 'Debes entrenar un perceptrón primero o seleccionar uno del historial.')
        return redirect('perceptron_app:inicio')
    
    if request.method == 'POST':
        form = PredictionForm(request.POST, input_columns=training_record.columnas_entrada)
        if form.is_valid():
            try:
                # Obtener valores de entrada
                valores_entrada = []
                for col in training_record.columnas_entrada:
                    valores_entrada.append(form.cleaned_data[col])
                
                # Crear perceptrón con los pesos entrenados
                perceptron = PerceptronSimple()
                perceptron.pesos = np.array(training_record.pesos_finales)
                perceptron.sesgo = training_record.sesgo_final
                
                # Hacer predicción
                prediccion = perceptron.predecir(np.array([valores_entrada], dtype=float))[0]
                
                # Guardar predicción en la base de datos
                Prediction.objects.create(
                    entrenamiento=training_record,
                    valores_entrada=valores_entrada,
                    salida_predicha=prediccion
                )
                
                messages.success(request, f'Predicción realizada: {prediccion}')
                
                # Pasar los datos de entrada al contexto para mostrar en el template
                datos_entrada = {col: form.cleaned_data[col] for col in training_record.columnas_entrada}
                context = {
                    'form': form,
                    'training_record': training_record,
                    'previous_predictions': Prediction.objects.filter(entrenamiento=training_record).order_by('-fecha_prediccion')[:10],
                    'prediction': prediccion,
                    'input_data': datos_entrada
                }
                return render(request, 'perceptron_app/make_prediction.html', context)
                
            except Exception as e:
                messages.error(request, f'Error al hacer la predicción: {str(e)}')
    else:
        form = PredictionForm(input_columns=training_record.columnas_entrada)
    
    # Obtener predicciones anteriores
    previous_predictions = Prediction.objects.filter(entrenamiento=training_record).order_by('-fecha_prediccion')[:10]
    
    context = {
        'form': form,
        'training_record': training_record,
        'previous_predictions': previous_predictions
    }
    
    return render(request, 'perceptron_app/make_prediction.html', context)


def historial_entrenamientos(request):
    """
    Vista para mostrar el historial de entrenamientos
    """
    entrenamientos = PerceptronTraining.objects.all().order_by('-fecha_creacion')
    
    context = {
        'trainings': entrenamientos
    }
    
    return render(request, 'perceptron_app/training_history.html', context)


def descargar_pesos(request, training_id):
    """
    Vista para descargar los pesos entrenados
    """
    training = get_object_or_404(PerceptronTraining, id=training_id)
    
    # Crear archivo JSON con los pesos
    datos_pesos = {
        'training_name': training.nombre,
        'created_at': training.fecha_creacion.isoformat(),
        'learning_rate': training.tasa_aprendizaje,
        'iteraciones': training.iteraciones,
        'input_columns': training.columnas_entrada,
        'output_columns': training.columnas_salida,
        'final_weights': training.pesos_finales,
        'final_bias': training.sesgo_final,
        'accuracy': training.precision
    }
    
    response = HttpResponse(
        json.dumps(datos_pesos, indent=2),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="perceptron_weights_{training.id}.json"'
    
    return response


def detalles_entrenamiento(request, training_id):
    """
    Vista para mostrar detalles completos de un entrenamiento con visualizaciones
    """
    training = get_object_or_404(PerceptronTraining, id=training_id)

    # Crear perceptrón con los pesos entrenados para generar visualizaciones
    perceptron = PerceptronSimple()
    perceptron.pesos = np.array(training.pesos_finales)
    perceptron.sesgo = training.sesgo_final
    perceptron.errores_entrenamiento = training.errores_entrenamiento
    perceptron.evolucion_pesos = training.evolucion_pesos

    # Generar visualizaciones
    error_plot = perceptron.crear_grafico_errores()
    weights_plot = perceptron.crear_grafico_pesos()
    network_diagram = perceptron.crear_diagrama_red()

    # Calcular estadísticas adicionales
    total_iterations = len(training.errores_entrenamiento)
    final_error = training.errores_entrenamiento[-1] if training.errores_entrenamiento else 0
    convergence_epoch = None

    if training.errores_entrenamiento:
        for i, error in enumerate(training.errores_entrenamiento):
            if error == 0:
                convergence_epoch = i + 1
                break

    # Preparar datos para gráficos de precisión
    accuracy_data = []
    if training.evolucion_pesos:
        for i, epoch_data in enumerate(training.evolucion_pesos):
            # Simular precisión basada en errores (esto es una aproximación)
            # En un escenario real, tendrías que recalcular con los datos originales
            error_rate = training.errores_entrenamiento[i] / len(training.columnas_entrada) if i < len(training.errores_entrenamiento) else 0
            accuracy = max(0, (1 - error_rate) * 100)
            accuracy_data.append({
                'epoch': epoch_data['iteracion'],
                'accuracy': round(accuracy, 2)
            })

    context = {
        'training': training,
        'error_plot': error_plot,
        'weights_plot': weights_plot,
        'network_diagram': network_diagram,
        'total_iterations': total_iterations,
        'final_error': final_error,
        'convergence_epoch': convergence_epoch,
        'accuracy_data': accuracy_data,
        'input_features': training.columnas_entrada,
        'output_feature': training.columnas_salida[0] if training.columnas_salida else None,
    }

    return render(request, 'perceptron_app/training_details.html', context)


def eliminar_entrenamiento(request, training_id):
    """
    Vista para eliminar un entrenamiento del historial
    """
    if request.method == 'POST':
        try:
            training = get_object_or_404(PerceptronTraining, id=training_id)
            training_name = training.nombre

            # Eliminar también las predicciones asociadas
            Prediction.objects.filter(entrenamiento=training).delete()

            # Eliminar el entrenamiento
            training.delete()

            messages.success(request, f'Entrenamiento "{training_name}" eliminado exitosamente.')

        except Exception as e:
            messages.error(request, f'Error al eliminar el entrenamiento: {str(e)}')

    return redirect('perceptron_app:historial_entrenamientos')


@csrf_exempt
@require_http_methods(["POST"])
def ajax_entrenar(request):
    """
    Vista AJAX para entrenar el perceptrón de forma asíncrona
    """
    try:
        data = json.loads(request.body)
        
        # Crear perceptrón
        perceptron = PerceptronSimple(
            tasa_aprendizaje=data['learning_rate'],
            max_iteraciones=data['iteraciones']
        )
        
        # Preparar datos
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        # Entrenar
        results = perceptron.entrenar(X, y)
        
        return JsonResponse({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })