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


def home(request):
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


def upload_data(request):
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
        return redirect('perceptron_app:upload_data')
    
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Procesar el archivo
            file = request.FILES['data_file']
            file_extension = file.name.split('.')[-1].lower()
            
            try:
                # Leer el archivo según su extensión
                if file_extension == 'csv':
                    df = pd.read_csv(file)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(file)
                elif file_extension == 'json':
                    df = pd.read_json(file)
                elif file_extension == 'txt':
                    # Asumir que es CSV con separador de tabulador
                    df = pd.read_csv(file, sep='\t')
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
                
                # Guardar información del archivo en la sesión (sin los datos completos)
                request.session['uploaded_data'] = {
                    'filename': file.name,
                    'columns': df.columns.tolist(),
                    'shape': df.shape,
                    'preview': df.head(10).to_html(classes='table table-striped', table_id='data-preview')
                }
                
                # Guardar la ruta del archivo
                request.session['file_path'] = file_path
                
                
                messages.success(request, f'Archivo cargado exitosamente. {df.shape[0]} filas, {df.shape[1]} columnas.')
                return redirect('perceptron_app:configure_training')
                
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


def configure_training(request):
    """
    Vista para configurar los parámetros de entrenamiento
    """
    if 'uploaded_data' not in request.session:
        messages.warning(request, 'Primero debes cargar un archivo de datos.')
        return redirect('perceptron_app:upload_data')
    
    # No limpiar el archivo aquí, se necesita para el entrenamiento
    
    uploaded_data = request.session['uploaded_data']
    columns = uploaded_data['columns']
    
    if request.method == 'POST':
        form = TrainingConfigForm(request.POST, columns=columns)
        if form.is_valid():
            # Guardar configuración en la sesión
            request.session['training_config'] = {
                'learning_rate': form.cleaned_data['learning_rate'],
                'epochs': form.cleaned_data['epochs'],
                'max_error': form.cleaned_data['max_error'],
                'input_columns': form.cleaned_data['input_columns'],
                'output_columns': form.cleaned_data['output_columns'],
                'training_name': form.cleaned_data['training_name']
            }
            
            messages.success(request, 'Configuración guardada. Listo para entrenar.')
            return redirect('perceptron_app:train_perceptron')
    else:
        form = TrainingConfigForm(columns=columns)
    
    context = {
        'form': form,
        'uploaded_data': uploaded_data,
        'columns': columns
    }
    
    return render(request, 'perceptron_app/configure_training.html', context)


def train_perceptron(request):
    """
    Vista para entrenar el perceptrón
    """
    if 'uploaded_data' not in request.session or 'training_config' not in request.session:
        messages.warning(request, 'Debes cargar datos y configurar el entrenamiento primero.')
        return redirect('perceptron_app:upload_data')
    
    uploaded_data = request.session['uploaded_data']
    training_config = request.session['training_config']
    
    if request.method == 'POST':
        try:
            # Cargar los datos desde el archivo guardado
            file_path = request.session.get('file_path')
            
            if not file_path:
                messages.error(request, 'No se encontró la ruta del archivo en la sesión.')
                return redirect('perceptron_app:upload_data')
                
            if not os.path.exists(file_path):
                messages.error(request, f'El archivo no existe en la ruta: {file_path}')
                return redirect('perceptron_app:upload_data')
            
            # Leer datos desde el archivo
            file_extension = uploaded_data['filename'].split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension == 'xlsx':
                df = pd.read_excel(file_path)
            elif file_extension == 'json':
                df = pd.read_json(file_path)
            elif file_extension == 'txt':
                df = pd.read_csv(file_path, sep='\t')
            
            
            # Preparar datos de entrenamiento
            X = df[training_config['input_columns']].values
            y = df[training_config['output_columns']].values.flatten()
            
            # Crear y entrenar el perceptrón
            perceptron = PerceptronSimple(
                learning_rate=training_config['learning_rate'],
                max_epochs=training_config['epochs'],
                max_error=training_config['max_error']
            )
            
            # Inicializar pesos con el número correcto de características
            perceptron._initialize_weights(X.shape[1])
            
            # Entrenar
            training_results = perceptron.fit(X, y)
            
            # Los datos ya vienen convertidos desde el perceptrón
            final_weights = training_results['final_weights']
            final_bias = training_results['final_bias']
            training_errors = training_results['training_errors']
            weight_evolution = training_results['weight_evolution']
            
            # Guardar en la base de datos
            training_record = PerceptronTraining.objects.create(
                name=training_config['training_name'],
                learning_rate=training_config['learning_rate'],
                epochs=training_config['epochs'],
                max_error=training_config['max_error'],
                input_columns=training_config['input_columns'],
                output_columns=training_config['output_columns'],
                final_weights=final_weights,
                final_bias=final_bias,
                accuracy=training_results['accuracy'],
                training_errors=training_errors,
                weight_evolution=weight_evolution,
                data_file=None  # No guardamos el archivo original en la base de datos
            )
            
            # Generar gráficos
            error_plot = perceptron.create_error_plot()
            weights_plot = perceptron.create_weights_plot()
            
            # Guardar gráficos en la sesión
            request.session['training_results'] = {
                'training_id': training_record.id,
                'error_plot': error_plot,
                'weights_plot': weights_plot,
                'summary': perceptron.get_training_summary(),
                'converged': training_results['converged'],
                'epochs_used': training_results['epochs_used']
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
            return redirect('perceptron_app:training_results')
            
        except Exception as e:
            messages.error(request, f'Error durante el entrenamiento: {str(e)}')
    
    context = {
        'uploaded_data': uploaded_data,
        'training_config': training_config
    }
    
    return render(request, 'perceptron_app/train_perceptron.html', context)


def training_results(request):
    """
    Vista para mostrar los resultados del entrenamiento
    """
    if 'training_results' not in request.session:
        messages.warning(request, 'No hay resultados de entrenamiento disponibles.')
        return redirect('perceptron_app:home')
    
    training_results = request.session['training_results']
    training_record = get_object_or_404(PerceptronTraining, id=training_results['training_id'])
    
    context = {
        'training_record': training_record,
        'error_plot': training_results['error_plot'],
        'weights_plot': training_results['weights_plot'],
        'summary': training_results['summary'],
        'converged': training_results['converged'],
        'epochs_used': training_results['epochs_used']
    }
    
    return render(request, 'perceptron_app/training_results.html', context)


def make_prediction(request, training_id=None):
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
        return redirect('perceptron_app:home')
    
    if request.method == 'POST':
        form = PredictionForm(request.POST, input_columns=training_record.input_columns)
        if form.is_valid():
            try:
                # Obtener valores de entrada
                input_values = []
                for col in training_record.input_columns:
                    input_values.append(form.cleaned_data[col])
                
                # Crear perceptrón con los pesos entrenados
                perceptron = PerceptronSimple()
                perceptron.weights = np.array(training_record.final_weights)
                perceptron.bias = training_record.final_bias
                
                # Hacer predicción
                prediction = perceptron.predict(np.array([input_values]))[0]
                
                # Guardar predicción en la base de datos
                Prediction.objects.create(
                    training=training_record,
                    input_values=input_values,
                    predicted_output=prediction
                )
                
                messages.success(request, f'Predicción realizada: {prediction}')
                
                # Pasar los datos de entrada al contexto para mostrar en el template
                input_data = {col: form.cleaned_data[col] for col in training_record.input_columns}
                context = {
                    'form': form,
                    'training_record': training_record,
                    'previous_predictions': Prediction.objects.filter(training=training_record).order_by('-created_at')[:10],
                    'prediction': prediction,
                    'input_data': input_data
                }
                return render(request, 'perceptron_app/make_prediction.html', context)
                
            except Exception as e:
                messages.error(request, f'Error al hacer la predicción: {str(e)}')
    else:
        form = PredictionForm(input_columns=training_record.input_columns)
    
    # Obtener predicciones anteriores
    previous_predictions = Prediction.objects.filter(training=training_record).order_by('-created_at')[:10]
    
    context = {
        'form': form,
        'training_record': training_record,
        'previous_predictions': previous_predictions
    }
    
    return render(request, 'perceptron_app/make_prediction.html', context)


def training_history(request):
    """
    Vista para mostrar el historial de entrenamientos
    """
    trainings = PerceptronTraining.objects.all().order_by('-created_at')
    
    context = {
        'trainings': trainings
    }
    
    return render(request, 'perceptron_app/training_history.html', context)


def download_weights(request, training_id):
    """
    Vista para descargar los pesos entrenados
    """
    training = get_object_or_404(PerceptronTraining, id=training_id)
    
    # Crear archivo JSON con los pesos
    weights_data = {
        'training_name': training.name,
        'created_at': training.created_at.isoformat(),
        'learning_rate': training.learning_rate,
        'epochs': training.epochs,
        'input_columns': training.input_columns,
        'output_columns': training.output_columns,
        'final_weights': training.final_weights,
        'final_bias': training.final_bias,
        'accuracy': training.accuracy
    }
    
    response = HttpResponse(
        json.dumps(weights_data, indent=2),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="perceptron_weights_{training.id}.json"'
    
    return response


def delete_training(request, training_id):
    """
    Vista para eliminar un entrenamiento del historial
    """
    if request.method == 'POST':
        try:
            training = get_object_or_404(PerceptronTraining, id=training_id)
            training_name = training.name
            
            # Eliminar también las predicciones asociadas
            Prediction.objects.filter(training=training).delete()
            
            # Eliminar el entrenamiento
            training.delete()
            
            messages.success(request, f'Entrenamiento "{training_name}" eliminado exitosamente.')
            
        except Exception as e:
            messages.error(request, f'Error al eliminar el entrenamiento: {str(e)}')
    
    return redirect('perceptron_app:training_history')


@csrf_exempt
@require_http_methods(["POST"])
def ajax_train(request):
    """
    Vista AJAX para entrenar el perceptrón de forma asíncrona
    """
    try:
        data = json.loads(request.body)
        
        # Crear perceptrón
        perceptron = PerceptronSimple(
            learning_rate=data['learning_rate'],
            max_epochs=data['epochs']
        )
        
        # Preparar datos
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        # Entrenar
        results = perceptron.fit(X, y)
        
        return JsonResponse({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })