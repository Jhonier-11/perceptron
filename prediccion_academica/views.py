"""
Vistas para la aplicación de Predicción del Rendimiento Académico
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.core.paginator import Paginator
from django.db.models import Q, Count, Avg, Max, Min
import traceback
import json
import pandas as pd
import numpy as np
import os
import uuid
from typing import Dict, Any

from .models import Estudiante, EntrenamientoMLP, PrediccionRendimiento, AlertaEstudiante
from .forms import CargaEstudiantesForm, EstudianteForm, ConfiguracionEntrenamientoMLPForm, PrediccionForm
from .mlp_engine import MLP
try:
    from .mlp_tensorflow import MLPTensorFlow
    TENSORFLOW_AVAILABLE = True
except ImportError:
    MLPTensorFlow = None
    TENSORFLOW_AVAILABLE = False
from .alertas import generar_alertas_estudiante, generar_alertas_todos_estudiantes
from .utils import (
    convertir_a_tipos_nativos,
    leer_csv_auto,
    preprocesar_datos_estudiantes,
    convertir_estudiante_a_caracteristicas
)
import os
from django.conf import settings


def dashboard(request):
    """
    Vista principal del dashboard
    """
    # Estadísticas generales
    total_estudiantes = Estudiante.objects.count()
    total_entrenamientos = EntrenamientoMLP.objects.count()
    total_predicciones = PrediccionRendimiento.objects.count()
    total_alertas = AlertaEstudiante.objects.filter(vista=False).count()
    
    # Estudiantes recientes
    estudiantes_recientes = Estudiante.objects.all()[:5]
    
    # Entrenamientos recientes
    entrenamientos_recientes = EntrenamientoMLP.objects.all()[:5]
    
    # Predicciones recientes
    predicciones_recientes = PrediccionRendimiento.objects.select_related(
        'estudiante', 'entrenamiento'
    ).all()[:10]
    
    # Alertas recientes
    alertas_recientes = AlertaEstudiante.objects.filter(vista=False).select_related(
        'estudiante'
    ).order_by('-prioridad', '-fecha_creacion')[:5]
    
    # Estadísticas de rendimiento
    estudiantes_con_calificaciones = Estudiante.objects.exclude(
        calificacion_g3__isnull=True
    )
    
    promedio_g3 = estudiantes_con_calificaciones.aggregate(Avg('calificacion_g3'))['calificacion_g3__avg']
    promedio_g3 = round(promedio_g3, 2) if promedio_g3 else None
    
    estudiantes_riesgo = estudiantes_con_calificaciones.filter(
        calificacion_g3__lt=10
    ).count()
    
    context = {
        'total_estudiantes': total_estudiantes,
        'total_entrenamientos': total_entrenamientos,
        'total_predicciones': total_predicciones,
        'total_alertas': total_alertas,
        'estudiantes_recientes': estudiantes_recientes,
        'entrenamientos_recientes': entrenamientos_recientes,
        'predicciones_recientes': predicciones_recientes,
        'alertas_recientes': alertas_recientes,
        'promedio_g3': promedio_g3,
        'estudiantes_riesgo': estudiantes_riesgo,
    }
    
    return render(request, 'prediccion_academica/dashboard.html', context)


def gestionar_estudiantes(request):
    """
    Vista para gestionar estudiantes (lista, crear, editar, eliminar)
    """
    # Búsqueda y filtrado
    busqueda = request.GET.get('busqueda', '')
    estudiantes = Estudiante.objects.all()
    
    if busqueda:
        estudiantes = estudiantes.filter(
            Q(nombre__icontains=busqueda) |
            Q(apellido__icontains=busqueda) |
            Q(identificacion__icontains=busqueda)
        )
    
    # Paginación
    paginador = Paginator(estudiantes, 20)
    pagina = request.GET.get('pagina', 1)
    estudiantes_pagina = paginador.get_page(pagina)
    
    if request.method == 'POST':
        if 'eliminar' in request.POST:
            estudiante_id = request.POST.get('estudiante_id')
            estudiante = get_object_or_404(Estudiante, id=estudiante_id)
            nombre = estudiante.get_nombre_completo()
            estudiante.delete()
            messages.success(request, f'Estudiante {nombre} eliminado exitosamente.')
            return redirect('prediccion_academica:estudiantes')
    
    context = {
        'estudiantes': estudiantes_pagina,
        'busqueda': busqueda,
    }
    
    return render(request, 'prediccion_academica/estudiantes.html', context)


def crear_estudiante(request):
    """
    Vista para crear un nuevo estudiante
    """
    if request.method == 'POST':
        form = EstudianteForm(request.POST)
        if form.is_valid():
            estudiante = form.save()
            messages.success(request, f'Estudiante {estudiante.get_nombre_completo()} creado exitosamente.')
            return redirect('prediccion_academica:estudiantes')
    else:
        form = EstudianteForm()
    
    context = {
        'form': form,
        'titulo': 'Crear Estudiante'
    }
    
    return render(request, 'prediccion_academica/estudiante_form.html', context)


def editar_estudiante(request, estudiante_id):
    """
    Vista para editar un estudiante existente
    """
    estudiante = get_object_or_404(Estudiante, id=estudiante_id)
    
    if request.method == 'POST':
        form = EstudianteForm(request.POST, instance=estudiante)
        if form.is_valid():
            estudiante = form.save()
            messages.success(request, f'Estudiante {estudiante.get_nombre_completo()} actualizado exitosamente.')
            return redirect('prediccion_academica:estudiante', estudiante_id=estudiante_id)
    else:
        form = EstudianteForm(instance=estudiante)
    
    context = {
        'form': form,
        'estudiante': estudiante,
        'titulo': 'Editar Estudiante'
    }
    
    return render(request, 'prediccion_academica/estudiante_form.html', context)


def vista_estudiante(request, estudiante_id):
    """
    Vista detallada de un estudiante
    """
    estudiante = get_object_or_404(Estudiante, id=estudiante_id)
    
    # Predicciones del estudiante
    predicciones = PrediccionRendimiento.objects.filter(
        estudiante=estudiante
    ).select_related('entrenamiento').order_by('-fecha_prediccion')
    
    # Alertas del estudiante
    alertas = AlertaEstudiante.objects.filter(
        estudiante=estudiante
    ).order_by('-prioridad', '-fecha_creacion')
    
    # Promedio de calificaciones
    promedio = estudiante.get_promedio()
    
    context = {
        'estudiante': estudiante,
        'predicciones': predicciones,
        'alertas': alertas,
        'promedio': promedio,
    }
    
    return render(request, 'prediccion_academica/estudiante.html', context)


def cargar_estudiantes(request):
    """
    Vista para cargar estudiantes desde un archivo CSV
    """
    if request.method == 'POST':
        form = CargaEstudiantesForm(request.POST, request.FILES)
        if form.is_valid():
            archivo = request.FILES['archivo']
            extension = archivo.name.split('.')[-1].lower()
            
            try:
                # Leer el archivo
                if extension == 'csv':
                    df = leer_csv_auto(archivo)
                elif extension == 'xlsx':
                    df = pd.read_excel(archivo)
                elif extension == 'json':
                    df = pd.read_json(archivo)
                else:
                    messages.error(request, 'Formato de archivo no soportado.')
                    return render(request, 'prediccion_academica/cargar_estudiantes.html', {'form': form})
                
                # Mapear columnas del CSV a campos del modelo
                # El CSV tiene columnas en minúsculas y con nombres específicos
                def convertir_valor_booleano(valor):
                    """Convierte valores yes/no a booleanos"""
                    if isinstance(valor, str):
                        valor_limpio = valor.strip().lower().strip('"').strip("'")
                        return valor_limpio in ['yes', 'si', '1', 'true', 'sí', 'y']
                    return bool(valor) if valor else False
                
                def convertir_valor_numerico(valor):
                    """Convierte valores a numéricos"""
                    if pd.isna(valor) or valor == '':
                        return None
                    try:
                        # Si es string, limpiarlo de comillas
                        if isinstance(valor, str):
                            valor = valor.strip().strip('"').strip("'")
                        return float(valor)
                    except:
                        return None
                
                # Crear estudiantes
                estudiantes_creados = 0
                estudiantes_actualizados = 0
                errores = []
                
                for index, row in df.iterrows():
                    try:
                        # Generar identificación única
                        identificacion = f"EST{index + 1:04d}"
                        
                        # Verificar si el estudiante ya existe
                        estudiante, creado = Estudiante.objects.get_or_create(
                            identificacion=identificacion,
                            defaults={
                                'nombre': f'Estudiante {index + 1}',
                                'apellido': '',
                            }
                        )
                        
                        # Mapear campos del CSV al modelo
                        if 'sex' in row or 'Sex' in row:
                            sexo_val = row.get('sex', row.get('Sex', 'M'))
                            if isinstance(sexo_val, str):
                                sexo_val = sexo_val.strip().strip('"').strip("'").upper()
                                estudiante.sexo = 'M' if sexo_val == 'M' else 'F'
                        
                        if 'age' in row or 'Age' in row:
                            edad_val = convertir_valor_numerico(row.get('age', row.get('Age')))
                            if edad_val is not None:
                                estudiante.edad = int(edad_val)
                        
                        if 'address' in row or 'Address' in row:
                            direccion_val = row.get('address', row.get('Address', 'U'))
                            if isinstance(direccion_val, str):
                                direccion_val = direccion_val.strip().strip('"').strip("'").upper()
                                estudiante.direccion = 'U' if direccion_val == 'U' else 'R'
                        
                        if 'famsize' in row or 'Famsize' in row:
                            famsize_val = row.get('famsize', row.get('Famsize', 'GT3'))
                            if isinstance(famsize_val, str):
                                famsize_val = famsize_val.strip().strip('"').strip("'").upper()
                                estudiante.tamano_familia = famsize_val
                        
                        if 'Pstatus' in row or 'pstatus' in row:
                            pstatus_val = row.get('Pstatus', row.get('pstatus', 'T'))
                            if isinstance(pstatus_val, str):
                                pstatus_val = pstatus_val.strip().strip('"').strip("'").upper()
                                estudiante.estado_padres = 'T' if pstatus_val == 'T' else 'A'
                        
                        if 'Medu' in row or 'medu' in row:
                            estudiante.educacion_madre = int(convertir_valor_numerico(row.get('Medu', row.get('medu', 0))) or 0)
                        
                        if 'Fedu' in row or 'fedu' in row:
                            estudiante.educacion_padre = int(convertir_valor_numerico(row.get('Fedu', row.get('fedu', 0))) or 0)
                        
                        if 'Mjob' in row or 'mjob' in row:
                            mjob_val = row.get('Mjob', row.get('mjob', 'other'))
                            if isinstance(mjob_val, str):
                                estudiante.trabajo_madre = mjob_val.strip().strip('"').strip("'")
                        
                        if 'Fjob' in row or 'fjob' in row:
                            fjob_val = row.get('Fjob', row.get('fjob', 'other'))
                            if isinstance(fjob_val, str):
                                estudiante.trabajo_padre = fjob_val.strip().strip('"').strip("'")
                        
                        if 'traveltime' in row or 'Traveltime' in row:
                            estudiante.tiempo_viaje = int(convertir_valor_numerico(row.get('traveltime', row.get('Traveltime', 1))) or 1)
                        
                        if 'studytime' in row or 'Studytime' in row:
                            estudiante.tiempo_estudio = int(convertir_valor_numerico(row.get('studytime', row.get('Studytime', 2))) or 2)
                        
                        if 'failures' in row or 'Failures' in row:
                            estudiante.fallos_previos = int(convertir_valor_numerico(row.get('failures', row.get('Failures', 0))) or 0)
                        
                        if 'schoolsup' in row or 'Schoolsup' in row:
                            estudiante.apoyo_escuela = convertir_valor_booleano(row.get('schoolsup', row.get('Schoolsup', 'no')))
                        
                        if 'famsup' in row or 'Famsup' in row:
                            estudiante.apoyo_familia = convertir_valor_booleano(row.get('famsup', row.get('Famsup', 'no')))
                        
                        if 'paid' in row or 'Paid' in row:
                            estudiante.clases_pagadas = convertir_valor_booleano(row.get('paid', row.get('Paid', 'no')))
                        
                        if 'activities' in row or 'Activities' in row:
                            estudiante.actividades_extra = convertir_valor_booleano(row.get('activities', row.get('Activities', 'no')))
                        
                        if 'nursery' in row or 'Nursery' in row:
                            estudiante.guarderia = convertir_valor_booleano(row.get('nursery', row.get('Nursery', 'no')))
                        
                        if 'higher' in row or 'Higher' in row:
                            estudiante.quiere_superior = convertir_valor_booleano(row.get('higher', row.get('Higher', 'yes')))
                        
                        if 'internet' in row or 'Internet' in row:
                            estudiante.internet = convertir_valor_booleano(row.get('internet', row.get('Internet', 'no')))
                        
                        if 'romantic' in row or 'Romantic' in row:
                            estudiante.relacion_romantica = convertir_valor_booleano(row.get('romantic', row.get('Romantic', 'no')))
                        
                        if 'famrel' in row or 'Famrel' in row:
                            estudiante.relacion_familiar = int(convertir_valor_numerico(row.get('famrel', row.get('Famrel', 4))) or 4)
                        
                        if 'freetime' in row or 'Freetime' in row:
                            estudiante.tiempo_libre = int(convertir_valor_numerico(row.get('freetime', row.get('Freetime', 3))) or 3)
                        
                        if 'goout' in row or 'Goout' in row:
                            estudiante.salidas = int(convertir_valor_numerico(row.get('goout', row.get('Goout', 3))) or 3)
                        
                        if 'Dalc' in row or 'dalc' in row:
                            estudiante.alcohol_semana = int(convertir_valor_numerico(row.get('Dalc', row.get('dalc', 1))) or 1)
                        
                        if 'Walc' in row or 'walc' in row:
                            estudiante.alcohol_fin_semana = int(convertir_valor_numerico(row.get('Walc', row.get('walc', 1))) or 1)
                        
                        if 'health' in row or 'Health' in row:
                            estudiante.salud = int(convertir_valor_numerico(row.get('health', row.get('Health', 3))) or 3)
                        
                        if 'absences' in row or 'Absences' in row:
                            estudiante.ausencias = int(convertir_valor_numerico(row.get('absences', row.get('Absences', 0))) or 0)
                        
                        if 'G1' in row or 'g1' in row:
                            estudiante.calificacion_g1 = convertir_valor_numerico(row.get('G1', row.get('g1')))
                        
                        if 'G2' in row or 'g2' in row:
                            estudiante.calificacion_g2 = convertir_valor_numerico(row.get('G2', row.get('g2')))
                        
                        if 'G3' in row or 'g3' in row:
                            estudiante.calificacion_g3 = convertir_valor_numerico(row.get('G3', row.get('g3')))
                        
                        estudiante.save()
                        
                        if creado:
                            estudiantes_creados += 1
                        else:
                            estudiantes_actualizados += 1
                    
                    except Exception as e:
                        errores.append(f"Fila {index + 1}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Mensajes
                if estudiantes_creados > 0:
                    messages.success(request, f'{estudiantes_creados} estudiantes creados exitosamente.')
                if estudiantes_actualizados > 0:
                    messages.info(request, f'{estudiantes_actualizados} estudiantes actualizados.')
                if errores:
                    messages.warning(request, f'Se encontraron {len(errores)} errores al procesar el archivo.')
                
                return redirect('prediccion_academica:estudiantes')
            
            except Exception as e:
                messages.error(request, f'Error al procesar el archivo: {str(e)}')
    
    else:
        form = CargaEstudiantesForm()
    
    context = {
        'form': form,
    }
    
    return render(request, 'prediccion_academica/cargar_estudiantes.html', context)


def vista_docentes(request):
    """
    Vista para docentes: muestra estudiantes con predicciones y alertas
    """
    # Filtros
    busqueda = request.GET.get('busqueda', '')
    filtro_riesgo = request.GET.get('riesgo', '')
    
    estudiantes = Estudiante.objects.all()
    
    if busqueda:
        estudiantes = estudiantes.filter(
            Q(nombre__icontains=busqueda) |
            Q(apellido__icontains=busqueda) |
            Q(identificacion__icontains=busqueda)
        )
    
    # Filtrar por riesgo
    if filtro_riesgo == 'alto':
        estudiantes = estudiantes.filter(
            Q(calificacion_g3__lt=10) | Q(ausencias__gt=10)
        )
    elif filtro_riesgo == 'medio':
        estudiantes = estudiantes.filter(
            calificacion_g3__gte=10,
            calificacion_g3__lt=14
        )
    
    # Obtener predicciones y alertas para cada estudiante
    estudiantes_con_datos = []
    for estudiante in estudiantes:
        # Última predicción
        ultima_prediccion = PrediccionRendimiento.objects.filter(
            estudiante=estudiante
        ).select_related('entrenamiento').order_by('-fecha_prediccion').first()
        
        # Alertas no vistas
        alertas_no_vistas = AlertaEstudiante.objects.filter(
            estudiante=estudiante,
            vista=False
        ).count()
        
        estudiantes_con_datos.append({
            'estudiante': estudiante,
            'ultima_prediccion': ultima_prediccion,
            'alertas_no_vistas': alertas_no_vistas,
        })
    
    # Paginación
    paginador = Paginator(estudiantes_con_datos, 20)
    pagina = request.GET.get('pagina', 1)
    estudiantes_pagina = paginador.get_page(pagina)
    
    # Alertas recientes
    alertas_recientes = AlertaEstudiante.objects.filter(
        vista=False
    ).select_related('estudiante').order_by('-prioridad', '-fecha_creacion')[:10]
    
    # Generar alertas para todos los estudiantes si no hay alertas recientes
    if alertas_recientes.count() == 0:
        try:
            generar_alertas_todos_estudiantes()
            alertas_recientes = AlertaEstudiante.objects.filter(
                vista=False
            ).select_related('estudiante').order_by('-prioridad', '-fecha_creacion')[:10]
        except Exception as e:
            pass  # Si hay error, continuar sin alertas
    
    context = {
        'estudiantes': estudiantes_pagina,
        'alertas_recientes': alertas_recientes,
        'busqueda': busqueda,
        'filtro_riesgo': filtro_riesgo,
    }
    
    return render(request, 'prediccion_academica/docentes.html', context)


def entrenar_mlp(request):
    """
    Vista para entrenar el modelo MLP
    """
    # Obtener estudiantes con al menos algunos datos
    estudiantes = Estudiante.objects.all()
    
    if estudiantes.count() == 0:
        messages.warning(request, 'No hay estudiantes registrados. Por favor, carga estudiantes primero.')
        return redirect('prediccion_academica:estudiantes')
    
    # Columnas disponibles (basadas en el modelo Estudiante)
    # Excluir G3 si es que no está disponible para todos
    columnas_disponibles = [
        'edad', 'sexo', 'direccion', 'tamano_familia', 'estado_padres',
        'educacion_madre', 'educacion_padre', 'trabajo_madre', 'trabajo_padre',
        'tiempo_viaje', 'tiempo_estudio', 'fallos_previos', 'apoyo_escuela',
        'apoyo_familia', 'clases_pagadas', 'actividades_extra', 'guarderia',
        'quiere_superior', 'internet', 'relacion_romantica', 'relacion_familiar',
        'tiempo_libre', 'salidas', 'alcohol_semana', 'alcohol_fin_semana',
        'salud', 'ausencias', 'G1', 'G2'
    ]
    
    if request.method == 'POST':
        form = ConfiguracionEntrenamientoMLPForm(
            request.POST,
            columnas_disponibles=columnas_disponibles
        )
        
        if form.is_valid():
            try:
                # Obtener datos del formulario
                nombre = form.cleaned_data['nombre']
                tipo_implementacion = form.cleaned_data.get('tipo_implementacion', 'numpy')  # Default a numpy para compatibilidad
                num_capas_ocultas = form.cleaned_data['num_capas_ocultas']
                neuronas_por_capa = form.get_neuronas_por_capa()
                funcion_activacion = form.cleaned_data['funcion_activacion']
                tasa_aprendizaje = form.cleaned_data['tasa_aprendizaje']
                iteraciones = form.cleaned_data['iteraciones']
                tamanio_batch = form.cleaned_data['tamanio_batch']
                porcentaje_entrenamiento = form.cleaned_data['porcentaje_entrenamiento']
                columnas_entrada = form.cleaned_data['columnas_entrada']
                columna_salida = form.cleaned_data['columna_salida']
                
                # Filtrar estudiantes que tengan la columna de salida
                if columna_salida == 'G1':
                    estudiantes_filtrados = estudiantes.exclude(calificacion_g1__isnull=True)
                elif columna_salida == 'G2':
                    estudiantes_filtrados = estudiantes.exclude(calificacion_g2__isnull=True)
                elif columna_salida == 'G3':
                    estudiantes_filtrados = estudiantes.exclude(calificacion_g3__isnull=True)
                else:
                    estudiantes_filtrados = estudiantes
                
                if estudiantes_filtrados.count() == 0:
                    messages.error(request, f'No hay estudiantes con datos para {columna_salida}. Por favor, carga estudiantes con esa información.')
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {'form': form, 'total_estudiantes': estudiantes.count()})
                
                # Crear DataFrame con datos de estudiantes
                datos_estudiantes = []
                for estudiante in estudiantes_filtrados:
                    datos = {
                        'edad': estudiante.edad,
                        'sexo': estudiante.sexo,
                        'direccion': estudiante.direccion,
                        'tamano_familia': estudiante.tamano_familia,
                        'estado_padres': estudiante.estado_padres,
                        'educacion_madre': estudiante.educacion_madre,
                        'educacion_padre': estudiante.educacion_padre,
                        'trabajo_madre': estudiante.trabajo_madre,
                        'trabajo_padre': estudiante.trabajo_padre,
                        'tiempo_viaje': estudiante.tiempo_viaje,
                        'tiempo_estudio': estudiante.tiempo_estudio,
                        'fallos_previos': estudiante.fallos_previos,
                        'apoyo_escuela': 1 if estudiante.apoyo_escuela else 0,
                        'apoyo_familia': 1 if estudiante.apoyo_familia else 0,
                        'clases_pagadas': 1 if estudiante.clases_pagadas else 0,
                        'actividades_extra': 1 if estudiante.actividades_extra else 0,
                        'guarderia': 1 if estudiante.guarderia else 0,
                        'quiere_superior': 1 if estudiante.quiere_superior else 0,
                        'internet': 1 if estudiante.internet else 0,
                        'relacion_romantica': 1 if estudiante.relacion_romantica else 0,
                        'relacion_familiar': estudiante.relacion_familiar,
                        'tiempo_libre': estudiante.tiempo_libre,
                        'salidas': estudiante.salidas,
                        'alcohol_semana': estudiante.alcohol_semana,
                        'alcohol_fin_semana': estudiante.alcohol_fin_semana,
                        'salud': estudiante.salud,
                        'ausencias': estudiante.ausencias,
                        'G1': estudiante.calificacion_g1 if estudiante.calificacion_g1 is not None else None,
                        'G2': estudiante.calificacion_g2 if estudiante.calificacion_g2 is not None else None,
                        'G3': estudiante.calificacion_g3 if estudiante.calificacion_g3 is not None else None,
                    }
                    datos_estudiantes.append(datos)
                
                df = pd.DataFrame(datos_estudiantes)
                
                # Verificar que el DataFrame no esté vacío
                if df.empty:
                    messages.error(request, 'No hay datos suficientes para entrenar el modelo.')
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {'form': form, 'total_estudiantes': estudiantes.count()})
                
                # Preprocesar datos
                X, y, info_preprocesamiento = preprocesar_datos_estudiantes(
                    df,
                    columnas_entrada=columnas_entrada,
                    columna_salida=columna_salida,
                    normalizar=True,
                    metodo_normalizacion='standard',
                    usar_one_hot=True,
                    manejar_faltantes=True,
                    estrategia_faltantes='media'
                )
                
                # Verificar que X y y no estén vacíos
                if X.shape[0] == 0 or y.shape[0] == 0:
                    messages.error(request, 'No hay datos suficientes después del preprocesamiento.')
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {'form': form, 'total_estudiantes': estudiantes.count()})
                
                # Dividir en entrenamiento y validación
                n_entrenamiento = int(len(X) * (porcentaje_entrenamiento / 100))
                if n_entrenamiento == 0:
                    n_entrenamiento = 1
                if n_entrenamiento >= len(X):
                    n_entrenamiento = len(X) - 1 if len(X) > 1 else 1
                
                indices = np.random.permutation(len(X))
                indices_entrenamiento = indices[:n_entrenamiento]
                indices_validacion = indices[n_entrenamiento:]
                
                X_entrenamiento = X[indices_entrenamiento]
                y_entrenamiento = y[indices_entrenamiento]
                X_validacion = X[indices_validacion] if len(indices_validacion) > 0 else None
                y_validacion = y[indices_validacion] if len(indices_validacion) > 0 else None
                
                # Crear arquitectura del MLP
                n_entradas = X.shape[1]  # Número de características después del preprocesamiento
                arquitectura = [n_entradas] + neuronas_por_capa + [1]
                
                # Preparar información de preprocesamiento sin el scaler (no serializable)
                info_preprocesamiento_serializable = {
                    'transformaciones': info_preprocesamiento.get('transformaciones', []),
                    'mapeo_categorias': info_preprocesamiento.get('mapeo_categorias', {}),
                    'mapeo_labels': info_preprocesamiento.get('mapeo_labels', {}),
                    'columnas_originales': info_preprocesamiento.get('columnas_originales', []),
                    'columnas_finales': info_preprocesamiento.get('columnas_finales', []),
                    'metodo_normalizacion': 'standard',
                }
                
                # Obtener número de características finales
                columnas_finales = info_preprocesamiento.get('columnas_finales', columnas_entrada)
                num_caracteristicas_finales = len(columnas_finales) if columnas_finales else len(columnas_entrada)
                
                # Extraer parámetros del scaler para guardarlos
                scaler_mean = None
                scaler_scale = None
                if info_preprocesamiento.get('scaler'):
                    scaler = info_preprocesamiento['scaler']
                    scaler_mean = convertir_a_tipos_nativos(scaler.mean_.tolist()) if hasattr(scaler, 'mean_') else None
                    scaler_scale = convertir_a_tipos_nativos(scaler.scale_.tolist()) if hasattr(scaler, 'scale_') else None
                
                # Seleccionar implementación según el tipo
                if tipo_implementacion == 'tensorflow':
                    # Verificar que TensorFlow esté disponible
                    if not TENSORFLOW_AVAILABLE or MLPTensorFlow is None:
                        messages.error(request, 'TensorFlow no está instalado. Por favor, instálalo usando: pip install tensorflow>=2.13.0')
                        return render(request, 'prediccion_academica/entrenar_mlp.html', {'form': form, 'total_estudiantes': estudiantes.count()})
                    
                    # Implementación con TensorFlow/Keras
                    mlp_tf = MLPTensorFlow(
                        arquitectura=arquitectura,
                        tasa_aprendizaje=tasa_aprendizaje,
                        funcion_activacion=funcion_activacion,
                        optimizer='adam'
                    )
                    
                    # Entrenar con TensorFlow
                    resultados = mlp_tf.entrenar(
                        X_entrenamiento=X_entrenamiento,
                        y_entrenamiento=y_entrenamiento,
                        X_validacion=X_validacion,
                        y_validacion=y_validacion,
                        epochs=iteraciones,
                        batch_size=tamanio_batch,
                        verbose=True,
                        patience=50
                    )
                    
                    # Guardar modelo TensorFlow en un archivo
                    os.makedirs(settings.MLP_MODELS_DIR, exist_ok=True)
                    nombre_archivo = f"modelo_{uuid.uuid4().hex[:8]}.h5"
                    ruta_modelo = settings.MLP_MODELS_DIR / nombre_archivo
                    mlp_tf.guardar_modelo(str(ruta_modelo))
                    
                    # Crear objeto de entrenamiento
                    entrenamiento = EntrenamientoMLP.objects.create(
                        nombre=nombre,
                        tipo_implementacion='tensorflow',
                        num_capas_ocultas=num_capas_ocultas,
                        neuronas_por_capa=neuronas_por_capa,
                        funcion_activacion=funcion_activacion,
                        tasa_aprendizaje=tasa_aprendizaje,
                        iteraciones=iteraciones,
                        tamanio_batch=tamanio_batch,
                        porcentaje_entrenamiento=porcentaje_entrenamiento,
                        columnas_entrada=columnas_entrada,
                        columna_salida=columna_salida,
                        num_caracteristicas_finales=num_caracteristicas_finales,
                        modelo_tensorflow=f"modelos_mlp/{nombre_archivo}",
                        historial_entrenamiento_tf=resultados.get('historial_entrenamiento', {}),
                        precision_entrenamiento=resultados['metricas_entrenamiento'].get('R2', 0.0),
                        precision_validacion=resultados['metricas_validacion'].get('R2', 0.0) if resultados['metricas_validacion'] else 0.0,
                        errores_entrenamiento=resultados['historial_errores_entrenamiento'],
                        errores_validacion=resultados['historial_errores_validacion'] if resultados['historial_errores_validacion'] else [],
                        precision_entrenamiento_historial=resultados['historial_precision_entrenamiento'],
                        precision_validacion_historial=resultados['historial_precision_validacion'] if resultados['historial_precision_validacion'] else [],
                        metricas=resultados['metricas_entrenamiento'],
                        info_preprocesamiento=info_preprocesamiento_serializable,
                        scaler_mean=scaler_mean,
                        scaler_scale=scaler_scale,
                        descripcion=f"Arquitectura: {arquitectura}, Función: {funcion_activacion}, TensorFlow/Keras"
                    )
                
                else:
                    # Implementación desde cero con NumPy (default)
                    mlp = MLP(
                        arquitectura=arquitectura,
                        tasa_aprendizaje=tasa_aprendizaje,
                        funcion_activacion=funcion_activacion,
                        inicializacion='xavier'
                    )
                    
                    # Entrenar
                    resultados = mlp.entrenar(
                        X_entrenamiento=X_entrenamiento,
                        y_entrenamiento=y_entrenamiento,
                        X_validacion=X_validacion,
                        y_validacion=y_validacion,
                        iteraciones=iteraciones,
                        tamanio_batch=tamanio_batch,
                        verbose=True,
                        paciencia=50
                    )
                    
                    # Guardar el entrenamiento en la base de datos
                    pesos_serializables = mlp.obtener_pesos_serializables()
                    
                    # Crear objeto de entrenamiento
                    entrenamiento = EntrenamientoMLP.objects.create(
                        nombre=nombre,
                        tipo_implementacion='numpy',
                        num_capas_ocultas=num_capas_ocultas,
                        neuronas_por_capa=neuronas_por_capa,
                        funcion_activacion=funcion_activacion,
                        tasa_aprendizaje=tasa_aprendizaje,
                        iteraciones=iteraciones,
                        tamanio_batch=tamanio_batch,
                        porcentaje_entrenamiento=porcentaje_entrenamiento,
                        columnas_entrada=columnas_entrada,
                        columna_salida=columna_salida,
                        num_caracteristicas_finales=num_caracteristicas_finales,
                        pesos_capas=pesos_serializables['pesos'],
                        sesgos_capas=pesos_serializables['sesgos'],
                        precision_entrenamiento=resultados['metricas_entrenamiento'].get('R2', 0.0),
                        precision_validacion=resultados['metricas_validacion'].get('R2', 0.0) if resultados['metricas_validacion'] else 0.0,
                        errores_entrenamiento=resultados['historial_errores_entrenamiento'],
                        errores_validacion=resultados['historial_errores_validacion'] if resultados['historial_errores_validacion'] else [],
                        precision_entrenamiento_historial=resultados['historial_precision_entrenamiento'],
                        precision_validacion_historial=resultados['historial_precision_validacion'] if resultados['historial_precision_validacion'] else [],
                        metricas=resultados['metricas_entrenamiento'],
                        info_preprocesamiento=info_preprocesamiento_serializable,
                        scaler_mean=scaler_mean,
                        scaler_scale=scaler_scale,
                        descripcion=f"Arquitectura: {arquitectura}, Función: {funcion_activacion}, NumPy"
                    )
                
                messages.success(request, f'Modelo entrenado exitosamente. R² entrenamiento: {resultados["metricas_entrenamiento"]["R2"]:.4f}')
                return redirect('prediccion_academica:resultados_entrenamiento', entrenamiento_id=entrenamiento.id)
            
            except Exception as e:
                messages.error(request, f'Error durante el entrenamiento: {str(e)}')
                import traceback
                traceback.print_exc()
    
    else:
        form = ConfiguracionEntrenamientoMLPForm(columnas_disponibles=columnas_disponibles)
    
    context = {
        'form': form,
        'total_estudiantes': estudiantes.count(),
    }
    
    return render(request, 'prediccion_academica/entrenar_mlp.html', context)


def resultados_entrenamiento(request, entrenamiento_id):
    """
    Vista para mostrar los resultados del entrenamiento
    """
    entrenamiento = get_object_or_404(EntrenamientoMLP, id=entrenamiento_id)
    
    # Determinar tipo de implementación (default a 'numpy' para compatibilidad)
    tipo_implementacion = getattr(entrenamiento, 'tipo_implementacion', 'numpy')
    
    # Generar gráficos según el tipo de implementación
    if tipo_implementacion == 'tensorflow':
        # Verificar que TensorFlow esté disponible
        if not TENSORFLOW_AVAILABLE or MLPTensorFlow is None:
            grafico_errores = None
            grafico_precision = None
        else:
            # Usar historial de TensorFlow para generar gráficos
            historial_tf = entrenamiento.historial_entrenamiento_tf or {}
            
            # Crear instancia temporal de MLPTensorFlow para generar gráficos
            if entrenamiento.num_caracteristicas_finales:
                n_entradas = entrenamiento.num_caracteristicas_finales
            else:
                info_preprocesamiento = entrenamiento.info_preprocesamiento or {}
                columnas_finales = info_preprocesamiento.get('columnas_finales', entrenamiento.columnas_entrada)
                n_entradas = len(columnas_finales) if columnas_finales else len(entrenamiento.columnas_entrada)
            
            arquitectura = [n_entradas] + entrenamiento.neuronas_por_capa + [1]
            try:
                mlp_tf = MLPTensorFlow(
                    arquitectura=arquitectura,
                    tasa_aprendizaje=entrenamiento.tasa_aprendizaje,
                    funcion_activacion=entrenamiento.funcion_activacion
                )
                
                # Usar historial guardado para generar gráficos
                grafico_errores = mlp_tf.crear_grafico_errores(historial_tf)
                grafico_precision = mlp_tf.crear_grafico_precision(historial_tf)
            except Exception as e:
                grafico_errores = None
                grafico_precision = None
        
    else:
        # Implementación NumPy (default)
        # Obtener información de preprocesamiento
        info_preprocesamiento = entrenamiento.info_preprocesamiento or {}
        
        # Obtener el número de características después del preprocesamiento
        # Usar el campo guardado o calcularlo
        if entrenamiento.num_caracteristicas_finales:
            n_entradas = entrenamiento.num_caracteristicas_finales
        else:
            columnas_finales = info_preprocesamiento.get('columnas_finales', entrenamiento.columnas_entrada)
            n_entradas = len(columnas_finales) if columnas_finales else len(entrenamiento.columnas_entrada)
        
        # Reconstruir el MLP para generar gráficos
        arquitectura = [n_entradas] + entrenamiento.neuronas_por_capa + [1]
        
        mlp = MLP(
            arquitectura=arquitectura,
            tasa_aprendizaje=entrenamiento.tasa_aprendizaje,
            funcion_activacion=entrenamiento.funcion_activacion
        )
        
        # Cargar pesos si están disponibles
        if entrenamiento.pesos_capas and entrenamiento.sesgos_capas:
            mlp.cargar_pesos(entrenamiento.pesos_capas, entrenamiento.sesgos_capas)
            
            # Restaurar historial de errores y precisiones para generar gráficos
            if entrenamiento.errores_entrenamiento:
                mlp.historial_errores_entrenamiento = entrenamiento.errores_entrenamiento
            if entrenamiento.errores_validacion:
                mlp.historial_errores_validacion = entrenamiento.errores_validacion
            if entrenamiento.precision_entrenamiento_historial:
                mlp.historial_precision_entrenamiento = entrenamiento.precision_entrenamiento_historial
            if entrenamiento.precision_validacion_historial:
                mlp.historial_precision_validacion = entrenamiento.precision_validacion_historial
            
            # Generar gráficos
            grafico_errores = mlp.crear_grafico_errores() if mlp.historial_errores_entrenamiento else None
            grafico_precision = mlp.crear_grafico_precision() if mlp.historial_precision_entrenamiento else None
        else:
            grafico_errores = None
            grafico_precision = None
    
    context = {
        'entrenamiento': entrenamiento,
        'tipo_implementacion': tipo_implementacion,
        'grafico_errores': grafico_errores,
        'grafico_precision': grafico_precision,
        'metricas': entrenamiento.metricas,
    }
    
    return render(request, 'prediccion_academica/resultados_entrenamiento.html', context)


def ver_predicciones(request):
    """
    Vista para ver todas las predicciones
    """
    # Filtros
    estudiante_id = request.GET.get('estudiante', '')
    entrenamiento_id = request.GET.get('entrenamiento', '')
    
    predicciones = PrediccionRendimiento.objects.select_related(
        'estudiante', 'entrenamiento'
    ).all()
    
    if estudiante_id:
        predicciones = predicciones.filter(estudiante_id=estudiante_id)
    
    if entrenamiento_id:
        predicciones = predicciones.filter(entrenamiento_id=entrenamiento_id)
    
    # Paginación
    paginador = Paginator(predicciones, 20)
    pagina = request.GET.get('pagina', 1)
    predicciones_pagina = paginador.get_page(pagina)
    
    context = {
        'predicciones': predicciones_pagina,
        'estudiantes': Estudiante.objects.all(),
        'entrenamientos': EntrenamientoMLP.objects.all(),
    }
    
    return render(request, 'prediccion_academica/predicciones.html', context)


def marcar_alerta_vista(request, alerta_id):
    """
    Vista para marcar una alerta como vista
    """
    alerta = get_object_or_404(AlertaEstudiante, id=alerta_id)
    alerta.marcar_como_vista()
    messages.success(request, 'Alerta marcada como vista.')
    return redirect('prediccion_academica:docentes')


def generar_alertas_manual(request):
    """
    Vista para generar alertas manualmente para todos los estudiantes
    """
    if request.method == 'POST':
        try:
            total_alertas = generar_alertas_todos_estudiantes()
            messages.success(request, f'Se generaron {total_alertas} alertas para estudiantes en riesgo.')
        except Exception as e:
            messages.error(request, f'Error al generar alertas: {str(e)}')
            traceback.print_exc()
    
    return redirect('prediccion_academica:docentes')


@csrf_exempt
@require_http_methods(["POST"])
def api_prediccion(request):
    """
    API para hacer predicciones
    """
    try:
        data = json.loads(request.body)
        estudiante_id = data.get('estudiante_id')
        entrenamiento_id = data.get('entrenamiento_id')
        
        estudiante = get_object_or_404(Estudiante, id=estudiante_id)
        entrenamiento = get_object_or_404(EntrenamientoMLP, id=entrenamiento_id)
        
        # Obtener información de preprocesamiento del modelo
        info_preprocesamiento = entrenamiento.info_preprocesamiento or {}
        
        # Agregar parámetros del scaler si están disponibles
        if entrenamiento.scaler_mean and entrenamiento.scaler_scale:
            info_preprocesamiento['scaler_mean'] = entrenamiento.scaler_mean
            info_preprocesamiento['scaler_scale'] = entrenamiento.scaler_scale
        
        # Convertir estudiante a características
        X = convertir_estudiante_a_caracteristicas(
            estudiante,
            entrenamiento.columnas_entrada,
            info_preprocesamiento
        )
        
        # Determinar tipo de implementación (default a 'numpy' para compatibilidad)
        tipo_implementacion = getattr(entrenamiento, 'tipo_implementacion', 'numpy')
        
        # Cargar modelo según el tipo de implementación
        if tipo_implementacion == 'tensorflow':
            # Verificar que TensorFlow esté disponible
            if not TENSORFLOW_AVAILABLE or MLPTensorFlow is None:
                return JsonResponse({
                    'success': False,
                    'error': 'TensorFlow no está instalado. Por favor, instálalo usando: pip install tensorflow>=2.13.0'
                }, status=400)
            
            # Cargar modelo TensorFlow
            if not entrenamiento.modelo_tensorflow:
                return JsonResponse({
                    'success': False,
                    'error': 'Modelo TensorFlow no encontrado'
                }, status=400)
            
            ruta_modelo = os.path.join(settings.MEDIA_ROOT, str(entrenamiento.modelo_tensorflow))
            if not os.path.exists(ruta_modelo):
                return JsonResponse({
                    'success': False,
                    'error': f'Archivo del modelo no encontrado: {ruta_modelo}'
                }, status=400)
            
            mlp_tf = MLPTensorFlow.cargar_modelo(ruta_modelo)
            
            # Hacer predicción
            prediccion = mlp_tf.predecir(X.reshape(1, -1))[0]
        
        else:
            # Implementación NumPy (default)
            # Obtener el número de características después del preprocesamiento
            # Usar el campo guardado o calcularlo
            if entrenamiento.num_caracteristicas_finales:
                n_entradas = entrenamiento.num_caracteristicas_finales
            else:
                columnas_finales = info_preprocesamiento.get('columnas_finales', entrenamiento.columnas_entrada)
                n_entradas = len(columnas_finales) if columnas_finales else len(entrenamiento.columnas_entrada)
            
            # Reconstruir el MLP
            arquitectura = [n_entradas] + entrenamiento.neuronas_por_capa + [1]
            
            mlp = MLP(
                arquitectura=arquitectura,
                tasa_aprendizaje=entrenamiento.tasa_aprendizaje,
                funcion_activacion=entrenamiento.funcion_activacion
            )
            
            # Cargar pesos
            if not entrenamiento.pesos_capas or not entrenamiento.sesgos_capas:
                return JsonResponse({
                    'success': False,
                    'error': 'Pesos y sesgos del modelo no encontrados'
                }, status=400)
            
            mlp.cargar_pesos(entrenamiento.pesos_capas, entrenamiento.sesgos_capas)
            
            # Hacer predicción
            prediccion = mlp.predecir(X)[0]
        
        # Guardar predicción
        caracteristicas_usadas = {col: getattr(estudiante, col, None) for col in entrenamiento.columnas_entrada}
        
        prediccion_obj = PrediccionRendimiento.objects.create(
            estudiante=estudiante,
            entrenamiento=entrenamiento,
            calificacion_predicha=float(prediccion),
            caracteristicas_usadas=caracteristicas_usadas
        )
        
        # Calcular error si hay calificación real
        calificacion_real = getattr(estudiante, f'calificacion_{entrenamiento.columna_salida.lower()}', None)
        if calificacion_real is not None:
            prediccion_obj.calificacion_real = calificacion_real
            prediccion_obj.calcular_error()
        
        # Generar alertas automáticamente después de la predicción
        alertas_count = 0
        try:
            alertas_generadas = generar_alertas_estudiante(estudiante, prediccion_obj)
            alertas_count = len(alertas_generadas)
        except Exception as e:
            # Si hay error al generar alertas, no interrumpir la respuesta
            traceback.print_exc()
        
        return JsonResponse({
            'success': True,
            'prediccion': float(prediccion),
            'calificacion_real': float(calificacion_real) if calificacion_real else None,
            'error': float(prediccion_obj.error_prediccion) if prediccion_obj.error_prediccion else None,
            'alertas_generadas': alertas_count,
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
