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
from django import forms
import traceback
import json
import pandas as pd
import numpy as np
import os
import uuid
from typing import Dict, Any

from .models import Estudiante, EntrenamientoMLP, PrediccionRendimiento, AlertaEstudiante, HistorialAcademico
from .forms import CargaEstudiantesForm, EstudianteForm, ConfiguracionEntrenamientoMLPForm, PrediccionForm, HistorialAcademicoForm
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
    estudiantes_con_promedios = Estudiante.objects.exclude(
        promedio_acumulado__isnull=True
    )
    
    promedio_acum = estudiantes_con_promedios.aggregate(Avg('promedio_acumulado'))['promedio_acumulado__avg']
    promedio_acum = round(promedio_acum, 2) if promedio_acum else None
    
    estudiantes_riesgo = estudiantes_con_promedios.filter(
        promedio_acumulado__lt=3.0
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
        'promedio_acumulado': promedio_acum,
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
    
    # Historial académico del estudiante
    historiales = HistorialAcademico.objects.filter(
        estudiante=estudiante
    ).order_by('semestre')
    
    # Promedio acumulado
    promedio = estudiante.promedio_acumulado
    
    context = {
        'estudiante': estudiante,
        'predicciones': predicciones,
        'alertas': alertas,
        'historiales': historiales,
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
                
                def obtener_valor_fila(row, columnas_posibles, valor_defecto=None):
                    """Obtiene un valor de una fila de pandas Series, buscando en varias columnas posibles"""
                    for col in columnas_posibles:
                        if col in row.index:
                            valor = row[col]
                            if pd.notna(valor):
                                return valor
                    return valor_defecto
                
                # Crear estudiantes
                estudiantes_creados = 0
                estudiantes_actualizados = 0
                errores = []
                
                for index, row in df.iterrows():
                    try:
                        # Generar identificación única
                        identificacion = f"EST{index + 1:04d}"
                        
                        # Mapear campos del CSV a un diccionario de defaults
                        defaults = {
                            'nombre': f'Estudiante {index + 1}',
                            'apellido': '',
                        }
                        
                        # Mapear campos requeridos con valores por defecto
                        # Sexo
                        sexo_val = None
                        if 'sex' in row.index:
                            sexo_val = row['sex']
                        elif 'Sex' in row.index:
                            sexo_val = row['Sex']
                        
                        if sexo_val is not None and pd.notna(sexo_val):
                            if isinstance(sexo_val, str):
                                sexo_val = str(sexo_val).strip().strip('"').strip("'").upper()
                            else:
                                sexo_val = str(sexo_val).upper()
                            defaults['sexo'] = 'M' if sexo_val == 'M' else 'F'
                        else:
                            defaults['sexo'] = 'M'
                        
                        # Edad
                        edad_val = None
                        if 'age' in row.index:
                            edad_val = convertir_valor_numerico(row['age'])
                        elif 'Age' in row.index:
                            edad_val = convertir_valor_numerico(row['Age'])
                        
                        if edad_val is not None and not pd.isna(edad_val):
                            defaults['edad'] = int(edad_val)
                        else:
                            defaults['edad'] = 18
                        
                        # Dirección
                        direccion_val = None
                        if 'address' in row.index:
                            direccion_val = row['address']
                        elif 'Address' in row.index:
                            direccion_val = row['Address']
                        
                        if direccion_val is not None and pd.notna(direccion_val):
                            if isinstance(direccion_val, str):
                                direccion_val = str(direccion_val).strip().strip('"').strip("'").upper()
                            else:
                                direccion_val = str(direccion_val).upper()
                            defaults['direccion'] = 'U' if direccion_val == 'U' else 'R'
                        else:
                            defaults['direccion'] = 'U'
                        
                        # Tamaño de familia
                        famsize_val = obtener_valor_fila(row, ['famsize', 'Famsize'], 'GT3')
                        if pd.notna(famsize_val):
                            if isinstance(famsize_val, str):
                                famsize_val = str(famsize_val).strip().strip('"').strip("'").upper()
                            else:
                                famsize_val = str(famsize_val).upper()
                            defaults['tamano_familia'] = famsize_val if famsize_val in ['GT3', 'LE3'] else 'GT3'
                        else:
                            defaults['tamano_familia'] = 'GT3'
                        
                        # Estado de los padres
                        pstatus_val = obtener_valor_fila(row, ['Pstatus', 'pstatus'], 'T')
                        if pd.notna(pstatus_val):
                            if isinstance(pstatus_val, str):
                                pstatus_val = str(pstatus_val).strip().strip('"').strip("'").upper()
                            else:
                                pstatus_val = str(pstatus_val).upper()
                            defaults['estado_padres'] = 'T' if pstatus_val == 'T' else 'A'
                        else:
                            defaults['estado_padres'] = 'T'
                        
                        # Mapear el resto de los campos requeridos a defaults
                        # Educacion padres
                        medu_val = convertir_valor_numerico(obtener_valor_fila(row, ['Medu', 'medu'], 0))
                        defaults['educacion_madre'] = int(medu_val) if medu_val is not None else 0
                        
                        fedu_val = convertir_valor_numerico(obtener_valor_fila(row, ['Fedu', 'fedu'], 0))
                        defaults['educacion_padre'] = int(fedu_val) if fedu_val is not None else 0
                        
                        # Trabajos padres
                        mjob_val = obtener_valor_fila(row, ['Mjob', 'mjob'], 'other')
                        if pd.notna(mjob_val):
                            defaults['trabajo_madre'] = str(mjob_val).strip().strip('"').strip("'")
                        else:
                            defaults['trabajo_madre'] = 'other'
                        
                        fjob_val = obtener_valor_fila(row, ['Fjob', 'fjob'], 'other')
                        if pd.notna(fjob_val):
                            defaults['trabajo_padre'] = str(fjob_val).strip().strip('"').strip("'")
                        else:
                            defaults['trabajo_padre'] = 'other'
                        
                        # Campos académicos con valores por defecto
                        traveltime_val = convertir_valor_numerico(obtener_valor_fila(row, ['traveltime', 'Traveltime'], 1))
                        defaults['tiempo_viaje'] = int(traveltime_val) if traveltime_val is not None else 1
                        
                        studytime_val = convertir_valor_numerico(obtener_valor_fila(row, ['studytime', 'Studytime'], 2))
                        defaults['tiempo_estudio'] = int(studytime_val) if studytime_val is not None else 2
                        
                        failures_val = convertir_valor_numerico(obtener_valor_fila(row, ['failures', 'Failures'], 0))
                        defaults['fallos_previos'] = int(failures_val) if failures_val is not None else 0
                        
                        famrel_val = convertir_valor_numerico(obtener_valor_fila(row, ['famrel', 'Famrel'], 4))
                        defaults['relacion_familiar'] = int(famrel_val) if famrel_val is not None else 4
                        
                        freetime_val = convertir_valor_numerico(obtener_valor_fila(row, ['freetime', 'Freetime'], 3))
                        defaults['tiempo_libre'] = int(freetime_val) if freetime_val is not None else 3
                        
                        goout_val = convertir_valor_numerico(obtener_valor_fila(row, ['goout', 'Goout'], 3))
                        defaults['salidas'] = int(goout_val) if goout_val is not None else 3
                        
                        dalc_val = convertir_valor_numerico(obtener_valor_fila(row, ['Dalc', 'dalc'], 1))
                        defaults['alcohol_semana'] = int(dalc_val) if dalc_val is not None else 1
                        
                        walc_val = convertir_valor_numerico(obtener_valor_fila(row, ['Walc', 'walc'], 1))
                        defaults['alcohol_fin_semana'] = int(walc_val) if walc_val is not None else 1
                        
                        health_val = convertir_valor_numerico(obtener_valor_fila(row, ['health', 'Health'], 3))
                        defaults['salud'] = int(health_val) if health_val is not None else 3
                        
                        absences_val = convertir_valor_numerico(obtener_valor_fila(row, ['absences', 'Absences'], 0))
                        defaults['ausencias'] = int(absences_val) if absences_val is not None else 0
                        
                        # Campos booleanos con valores por defecto
                        schoolsup_val = obtener_valor_fila(row, ['schoolsup', 'Schoolsup'], 'no')
                        defaults['apoyo_escuela'] = convertir_valor_booleano(schoolsup_val) if pd.notna(schoolsup_val) else False
                        
                        famsup_val = obtener_valor_fila(row, ['famsup', 'Famsup'], 'no')
                        defaults['apoyo_familia'] = convertir_valor_booleano(famsup_val) if pd.notna(famsup_val) else False
                        
                        paid_val = obtener_valor_fila(row, ['paid', 'Paid'], 'no')
                        defaults['clases_pagadas'] = convertir_valor_booleano(paid_val) if pd.notna(paid_val) else False
                        
                        activities_val = obtener_valor_fila(row, ['activities', 'Activities'], 'no')
                        defaults['actividades_extra'] = convertir_valor_booleano(activities_val) if pd.notna(activities_val) else False
                        
                        nursery_val = obtener_valor_fila(row, ['nursery', 'Nursery'], 'no')
                        defaults['guarderia'] = convertir_valor_booleano(nursery_val) if pd.notna(nursery_val) else False
                        
                        higher_val = obtener_valor_fila(row, ['higher', 'Higher'], 'yes')
                        defaults['quiere_superior'] = convertir_valor_booleano(higher_val) if pd.notna(higher_val) else True
                        
                        internet_val = obtener_valor_fila(row, ['internet', 'Internet'], 'no')
                        defaults['internet'] = convertir_valor_booleano(internet_val) if pd.notna(internet_val) else False
                        
                        romantic_val = obtener_valor_fila(row, ['romantic', 'Romantic'], 'no')
                        defaults['relacion_romantica'] = convertir_valor_booleano(romantic_val) if pd.notna(romantic_val) else False
                        
                        # Verificar si el estudiante ya existe
                        estudiante, creado = Estudiante.objects.get_or_create(
                            identificacion=identificacion,
                            defaults=defaults
                        )
                        
                        # Actualizar campos adicionales (tanto si fue creado como si ya existía)
                        # Estos campos pueden actualizarse después porque no son requeridos
                        if 'Medu' in row or 'medu' in row:
                            estudiante.educacion_madre = int(convertir_valor_numerico(row.get('Medu', row.get('medu', 0))) or 0)
                        
                        if 'Fedu' in row or 'fedu' in row:
                            estudiante.educacion_padre = int(convertir_valor_numerico(row.get('Fedu', row.get('fedu', 0))) or 0)
                        
                        if 'Mjob' in row or 'mjob' in row:
                            mjob_val = row.get('Mjob', row.get('mjob', 'other'))
                            if isinstance(mjob_val, str) and pd.notna(mjob_val):
                                estudiante.trabajo_madre = str(mjob_val).strip().strip('"').strip("'")
                        
                        if 'Fjob' in row or 'fjob' in row:
                            fjob_val = row.get('Fjob', row.get('fjob', 'other'))
                            if isinstance(fjob_val, str) and pd.notna(fjob_val):
                                estudiante.trabajo_padre = str(fjob_val).strip().strip('"').strip("'")
                        
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
                        
                        # Campos académicos universitarios (si existen en el CSV)
                        if 'semestre_actual' in row or 'Semestre' in row:
                            estudiante.semestre_actual = int(convertir_valor_numerico(row.get('semestre_actual', row.get('Semestre', 1))) or 1)
                        
                        if 'puntaje_icfes_global' in row or 'Puntaje ICFES' in row or 'icfes' in row:
                            estudiante.puntaje_icfes_global = int(convertir_valor_numerico(row.get('puntaje_icfes_global', row.get('Puntaje ICFES', row.get('icfes')))) or 0)
                        
                        if 'estrato' in row or 'Estrato' in row:
                            estudiante.estrato = int(convertir_valor_numerico(row.get('estrato', row.get('Estrato', 3))) or 3)
                        
                        if 'programa_academico' in row or 'Programa' in row or 'carrera' in row:
                            estudiante.programa_academico = str(row.get('programa_academico', row.get('Programa', row.get('carrera', '')))).strip()
                        
                        if 'trabaja_actualmente' in row or 'trabaja' in row:
                            trabaja_val = row.get('trabaja_actualmente', row.get('trabaja', 'no'))
                            estudiante.trabaja_actualmente = convertir_valor_booleano(trabaja_val)
                        
                        if 'horas_trabajo_sem' in row or 'horas_trabajo' in row:
                            estudiante.horas_trabajo_sem = int(convertir_valor_numerico(row.get('horas_trabajo_sem', row.get('horas_trabajo', 0))) or 0)
                        
                        if 'promedio_semestre_anterior' in row or 'promedio_sem_ant' in row:
                            estudiante.promedio_semestre_anterior = convertir_valor_numerico(row.get('promedio_semestre_anterior', row.get('promedio_sem_ant')))
                        
                        if 'promedio_acumulado' in row or 'promedio_acum' in row:
                            estudiante.promedio_acumulado = convertir_valor_numerico(row.get('promedio_acumulado', row.get('promedio_acum')))
                        
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
    
    # Filtrar por riesgo (basado en promedio acumulado)
    if filtro_riesgo == 'alto':
        estudiantes = estudiantes.filter(
            Q(promedio_acumulado__lt=3.0) | Q(ausencias__gt=10)
        )
    elif filtro_riesgo == 'medio':
        estudiantes = estudiantes.filter(
            promedio_acumulado__gte=3.0,
            promedio_acumulado__lt=3.5
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
    
    # Columnas disponibles (basadas en el modelo Estudiante y el data_pipeline)
    columnas_disponibles = [
        'edad', 'sexo', 'direccion', 'tamano_familia', 'estado_padres',
        'educacion_madre', 'educacion_padre', 'trabajo_madre', 'trabajo_padre',
        'tiempo_viaje', 'tiempo_estudio', 'fallos_previos', 'apoyo_escuela',
        'apoyo_familia', 'clases_pagadas', 'actividades_extra', 'guarderia',
        'quiere_superior', 'internet', 'relacion_romantica', 'relacion_familiar',
        'tiempo_libre', 'salidas', 'alcohol_semana', 'alcohol_fin_semana',
        'salud', 'ausencias',
        # Campos académicos universitarios
        'semestre_actual', 'puntaje_icfes_global', 'estrato', 'programa_academico',
        'trabaja_actualmente', 'horas_trabajo_sem', 'promedio_semestre_anterior',
        'promedio_acumulado',
        # Campos del data_pipeline
        'promedio_semestre_ant', 'materias_reprobadas_sem_ant', 'tendencia_rendimiento',
        'porcentaje_asistencia_sem_ant'
    ]
    
    if request.method == 'POST':
        form = ConfiguracionEntrenamientoMLPForm(
            request.POST,
            columnas_disponibles=columnas_disponibles
        )
        
        if not form.is_valid():
            # Validar errores del formulario
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'Error en {field}: {error}')
            return render(request, 'prediccion_academica/entrenar_mlp.html', {
                'form': form, 
                'total_estudiantes': estudiantes.count()
            })
        
        try:
            # Obtener datos del formulario
            nombre = form.cleaned_data['nombre']
            tipo_implementacion = form.cleaned_data.get('tipo_implementacion', 'numpy')
            num_capas_ocultas = form.cleaned_data['num_capas_ocultas']
            neuronas_por_capa = form.get_neuronas_por_capa()
            funcion_activacion = form.cleaned_data['funcion_activacion']
            tasa_aprendizaje = form.cleaned_data['tasa_aprendizaje']
            iteraciones = form.cleaned_data['iteraciones']
            tamanio_batch = form.cleaned_data['tamanio_batch']
            porcentaje_entrenamiento = form.cleaned_data['porcentaje_entrenamiento']
            columnas_entrada = form.cleaned_data['columnas_entrada']
            columna_salida = form.cleaned_data['columna_salida']
            
            # Validaciones adicionales
            if not columnas_entrada or len(columnas_entrada) == 0:
                messages.error(request, 'Debe seleccionar al menos una columna de entrada.')
                return render(request, 'prediccion_academica/entrenar_mlp.html', {
                    'form': form, 
                    'total_estudiantes': estudiantes.count()
                })
            
            if not neuronas_por_capa or len(neuronas_por_capa) == 0:
                messages.error(request, 'Debe especificar al menos una capa oculta con neuronas.')
                return render(request, 'prediccion_academica/entrenar_mlp.html', {
                    'form': form, 
                    'total_estudiantes': estudiantes.count()
                })
            
            if tamanio_batch > len(estudiantes):
                messages.warning(request, f'El tamaño de batch ({tamanio_batch}) es mayor que el número de estudiantes ({len(estudiantes)}). Se ajustará al número de estudiantes.')
                tamanio_batch = min(tamanio_batch, len(estudiantes))
            
            if iteraciones <= 0:
                messages.error(request, 'El número de iteraciones debe ser mayor que 0.')
                return render(request, 'prediccion_academica/entrenar_mlp.html', {
                    'form': form, 
                    'total_estudiantes': estudiantes.count()
                })
            
            # Validar rangos de valores
            if tasa_aprendizaje <= 0 or tasa_aprendizaje > 1:
                messages.error(request, f'La tasa de aprendizaje debe estar entre 0 y 1. Valor ingresado: {tasa_aprendizaje}')
                return render(request, 'prediccion_academica/entrenar_mlp.html', {
                    'form': form, 
                    'total_estudiantes': estudiantes.count()
                })
            
            if porcentaje_entrenamiento < 50 or porcentaje_entrenamiento > 90:
                messages.error(request, f'El porcentaje de entrenamiento debe estar entre 50% y 90%. Valor ingresado: {porcentaje_entrenamiento}%')
                return render(request, 'prediccion_academica/entrenar_mlp.html', {
                    'form': form, 
                    'total_estudiantes': estudiantes.count()
                })
            
            # Validar neuronas por capa
            for i, neuronas in enumerate(neuronas_por_capa, 1):
                if neuronas <= 0:
                    messages.error(request, f'La capa oculta {i} debe tener al menos 1 neurona. Valor ingresado: {neuronas}')
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
            
            # Generar dataset según el tipo de columna de salida
            if columna_salida == 'Y_promedio_sem_siguiente':
                # Usar el nuevo pipeline de datos
                from .data_pipeline import generar_dataset_rendimiento
                from .models import HistorialAcademico
                
                # Validar que haya historiales académicos
                total_historiales = HistorialAcademico.objects.count()
                if total_historiales == 0:
                    messages.error(
                        request, 
                        'No hay historiales académicos registrados. Por favor, crea historiales académicos para los estudiantes antes de entrenar el modelo.'
                    )
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
                estudiantes_con_historial = Estudiante.objects.filter(
                    historial_academico__isnull=False
                ).distinct().count()
                
                if estudiantes_con_historial == 0:
                    messages.error(
                        request, 
                        'No hay estudiantes con historial académico. Por favor, crea historiales académicos para al menos algunos estudiantes.'
                    )
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
                try:
                    df = generar_dataset_rendimiento()
                except Exception as e:
                    messages.error(request, f'Error al generar el dataset: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
                # Validar que el dataset no esté vacío
                if df.empty or len(df) == 0:
                    messages.error(
                        request, 
                        'El dataset generado está vacío. Asegúrate de que los estudiantes tengan al menos 2 semestres de historial académico.'
                    )
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
                # Validar que las columnas seleccionadas existan en el dataset
                columnas_faltantes = [col for col in columnas_entrada if col not in df.columns]
                if columnas_faltantes:
                    messages.error(
                        request, 
                        f'Las siguientes columnas seleccionadas no están disponibles en el dataset: {", ".join(columnas_faltantes)}'
                    )
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
                # Validar que la columna de salida exista
                if columna_salida not in df.columns:
                    messages.error(
                        request, 
                        f'La columna de salida "{columna_salida}" no está disponible en el dataset.'
                    )
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
            else:
                # Método antiguo: crear DataFrame desde estudiantes
                estudiantes_filtrados = estudiantes
                
                if estudiantes_filtrados.count() == 0:
                    messages.error(request, f'No hay estudiantes registrados para entrenar el modelo.')
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
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
                        # Campos académicos universitarios nuevos
                        'semestre_actual': estudiante.semestre_actual if estudiante.semestre_actual else 1,
                        'puntaje_icfes_global': estudiante.puntaje_icfes_global if estudiante.puntaje_icfes_global else None,
                        'estrato': estudiante.estrato if estudiante.estrato else None,
                        'programa_academico': estudiante.programa_academico if estudiante.programa_academico else '',
                        'trabaja_actualmente': 1 if estudiante.trabaja_actualmente else 0,
                        'horas_trabajo_sem': estudiante.horas_trabajo_sem if estudiante.horas_trabajo_sem else 0,
                        'promedio_semestre_anterior': estudiante.promedio_semestre_anterior if estudiante.promedio_semestre_anterior else None,
                        'promedio_acumulado': estudiante.promedio_acumulado if estudiante.promedio_acumulado else None,
                    }
                    datos_estudiantes.append(datos)
                
                df = pd.DataFrame(datos_estudiantes)
                
                # Verificar que el DataFrame no esté vacío
                if df.empty or len(df) == 0:
                    messages.error(request, 'No hay datos suficientes para entrenar el modelo.')
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
                # Validar que las columnas seleccionadas existan en el dataset
                columnas_faltantes = [col for col in columnas_entrada if col not in df.columns]
                if columnas_faltantes:
                    messages.error(
                        request, 
                        f'Las siguientes columnas seleccionadas no están disponibles en el dataset: {", ".join(columnas_faltantes)}'
                    )
                    return render(request, 'prediccion_academica/entrenar_mlp.html', {
                        'form': form, 
                        'total_estudiantes': estudiantes.count()
                    })
                
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
            return render(request, 'prediccion_academica/entrenar_mlp.html', {
                'form': form, 
                'total_estudiantes': estudiantes.count()
            })
    
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


# ==================== VISTAS DE HISTORIAL ACADÉMICO ====================

def listar_historiales(request, estudiante_id=None):
    """
    Vista para listar todos los historiales académicos o los de un estudiante específico
    """
    estudiante = None
    if estudiante_id is not None:
        estudiante = get_object_or_404(Estudiante, id=estudiante_id)
        historiales = HistorialAcademico.objects.filter(estudiante=estudiante).order_by('-semestre')
        titulo = f"Historial Académico - {estudiante.get_nombre_completo()}"
    else:
        historiales = HistorialAcademico.objects.all().select_related('estudiante').order_by('-semestre', 'estudiante')
        titulo = "Historial Académico - Todos los Estudiantes"
    
    # Paginación
    paginator = Paginator(historiales, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'historiales': page_obj,
        'estudiante': estudiante,
        'titulo': titulo,
    }
    
    return render(request, 'prediccion_academica/historiales/listar.html', context)


def crear_historial(request, estudiante_id=None):
    """
    Vista para crear un nuevo historial académico
    """
    estudiante = None
    if estudiante_id:
        estudiante = get_object_or_404(Estudiante, id=estudiante_id)
    
    if request.method == 'POST':
        form = HistorialAcademicoForm(request.POST)
        if form.is_valid():
            historial = form.save()
            messages.success(request, f'Historial académico del semestre {historial.semestre} creado exitosamente.')
            if estudiante:
                return redirect('prediccion_academica:historiales_estudiante', estudiante_id=estudiante.id)
            return redirect('prediccion_academica:listar_historiales')
    else:
        initial = {}
        if estudiante:
            # Obtener el siguiente semestre
            ultimo_historial = HistorialAcademico.objects.filter(estudiante=estudiante).order_by('-semestre').first()
            siguiente_semestre = (ultimo_historial.semestre + 1) if ultimo_historial else 1
            initial = {
                'estudiante': estudiante,
                'semestre': siguiente_semestre,
                'porcentaje_asistencia': 1.0,
            }
        form = HistorialAcademicoForm(initial=initial)
        if estudiante:
            form.fields['estudiante'].widget = forms.HiddenInput()
    
    context = {
        'form': form,
        'estudiante': estudiante,
        'titulo': f'Crear Historial Académico' + (f' - {estudiante.get_nombre_completo()}' if estudiante else ''),
    }
    
    return render(request, 'prediccion_academica/historiales/crear.html', context)


def editar_historial(request, historial_id):
    """
    Vista para editar un historial académico existente
    """
    historial = get_object_or_404(HistorialAcademico, id=historial_id)
    estudiante = historial.estudiante
    
    if request.method == 'POST':
        form = HistorialAcademicoForm(request.POST, instance=historial)
        if form.is_valid():
            form.save()
            messages.success(request, f'Historial académico del semestre {historial.semestre} actualizado exitosamente.')
            return redirect('prediccion_academica:historiales_estudiante', estudiante_id=estudiante.id)
    else:
        form = HistorialAcademicoForm(instance=historial)
    
    context = {
        'form': form,
        'historial': historial,
        'estudiante': estudiante,
        'titulo': f'Editar Historial Académico - Semestre {historial.semestre}',
    }
    
    return render(request, 'prediccion_academica/historiales/editar.html', context)


def eliminar_historial(request, historial_id):
    """
    Vista para eliminar un historial académico
    """
    historial = get_object_or_404(HistorialAcademico, id=historial_id)
    estudiante = historial.estudiante
    semestre = historial.semestre
    
    if request.method == 'POST':
        historial.delete()
        messages.success(request, f'Historial académico del semestre {semestre} eliminado exitosamente.')
        return redirect('prediccion_academica:historiales_estudiante', estudiante_id=estudiante.id)
    
    context = {
        'historial': historial,
        'estudiante': estudiante,
    }
    
    return render(request, 'prediccion_academica/historiales/eliminar.html', context)
