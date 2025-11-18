"""
Pipeline de datos para generar el dataset de entrenamiento de rendimiento académico
"""

import pandas as pd
from django.db.models import Avg
from .models import Estudiante, HistorialAcademico


def generar_dataset_rendimiento():
    """
    Genera el dataset de entrenamiento para predecir Y_promedio_sem_siguiente
    
    Retorna:
        pandas.DataFrame: DataFrame con las features (X) y el target (Y_promedio_sem_siguiente)
    """
    # Definir las features del dataset ideal
    FEATURES = [
        'promedio_semestre_ant',
        'promedio_acumulado',
        'materias_reprobadas_sem_ant',
        'tendencia_rendimiento',
        'puntaje_icfes_global',
        'trabaja_actualmente',
        'horas_trabajo_sem',
        'estrato',
        'semestre_actual',
        'porcentaje_asistencia_sem_ant',
        'programa_academico'
    ]
    
    TARGET = 'Y_promedio_sem_siguiente'
    
    # Lista para almacenar las filas de datos
    data_rows = []
    
    # Iterar sobre todos los estudiantes
    for estudiante in Estudiante.objects.all():
        # Obtener todos los historiales académicos del estudiante ordenados por semestre
        historiales = HistorialAcademico.objects.filter(
            estudiante=estudiante
        ).order_by('semestre')
        
        # Si el estudiante no tiene historial, saltarlo
        if not historiales.exists():
            continue
        
        # Obtener el semestre actual del estudiante
        semestre_actual = estudiante.semestre_actual if estudiante.semestre_actual else 1
        
        # Iterar sobre semestres históricos (desde 1 hasta semestre_actual - 1)
        # Solo procesamos semestres que tienen un semestre siguiente con datos
        for semestre_historico in range(1, semestre_actual):
            # Buscar el historial del semestre siguiente (el que queremos predecir)
            historial_siguiente = historiales.filter(semestre=semestre_historico + 1).first()
            
            # Si no existe el semestre siguiente, saltar este semestre histórico
            if not historial_siguiente:
                continue
            
            # Construir x_row (features) calculando valores como si el estudiante estuviera en semestre_historico
            x_row = {}
            
            # promedio_semestre_ant: promedio del HistorialAcademico del semestre_historico - 1
            historial_sem_ant = historiales.filter(semestre=semestre_historico - 1).first()
            if historial_sem_ant:
                x_row['promedio_semestre_ant'] = historial_sem_ant.promedio
            else:
                # Si no hay semestre anterior, usar None o 0.0
                x_row['promedio_semestre_ant'] = None
            
            # promedio_acumulado: promedio ponderado de todos los HistorialAcademico hasta semestre_historico
            historiales_hasta_semestre = historiales.filter(semestre__lte=semestre_historico)
            if historiales_hasta_semestre.exists():
                # Calcular promedio ponderado por créditos si están disponibles
                total_creditos = sum(h.creditos_inscritos for h in historiales_hasta_semestre if h.creditos_inscritos > 0)
                if total_creditos > 0:
                    promedio_ponderado = sum(
                        h.promedio * h.creditos_inscritos 
                        for h in historiales_hasta_semestre 
                        if h.creditos_inscritos > 0
                    ) / total_creditos
                else:
                    # Si no hay créditos, calcular promedio simple
                    promedios = [h.promedio for h in historiales_hasta_semestre]
                    promedio_ponderado = sum(promedios) / len(promedios) if promedios else 0.0
                x_row['promedio_acumulado'] = promedio_ponderado
            else:
                x_row['promedio_acumulado'] = 0.0
            
            # materias_reprobadas_sem_ant: materias_reprobadas del HistorialAcademico del semestre_historico - 1
            if historial_sem_ant:
                x_row['materias_reprobadas_sem_ant'] = historial_sem_ant.materias_reprobadas
            else:
                x_row['materias_reprobadas_sem_ant'] = 0
            
            # tendencia_rendimiento: (promedio_semestre_ant - promedio_acumulado)
            if x_row['promedio_semestre_ant'] is not None:
                x_row['tendencia_rendimiento'] = x_row['promedio_semestre_ant'] - x_row['promedio_acumulado']
            else:
                x_row['tendencia_rendimiento'] = 0.0
            
            # Valores estáticos del estudiante
            x_row['puntaje_icfes_global'] = estudiante.puntaje_icfes_global if estudiante.puntaje_icfes_global else None
            x_row['trabaja_actualmente'] = 1 if estudiante.trabaja_actualmente else 0
            x_row['horas_trabajo_sem'] = estudiante.horas_trabajo_sem if estudiante.horas_trabajo_sem else 0
            x_row['estrato'] = estudiante.estrato if estudiante.estrato else None
            x_row['programa_academico'] = estudiante.programa_academico if estudiante.programa_academico else None
            
            # semestre_actual: semestre_historico (el semestre en el que estamos "simulando")
            x_row['semestre_actual'] = semestre_historico
            
            # porcentaje_asistencia_sem_ant: porcentaje_asistencia del HistorialAcademico del semestre_historico - 1
            if historial_sem_ant:
                x_row['porcentaje_asistencia_sem_ant'] = historial_sem_ant.porcentaje_asistencia
            else:
                x_row['porcentaje_asistencia_sem_ant'] = 1.0
            
            # Construir y_row: promedio del HistorialAcademico del semestre_historico + 1
            y_row = historial_siguiente.promedio
            
            # Si tenemos todos los datos necesarios, agregar la fila
            # Solo agregamos si y_row no es None
            if y_row is not None:
                fila_completa = {**x_row, TARGET: y_row}
                data_rows.append(fila_completa)
    
    # Convertir a DataFrame
    if not data_rows:
        # Retornar DataFrame vacío con las columnas correctas
        df = pd.DataFrame(columns=FEATURES + [TARGET])
    else:
        df = pd.DataFrame(data_rows)
        # Asegurar que todas las columnas estén presentes
        for col in FEATURES + [TARGET]:
            if col not in df.columns:
                df[col] = None
    
    return df

