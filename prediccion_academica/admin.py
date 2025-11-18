"""
Configuración del admin para la aplicación de Predicción del Rendimiento Académico
"""

from django.contrib import admin
from .models import Estudiante, EntrenamientoMLP, PrediccionRendimiento, AlertaEstudiante, HistorialAcademico


@admin.register(Estudiante)
class EstudianteAdmin(admin.ModelAdmin):
    list_display = ['identificacion', 'nombre', 'apellido', 'edad', 'sexo', 'semestre_actual', 'programa_academico', 'fecha_creacion']
    list_filter = ['sexo', 'direccion', 'estado_padres', 'programa_academico', 'trabaja_actualmente', 'fecha_creacion']
    search_fields = ['identificacion', 'nombre', 'apellido', 'programa_academico']
    readonly_fields = ['fecha_creacion', 'fecha_actualizacion', 'promedio_semestre_anterior', 'promedio_acumulado']
    fieldsets = (
        ('Información Personal', {
            'fields': ('identificacion', 'nombre', 'apellido', 'edad', 'sexo', 'direccion')
        }),
        ('Información Familiar', {
            'fields': ('tamano_familia', 'estado_padres', 'educacion_madre', 'educacion_padre', 
                      'trabajo_madre', 'trabajo_padre')
        }),
        ('Características Académicas', {
            'fields': ('tiempo_viaje', 'tiempo_estudio', 'fallos_previos', 'apoyo_escuela', 
                      'apoyo_familia', 'clases_pagadas', 'actividades_extra', 'guarderia', 
                      'quiere_superior', 'internet', 'relacion_romantica', 'relacion_familiar',
                      'tiempo_libre', 'salidas', 'alcohol_semana', 'alcohol_fin_semana', 
                      'salud', 'ausencias')
        }),
        ('Información Académica Universitaria', {
            'fields': ('semestre_actual', 'puntaje_icfes_global', 'estrato', 'programa_academico',
                      'trabaja_actualmente', 'horas_trabajo_sem', 'promedio_semestre_anterior', 
                      'promedio_acumulado')
        }),
        ('Fechas', {
            'fields': ('fecha_creacion', 'fecha_actualizacion')
        }),
    )


@admin.register(EntrenamientoMLP)
class EntrenamientoMLPAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'columna_salida', 'precision_entrenamiento', 'precision_validacion', 'fecha_creacion']
    list_filter = ['columna_salida', 'funcion_activacion', 'fecha_creacion']
    search_fields = ['nombre', 'descripcion']
    readonly_fields = ['fecha_creacion']
    fieldsets = (
        ('Información General', {
            'fields': ('nombre', 'descripcion', 'fecha_creacion')
        }),
        ('Arquitectura', {
            'fields': ('num_capas_ocultas', 'neuronas_por_capa', 'funcion_activacion')
        }),
        ('Parámetros de Entrenamiento', {
            'fields': ('tasa_aprendizaje', 'iteraciones', 'tamanio_batch', 'porcentaje_entrenamiento')
        }),
        ('Datos', {
            'fields': ('columnas_entrada', 'columna_salida')
        }),
        ('Resultados', {
            'fields': ('pesos_capas', 'sesgos_capas', 'precision_entrenamiento', 'precision_validacion',
                      'errores_entrenamiento', 'errores_validacion', 'metricas', 'archivo_modelo')
        }),
    )


@admin.register(PrediccionRendimiento)
class PrediccionRendimientoAdmin(admin.ModelAdmin):
    list_display = ['estudiante', 'entrenamiento', 'calificacion_predicha', 'calificacion_real', 'error_prediccion', 'fecha_prediccion']
    list_filter = ['entrenamiento', 'fecha_prediccion']
    search_fields = ['estudiante__nombre', 'estudiante__apellido', 'estudiante__identificacion']
    readonly_fields = ['fecha_prediccion']
    date_hierarchy = 'fecha_prediccion'


@admin.register(HistorialAcademico)
class HistorialAcademicoAdmin(admin.ModelAdmin):
    list_display = ['estudiante', 'semestre', 'promedio', 'porcentaje_asistencia', 'materias_reprobadas', 'creditos_inscritos', 'fecha_creacion']
    list_filter = ['semestre', 'materias_reprobadas', 'fecha_creacion']
    search_fields = ['estudiante__nombre', 'estudiante__apellido', 'estudiante__identificacion']
    readonly_fields = ['fecha_creacion', 'fecha_actualizacion']
    date_hierarchy = 'fecha_creacion'


@admin.register(AlertaEstudiante)
class AlertaEstudianteAdmin(admin.ModelAdmin):
    list_display = ['estudiante', 'tipo_alerta', 'prioridad', 'vista', 'fecha_creacion']
    list_filter = ['tipo_alerta', 'prioridad', 'vista', 'fecha_creacion']
    search_fields = ['estudiante__nombre', 'estudiante__apellido', 'mensaje']
    readonly_fields = ['fecha_creacion', 'fecha_vista']
    date_hierarchy = 'fecha_creacion'
