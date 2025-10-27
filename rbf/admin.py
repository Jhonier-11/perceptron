"""
Configuración del admin para la app RBF
"""
from django.contrib import admin
from .models import RBFTraining, RBFPrediction


@admin.register(RBFTraining)
class RBFTrainingAdmin(admin.ModelAdmin):
    list_display = ('nombre', 'fecha_creacion', 'num_centros', 'porcentaje_entrenamiento', 'convergencia')
    list_filter = ('fecha_creacion', 'convergencia', 'num_centros')
    search_fields = ('nombre',)
    readonly_fields = ('fecha_creacion',)
    
    fieldsets = (
        ('Información General', {
            'fields': ('nombre', 'fecha_creacion')
        }),
        ('Parámetros de Entrenamiento', {
            'fields': ('num_centros', 'porcentaje_entrenamiento', 'error_aproximacion')
        }),
        ('Datos', {
            'fields': ('columnas_entrada', 'columnas_salida', 'archivo_datos')
        }),
        ('Resultados', {
            'fields': ('centros_radiales', 'pesos_finales', 'umbral', 'convergencia')
        }),
        ('Métricas', {
            'fields': ('metricas_entrenamiento', 'metricas_prueba', 'estadisticas_normalizacion')
        }),
    )


@admin.register(RBFPrediction)
class RBFPredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'entrenamiento', 'salida_predicha', 'fecha_prediccion')
    list_filter = ('fecha_prediccion', 'entrenamiento')
    search_fields = ('entrenamiento__nombre',)
    readonly_fields = ('fecha_prediccion',)
    
    fieldsets = (
        ('Información General', {
            'fields': ('entrenamiento', 'fecha_prediccion')
        }),
        ('Predicción', {
            'fields': ('valores_entrada', 'salida_predicha')
        }),
    )