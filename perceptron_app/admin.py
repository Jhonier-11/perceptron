"""
Configuración del admin de Django para la aplicación del perceptrón
"""

from django.contrib import admin
from .models import PerceptronTraining, Prediction


@admin.register(PerceptronTraining)
class PerceptronTrainingAdmin(admin.ModelAdmin):
    """
    Configuración del admin para el modelo PerceptronTraining
    """
    list_display = [
        'nombre', 'fecha_creacion', 'tasa_aprendizaje', 'iteraciones', 
        'precision', 'converged', 'input_columns_display', 'output_columns_display'
    ]
    list_filter = ['fecha_creacion', 'tasa_aprendizaje', 'iteraciones', 'precision']
    search_fields = ['nombre', 'columnas_entrada', 'columnas_salida']
    readonly_fields = ['fecha_creacion', 'pesos_finales', 'sesgo_final', 'precision', 'errores_entrenamiento', 'evolucion_pesos']
    ordering = ['-fecha_creacion']
    
    fieldsets = (
        ('Información General', {
            'fields': ('nombre', 'fecha_creacion', 'archivo_datos')
        }),
        ('Parámetros de Entrenamiento', {
            'fields': ('tasa_aprendizaje', 'iteraciones', 'columnas_entrada', 'columnas_salida')
        }),
        ('Resultados', {
            'fields': ('pesos_finales', 'sesgo_final', 'precision', 'errores_entrenamiento', 'evolucion_pesos'),
            'classes': ('collapse',)
        }),
    )
    
    def input_columns_display(self, obj):
        """Mostrar columnas de entrada de forma legible"""
        return ', '.join(obj.columnas_entrada) if obj.columnas_entrada else '-'
    input_columns_display.short_description = 'Columnas de Entrada'
    
    def output_columns_display(self, obj):
        """Mostrar columnas de salida de forma legible"""
        return ', '.join(obj.columnas_salida) if obj.columnas_salida else '-'
    output_columns_display.short_description = 'Columnas de Salida'
    
    def converged(self, obj):
        """Indicar si el entrenamiento convergió"""
        if obj.errores_entrenamiento:
            return obj.errores_entrenamiento[-1] == 0
        return False
    converged.boolean = True
    converged.short_description = 'Convergió'


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """
    Configuración del admin para el modelo Prediction
    """
    list_display = [
        'id', 'entrenamiento', 'input_values_display', 'salida_predicha', 'fecha_prediccion'
    ]
    list_filter = ['fecha_prediccion', 'entrenamiento', 'salida_predicha']
    search_fields = ['entrenamiento__nombre', 'valores_entrada']
    readonly_fields = ['fecha_prediccion']
    ordering = ['-fecha_prediccion']
    
    def input_values_display(self, obj):
        """Mostrar valores de entrada de forma legible"""
        if obj.valores_entrada:
            return ', '.join([f"{k}: {v}" for k, v in obj.valores_entrada.items()])
        return '-'
    input_values_display.short_description = 'Valores de Entrada'