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
        'name', 'created_at', 'learning_rate', 'epochs', 
        'accuracy', 'converged', 'input_columns_display', 'output_columns_display'
    ]
    list_filter = ['created_at', 'learning_rate', 'epochs', 'accuracy']
    search_fields = ['name', 'input_columns', 'output_columns']
    readonly_fields = ['created_at', 'final_weights', 'final_bias', 'accuracy', 'training_errors', 'weight_evolution']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Información General', {
            'fields': ('name', 'created_at', 'data_file')
        }),
        ('Parámetros de Entrenamiento', {
            'fields': ('learning_rate', 'epochs', 'input_columns', 'output_columns')
        }),
        ('Resultados', {
            'fields': ('final_weights', 'final_bias', 'accuracy', 'training_errors', 'weight_evolution'),
            'classes': ('collapse',)
        }),
    )
    
    def input_columns_display(self, obj):
        """Mostrar columnas de entrada de forma legible"""
        return ', '.join(obj.input_columns) if obj.input_columns else '-'
    input_columns_display.short_description = 'Columnas de Entrada'
    
    def output_columns_display(self, obj):
        """Mostrar columnas de salida de forma legible"""
        return ', '.join(obj.output_columns) if obj.output_columns else '-'
    output_columns_display.short_description = 'Columnas de Salida'
    
    def converged(self, obj):
        """Indicar si el entrenamiento convergió"""
        if obj.training_errors:
            return obj.training_errors[-1] == 0
        return False
    converged.boolean = True
    converged.short_description = 'Convergió'


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """
    Configuración del admin para el modelo Prediction
    """
    list_display = [
        'id', 'training', 'input_values_display', 'predicted_output', 'created_at'
    ]
    list_filter = ['created_at', 'training', 'predicted_output']
    search_fields = ['training__name', 'input_values']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    def input_values_display(self, obj):
        """Mostrar valores de entrada de forma legible"""
        if obj.input_values:
            return ', '.join([f"{k}: {v}" for k, v in obj.input_values.items()])
        return '-'
    input_values_display.short_description = 'Valores de Entrada'