"""
Modelos para la aplicación del perceptrón simple
"""

from django.db import models
from django.core.validators import MaxValueValidator
import json


class PerceptronTraining(models.Model):
    """
    Modelo para almacenar los resultados de entrenamiento del perceptrón
    """
    name = models.CharField(max_length=100, verbose_name="Nombre del entrenamiento")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
    
    # Parámetros de entrenamiento
    learning_rate = models.FloatField(verbose_name="Tasa de aprendizaje")
    epochs = models.IntegerField(verbose_name="Número de épocas")
    max_error = models.FloatField(verbose_name="Error máximo permitido", default=0.1, validators=[MaxValueValidator(0.1)])
    
    # Datos de entrada
    input_columns = models.JSONField(verbose_name="Columnas de entrada")
    output_columns = models.JSONField(verbose_name="Columnas de salida")
    
    # Resultados del entrenamiento
    final_weights = models.JSONField(verbose_name="Pesos finales")
    final_bias = models.FloatField(verbose_name="Sesgo final")
    accuracy = models.FloatField(verbose_name="Precisión final")
    training_errors = models.JSONField(verbose_name="Errores por época")
    weight_evolution = models.JSONField(verbose_name="Evolución de pesos")
    
    # Archivo de datos original
    data_file = models.FileField(upload_to='training_data/', verbose_name="Archivo de datos", null=True, blank=True)
    
    class Meta:
        verbose_name = "Entrenamiento de Perceptrón"
        verbose_name_plural = "Entrenamientos de Perceptrón"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.created_at.strftime('%d/%m/%Y %H:%M')}"


class Prediction(models.Model):
    """
    Modelo para almacenar predicciones individuales
    """
    training = models.ForeignKey(PerceptronTraining, on_delete=models.CASCADE, 
                               related_name='predictions', verbose_name="Entrenamiento")
    input_values = models.JSONField(verbose_name="Valores de entrada")
    predicted_output = models.FloatField(verbose_name="Salida predicha")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de predicción")
    
    class Meta:
        verbose_name = "Predicción"
        verbose_name_plural = "Predicciones"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Predicción {self.id} - {self.training.name}"