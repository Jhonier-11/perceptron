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
    nombre = models.CharField(max_length=100, verbose_name="Nombre del entrenamiento")
    fecha_creacion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
    
    # Parámetros de entrenamiento
    tasa_aprendizaje = models.FloatField(verbose_name="Tasa de aprendizaje")
    iteraciones = models.IntegerField(verbose_name="Número de iteraciones")
    error_maximo = models.FloatField(verbose_name="Error máximo permitido", default=0.1, validators=[MaxValueValidator(0.1)])
    
    # Datos de entrada
    columnas_entrada = models.JSONField(verbose_name="Columnas de entrada")
    columnas_salida = models.JSONField(verbose_name="Columnas de salida")
    
    # Resultados del entrenamiento
    pesos_finales = models.JSONField(verbose_name="Pesos finales")
    sesgo_final = models.FloatField(verbose_name="Sesgo final")
    precision = models.FloatField(verbose_name="Precisión final")
    errores_entrenamiento = models.JSONField(verbose_name="Errores por época")
    evolucion_pesos = models.JSONField(verbose_name="Evolución de pesos")
    
    # Archivo de datos original
    archivo_datos = models.FileField(upload_to='training_data/', verbose_name="Archivo de datos", null=True, blank=True)
    
    class Meta:
        verbose_name = "Entrenamiento de Perceptrón"
        verbose_name_plural = "Entrenamientos de Perceptrón"
        ordering = ['-fecha_creacion']
    
    def __str__(self):
        return f"{self.nombre} - {self.fecha_creacion.strftime('%d/%m/%Y %H:%M')}"


class Prediction(models.Model):
    """
    Modelo para almacenar predicciones individuales
    """
    entrenamiento = models.ForeignKey(PerceptronTraining, on_delete=models.CASCADE, 
                               related_name='predicciones', verbose_name="Entrenamiento")
    valores_entrada = models.JSONField(verbose_name="Valores de entrada")
    salida_predicha = models.FloatField(verbose_name="Salida predicha")
    fecha_prediccion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de predicción")
    
    class Meta:
        verbose_name = "Predicción"
        verbose_name_plural = "Predicciones"
        ordering = ['-fecha_prediccion']
    
    def __str__(self):
        return f"Predicción {self.id} - {self.entrenamiento.nombre}"