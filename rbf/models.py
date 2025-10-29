"""
Modelos para la aplicación de Red Neuronal RBF
"""
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
import json


class RBFTraining(models.Model):
    """
    Modelo para almacenar los resultados de entrenamiento de la red RBF
    """
    nombre = models.CharField(max_length=100, verbose_name="Nombre del entrenamiento")
    fecha_creacion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de creación")
    
    # Parámetros de entrenamiento
    num_centros = models.IntegerField(
        verbose_name="Número de centros radiales",
        validators=[MinValueValidator(2), MaxValueValidator(50)]
    )
    porcentaje_entrenamiento = models.FloatField(
        verbose_name="Porcentaje de entrenamiento",
        validators=[MinValueValidator(50.0), MaxValueValidator(90.0)],
        default=70.0
    )
    error_aproximacion = models.FloatField(
        verbose_name="Error de aproximación óptimo",
        validators=[MaxValueValidator(1.0)],
        default=0.1
    )
    
    # Datos de entrada
    columnas_entrada = models.JSONField(verbose_name="Columnas de entrada")
    columnas_salida = models.JSONField(verbose_name="Columnas de salida")
    
    # Resultados del entrenamiento
    centros_radiales = models.JSONField(verbose_name="Centros radiales (Rj)")
    pesos_finales = models.JSONField(verbose_name="Pesos finales (W)")
    umbral = models.FloatField(verbose_name="Umbral (W0)")
    
    # Métricas
    metricas_entrenamiento = models.JSONField(verbose_name="Métricas de entrenamiento (EG, MAE, RMSE)")
    metricas_prueba = models.JSONField(verbose_name="Métricas de prueba (EG, MAE, RMSE)")
    
    # Valores reales y predichos para regenerar gráficos (guardados como listas)
    y_train_real = models.JSONField(
        verbose_name="Valores reales de entrenamiento",
        null=True,
        blank=True,
        help_text="Valores reales (Y) del conjunto de entrenamiento"
    )
    y_train_pred = models.JSONField(
        verbose_name="Valores predichos de entrenamiento",
        null=True,
        blank=True,
        help_text="Valores predichos (Y_pred) del conjunto de entrenamiento"
    )
    y_test_real = models.JSONField(
        verbose_name="Valores reales de prueba",
        null=True,
        blank=True,
        help_text="Valores reales (Y) del conjunto de prueba"
    )
    y_test_pred = models.JSONField(
        verbose_name="Valores predichos de prueba",
        null=True,
        blank=True,
        help_text="Valores predichos (Y_pred) del conjunto de prueba"
    )
    
    # Estado de convergencia
    convergencia = models.BooleanField(
        verbose_name="Convergencia alcanzada",
        default=False
    )
    
    # Estadísticas de normalización (para poder desnormalizar datos)
    estadisticas_normalizacion = models.JSONField(
        verbose_name="Estadísticas de normalización (mean, std)",
        null=True,
        blank=True,
        default=dict
    )
    
    # Archivo de datos original
    archivo_datos = models.FileField(
        upload_to='rbf_training_data/',
        verbose_name="Archivo de datos",
        null=True,
        blank=True
    )
    
    class Meta:
        verbose_name = "Entrenamiento RBF"
        verbose_name_plural = "Entrenamientos RBF"
        ordering = ['-fecha_creacion']
    
    def __str__(self):
        return f"{self.nombre} - {self.fecha_creacion.strftime('%d/%m/%Y %H:%M')}"


class RBFPrediction(models.Model):
    """
    Modelo para almacenar predicciones individuales realizadas con la red RBF
    """
    entrenamiento = models.ForeignKey(RBFTraining, on_delete=models.CASCADE, 
                                      related_name='predicciones', 
                                      verbose_name="Entrenamiento")
    valores_entrada = models.JSONField(verbose_name="Valores de entrada")
    salida_predicha = models.FloatField(verbose_name="Salida predicha")
    fecha_prediccion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de predicción")
    
    class Meta:
        verbose_name = "Predicción RBF"
        verbose_name_plural = "Predicciones RBF"
        ordering = ['-fecha_prediccion']
    
    def __str__(self):
        return f"Predicción {self.id} - {self.entrenamiento.nombre}"