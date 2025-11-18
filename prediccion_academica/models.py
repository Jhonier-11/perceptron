"""
Modelos para la aplicación de Predicción del Rendimiento Académico
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import json


class Estudiante(models.Model):
    """
    Modelo para almacenar información de estudiantes
    """
    identificacion = models.CharField(max_length=20, unique=True, verbose_name="Identificación")
    nombre = models.CharField(max_length=100, verbose_name="Nombre")
    apellido = models.CharField(max_length=100, verbose_name="Apellido")
    edad = models.IntegerField(verbose_name="Edad", validators=[MinValueValidator(10), MaxValueValidator(30)])
    sexo = models.CharField(
        max_length=1, 
        choices=[('M', 'Masculino'), ('F', 'Femenino')],
        verbose_name="Sexo"
    )
    direccion = models.CharField(
        max_length=10,
        choices=[('U', 'Urbano'), ('R', 'Rural')],
        verbose_name="Dirección"
    )
    
    # Características familiares
    tamano_familia = models.CharField(
        max_length=3,
        choices=[('GT3', 'Mayor 3'), ('LE3', 'Menor igual 3')],
        verbose_name="Tamaño de Familia"
    )
    estado_padres = models.CharField(
        max_length=1,
        choices=[('T', 'Juntos'), ('S', 'Separados')],
        verbose_name="Estado de los Padres"
    )
    educacion_madre = models.IntegerField(
        choices=[(0, 'Ninguna'), (1, 'Primaria'), (2, '5to-9no'), (3, 'Secundaria'), (4, 'Superior')],
        verbose_name="Educación de la Madre"
    )
    educacion_padre = models.IntegerField(
        choices=[(0, 'Ninguna'), (1, 'Primaria'), (2, '5to-9no'), (3, 'Secundaria'), (4, 'Superior')],
        verbose_name="Educación del Padre"
    )
    trabajo_madre = models.CharField(max_length=50, verbose_name="Trabajo de la Madre")
    trabajo_padre = models.CharField(max_length=50, verbose_name="Trabajo del Padre")
    
    # Características académicas
    tiempo_viaje = models.IntegerField(
        verbose_name="Tiempo de Viaje",
        validators=[MinValueValidator(1), MaxValueValidator(4)],
        help_text="1 (<15 min), 2 (15-30 min), 3 (30-60 min), 4 (>60 min)"
    )
    tiempo_estudio = models.IntegerField(
        verbose_name="Tiempo de Estudio",
        validators=[MinValueValidator(1), MaxValueValidator(4)],
        help_text="1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)"
    )
    fallos_previos = models.IntegerField(
        default=0,
        verbose_name="Fallos Previos",
        validators=[MinValueValidator(0)]
    )
    apoyo_escuela = models.BooleanField(default=False, verbose_name="Apoyo de la Escuela")
    apoyo_familia = models.BooleanField(default=False, verbose_name="Apoyo de la Familia")
    clases_pagadas = models.BooleanField(default=False, verbose_name="Clases Pagadas")
    actividades_extra = models.BooleanField(default=False, verbose_name="Actividades Extraescolares")
    guarderia = models.BooleanField(default=False, verbose_name="Guardería")
    quiere_superior = models.BooleanField(default=False, verbose_name="Quiere Estudios Superiores")
    internet = models.BooleanField(default=False, verbose_name="Tiene Internet")
    relacion_romantica = models.BooleanField(default=False, verbose_name="Relación Romántica")
    relacion_familiar = models.IntegerField(
        verbose_name="Relación Familiar",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1 (muy mala) - 5 (excelente)"
    )
    tiempo_libre = models.IntegerField(
        verbose_name="Tiempo Libre",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1 (muy poco) - 5 (mucho)"
    )
    salidas = models.IntegerField(
        verbose_name="Salidas con Amigos",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1 (muy poco) - 5 (mucho)"
    )
    alcohol_semana = models.IntegerField(
        verbose_name="Consumo de Alcohol (Semana)",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1 (muy bajo) - 5 (muy alto)"
    )
    alcohol_fin_semana = models.IntegerField(
        verbose_name="Consumo de Alcohol (Fin de Semana)",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1 (muy bajo) - 5 (muy alto)"
    )
    salud = models.IntegerField(
        verbose_name="Salud",
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1 (muy mala) - 5 (excelente)"
    )
    ausencias = models.IntegerField(
        default=0,
        verbose_name="Ausencias",
        validators=[MinValueValidator(0)]
    )
    
    # Información académica universitaria
    horas_trabajo_sem = models.PositiveIntegerField(
        default=0,
        verbose_name="Horas de Trabajo Semanales"
    )
    promedio_semestre_anterior = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Promedio Semestre Anterior"
    )
    promedio_acumulado = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Promedio Acumulado"
    )
    semestre_actual = models.PositiveIntegerField(
        default=1,
        verbose_name="Semestre Actual"
    )
    puntaje_icfes_global = models.IntegerField(
        null=True,
        blank=True,
        verbose_name="Puntaje ICFES Global"
    )
    estrato = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(6)],
        verbose_name="Estrato"
    )
    programa_academico = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        verbose_name="Programa Académico"
    )
    trabaja_actualmente = models.BooleanField(
        default=False,
        verbose_name="Trabaja Actualmente"
    )
    
    fecha_creacion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de Creación")
    fecha_actualizacion = models.DateTimeField(auto_now=True, verbose_name="Fecha de Actualización")
    
    class Meta:
        verbose_name = "Estudiante"
        verbose_name_plural = "Estudiantes"
        ordering = ['-fecha_creacion']
        indexes = [
            models.Index(fields=['identificacion']),
            models.Index(fields=['-fecha_creacion']),
        ]
    
    def __str__(self):
        return f"{self.nombre} {self.apellido} ({self.identificacion})"
    
    def get_nombre_completo(self):
        """Retorna el nombre completo del estudiante"""
        return f"{self.nombre} {self.apellido}"
    
    @property
    def tendencia_rendimiento(self):
        """Calcula la tendencia de rendimiento: (promedio_semestre_anterior - promedio_acumulado)"""
        if self.promedio_semestre_anterior is not None and self.promedio_acumulado is not None:
            return self.promedio_semestre_anterior - self.promedio_acumulado
        return 0.0
    
    @property
    def porcentaje_asistencia_sem_ant(self):
        """Calcula el promedio de porcentaje_asistencia del semestre anterior"""
        semestre_anterior = self.semestre_actual - 1
        if semestre_anterior < 1:
            return 1.0
        
        historiales = self.historial_academico.filter(semestre=semestre_anterior)
        if not historiales.exists():
            return 1.0
        
        porcentajes = [h.porcentaje_asistencia for h in historiales if h.porcentaje_asistencia is not None]
        if not porcentajes:
            return 1.0
        
        return sum(porcentajes) / len(porcentajes)


class HistorialAcademico(models.Model):
    """
    Modelo para almacenar el historial académico por semestre de cada estudiante
    """
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.CASCADE,
        related_name='historial_academico',
        verbose_name="Estudiante"
    )
    semestre = models.IntegerField(
        verbose_name="Semestre",
        validators=[MinValueValidator(1)]
    )
    promedio = models.FloatField(
        verbose_name="Promedio Ponderado",
        validators=[MinValueValidator(0), MaxValueValidator(5)],
        help_text="Promedio ponderado del semestre (0-5)"
    )
    porcentaje_asistencia = models.FloatField(
        default=1.0,
        verbose_name="Porcentaje de Asistencia",
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Porcentaje de asistencia del semestre (0-1)"
    )
    materias_reprobadas = models.IntegerField(
        default=0,
        verbose_name="Materias Reprobadas",
        validators=[MinValueValidator(0)]
    )
    creditos_inscritos = models.IntegerField(
        default=0,
        verbose_name="Créditos Inscritos",
        validators=[MinValueValidator(0)]
    )
    
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Fecha de Creación"
    )
    fecha_actualizacion = models.DateTimeField(
        auto_now=True,
        verbose_name="Fecha de Actualización"
    )
    
    class Meta:
        verbose_name = "Historial Académico"
        verbose_name_plural = "Historiales Académicos"
        ordering = ['estudiante', 'semestre']
        unique_together = [['estudiante', 'semestre']]
        indexes = [
            models.Index(fields=['estudiante', 'semestre']),
            models.Index(fields=['semestre']),
        ]
    
    def __str__(self):
        return f"{self.estudiante.nombre} - Semestre {self.semestre} - Promedio: {self.promedio:.2f}"


class EntrenamientoMLP(models.Model):
    """
    Modelo para almacenar los resultados de entrenamiento del MLP
    """
    nombre = models.CharField(max_length=100, verbose_name="Nombre del Entrenamiento")
    fecha_creacion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de Creación")
    
    # Tipo de implementación
    tipo_implementacion = models.CharField(
        max_length=20,
        default='numpy',
        choices=[
            ('numpy', 'Implementación desde cero (NumPy)'),
            ('tensorflow', 'TensorFlow/Keras'),
        ],
        verbose_name="Tipo de Implementación",
        help_text="Tipo de implementación utilizada para entrenar el modelo"
    )
    
    # Parámetros de arquitectura
    num_capas_ocultas = models.IntegerField(
        default=1,
        verbose_name="Número de Capas Ocultas",
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    neuronas_por_capa = models.JSONField(
        verbose_name="Neuronas por Capa",
        help_text="Lista con el número de neuronas por cada capa oculta, ej: [32, 16]"
    )
    funcion_activacion = models.CharField(
        max_length=20,
        default='relu',
        choices=[
            ('relu', 'ReLU'),
            ('sigmoid', 'Sigmoid'),
            ('tanh', 'Tanh'),
        ],
        verbose_name="Función de Activación"
    )
    
    # Parámetros de entrenamiento
    tasa_aprendizaje = models.FloatField(
        verbose_name="Tasa de Aprendizaje",
        validators=[MinValueValidator(0.0001), MaxValueValidator(1.0)]
    )
    iteraciones = models.IntegerField(
        verbose_name="Número de Iteraciones",
        validators=[MinValueValidator(1), MaxValueValidator(10000)]
    )
    porcentaje_entrenamiento = models.FloatField(
        default=70.0,
        verbose_name="Porcentaje de Entrenamiento",
        validators=[MinValueValidator(50.0), MaxValueValidator(90.0)]
    )
    tamanio_batch = models.IntegerField(
        default=32,
        verbose_name="Tamaño de Batch",
        validators=[MinValueValidator(1), MaxValueValidator(256)]
    )
    
    # Datos de entrada/salida
    columnas_entrada = models.JSONField(
        verbose_name="Columnas de Entrada",
        help_text="Lista de nombres de columnas usadas como características"
    )
    columna_salida = models.CharField(
        max_length=50,
        choices=[('Y_promedio_sem_siguiente', 'Y_promedio_sem_siguiente')],
        verbose_name="Columna de Salida",
        default='Y_promedio_sem_siguiente'
    )
    num_caracteristicas_finales = models.IntegerField(
        verbose_name="Número de Características Finales",
        help_text="Número de características después del preprocesamiento (one-hot encoding)",
        null=True,
        blank=True
    )
    
    # Resultados del entrenamiento
    pesos_capas = models.JSONField(
        verbose_name="Pesos de las Capas",
        help_text="Lista de matrices de pesos para cada capa (solo para implementación NumPy)",
        null=True,
        blank=True
    )
    sesgos_capas = models.JSONField(
        verbose_name="Sesgos de las Capas",
        help_text="Lista de vectores de sesgos para cada capa (solo para implementación NumPy)",
        null=True,
        blank=True
    )
    modelo_tensorflow = models.FileField(
        upload_to='modelos_mlp/',
        null=True,
        blank=True,
        verbose_name="Modelo TensorFlow",
        help_text="Archivo del modelo TensorFlow serializado (.h5 o .keras)"
    )
    historial_entrenamiento_tf = models.JSONField(
        verbose_name="Historial de Entrenamiento TensorFlow",
        help_text="Historial de entrenamiento de TensorFlow (loss, metrics por epoch)",
        null=True,
        blank=True,
        default=dict
    )
    precision_entrenamiento = models.FloatField(
        verbose_name="Precisión de Entrenamiento",
        help_text="R² o precisión en el conjunto de entrenamiento"
    )
    precision_validacion = models.FloatField(
        verbose_name="Precisión de Validación",
        help_text="R² o precisión en el conjunto de validación"
    )
    errores_entrenamiento = models.JSONField(
        verbose_name="Errores de Entrenamiento",
        help_text="Lista de errores por iteración en entrenamiento"
    )
    errores_validacion = models.JSONField(
        verbose_name="Errores de Validación",
        help_text="Lista de errores por iteración en validación",
        null=True,
        blank=True
    )
    precision_entrenamiento_historial = models.JSONField(
        verbose_name="Historial de Precisión de Entrenamiento",
        help_text="Lista de precisión (R²) por iteración en entrenamiento",
        null=True,
        blank=True
    )
    precision_validacion_historial = models.JSONField(
        verbose_name="Historial de Precisión de Validación",
        help_text="Lista de precisión (R²) por iteración en validación",
        null=True,
        blank=True
    )
    metricas = models.JSONField(
        verbose_name="Métricas",
        help_text="Diccionario con métricas: MAE, RMSE, R², etc.",
        default=dict
    )
    
    # Archivo del modelo (opcional)
    archivo_modelo = models.FileField(
        upload_to='modelos_mlp/',
        null=True,
        blank=True,
        verbose_name="Archivo del Modelo"
    )
    
    # Información de preprocesamiento (para hacer predicciones)
    info_preprocesamiento = models.JSONField(
        verbose_name="Información de Preprocesamiento",
        help_text="Información sobre transformaciones aplicadas (mapeo_categorias, mapeo_labels, columnas_finales, etc.)",
        default=dict,
        null=True,
        blank=True
    )
    
    # Parámetros del scaler (media y desviación estándar)
    scaler_mean = models.JSONField(
        verbose_name="Media del Scaler",
        help_text="Media de cada característica para normalización",
        null=True,
        blank=True
    )
    scaler_scale = models.JSONField(
        verbose_name="Escala del Scaler",
        help_text="Desviación estándar de cada característica para normalización",
        null=True,
        blank=True
    )
    
    # Información adicional
    descripcion = models.TextField(
        blank=True,
        verbose_name="Descripción",
        help_text="Descripción opcional del entrenamiento"
    )
    
    class Meta:
        verbose_name = "Entrenamiento MLP"
        verbose_name_plural = "Entrenamientos MLP"
        ordering = ['-fecha_creacion']
        indexes = [
            models.Index(fields=['-fecha_creacion']),
            models.Index(fields=['columna_salida']),
        ]
    
    def __str__(self):
        return f"{self.nombre} - {self.columna_salida} ({self.fecha_creacion.strftime('%d/%m/%Y %H:%M')})"
    
    def get_arquitectura(self):
        """Retorna una descripción de la arquitectura"""
        return f"{len(self.columnas_entrada)} -> {' -> '.join(map(str, self.neuronas_por_capa))} -> 1"


class PrediccionRendimiento(models.Model):
    """
    Modelo para almacenar predicciones de rendimiento académico
    """
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.CASCADE,
        related_name='predicciones',
        verbose_name="Estudiante"
    )
    entrenamiento = models.ForeignKey(
        EntrenamientoMLP,
        on_delete=models.CASCADE,
        related_name='predicciones',
        verbose_name="Modelo de Entrenamiento"
    )
    calificacion_predicha = models.FloatField(
        verbose_name="Calificación Predicha",
        validators=[MinValueValidator(0), MaxValueValidator(20)]
    )
    calificacion_real = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Calificación Real",
        validators=[MinValueValidator(0), MaxValueValidator(20)],
        help_text="Calificación real si está disponible para comparación"
    )
    fecha_prediccion = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Fecha de Predicción"
    )
    caracteristicas_usadas = models.JSONField(
        verbose_name="Características Usadas",
        help_text="Diccionario con los valores de entrada usados para la predicción"
    )
    error_prediccion = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Error de Predicción",
        help_text="Diferencia absoluta entre predicción y valor real (si está disponible)"
    )
    
    class Meta:
        verbose_name = "Predicción de Rendimiento"
        verbose_name_plural = "Predicciones de Rendimiento"
        ordering = ['-fecha_prediccion']
        indexes = [
            models.Index(fields=['-fecha_prediccion']),
            models.Index(fields=['estudiante']),
            models.Index(fields=['entrenamiento']),
        ]
    
    def __str__(self):
        return f"Predicción para {self.estudiante.nombre} - {self.calificacion_predicha:.2f}"
    
    def calcular_error(self):
        """Calcula el error de predicción si hay calificación real"""
        if self.calificacion_real is not None:
            self.error_prediccion = abs(self.calificacion_predicha - self.calificacion_real)
            self.save()
        return self.error_prediccion


class AlertaEstudiante(models.Model):
    """
    Modelo para almacenar alertas sobre estudiantes en riesgo
    """
    TIPO_ALERTA_CHOICES = [
        ('bajo_rendimiento', 'Bajo Rendimiento'),
        ('alto_riesgo', 'Alto Riesgo de Reprobación'),
        ('mejora_necesaria', 'Mejora Necesaria'),
        ('ausencias_elevadas', 'Ausencias Elevadas'),
        ('prediccion_baja', 'Predicción de Calificación Baja'),
    ]
    
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.CASCADE,
        related_name='alertas',
        verbose_name="Estudiante"
    )
    tipo_alerta = models.CharField(
        max_length=50,
        choices=TIPO_ALERTA_CHOICES,
        verbose_name="Tipo de Alerta"
    )
    mensaje = models.TextField(
        verbose_name="Mensaje",
        help_text="Mensaje descriptivo de la alerta"
    )
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Fecha de Creación"
    )
    vista = models.BooleanField(
        default=False,
        verbose_name="Vista",
        help_text="Indica si la alerta ha sido vista por un docente"
    )
    fecha_vista = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Fecha de Vista",
        help_text="Fecha en que la alerta fue vista"
    )
    prioridad = models.IntegerField(
        default=1,
        choices=[(1, 'Baja'), (2, 'Media'), (3, 'Alta')],
        verbose_name="Prioridad"
    )
    
    # Relación con predicción (opcional)
    prediccion_relacionada = models.ForeignKey(
        PrediccionRendimiento,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='alertas',
        verbose_name="Predicción Relacionada"
    )
    
    class Meta:
        verbose_name = "Alerta de Estudiante"
        verbose_name_plural = "Alertas de Estudiantes"
        ordering = ['-prioridad', '-fecha_creacion']
        indexes = [
            models.Index(fields=['-fecha_creacion']),
            models.Index(fields=['estudiante']),
            models.Index(fields=['vista']),
            models.Index(fields=['-prioridad']),
        ]
    
    def __str__(self):
        return f"Alerta {self.get_tipo_alerta_display()} - {self.estudiante.nombre}"
    
    def marcar_como_vista(self):
        """Marca la alerta como vista"""
        from django.utils import timezone
        self.vista = True
        self.fecha_vista = timezone.now()
        self.save()
