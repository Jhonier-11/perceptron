"""
Sistema de alertas automáticas para estudiantes en riesgo
"""

from .models import Estudiante, PrediccionRendimiento, AlertaEstudiante
from django.utils import timezone
from typing import List, Optional


def generar_alertas_estudiante(estudiante: Estudiante, prediccion: Optional[PrediccionRendimiento] = None) -> List[AlertaEstudiante]:
    """
    Genera alertas automáticas para un estudiante basado en sus datos y predicciones
    
    Args:
        estudiante: Instancia del modelo Estudiante
        prediccion: Predicción reciente (opcional)
        
    Returns:
        Lista de alertas generadas
    """
    alertas_generadas = []
    
    # 1. Alerta por predicción baja (< 10)
    if prediccion and prediccion.calificacion_predicha < 10:
        # Verificar si ya existe una alerta similar reciente (últimos 7 días)
        alerta_existente = AlertaEstudiante.objects.filter(
            estudiante=estudiante,
            tipo_alerta='prediccion_baja',
            fecha_creacion__gte=timezone.now() - timezone.timedelta(days=7)
        ).exists()
        
        if not alerta_existente:
            alerta = AlertaEstudiante.objects.create(
                estudiante=estudiante,
                tipo_alerta='prediccion_baja',
                mensaje=f"El estudiante tiene una predicción baja ({prediccion.calificacion_predicha:.2f}/20) en {prediccion.entrenamiento.columna_salida}. Se recomienda intervención temprana.",
                prioridad=2 if prediccion.calificacion_predicha >= 8 else 3,
                prediccion_relacionada=prediccion
            )
            alertas_generadas.append(alerta)
    
    # 2. Alerta por alto riesgo de reprobación (< 8)
    if prediccion and prediccion.calificacion_predicha < 8:
        alerta_existente = AlertaEstudiante.objects.filter(
            estudiante=estudiante,
            tipo_alerta='alto_riesgo',
            fecha_creacion__gte=timezone.now() - timezone.timedelta(days=7)
        ).exists()
        
        if not alerta_existente:
            alerta = AlertaEstudiante.objects.create(
                estudiante=estudiante,
                tipo_alerta='alto_riesgo',
                mensaje=f"⚠️ ALTO RIESGO: El estudiante tiene una predicción muy baja ({prediccion.calificacion_predicha:.2f}/20) en {prediccion.entrenamiento.columna_salida}. Riesgo alto de reprobación. Se requiere intervención inmediata.",
                prioridad=3,
                prediccion_relacionada=prediccion
            )
            alertas_generadas.append(alerta)
    
    # 3. Alerta por ausencias elevadas (> 10)
    if estudiante.ausencias > 10:
        alerta_existente = AlertaEstudiante.objects.filter(
            estudiante=estudiante,
            tipo_alerta='ausencias_elevadas',
            fecha_creacion__gte=timezone.now() - timezone.timedelta(days=30)
        ).exists()
        
        if not alerta_existente:
            alerta = AlertaEstudiante.objects.create(
                estudiante=estudiante,
                tipo_alerta='ausencias_elevadas',
                mensaje=f"El estudiante tiene {estudiante.ausencias} ausencias. Esto puede afectar significativamente su rendimiento académico.",
                prioridad=3 if estudiante.ausencias > 15 else 2
            )
            alertas_generadas.append(alerta)
    
    # 4. Alerta por fallos previos (> 2)
    if estudiante.fallos_previos > 2:
        alerta_existente = AlertaEstudiante.objects.filter(
            estudiante=estudiante,
            tipo_alerta='bajo_rendimiento',
            fecha_creacion__gte=timezone.now() - timezone.timedelta(days=30)
        ).exists()
        
        if not alerta_existente:
            alerta = AlertaEstudiante.objects.create(
                estudiante=estudiante,
                tipo_alerta='bajo_rendimiento',
                mensaje=f"El estudiante tiene {estudiante.fallos_previos} fallos previos. Historial de bajo rendimiento que requiere atención.",
                prioridad=3
            )
            alertas_generadas.append(alerta)
    
    # 5. Alerta por discrepancia grande entre predicción y realidad
    if prediccion and prediccion.calificacion_real is not None:
        diferencia = abs(prediccion.calificacion_predicha - prediccion.calificacion_real)
        if diferencia > 3:  # Diferencia mayor a 3 puntos
            alerta_existente = AlertaEstudiante.objects.filter(
                estudiante=estudiante,
                tipo_alerta='mejora_necesaria',
                fecha_creacion__gte=timezone.now() - timezone.timedelta(days=7)
            ).exists()
            
            if not alerta_existente:
                if prediccion.calificacion_real < prediccion.calificacion_predicha:
                    mensaje = f"La calificación real ({prediccion.calificacion_real:.2f}) es significativamente menor que la predicción ({prediccion.calificacion_predicha:.2f}). El estudiante necesita apoyo adicional."
                else:
                    mensaje = f"La calificación real ({prediccion.calificacion_real:.2f}) es mejor que la predicción ({prediccion.calificacion_predicha:.2f}). Posible mejora en el rendimiento."
                
                alerta = AlertaEstudiante.objects.create(
                    estudiante=estudiante,
                    tipo_alerta='mejora_necesaria',
                    mensaje=mensaje,
                    prioridad=2,
                    prediccion_relacionada=prediccion
                )
                alertas_generadas.append(alerta)
    
    # 6. Alerta por bajo tiempo de estudio (< 2)
    if estudiante.tiempo_estudio < 2:
        alerta_existente = AlertaEstudiante.objects.filter(
            estudiante=estudiante,
            tipo_alerta='mejora_necesaria',
            fecha_creacion__gte=timezone.now() - timezone.timedelta(days=30)
        ).exists()
        
        if not alerta_existente:
            alerta = AlertaEstudiante.objects.create(
                estudiante=estudiante,
                tipo_alerta='mejora_necesaria',
                mensaje=f"El estudiante reporta un tiempo de estudio bajo ({estudiante.tiempo_estudio}/4). Se recomienda aumentar el tiempo de estudio semanal.",
                prioridad=1
            )
            alertas_generadas.append(alerta)
    
    # 7. Alerta por calificaciones actuales bajas
    calificaciones = []
    if estudiante.calificacion_g1 is not None:
        calificaciones.append(estudiante.calificacion_g1)
    if estudiante.calificacion_g2 is not None:
        calificaciones.append(estudiante.calificacion_g2)
    if estudiante.calificacion_g3 is not None:
        calificaciones.append(estudiante.calificacion_g3)
    
    if calificaciones:
        promedio = sum(calificaciones) / len(calificaciones)
        if promedio < 10:
            alerta_existente = AlertaEstudiante.objects.filter(
                estudiante=estudiante,
                tipo_alerta='bajo_rendimiento',
                fecha_creacion__gte=timezone.now() - timezone.timedelta(days=30)
            ).exists()
            
            if not alerta_existente:
                alerta = AlertaEstudiante.objects.create(
                    estudiante=estudiante,
                    tipo_alerta='bajo_rendimiento',
                    mensaje=f"El estudiante tiene un promedio de calificaciones bajo ({promedio:.2f}/20). Se requiere apoyo académico.",
                    prioridad=3 if promedio < 8 else 2
                )
                alertas_generadas.append(alerta)
    
    return alertas_generadas


def generar_alertas_todos_estudiantes() -> int:
    """
    Genera alertas para todos los estudiantes que cumplan los criterios
    
    Returns:
        Número de alertas generadas
    """
    estudiantes = Estudiante.objects.all()
    total_alertas = 0
    
    for estudiante in estudiantes:
        # Obtener la última predicción si existe
        ultima_prediccion = PrediccionRendimiento.objects.filter(
            estudiante=estudiante
        ).select_related('entrenamiento').order_by('-fecha_prediccion').first()
        
        alertas = generar_alertas_estudiante(estudiante, ultima_prediccion)
        total_alertas += len(alertas)
    
    return total_alertas


def limpiar_alertas_antiguas(dias: int = 90):
    """
    Elimina alertas antiguas que ya fueron vistas
    
    Args:
        dias: Número de días de antigüedad para considerar una alerta como antigua
    """
    fecha_limite = timezone.now() - timezone.timedelta(days=dias)
    alertas_eliminadas = AlertaEstudiante.objects.filter(
        vista=True,
        fecha_creacion__lt=fecha_limite
    ).delete()
    
    return alertas_eliminadas[0] if alertas_eliminadas else 0

