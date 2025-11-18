"""
URLs para la aplicación de Predicción del Rendimiento Académico
"""

from django.urls import path
from . import views

app_name = 'prediccion_academica'

urlpatterns = [
    # Landing Page
    path('', views.landing_page, name='landing'),
    
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # Gestión de estudiantes
    path('estudiantes/', views.gestionar_estudiantes, name='estudiantes'),
    path('estudiantes/crear/', views.crear_estudiante, name='crear_estudiante'),
    path('estudiantes/<int:estudiante_id>/', views.vista_estudiante, name='estudiante'),
    path('estudiantes/<int:estudiante_id>/editar/', views.editar_estudiante, name='editar_estudiante'),
    path('estudiantes/cargar/', views.cargar_estudiantes, name='cargar_estudiantes'),
    
    # Gestión de historiales académicos
    path('historiales/', views.listar_historiales, name='listar_historiales'),
    path('historiales/crear/', views.crear_historial, name='crear_historial'),
    path('historiales/crear/<int:estudiante_id>/', views.crear_historial, name='crear_historial_estudiante'),
    path('historiales/<int:historial_id>/editar/', views.editar_historial, name='editar_historial'),
    path('historiales/<int:historial_id>/eliminar/', views.eliminar_historial, name='eliminar_historial'),
    path('estudiantes/<int:estudiante_id>/historiales/', views.listar_historiales, name='historiales_estudiante'),
    
    # Vista para docentes
    path('docentes/', views.vista_docentes, name='docentes'),
    
    # Entrenamiento MLP
    path('entrenar/', views.entrenar_mlp, name='entrenar_mlp'),
    path('entrenar/<int:entrenamiento_id>/resultados/', views.resultados_entrenamiento, name='resultados_entrenamiento'),
    
    # Predicciones
    path('predicciones/', views.ver_predicciones, name='predicciones'),
    
    # API
    path('api/prediccion/', views.api_prediccion, name='api_prediccion'),
    
    # Alertas
    path('alertas/<int:alerta_id>/marcar-vista/', views.marcar_alerta_vista, name='marcar_alerta_vista'),
    path('alertas/generar/', views.generar_alertas_manual, name='generar_alertas_manual'),
]

