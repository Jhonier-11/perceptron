"""
URLs para la aplicación del perceptrón simple
"""

from django.urls import path
from . import views

app_name = 'perceptron_app'

urlpatterns = [
    # Vista principal
    path('', views.inicio, name='inicio'),
    
    # Flujo de entrenamiento
    path('cargar-datos/', views.cargar_datos, name='cargar_datos'),
    path('configurar-entrenamiento/', views.configurar_entrenamiento, name='configurar_entrenamiento'),
    path('entrenar-perceptron/', views.entrenar_perceptron, name='entrenar_perceptron'),
    path('resultados-entrenamiento/', views.resultados_entrenamiento, name='resultados_entrenamiento'),
    
    # Predicciones
    path('hacer-prediccion/', views.hacer_prediccion, name='hacer_prediccion'),
    path('hacer-prediccion/<int:training_id>/', views.hacer_prediccion, name='hacer_prediccion_con_id'),
    
    # Historial y gestión
    path('historial-entrenamientos/', views.historial_entrenamientos, name='historial_entrenamientos'),
    path('descargar-pesos/<int:training_id>/', views.descargar_pesos, name='descargar_pesos'),
    path('eliminar-entrenamiento/<int:training_id>/', views.eliminar_entrenamiento, name='eliminar_entrenamiento'),
    
    # APIs AJAX
    path('api/entrenar/', views.ajax_entrenar, name='ajax_entrenar'),
]
