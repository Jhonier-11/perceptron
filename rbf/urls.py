"""
URL configuration for rbf app
"""
from django.urls import path
from . import views

app_name = 'rbf'

urlpatterns = [
    path('', views.inicio_rbf, name='inicio_rbf'),
    path('cargar-datos/', views.cargar_datos_rbf, name='cargar_datos_rbf'),
    path('configurar/', views.configurar_rbf, name='configurar_rbf'),
    path('entrenar/', views.entrenar_rbf, name='entrenar_rbf'),
    path('resultados/', views.resultados_rbf, name='resultados_rbf'),
    path('historial/', views.historial_rbf, name='historial_rbf'),
    path('detalles/<int:training_id>/', views.detalles_rbf, name='detalles_rbf'),
    path('predecir/<int:training_id>/', views.predecir_rbf, name='predecir_rbf'),
    path('historial-predicciones/<int:training_id>/', views.historial_predicciones_rbf, name='historial_predicciones'),
    path('descargar/<int:training_id>/', views.descargar_modelo_rbf, name='descargar_modelo_rbf'),
    path('eliminar/<int:training_id>/', views.eliminar_rbf, name='eliminar_rbf'),
]

