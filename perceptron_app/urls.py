"""
URLs para la aplicación del perceptrón simple
"""

from django.urls import path
from . import views

app_name = 'perceptron_app'

urlpatterns = [
    # Vista principal
    path('', views.home, name='home'),
    
    # Flujo de entrenamiento
    path('upload/', views.upload_data, name='upload_data'),
    path('configure/', views.configure_training, name='configure_training'),
    path('train/', views.train_perceptron, name='train_perceptron'),
    path('results/', views.training_results, name='training_results'),
    
    # Predicciones
    path('predict/', views.make_prediction, name='make_prediction'),
    path('predict/<int:training_id>/', views.make_prediction, name='make_prediction_with_id'),
    
    # Historial y gestión
    path('history/', views.training_history, name='training_history'),
    path('download/<int:training_id>/', views.download_weights, name='download_weights'),
    path('delete/<int:training_id>/', views.delete_training, name='delete_training'),
    
    # APIs AJAX
    path('api/train/', views.ajax_train, name='ajax_train'),
]
