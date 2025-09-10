#!/usr/bin/env python3
"""
Script para probar que todas las URLs funcionan correctamente
"""

import os
import sys
import django

# Configurar Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'percep.settings')
django.setup()

from django.urls import reverse
from django.test import Client

def test_urls():
    """Probar que todas las URLs funcionan"""
    print("🧪 Probando URLs de la aplicación...")
    
    # URLs a probar
    urls_to_test = [
        'perceptron_app:home',
        'perceptron_app:upload_data',
        'perceptron_app:configure_training',
        'perceptron_app:train_perceptron',
        'perceptron_app:training_results',
        'perceptron_app:make_prediction',
        'perceptron_app:training_history',
    ]
    
    print("\n📋 Verificando URLs:")
    for url_name in urls_to_test:
        try:
            url = reverse(url_name)
            print(f"✅ {url_name}: {url}")
        except Exception as e:
            print(f"❌ {url_name}: ERROR - {e}")
    
    print("\n🌐 Probando respuestas HTTP:")
    client = Client()
    
    # Probar página principal
    try:
        response = client.get('/')
        print(f"✅ Página principal: {response.status_code}")
    except Exception as e:
        print(f"❌ Página principal: ERROR - {e}")
    
    # Probar carga de datos
    try:
        response = client.get('/upload/')
        print(f"✅ Carga de datos: {response.status_code}")
    except Exception as e:
        print(f"❌ Carga de datos: ERROR - {e}")
    
    # Probar historial
    try:
        response = client.get('/history/')
        print(f"✅ Historial: {response.status_code}")
    except Exception as e:
        print(f"❌ Historial: ERROR - {e}")
    
    print("\n🎉 Pruebas de URLs completadas!")

if __name__ == "__main__":
    test_urls()
