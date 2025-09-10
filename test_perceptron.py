#!/usr/bin/env python3
"""
Script de prueba para verificar que el perceptrón funciona correctamente
"""

import sys
import os
import django

# Configurar Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'percep.settings')
django.setup()

import numpy as np
from perceptron_app.perceptron import PerceptronSimple

def test_and_gate():
    """Probar el perceptrón con la compuerta AND"""
    print("🧪 Probando compuerta lógica AND...")
    
    # Datos de la compuerta AND
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    # Crear y entrenar perceptrón
    perceptron = PerceptronSimple(learning_rate=0.5, max_epochs=100)
    results = perceptron.fit(X, y)
    
    # Verificar resultados
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"✅ Precisión: {accuracy:.1f}%")
    print(f"✅ Convergió: {results['converged']}")
    print(f"✅ Épocas utilizadas: {results['epochs_used']}")
    print(f"✅ Pesos finales: {[f'{w:.3f}' for w in results['final_weights']]}")
    print(f"✅ Sesgo final: {results['final_bias']:.3f}")
    
    return accuracy == 100.0

def test_or_gate():
    """Probar el perceptrón con la compuerta OR"""
    print("\n🧪 Probando compuerta lógica OR...")
    
    # Datos de la compuerta OR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    # Crear y entrenar perceptrón
    perceptron = PerceptronSimple(learning_rate=0.5, max_epochs=100)
    results = perceptron.fit(X, y)
    
    # Verificar resultados
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"✅ Precisión: {accuracy:.1f}%")
    print(f"✅ Convergió: {results['converged']}")
    print(f"✅ Épocas utilizadas: {results['epochs_used']}")
    print(f"✅ Pesos finales: {[f'{w:.3f}' for w in results['final_weights']]}")
    print(f"✅ Sesgo final: {results['final_bias']:.3f}")
    
    return accuracy == 100.0

def test_xor_gate():
    """Probar el perceptrón con la compuerta XOR (debería fallar)"""
    print("\n🧪 Probando compuerta lógica XOR (debería fallar)...")
    
    # Datos de la compuerta XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Crear y entrenar perceptrón
    perceptron = PerceptronSimple(learning_rate=0.5, max_epochs=1000)
    results = perceptron.fit(X, y)
    
    # Verificar resultados
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"❌ Precisión: {accuracy:.1f}% (esperado: < 100%)")
    print(f"❌ Convergió: {results['converged']} (esperado: False)")
    print(f"❌ Épocas utilizadas: {results['epochs_used']}")
    
    return accuracy < 100.0  # XOR no es linealmente separable

def test_prediction():
    """Probar la función de predicción individual"""
    print("\n🧪 Probando función de predicción...")
    
    # Entrenar con AND
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    perceptron = PerceptronSimple(learning_rate=0.5, max_epochs=100)
    perceptron.fit(X, y)
    
    # Probar predicciones individuales
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]
    
    all_correct = True
    for inputs, expected in test_cases:
        prediction = perceptron._predict_single(np.array(inputs))
        correct = prediction == expected
        all_correct = all_correct and correct
        print(f"  Entrada {inputs} → Predicción: {prediction}, Esperado: {expected} {'✅' if correct else '❌'}")
    
    return all_correct

def main():
    """Ejecutar todas las pruebas"""
    print("🚀 Iniciando pruebas del perceptrón simple...")
    print("=" * 50)
    
    tests = [
        ("Compuerta AND", test_and_gate),
        ("Compuerta OR", test_or_gate),
        ("Compuerta XOR (debería fallar)", test_xor_gate),
        ("Función de predicción", test_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{len(results)} pruebas pasaron")
    
    if passed == len(results):
        print("🎉 ¡Todas las pruebas pasaron! El perceptrón funciona correctamente.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa la implementación.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
