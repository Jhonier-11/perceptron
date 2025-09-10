"""
Implementación del Perceptrón Simple desde cero
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI para evitar problemas de hilos
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Tuple, Dict, Any


class PerceptronSimple:
    """
    Implementación del Perceptrón Simple desde cero usando solo NumPy
    """
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 100, max_error: float = 0.1):
        """
        Inicializa el perceptrón simple
        
        Args:
            learning_rate (float): Tasa de aprendizaje (eta)
            max_epochs (int): Número máximo de épocas de entrenamiento
            max_error (float): Error máximo permitido para detener el entrenamiento
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.max_error = max_error
        self.weights = None
        self.bias = 0.0
        self.training_errors = []
        self.weight_evolution = []
        
    def _step_function(self, x: float) -> int:
        """
        Función de activación escalón (step function)
        
        Args:
            x (float): Valor de entrada
            
        Returns:
            int: 1 si x >= 0, 0 en caso contrario
        """
        return 1 if x >= 0 else 0
    
    def _initialize_weights(self, n_features: int) -> None:
        """
        Inicializa los pesos aleatoriamente
        
        Args:
            n_features (int): Número de características de entrada
        """
        # Inicialización aleatoria de pesos entre -0.5 y 0.5
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        self.bias = np.random.uniform(-0.5, 0.5)
        
    def _predict_single(self, X: np.ndarray) -> int:
        """
        Realiza una predicción para una sola muestra
        
        Args:
            X (np.ndarray): Vector de características de entrada
            
        Returns:
            int: Predicción (0 o 1)
        """
        # Cálculo de la suma ponderada: w1*x1 + w2*x2 + ... + wn*xn + bias
        linear_output = np.dot(self.weights, X) + self.bias
        
        # Aplicar función de activación escalón
        return self._step_function(linear_output)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones para múltiples muestras
        
        Args:
            X (np.ndarray): Matriz de características de entrada
            
        Returns:
            np.ndarray: Array de predicciones
        """
        predictions = []
        for sample in X:
            predictions.append(self._predict_single(sample))
        return np.array(predictions)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Entrena el perceptrón usando la regla del perceptrón
        
        Args:
            X (np.ndarray): Matriz de características de entrada
            y (np.ndarray): Vector de etiquetas objetivo
            
        Returns:
            Dict[str, Any]: Diccionario con información del entrenamiento
        """
        n_samples, n_features = X.shape
        
        # Inicializar pesos aleatoriamente solo si no están ya inicializados
        if not hasattr(self, 'weights') or len(self.weights) != n_features:
            self._initialize_weights(n_features)
        
        # Listas para almacenar el progreso del entrenamiento
        self.training_errors = []
        self.weight_evolution = []
        
        print(f"Iniciando entrenamiento del perceptrón...")
        print(f"Tasa de aprendizaje: {self.learning_rate}")
        print(f"Número máximo de épocas: {self.max_epochs}")
        print(f"Error máximo permitido: {self.max_error}")
        print(f"Número de características: {n_features}")
        print(f"Número de muestras: {n_samples}")
        print("-" * 50)
        
        # Entrenamiento por épocas
        for epoch in range(self.max_epochs):
            epoch_errors = 0
            
            # Guardar pesos actuales para visualización
            self.weight_evolution.append({
                'epoch': epoch + 1,
                'weights': self.weights.copy().tolist(),
                'bias': float(self.bias)
            })
            
            # Entrenar con cada muestra
            for i in range(n_samples):
                # Predicción actual
                prediction = self._predict_single(X[i])
                
                # Calcular error
                error = y[i] - prediction
                
                # Actualizar pesos y sesgo si hay error
                if error != 0:
                    # Regla del perceptrón: w = w + eta * error * x
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    epoch_errors += 1
            
            # Guardar error de la época
            self.training_errors.append(epoch_errors)
            
            # Calcular error relativo (errores por muestra)
            error_rate = epoch_errors / n_samples
            
            # Mostrar progreso cada 10 épocas
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Época {epoch + 1:3d}: Errores = {epoch_errors:2d}, "
                      f"Tasa de error = {error_rate:.3f}, "
                      f"Pesos = {[f'{w:.3f}' for w in self.weights]}, "
                      f"Sesgo = {self.bias:.3f}")
            
            # Parar si se alcanza el error máximo permitido o convergencia total
            if error_rate <= self.max_error:
                if epoch_errors == 0:
                    print(f"\n¡Convergencia total alcanzada en la época {epoch + 1}!")
                else:
                    print(f"\n¡Error objetivo alcanzado en la época {epoch + 1}! "
                          f"Tasa de error: {error_rate:.3f} <= {self.max_error}")
                break
        
        # Calcular precisión final
        final_predictions = self.predict(X)
        accuracy = np.mean(final_predictions == y) * 100
        
        print(f"\nEntrenamiento completado!")
        print(f"Precisión final: {accuracy:.2f}%")
        print(f"Pesos finales: {[f'{w:.3f}' for w in self.weights]}")
        print(f"Sesgo final: {self.bias:.3f}")
        
        return {
            'final_weights': self.weights.tolist(),
            'final_bias': float(self.bias),
            'accuracy': accuracy,
            'training_errors': self.training_errors,
            'weight_evolution': self.weight_evolution,
            'converged': epoch_errors == 0,
            'epochs_used': epoch + 1
        }
    
    def get_training_summary(self) -> str:
        """
        Genera un resumen del entrenamiento en formato texto
        
        Returns:
            str: Resumen del entrenamiento
        """
        if not self.training_errors:
            return "No se ha realizado entrenamiento aún."
        
        summary = f"""
=== RESUMEN DEL ENTRENAMIENTO ===
Tasa de aprendizaje: {self.learning_rate}
Épocas utilizadas: {len(self.training_errors)}
Pesos finales: {[f'{w:.3f}' for w in self.weights]}
Sesgo final: {self.bias:.3f}
Precisión final: {np.mean(self.predict(np.array([[0,0],[0,1],[1,0],[1,1]])) == np.array([0,0,0,1])) * 100:.2f}%

Evolución de errores por época:
"""
        for i, errors in enumerate(self.training_errors):
            summary += f"Época {i+1:3d}: {errors:2d} errores\n"
        
        return summary
    
    def create_error_plot(self) -> str:
        """
        Crea un gráfico de la evolución del error durante el entrenamiento
        
        Returns:
            str: Imagen codificada en base64
        """
        if not self.training_errors:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.training_errors) + 1), self.training_errors, 'b-', linewidth=2)
        plt.title('Evolución del Error durante el Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Número de Errores', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def create_weights_plot(self) -> str:
        """
        Crea un gráfico de la evolución de los pesos durante el entrenamiento
        
        Returns:
            str: Imagen codificada en base64
        """
        if not self.weight_evolution:
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Extraer datos de pesos
        epochs = [w['epoch'] for w in self.weight_evolution]
        weights_data = np.array([w['weights'] for w in self.weight_evolution])
        bias_data = [w['bias'] for w in self.weight_evolution]
        
        # Graficar cada peso
        for i in range(weights_data.shape[1]):
            plt.plot(epochs, weights_data[:, i], label=f'Peso w{i+1}', linewidth=2, marker='o', markersize=4)
        
        # Graficar sesgo
        plt.plot(epochs, bias_data, label='Sesgo (bias)', linewidth=2, marker='s', markersize=4)
        
        plt.title('Evolución de los Pesos durante el Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Valor del Peso', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
