import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI para evitar problemas de hilos
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Tuple, Dict, Any


class PerceptronSimple:
    """
    Implementación del Perceptrón Simple
    """
    
    def __init__(self, tasa_aprendizaje: float = 0.1, max_iteraciones: int = 100, error_maximo: float = 0.1):
        """
        Inicializa el perceptrón simple
        
        Args:
            tasa_aprendizaje (float): Tasa de aprendizaje 
            max_iteraciones (int): Número máximo de iteraciones de entrenamiento
            error_maximo (float): Error máximo permitido para detener el entrenamiento
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_iteraciones = max_iteraciones
        self.error_maximo = error_maximo
        self.pesos = None
        self.sesgo = 0.0
        self.errores_entrenamiento = []
        self.evolucion_pesos = []
        
    def _funcion_escalon(self, x: float) -> int:
        """
        Función de activación
        
        Args:
            x (float): Valor de entrada
            
        Returns:
            int: 1 si x >= 0, 0 en caso contrario
        """
        return 1 if x >= 0 else 0
    
    def _inicializar_pesos(self, num_caracteristicas: int) -> None:
        """
        Inicializa los pesos aleatoriamente
        
        Args:
            num_caracteristicas (int): Número de características de entrada
        """
        # Inicialización aleatoria de pesos entre -1 y 1
        self.pesos = np.random.uniform(-1, 1, num_caracteristicas)
        self.sesgo = np.random.uniform(-1, 1)
        
    def _predecir_individual(self, X: np.ndarray) -> int:
        """
        Realiza una predicción para una sola muestra
        
        Args:
            X (np.ndarray): Vector de características de entrada
            
        Returns:
            int: Predicción (0 o 1)
        """
        # Asegurar que X es un array de NumPy
        X = np.array(X, dtype=float)
        
        # Cálculo de la suma ponderada: w1*x1 + w2*x2 + ... + wn*xn + sesgo
        salida_lineal = np.dot(self.pesos, X) + self.sesgo
        
        # Aplicar función de activación escalón
        return self._funcion_escalon(salida_lineal)
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones para múltiples muestras
        
        Args:
            X (np.ndarray): Matriz de características de entrada
            
        Returns:
            np.ndarray: Array de predicciones
        """
        # Asegurar que X es un array de NumPy
        X = np.array(X, dtype=float)
        predicciones = []
        for muestra in X:
            predicciones.append(self._predecir_individual(muestra))
        return np.array(predicciones)
    
    def entrenar(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Entrena el perceptrón usando la regla del perceptrón
        
        Args:
            X (np.ndarray): Matriz de características de entrada
            y (np.ndarray): Vector de etiquetas objetivo
            
        Returns:
            Dict[str, Any]: Diccionario con información del entrenamiento
        """
        # Asegurar que X e y son arrays de NumPy
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        num_muestras, num_caracteristicas = X.shape
        
        # Inicializar pesos aleatoriamente solo si no están ya inicializados
        if not hasattr(self, 'pesos') or len(self.pesos) != num_caracteristicas:
            self._inicializar_pesos(num_caracteristicas)
        
        # Listas para almacenar el progreso del entrenamiento
        self.errores_entrenamiento = []
        self.evolucion_pesos = []
        
        print(f"Iniciando entrenamiento del perceptrón...")
        print(f"Tasa de aprendizaje: {self.tasa_aprendizaje}")
        print(f"Número máximo de iteraciones: {self.max_iteraciones}")
        print(f"Error máximo permitido: {self.error_maximo}")
        print(f"Número de características: {num_caracteristicas}")
        print(f"Número de muestras: {num_muestras}")
        print("-" * 50)
        
        # Entrenamiento por iteraciones
        for iteracion in range(self.max_iteraciones):
            errores_iteracion = 0
            
            # Guardar pesos actuales para visualización
            self.evolucion_pesos.append({
                'iteracion': iteracion + 1,
                'pesos': self.pesos.copy().tolist(),
                'sesgo': float(self.sesgo)
            })
            
            # Entrenar con cada muestra
            for i in range(num_muestras):
                # Predicción actual
                prediccion = self._predecir_individual(X[i])
                
                # Calcular error
                error = y[i] - prediccion
                
                # Actualizar pesos y sesgo si hay error
                if error != 0:
                    # Regla del perceptrón: w = w + eta * error * x
                    self.pesos += self.tasa_aprendizaje * error * X[i]
                    self.sesgo += self.tasa_aprendizaje * error
                    errores_iteracion += 1
            
            # Guardar error de la iteración
            self.errores_entrenamiento.append(errores_iteracion)
            
            # Calcular error relativo (errores por muestra)
            tasa_error = errores_iteracion / num_muestras
            
            # Mostrar progreso cada 10 iteraciones
            if (iteracion + 1) % 10 == 0 or iteracion == 0:
                print(f"Iteración {iteracion + 1:3d}: Errores = {errores_iteracion:2d}, "
                      f"Tasa de error = {tasa_error:.3f}, "
                      f"Pesos = {[f'{w:.3f}' for w in self.pesos]}, "
                      f"Sesgo = {self.sesgo:.3f}")
            
            # Parar si se alcanza el error máximo permitido o convergencia total
            if tasa_error <= self.error_maximo:
                if errores_iteracion == 0:
                    print(f"\n¡Convergencia total alcanzada en la iteración {iteracion + 1}!")
                else:
                    print(f"\n¡Error objetivo alcanzado en la iteración {iteracion + 1}! "
                          f"Tasa de error: {tasa_error:.3f} <= {self.error_maximo}")
                break
        
        # Calcular precisión final
        predicciones_finales = self.predecir(X)
        precision = np.mean(predicciones_finales == y) * 100
        
        print(f"\nEntrenamiento completado!")
        print(f"Precisión final: {precision:.2f}%")
        print(f"Pesos finales: {[f'{w:.3f}' for w in self.pesos]}")
        print(f"Sesgo final: {self.sesgo:.3f}")
        
        return {
            'pesos_finales': self.pesos.tolist(),
            'sesgo_final': float(self.sesgo),
            'precision': precision,
            'errores_entrenamiento': self.errores_entrenamiento,
            'evolucion_pesos': self.evolucion_pesos,
            'converged': errores_iteracion == 0,
            'iteraciones_utilizadas': iteracion + 1
        }
    
    def obtener_resumen_entrenamiento(self) -> str:
        """
        Genera un resumen del entrenamiento en formato texto
        
        Returns:
            str: Resumen del entrenamiento
        """
        if not self.errores_entrenamiento:
            return "No se ha realizado entrenamiento aún."
        
        resumen = f"""
                    === RESUMEN DEL ENTRENAMIENTO ===
                    Tasa de aprendizaje: {self.tasa_aprendizaje}
                    Iteraciones utilizadas: {len(self.errores_entrenamiento)}
                    Pesos finales: {[f'{w:.3f}' for w in self.pesos]}
                    Sesgo final: {self.sesgo:.3f}
                    Precisión final: {np.mean(self.predecir(np.array([[0,0],[0,1],[1,0],[1,1]])) == np.array([0,0,0,1])) * 100:.2f}%

                    Evolución de errores por iteración:
                    """
        for i, errores in enumerate(self.errores_entrenamiento):
            resumen += f"Iteración {i+1:3d}: {errores:2d} errores\n"
        
        return resumen
    
    def crear_grafico_errores(self) -> str:
        """
        Crea un gráfico de la evolución del error durante el entrenamiento
        
        Returns:
            str: Imagen codificada en base64
        """
        if not self.errores_entrenamiento:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errores_entrenamiento) + 1), self.errores_entrenamiento, 'b-', linewidth=2)
        plt.title('Evolución del Error durante el Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Iteración', fontsize=12)
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
    
    def crear_grafico_pesos(self) -> str:
        """
        Crea un gráfico de la evolución de los pesos durante el entrenamiento
        
        Returns:
            str: Imagen codificada en base64
        """
        if not self.evolucion_pesos:
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Extraer datos de pesos
        iteraciones = [w['iteracion'] for w in self.evolucion_pesos]
        datos_pesos = np.array([w['pesos'] for w in self.evolucion_pesos])
        datos_sesgo = [w['sesgo'] for w in self.evolucion_pesos]
        
        # Graficar cada peso
        for i in range(datos_pesos.shape[1]):
            plt.plot(iteraciones, datos_pesos[:, i], label=f'Peso w{i+1}', linewidth=2, marker='o', markersize=4)
        
        # Graficar sesgo
        plt.plot(iteraciones, datos_sesgo, label='Sesgo (bias)', linewidth=2, marker='s', markersize=4)
        
        plt.title('Evolución de los Pesos durante el Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Iteración', fontsize=12)
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
