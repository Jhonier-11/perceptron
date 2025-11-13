"""
Implementación del Perceptrón Multicapa (MLP) desde cero usando NumPy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Tuple, Dict, Any, Optional
import json


class MLP:
    """
    Implementación del Perceptrón Multicapa (MLP) desde cero
    """
    
    def __init__(
        self,
        arquitectura: List[int],
        tasa_aprendizaje: float = 0.01,
        funcion_activacion: str = 'relu',
        inicializacion: str = 'xavier'
    ):
        """
        Inicializa el MLP
        
        Args:
            arquitectura: Lista con el número de neuronas por capa [input, hidden1, hidden2, ..., output]
            tasa_aprendizaje: Tasa de aprendizaje
            funcion_activacion: Función de activación ('relu', 'sigmoid', 'tanh')
            inicializacion: Método de inicialización ('xavier', 'he', 'random')
        """
        self.arquitectura = arquitectura
        self.tasa_aprendizaje = tasa_aprendizaje
        self.funcion_activacion = funcion_activacion
        self.inicializacion = inicializacion
        
        # Inicializar pesos y sesgos
        self.pesos = []
        self.sesgos = []
        self._inicializar_pesos()
        
        # Historial de entrenamiento
        self.historial_errores_entrenamiento = []
        self.historial_errores_validacion = []
        self.historial_precision_entrenamiento = []
        self.historial_precision_validacion = []
        
    def _inicializar_pesos(self):
        """Inicializa los pesos y sesgos según la arquitectura"""
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(self.arquitectura) - 1):
            n_entrada = self.arquitectura[i]
            n_salida = self.arquitectura[i + 1]
            
            # Inicializar pesos según el método seleccionado
            if self.inicializacion == 'xavier':
                # Xavier/Glorot initialization
                limite = np.sqrt(6.0 / (n_entrada + n_salida))
                peso = np.random.uniform(-limite, limite, (n_entrada, n_salida))
            elif self.inicializacion == 'he':
                # He initialization (para ReLU)
                limite = np.sqrt(2.0 / n_entrada)
                peso = np.random.normal(0, limite, (n_entrada, n_salida))
            else:
                # Random initialization
                peso = np.random.uniform(-0.5, 0.5, (n_entrada, n_salida))
            
            # Inicializar sesgos en cero
            sesgo = np.zeros((1, n_salida))
            
            self.pesos.append(peso)
            self.sesgos.append(sesgo)
    
    def _activacion(self, x: np.ndarray, derivada: bool = False) -> np.ndarray:
        """
        Aplica la función de activación
        
        Args:
            x: Entrada
            derivada: Si es True, retorna la derivada de la función de activación
        
        Returns:
            Salida de la función de activación
        """
        if self.funcion_activacion == 'relu':
            if derivada:
                return (x > 0).astype(float)
            return np.maximum(0, x)
        
        elif self.funcion_activacion == 'sigmoid':
            # Usar versión estable numéricamente
            x = np.clip(x, -500, 500)
            sig = 1 / (1 + np.exp(-x))
            if derivada:
                return sig * (1 - sig)
            return sig
        
        elif self.funcion_activacion == 'tanh':
            if derivada:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)
        
        else:
            raise ValueError(f"Función de activación '{self.funcion_activacion}' no soportada")
    
    def _activacion_salida(self, x: np.ndarray) -> np.ndarray:
        """
        Función de activación para la capa de salida (lineal para regresión)
        
        Args:
            x: Entrada
        
        Returns:
            Salida (sin transformación)
        """
        return x
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza la propagación hacia adelante
        
        Args:
            X: Datos de entrada (n_muestras, n_caracteristicas)
        
        Returns:
            Tupla (activaciones, z_valores) donde:
            - activaciones: Lista de activaciones de cada capa
            - z_valores: Lista de valores antes de la activación
        """
        activaciones = [X]
        z_valores = []
        
        # Capas ocultas
        for i in range(len(self.pesos) - 1):
            z = np.dot(activaciones[-1], self.pesos[i]) + self.sesgos[i]
            z_valores.append(z)
            activacion = self._activacion(z)
            activaciones.append(activacion)
        
        # Capa de salida (sin activación para regresión)
        z = np.dot(activaciones[-1], self.pesos[-1]) + self.sesgos[-1]
        z_valores.append(z)
        activacion = self._activacion_salida(z)
        activaciones.append(activacion)
        
        return activaciones, z_valores
    
    def backward_propagation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activaciones: List[np.ndarray],
        z_valores: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza la retropropagación del error
        
        Args:
            X: Datos de entrada
            y: Valores reales
            activaciones: Activaciones de cada capa
            z_valores: Valores antes de la activación
        
        Returns:
            Tupla (gradientes_pesos, gradientes_sesgos)
        """
        m = X.shape[0]  # Número de muestras
        
        # Calcular error en la capa de salida
        y_pred = activaciones[-1]
        error = y_pred - y.reshape(-1, 1)
        
        # Gradientes de pesos y sesgos
        gradientes_pesos = []
        gradientes_sesgos = []
        
        # Gradiente de la capa de salida
        delta = error / m  # Derivada del error cuadrático
        gradiente_peso = np.dot(activaciones[-2].T, delta)
        gradiente_sesgo = np.sum(delta, axis=0, keepdims=True)
        gradientes_pesos.insert(0, gradiente_peso)
        gradientes_sesgos.insert(0, gradiente_sesgo)
        
        # Retropropagar a través de las capas ocultas
        for i in range(len(self.pesos) - 2, -1, -1):
            # Calcular delta para la capa actual
            delta = np.dot(delta, self.pesos[i + 1].T) * self._activacion(z_valores[i], derivada=True)
            
            # Calcular gradientes
            gradiente_peso = np.dot(activaciones[i].T, delta)
            gradiente_sesgo = np.sum(delta, axis=0, keepdims=True)
            gradientes_pesos.insert(0, gradiente_peso)
            gradientes_sesgos.insert(0, gradiente_sesgo)
        
        return gradientes_pesos, gradientes_sesgos
    
    def actualizar_pesos(
        self,
        gradientes_pesos: List[np.ndarray],
        gradientes_sesgos: List[np.ndarray]
    ):
        """Actualiza los pesos y sesgos usando los gradientes"""
        for i in range(len(self.pesos)):
            self.pesos[i] -= self.tasa_aprendizaje * gradientes_pesos[i]
            self.sesgos[i] -= self.tasa_aprendizaje * gradientes_sesgos[i]
    
    def entrenar(
        self,
        X_entrenamiento: np.ndarray,
        y_entrenamiento: np.ndarray,
        X_validacion: Optional[np.ndarray] = None,
        y_validacion: Optional[np.ndarray] = None,
        iteraciones: int = 1000,
        tamanio_batch: int = 32,
        verbose: bool = True,
        paciencia: int = 50
    ) -> Dict[str, Any]:
        """
        Entrena el modelo MLP
        
        Args:
            X_entrenamiento: Datos de entrenamiento
            y_entrenamiento: Valores objetivo de entrenamiento
            X_validacion: Datos de validación (opcional)
            y_validacion: Valores objetivo de validación (opcional)
            iteraciones: Número máximo de iteraciones
            tamanio_batch: Tamaño del batch para entrenamiento
            verbose: Si es True, muestra progreso
            paciencia: Número de iteraciones sin mejora antes de parar (early stopping)
        
        Returns:
            Diccionario con métricas de entrenamiento
        """
        n_muestras = X_entrenamiento.shape[0]
        mejor_error_validacion = float('inf')
        iteraciones_sin_mejora = 0
        
        # Limpiar historial
        self.historial_errores_entrenamiento = []
        self.historial_errores_validacion = []
        self.historial_precision_entrenamiento = []
        self.historial_precision_validacion = []
        
        for iteracion in range(iteraciones):
            # Crear batches
            indices = np.random.permutation(n_muestras)
            X_entrenamiento_shuffled = X_entrenamiento[indices]
            y_entrenamiento_shuffled = y_entrenamiento[indices]
            
            # Entrenar por batches
            for inicio in range(0, n_muestras, tamanio_batch):
                fin = min(inicio + tamanio_batch, n_muestras)
                X_batch = X_entrenamiento_shuffled[inicio:fin]
                y_batch = y_entrenamiento_shuffled[inicio:fin]
                
                # Forward propagation
                activaciones, z_valores = self.forward_propagation(X_batch)
                
                # Backward propagation
                gradientes_pesos, gradientes_sesgos = self.backward_propagation(
                    X_batch, y_batch, activaciones, z_valores
                )
                
                # Actualizar pesos
                self.actualizar_pesos(gradientes_pesos, gradientes_sesgos)
            
            # Calcular métricas de entrenamiento
            y_pred_entrenamiento = self.predecir(X_entrenamiento)
            error_entrenamiento = self._calcular_error_cuadratico_medio(y_entrenamiento, y_pred_entrenamiento)
            precision_entrenamiento = self._calcular_r2(y_entrenamiento, y_pred_entrenamiento)
            
            self.historial_errores_entrenamiento.append(error_entrenamiento)
            self.historial_precision_entrenamiento.append(precision_entrenamiento)
            
            # Calcular métricas de validación si hay datos de validación
            if X_validacion is not None and y_validacion is not None:
                y_pred_validacion = self.predecir(X_validacion)
                error_validacion = self._calcular_error_cuadratico_medio(y_validacion, y_pred_validacion)
                precision_validacion = self._calcular_r2(y_validacion, y_pred_validacion)
                
                self.historial_errores_validacion.append(error_validacion)
                self.historial_precision_validacion.append(precision_validacion)
                
                # Early stopping
                if error_validacion < mejor_error_validacion:
                    mejor_error_validacion = error_validacion
                    iteraciones_sin_mejora = 0
                else:
                    iteraciones_sin_mejora += 1
                    if iteraciones_sin_mejora >= paciencia:
                        if verbose:
                            print(f"Early stopping en iteración {iteracion + 1}")
                        break
            
            # Mostrar progreso
            if verbose and (iteracion + 1) % 100 == 0:
                msg = f"Iteración {iteracion + 1}/{iteraciones} - Error entrenamiento: {error_entrenamiento:.4f}"
                if X_validacion is not None:
                    msg += f" - Error validación: {error_validacion:.4f}"
                print(msg)
        
        # Calcular métricas finales
        y_pred_final_entrenamiento = self.predecir(X_entrenamiento)
        metricas_entrenamiento = self._calcular_metricas(y_entrenamiento, y_pred_final_entrenamiento)
        
        metricas_validacion = None
        if X_validacion is not None and y_validacion is not None:
            y_pred_final_validacion = self.predecir(X_validacion)
            metricas_validacion = self._calcular_metricas(y_validacion, y_pred_final_validacion)
        
        return {
            'metricas_entrenamiento': metricas_entrenamiento,
            'metricas_validacion': metricas_validacion,
            'historial_errores_entrenamiento': self.historial_errores_entrenamiento,
            'historial_errores_validacion': self.historial_errores_validacion,
            'historial_precision_entrenamiento': self.historial_precision_entrenamiento,
            'historial_precision_validacion': self.historial_precision_validacion,
            'iteraciones_realizadas': iteracion + 1
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        
        Args:
            X: Datos de entrada
        
        Returns:
            Predicciones
        """
        activaciones, _ = self.forward_propagation(X)
        return activaciones[-1].flatten()
    
    def _calcular_error_cuadratico_medio(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula el Error Cuadrático Medio (MSE)"""
        return np.mean((y_real - y_pred) ** 2)
    
    def _calcular_raiz_error_cuadratico_medio(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la Raíz del Error Cuadrático Medio (RMSE)"""
        return np.sqrt(self._calcular_error_cuadratico_medio(y_real, y_pred))
    
    def _calcular_error_absoluto_medio(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula el Error Absoluto Medio (MAE)"""
        return np.mean(np.abs(y_real - y_pred))
    
    def _calcular_r2(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula el coeficiente de determinación R²"""
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)
    
    def _calcular_metricas(self, y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula todas las métricas"""
        return {
            'MSE': float(self._calcular_error_cuadratico_medio(y_real, y_pred)),
            'RMSE': float(self._calcular_raiz_error_cuadratico_medio(y_real, y_pred)),
            'MAE': float(self._calcular_error_absoluto_medio(y_real, y_pred)),
            'R2': float(self._calcular_r2(y_real, y_pred))
        }
    
    def obtener_pesos_serializables(self) -> Dict[str, Any]:
        """Convierte los pesos y sesgos a formato serializable (listas)"""
        pesos_serializables = [peso.tolist() for peso in self.pesos]
        sesgos_serializables = [sesgo.tolist() for sesgo in self.sesgos]
        
        return {
            'pesos': pesos_serializables,
            'sesgos': sesgos_serializables,
            'arquitectura': self.arquitectura,
            'funcion_activacion': self.funcion_activacion
        }
    
    def cargar_pesos(self, pesos: List[List[List[float]]], sesgos: List[List[List[float]]]):
        """Carga pesos y sesgos desde listas"""
        self.pesos = [np.array(p) for p in pesos]
        self.sesgos = [np.array(s) for s in sesgos]
    
    def crear_grafico_errores(self) -> str:
        """Crea un gráfico de la evolución de los errores"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.historial_errores_entrenamiento:
            iteraciones = range(1, len(self.historial_errores_entrenamiento) + 1)
            ax.plot(iteraciones, self.historial_errores_entrenamiento, label='Entrenamiento', color='blue')
        
        if self.historial_errores_validacion:
            ax.plot(iteraciones, self.historial_errores_validacion, label='Validación', color='red')
        
        ax.set_xlabel('Iteración')
        ax.set_ylabel('Error Cuadrático Medio (MSE)')
        ax.set_title('Evolución del Error durante el Entrenamiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        imagen_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return imagen_base64
    
    def crear_grafico_precision(self) -> str:
        """Crea un gráfico de la evolución de la precisión (R²)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determinar el número de iteraciones
        if self.historial_precision_entrenamiento:
            iteraciones = range(1, len(self.historial_precision_entrenamiento) + 1)
            ax.plot(iteraciones, self.historial_precision_entrenamiento, label='Entrenamiento', color='blue')
        elif self.historial_precision_validacion:
            iteraciones = range(1, len(self.historial_precision_validacion) + 1)
        else:
            # Si no hay historial, retornar gráfico vacío
            ax.text(0.5, 0.5, 'No hay datos de precisión disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
        
        if self.historial_precision_validacion:
            if not self.historial_precision_entrenamiento:
                iteraciones = range(1, len(self.historial_precision_validacion) + 1)
            ax.plot(iteraciones, self.historial_precision_validacion, label='Validación', color='red')
        
        ax.set_xlabel('Iteración')
        ax.set_ylabel('Coeficiente de Determinación (R²)')
        ax.set_title('Evolución de la Precisión durante el Entrenamiento')
        if self.historial_precision_entrenamiento or self.historial_precision_validacion:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        imagen_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return imagen_base64
    
    def crear_grafico_predicciones_vs_reales(self, y_real: np.ndarray, y_pred: np.ndarray) -> str:
        """Crea un gráfico de predicciones vs valores reales"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y_real, y_pred, alpha=0.5)
        
        # Línea perfecta (y = x)
        min_val = min(y_real.min(), y_pred.min())
        max_val = max(y_real.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Perfecta')
        
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Valores Predichos')
        ax.set_title('Predicciones vs Valores Reales')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        imagen_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return imagen_base64

