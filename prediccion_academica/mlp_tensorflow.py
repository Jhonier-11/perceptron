"""
Implementación del Perceptrón Multicapa (MLP) usando TensorFlow/Keras
"""

import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Crear placeholders para evitar errores de importación
    tf = None
    keras = None

import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import io
import base64
import os
from typing import List, Tuple, Dict, Any, Optional
import json


class MLPTensorFlow:
    """
    Implementación del Perceptrón Multicapa usando TensorFlow/Keras
    """
    
    def __init__(
        self,
        arquitectura: List[int],
        tasa_aprendizaje: float = 0.01,
        funcion_activacion: str = 'relu',
        optimizer: str = 'adam'
    ):
        """
        Inicializa el MLP usando TensorFlow/Keras
        
        Args:
            arquitectura: Lista con el número de neuronas por capa [input, hidden1, hidden2, ..., output]
            tasa_aprendizaje: Tasa de aprendizaje
            funcion_activacion: Función de activación ('relu', 'sigmoid', 'tanh')
            optimizer: Optimizador a usar ('adam', 'sgd', 'rmsprop')
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow no está instalado. Por favor, instálalo usando: "
                "pip install tensorflow>=2.13.0"
            )
        
        self.arquitectura = arquitectura
        self.tasa_aprendizaje = tasa_aprendizaje
        self.funcion_activacion = funcion_activacion
        self.optimizer = optimizer
        self.modelo = None
        self.historial_entrenamiento = None
        
        # Mapeo de funciones de activación
        self.activacion_map = {
            'relu': 'relu',
            'sigmoid': 'sigmoid',
            'tanh': 'tanh',
        }
        
        # Crear el modelo
        self._crear_modelo()
    
    def _crear_modelo(self):
        """Crea el modelo Sequential de Keras"""
        modelo = models.Sequential()
        
        # Capa de entrada y primera capa oculta
        modelo.add(layers.Dense(
            self.arquitectura[1],
            activation=self.activacion_map.get(self.funcion_activacion, 'relu'),
            input_shape=(self.arquitectura[0],),
            kernel_initializer='glorot_uniform',  # Xavier initialization
            name='dense_input'
        ))
        
        # Capas ocultas adicionales
        for i in range(2, len(self.arquitectura) - 1):
            modelo.add(layers.Dense(
                self.arquitectura[i],
                activation=self.activacion_map.get(self.funcion_activacion, 'relu'),
                kernel_initializer='glorot_uniform',
                name=f'dense_hidden_{i-1}'
            ))
        
        # Capa de salida (sin activación para regresión)
        modelo.add(layers.Dense(
            self.arquitectura[-1],
            activation=None,  # Regresión lineal
            name='dense_output'
        ))
        
        # Configurar optimizador
        if self.optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.tasa_aprendizaje)
        elif self.optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=self.tasa_aprendizaje)
        else:
            opt = keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje)
        
        # Métrica personalizada para R²
        def r_squared(y_true, y_pred):
            """Calcula R² como métrica personalizada"""
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
            ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))
        
        # Compilar el modelo
        modelo.compile(
            optimizer=opt,
            loss='mse',  # Mean Squared Error para regresión
            metrics=['mae', r_squared]  # Mean Absolute Error y R²
        )
        
        self.modelo = modelo
    
    def entrenar(
        self,
        X_entrenamiento: np.ndarray,
        y_entrenamiento: np.ndarray,
        X_validacion: Optional[np.ndarray] = None,
        y_validacion: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        patience: int = 50,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Entrena el modelo
        
        Args:
            X_entrenamiento: Datos de entrenamiento
            y_entrenamiento: Etiquetas de entrenamiento
            X_validacion: Datos de validación (opcional)
            y_validacion: Etiquetas de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño del batch
            verbose: Mostrar progreso
            patience: Paciencia para early stopping
            validation_split: Porcentaje de datos para validación si no se proporciona X_validacion
        
        Returns:
            Diccionario con métricas y historial de entrenamiento
        """
        # Preparar datos de validación
        if X_validacion is not None and y_validacion is not None:
            validation_data = (X_validacion, y_validacion)
            validation_split = None
        else:
            validation_data = None
        
        # Callbacks
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1 if verbose else 0
        )
        callbacks_list.append(early_stopping)
        
        # Reducir learning rate en plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1 if verbose else 0
        )
        callbacks_list.append(reduce_lr)
        
        # Entrenar el modelo
        self.historial_entrenamiento = self.modelo.fit(
            X_entrenamiento,
            y_entrenamiento,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1 if verbose else 0
        )
        
        # Calcular métricas
        metricas_entrenamiento = self._calcular_metricas(
            y_entrenamiento,
            self.modelo.predict(X_entrenamiento, verbose=0)
        )
        
        metricas_validacion = {}
        if validation_data:
            y_pred_validacion = self.modelo.predict(X_validacion, verbose=0)
            metricas_validacion = self._calcular_metricas(
                y_validacion,
                y_pred_validacion
            )
        
        # Preparar historial para serialización
        historial = {
            'loss': [float(x) for x in self.historial_entrenamiento.history['loss']],
            'mae': [float(x) for x in self.historial_entrenamiento.history['mae']],
        }
        
        # Obtener R² del historial si está disponible
        # TensorFlow puede usar nombres diferentes para las métricas personalizadas
        r2_key = None
        for key in self.historial_entrenamiento.history.keys():
            if 'r_squared' in key.lower() or 'r2' in key.lower():
                r2_key = key
                break
        
        if r2_key and r2_key in self.historial_entrenamiento.history:
            historial['r2_entrenamiento'] = [float(x) for x in self.historial_entrenamiento.history[r2_key]]
        else:
            # Si no hay R² en el historial, calcularlo final y crear una lista
            historial['r2_entrenamiento'] = []
        
        if 'val_loss' in self.historial_entrenamiento.history:
            historial['val_loss'] = [float(x) for x in self.historial_entrenamiento.history['val_loss']]
            historial['val_mae'] = [float(x) for x in self.historial_entrenamiento.history['val_mae']]
            if 'val_r_squared' in self.historial_entrenamiento.history:
                historial['r2_validacion'] = [float(x) for x in self.historial_entrenamiento.history['val_r_squared']]
        
        # Calcular R² final (esto lo hacemos de todas formas para asegurarnos)
        y_pred_train = self.modelo.predict(X_entrenamiento, verbose=0)
        r2_train = self._calcular_r2(y_entrenamiento, y_pred_train)
        historial['r2_entrenamiento_final'] = float(r2_train)
        
        # Si tenemos R² en el historial, usarlo; si no, crear una lista con el valor final
        if 'r2_entrenamiento' not in historial or not historial['r2_entrenamiento']:
            historial['r2_entrenamiento'] = [r2_train] * len(historial['loss'])
        
        if validation_data:
            y_pred_val = self.modelo.predict(X_validacion, verbose=0)
            r2_val = self._calcular_r2(y_validacion, y_pred_val)
            historial['r2_validacion_final'] = float(r2_val)
            
            # Si no tenemos R² de validación en el historial, crear una lista con el valor final
            if 'r2_validacion' not in historial or not historial['r2_validacion']:
                historial['r2_validacion'] = [r2_val] * len(historial['val_loss'])
        
        return {
            'metricas_entrenamiento': metricas_entrenamiento,
            'metricas_validacion': metricas_validacion,
            'historial_entrenamiento': historial,
            'historial_errores_entrenamiento': historial['loss'],
            'historial_errores_validacion': historial.get('val_loss', []),
            'historial_precision_entrenamiento': historial['r2_entrenamiento'],
            'historial_precision_validacion': historial.get('r2_validacion', []) if validation_data else [],
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Hace predicciones
        
        Args:
            X: Datos de entrada
        
        Returns:
            Predicciones
        """
        if self.modelo is None:
            raise ValueError("El modelo no ha sido entrenado o cargado")
        
        predicciones = self.modelo.predict(X, verbose=0)
        return predicciones.flatten() if predicciones.ndim > 1 else predicciones
    
    def guardar_modelo(self, ruta: str):
        """
        Guarda el modelo en un archivo
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        
        # Guardar modelo en formato HDF5
        self.modelo.save(ruta)
    
    @staticmethod
    def cargar_modelo(ruta: str) -> 'MLPTensorFlow':
        """
        Carga un modelo desde un archivo
        
        Args:
            ruta: Ruta del archivo del modelo
        
        Returns:
            Instancia de MLPTensorFlow con el modelo cargado
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow no está instalado. Por favor, instálalo usando: "
                "pip install tensorflow>=2.13.0"
            )
        
        modelo = keras.models.load_model(ruta)
        
        # Crear instancia (necesitamos la arquitectura, pero la podemos obtener del modelo)
        arquitectura = [modelo.input_shape[1]]  # Input
        for layer in modelo.layers:
            if hasattr(layer, 'units'):
                arquitectura.append(layer.units)
        
        # Crear instancia
        mlp = MLPTensorFlow(
            arquitectura=arquitectura,
            tasa_aprendizaje=0.01,  # No podemos obtener esto del modelo guardado fácilmente
            funcion_activacion='relu'  # Tampoco podemos obtener esto fácilmente
        )
        mlp.modelo = modelo
        
        return mlp
    
    def _calcular_metricas(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de evaluación
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
        
        Returns:
            Diccionario con métricas
        """
        mse = np.mean((y_true - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred.flatten()))
        r2 = self._calcular_r2(y_true, y_pred)
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2)
        }
    
    def _calcular_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula el coeficiente de determinación R²
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
        
        Returns:
            R²
        """
        ss_res = np.sum((y_true - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def obtener_historial(self) -> Dict[str, List[float]]:
        """
        Obtiene el historial de entrenamiento
        
        Returns:
            Diccionario con el historial
        """
        if self.historial_entrenamiento is None:
            return {}
        
        historial = {
            'loss': [float(x) for x in self.historial_entrenamiento.history['loss']],
            'mae': [float(x) for x in self.historial_entrenamiento.history['mae']],
        }
        
        # Buscar R² en el historial
        r2_key = None
        for key in self.historial_entrenamiento.history.keys():
            if 'r_squared' in key.lower() or 'r2' in key.lower():
                if 'val' not in key.lower():
                    r2_key = key
                    break
        
        if r2_key and r2_key in self.historial_entrenamiento.history:
            historial['r2_entrenamiento'] = [float(x) for x in self.historial_entrenamiento.history[r2_key]]
        
        if 'val_loss' in self.historial_entrenamiento.history:
            historial['val_loss'] = [float(x) for x in self.historial_entrenamiento.history['val_loss']]
            historial['val_mae'] = [float(x) for x in self.historial_entrenamiento.history['val_mae']]
            
            # Buscar R² de validación
            val_r2_key = None
            for key in self.historial_entrenamiento.history.keys():
                if 'val' in key.lower() and ('r_squared' in key.lower() or 'r2' in key.lower()):
                    val_r2_key = key
                    break
            
            if val_r2_key and val_r2_key in self.historial_entrenamiento.history:
                historial['r2_validacion'] = [float(x) for x in self.historial_entrenamiento.history[val_r2_key]]
        
        return historial
    
    def crear_grafico_errores(self, historial: Optional[Dict[str, List[float]]] = None) -> str:
        """
        Crea un gráfico de evolución de errores
        
        Args:
            historial: Historial de entrenamiento (opcional)
        
        Returns:
            Imagen en base64
        """
        if historial is None:
            historial = self.obtener_historial()
        
        if not historial:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(historial['loss']) + 1)
        ax.plot(epochs, historial['loss'], label='Entrenamiento', color='blue')
        
        if 'val_loss' in historial and historial['val_loss']:
            ax.plot(epochs, historial['val_loss'], label='Validación', color='red')
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Error Cuadrático Medio (MSE)')
        ax.set_title('Evolución del Error durante el Entrenamiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        imagen_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return imagen_base64
    
    def crear_grafico_precision(self, historial: Optional[Dict[str, List[float]]] = None) -> str:
        """
        Crea un gráfico de evolución de precisión (R²)
        
        Args:
            historial: Historial de entrenamiento con R² (opcional)
        
        Returns:
            Imagen en base64
        """
        if historial is None:
            historial = self.obtener_historial()
        
        if not historial or 'r2_entrenamiento' not in historial:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(historial['r2_entrenamiento']) + 1)
        ax.plot(epochs, historial['r2_entrenamiento'], label='Entrenamiento', color='blue')
        
        if 'r2_validacion' in historial and historial['r2_validacion']:
            ax.plot(epochs, historial['r2_validacion'], label='Validación', color='red')
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Coeficiente de Determinación (R²)')
        ax.set_title('Evolución de la Precisión durante el Entrenamiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        imagen_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return imagen_base64
    
    def crear_grafico_predicciones_vs_reales(self, y_real: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Crea un gráfico de predicciones vs valores reales
        
        Args:
            y_real: Valores reales
            y_pred: Valores predichos
        
        Returns:
            Imagen en base64
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y_real, y_pred, alpha=0.5)
        
        min_val = min(y_real.min(), y_pred.min())
        max_val = max(y_real.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Perfecta')
        
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Valores Predichos')
        ax.set_title('Predicciones vs Valores Reales')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        imagen_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return imagen_base64

