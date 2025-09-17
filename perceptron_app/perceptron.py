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
    
    def __init__(self, tasa_aprendizaje: float = 0.1, max_iteraciones: int = 100, error_maximo: float = 0.1,
                 pesos_iniciales: List[float] = None, sesgo_inicial: float = None):
        """
        Inicializa el perceptrón simple

        Args:
            tasa_aprendizaje (float): Tasa de aprendizaje
            max_iteraciones (int): Número máximo de iteraciones de entrenamiento
            error_maximo (float): Error máximo permitido para detener el entrenamiento
            pesos_iniciales (List[float], optional): Pesos iniciales predefinidos
            sesgo_inicial (float, optional): Sesgo inicial predefinido
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_iteraciones = max_iteraciones
        self.error_maximo = error_maximo
        self.pesos = pesos_iniciales
        self.sesgo = sesgo_inicial if sesgo_inicial is not None else 0.0
        self.errores_entrenamiento = []
        self.errores_patron = []
        self.evolucion_pesos = []
        print(f"Perceptrón inicializado - pesos: {self.pesos}")
    
    def _limpiar_estado(self):
        """
        Limpia el estado del perceptrón para un nuevo entrenamiento
        """
        print(f"Limpiando estado - pesos antes: {self.pesos}")
        self.pesos = None
        self.sesgo = 0.0
        self.errores_entrenamiento = []
        self.errores_patron = []
        self.evolucion_pesos = []
        print(f"Estado limpiado - pesos después: {self.pesos}")
        
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
        Inicializa los pesos (aleatoriamente si no están predefinidos)

        Args:
            num_caracteristicas (int): Número de características de entrada
        """
        print(f"_inicializar_pesos llamado con num_caracteristicas = {num_caracteristicas}")
        print(f"Pesos antes de inicializar: {self.pesos}")

        # Si no hay pesos predefinidos, inicializar aleatoriamente
        if self.pesos is None:
            print("Inicializando pesos aleatoriamente...")
            self.pesos = np.random.uniform(-1, 1, num_caracteristicas)
            self.sesgo = np.random.uniform(-1, 1)
        else:
            print("Usando pesos predefinidos...")
            # Validar que los pesos predefinidos coincidan con el número de características
            if len(self.pesos) != num_caracteristicas:
                raise ValueError(f"Los pesos predefinidos tienen {len(self.pesos)} elementos, "
                               f"pero se necesitan {num_caracteristicas} para las características de entrada.")

            # Convertir a numpy array si no lo es
            self.pesos = np.array(self.pesos, dtype=float)

        print(f"Pesos después de inicializar: {self.pesos}")
        print(f"Longitud de pesos: {len(self.pesos)}")
        print(f"Sesgo: {self.sesgo}")
        
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
        
        # Debug: verificar dimensiones
        if self.pesos is None:
            raise ValueError("Los pesos no han sido inicializados")
        
        if len(self.pesos) != len(X):
            print(f"Error de dimensiones: pesos tiene {len(self.pesos)} elementos, X tiene {len(X)} elementos")
            print(f"Pesos: {self.pesos}")
            print(f"X: {X}")
            raise ValueError(f"Dimensiones no coinciden: pesos({len(self.pesos)}) vs X({len(X)})")
        
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
        
        # Limpiar estado anterior y inicializar pesos para el nuevo entrenamiento
        self._limpiar_estado()
        print(f"Inicializando pesos para {num_caracteristicas} características")
        self._inicializar_pesos(num_caracteristicas)
        
        # Listas para almacenar el progreso del entrenamiento
        self.errores_entrenamiento = []
        self.errores_patron = []
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
            error_patron = 0
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
                
                # Calcular error lineal
                error = y[i] - prediccion
                
                # Actualizar pesos y sesgo si hay error
                if error != 0:
                    # Regla del perceptrón: w = w + eta * error * x
                    self.pesos += self.tasa_aprendizaje * error * X[i]
                    self.sesgo += self.tasa_aprendizaje * error
                    errores_iteracion += 1
                    error_patron += error
            
            # Guardar error de la iteración
            self.errores_entrenamiento.append(errores_iteracion)
            self.errores_patron.append(error_patron)
            # Calcular error del patron (errores por muestra)
            tasa_error = error_patron / num_muestras
            
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
            'errores_patron': self.errores_patron,
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
                    Errores patrón: {self.errores_patron}
                    Pesos finales: {[f'{w:.3f}' for w in self.pesos]}
                    Sesgo final: {self.sesgo:.3f}
                    Número de características: {len(self.pesos)}

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

    def crear_diagrama_red(self) -> str:
        """
        Crea un diagrama simple de la estructura de la red neuronal

        Returns:
            str: Imagen codificada en base64
        """
        if self.pesos is None:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Número de neuronas de entrada y salida
        num_inputs = len(self.pesos)
        num_outputs = 1  # Perceptrón simple tiene una salida

        # Posiciones de las neuronas
        input_positions = [(1.5, 9 - i * 8 / max(1, num_inputs - 1)) for i in range(num_inputs)]
        hidden_position = (4, 5)
        output_position = (6.5, 5)

        # Dibujar neuronas de entrada
        for i, (x, y) in enumerate(input_positions):
            circle = plt.Circle((x, y), 0.3, fill=True, color='lightblue', ec='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f'X{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')

        # Dibujar neurona oculta (perceptrón)
        circle = plt.Circle(hidden_position, 0.4, fill=True, color='lightgreen', ec='green', linewidth=2)
        ax.add_patch(circle)
        ax.text(hidden_position[0], hidden_position[1], '∑', ha='center', va='center', fontsize=12, fontweight='bold')

        # Dibujar neurona de salida
        circle = plt.Circle(output_position, 0.3, fill=True, color='lightcoral', ec='red', linewidth=2)
        ax.add_patch(circle)
        ax.text(output_position[0], output_position[1], 'Y', ha='center', va='center', fontsize=10, fontweight='bold')

        # Dibujar conexiones con colores basados en pesos
        for i, (x, y) in enumerate(input_positions):
            weight = self.pesos[i]
            color = 'red' if weight < 0 else 'blue'
            # Asegurar que las líneas sean visibles con transparencia mínima
            alpha = min(1.0, abs(weight) / 2.0)  # Transparencia basada en magnitud del peso
            linewidth = 1 + abs(weight) * 2  # Grosor basado en magnitud del peso       

            # alpha = max(0.6, min(1.0, abs(weight) / 2.0))  # Transparencia basada en magnitud del peso
            # linewidth = max(1.5, 1 + abs(weight) * 2)  # Grosor mínimo garantizado

            ax.plot([x, hidden_position[0]], [y, hidden_position[1]],
                   color=color, linewidth=linewidth, alpha=alpha)

            # Etiqueta del peso
            mid_x = (x + hidden_position[0]) / 2
            mid_y = (y + hidden_position[1]) / 2
            ax.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # Conexión a la salida
        ax.plot([hidden_position[0], output_position[0]], [hidden_position[1], output_position[1]],
               color='purple', linewidth=3, alpha=0.8)

        # Etiqueta del sesgo
        mid_x = (hidden_position[0] + output_position[0]) / 2
        mid_y = (hidden_position[1] + output_position[1]) / 2
        ax.text(mid_x, mid_y, f'bias: {self.sesgo:.2f}', fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

        # Título
        ax.set_title('Estructura del Perceptrón Simple', fontsize=16, fontweight='bold', pad=20)

        # Leyenda
        ax.text(1, 1, 'Entrada', fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.text(4, 9.5, 'Perceptrón', fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax.text(6.5, 9.5, 'Salida', fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    def crear_grafico_precision(self) -> str:
        """
        Crea un gráfico de la evolución de la precisión durante el entrenamiento

        Returns:
            str: Imagen codificada en base64
        """
        if not self.errores_entrenamiento:
            return None

        # Calcular precisión aproximada basada en errores
        precisiones = []
        for errores in self.errores_entrenamiento:
            # Asumiendo que cada error representa un error por muestra
            # En un escenario real, esto debería calcularse con los datos reales
            if hasattr(self, '_num_muestras') and self._num_muestras > 0:
                tasa_error = errores / self._num_muestras
            else:
                # Estimación basada en número de características
                tasa_error = min(1.0, errores / len(self.pesos)) if self.pesos else 0
            precision = max(0, (1 - tasa_error) * 100)
            precisiones.append(precision)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(precisiones) + 1), precisiones, 'g-', linewidth=2, marker='o', markersize=4)
        plt.title('Evolución de la Precisión durante el Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Precisión (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)  # Asegurar que el eje Y vaya de 0 a 105%
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

    def crear_grafico_error_patron(self) -> str:
        """
        Crea un gráfico de la evolución del error patrón durante el entrenamiento
        
        Returns:
            str: Imagen codificada en base64
        """
        if not self.errores_patron:
            return None
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(self.errores_patron) + 1), self.errores_patron, 'b-', linewidth=2)
        plt.title('Evolución del Error por Patrón durante el Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Error por Patrón', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
