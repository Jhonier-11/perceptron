"""
Motor de la Red Neuronal de Función de Base Radial (RBF)
Implementado completamente desde cero usando solo Python y NumPy
"""
import numpy as np
import pandas as pd


def dividir_entrenamiento_prueba(X, y, porcentaje_entrenamiento=0.7):
    """
    Divide manualmente el dataset en conjuntos de entrenamiento y prueba
    
    Args:
        X (np.ndarray): Matriz de características
        y (np.ndarray): Vector de etiquetas
        porcentaje_entrenamiento (float): Porcentaje para entrenamiento (0.5 a 0.9)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Validar entrada
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=float)
    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=float)
    
    num_muestras = X.shape[0]
    
    # Calcular índices
    num_entrenamiento = int(num_muestras * porcentaje_entrenamiento)
    
    # Mezclar índices aleatoriamente
    indices = np.arange(num_muestras)
    np.random.seed(42)  # Para reproducibilidad
    np.random.shuffle(indices)
    
    # Dividir índices
    indices_train = indices[:num_entrenamiento]
    indices_test = indices[num_entrenamiento:]
    
    # Extraer subconjuntos
    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]
    
    return X_train, X_test, y_train, y_test


def normalizar_datos(X):
    """
    Normaliza los datos usando estandarización (mean=0, std=1)
    
    Args:
        X (np.ndarray): Matriz de características
        
    Returns:
        tuple: (X_normalizado, mean, std)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=float)
    
    # Calcular media y desviación estándar para cada columna
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    
    # Evitar división por cero
    std = np.where(std == 0, 1, std)
    
    # Estandarizar
    X_normalizado = (X - mean) / std
    
    return X_normalizado, mean.flatten(), std.flatten()


def desnormalizar_datos(X, mean, std):
    """
    Revierte la normalización de los datos
    
    Args:
        X (np.ndarray): Matriz normalizada
        mean (np.ndarray): Media original
        std (np.ndarray): Desviación estándar original
        
    Returns:
        np.ndarray: Datos desnormalizados
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=float)
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    
    # Revertir la normalización
    X_original = X * std + mean
    
    return X_original


def calcular_eg(y_deseado, y_real):
    """
    Calcula el Error General (EG)
    
    EG = (suma de |Yd - Yr|) / N
    
    Args:
        y_deseado (np.ndarray): Valores deseados
        y_real (np.ndarray): Valores reales/predichos
        
    Returns:
        float: Error General
    """
    if not isinstance(y_deseado, np.ndarray):
        y_deseado = np.array(y_deseado, dtype=float)
    if not isinstance(y_real, np.ndarray):
        y_real = np.array(y_real, dtype=float)
    
    n = len(y_deseado)
    if n == 0:
        return 0.0
    
    error = np.sum(np.abs(y_deseado - y_real)) / n
    return float(error)


def calcular_mae(y_deseado, y_real):
    """
    Calcula el Error Absoluto Medio (MAE)
    
    MAE = (1/N) * suma(|Yd - Yr|)
    
    Args:
        y_deseado (np.ndarray): Valores deseados
        y_real (np.ndarray): Valores reales/predichos
        
    Returns:
        float: Error Absoluto Medio
    """
    if not isinstance(y_deseado, np.ndarray):
        y_deseado = np.array(y_deseado, dtype=float)
    if not isinstance(y_real, np.ndarray):
        y_real = np.array(y_real, dtype=float)
    
    n = len(y_deseado)
    if n == 0:
        return 0.0
    
    mae = np.mean(np.abs(y_deseado - y_real))
    return float(mae)


def calcular_rmse(y_deseado, y_real):
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE)
    
    RMSE = sqrt((1/N) * suma((Yd - Yr)^2))
    
    Args:
        y_deseado (np.ndarray): Valores deseados
        y_real (np.ndarray): Valores reales/predichos
        
    Returns:
        float: Raíz del Error Cuadrático Medio
    """
    if not isinstance(y_deseado, np.ndarray):
        y_deseado = np.array(y_deseado, dtype=float)
    if not isinstance(y_real, np.ndarray):
        y_real = np.array(y_real, dtype=float)
    
    n = len(y_deseado)
    if n == 0:
        return 0.0
    
    mse = np.mean((y_deseado - y_real) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)


def verificar_convergencia(eg, error_objetivo=0.1):
    """
    Verifica si la red ha convergido comparando EG con el error objetivo
    
    Args:
        eg (float): Error General calculado
        error_objetivo (float): Error objetivo (por defecto 0.1)
        
    Returns:
        bool: True si convergió (EG <= error_objetivo)
    """
    return eg <= error_objetivo


def label_encoding(serie):
    """
    Codificación Label Encoding para variables categóricas
    
    Args:
        serie (pd.Series): Serie con valores categóricos
        
    Returns:
        tuple: (serie_codificada, mapping)
    """
    valores_unicos = serie.unique()
    mapping = {valor: i for i, valor in enumerate(valores_unicos)}
    
    serie_codificada = serie.map(mapping)
    
    return serie_codificada, mapping


def one_hot_encoding(df, columnas):
    """
    Codificación One-Hot para variables categóricas
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas a codificar
        columnas (list): Lista de nombres de columnas categóricas
        
    Returns:
        pd.DataFrame: DataFrame con las columnas codificadas
    """
    df_resultado = df.copy()
    
    for col in columnas:
        if col in df_resultado.columns:
            # Obtener dummies
            dummies = pd.get_dummies(df_resultado[col], prefix=col, prefix_sep='_')
            # Eliminar columna original
            df_resultado = df_resultado.drop(col, axis=1)
            # Concatenar dummies
            df_resultado = pd.concat([df_resultado, dummies], axis=1)
    
    return df_resultado


def preprocesar_datos(df, columnas_entrada, columnas_salida):
    """
    Preprocesa automáticamente los datos: convierte columnas categóricas a numéricas
    y detecta si necesita normalización
    
    Args:
        df (pd.DataFrame): DataFrame original
        columnas_entrada (list): Lista de columnas de entrada
        columnas_salida (list): Lista de columnas de salida
        
    Returns:
        tuple: (info_preprocesamiento, df_procesado, df_original_con_comparacion)
    """
    info_preprocesamiento = {
        'transformaciones_realizadas': [],
        'columnas_codificadas': [],
        'necesita_normalizacion': False,
        'escalas_diferentes': False,
        'tipos_originales': {},
        'tipos_tras_preprocesamiento': {},
        'datos_comparacion': []  # Para mostrar comparación en la tabla
    }
    
    df_original = df.copy()
    df_procesado = df.copy()
    
    # Analizar tipos de datos originales
    for col in columnas_entrada + columnas_salida:
        if col in df.columns:
            tipo = df[col].dtype
            info_preprocesamiento['tipos_originales'][col] = str(tipo)
    
    # Detectar columnas no numéricas
    columnas_no_numericas = []
    for col in columnas_entrada + columnas_salida:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                columnas_no_numericas.append(col)
    
    # Si hay columnas no numéricas, aplicar transformaciones
    if columnas_no_numericas:
        # Decidir: Label Encoding para ordinales/categóricas simples
        # One-Hot para variables nominales con pocas categorías
        
        for col in columnas_no_numericas:
            if col in df_procesado.columns:
                num_unicos = df_procesado[col].nunique()
                
                # Si tiene menos de 10 categorías, usar Label Encoding (más simple)
                if num_unicos < 10:
                    df_procesado[col], mapping = label_encoding(df_procesado[col])
                    info_preprocesamiento['transformaciones_realizadas'].append({
                        'columna': col,
                        'tipo': 'Label Encoding',
                        'num_categorias': num_unicos,
                        'mapping': str(mapping)
                    })
                    info_preprocesamiento['columnas_codificadas'].append({
                        'columna': col,
                        'tipo': 'Label Encoding',
                        'mapping': mapping
                    })
                else:
                    # Para muchas categorías, usar Label Encoding también
                    df_procesado[col], mapping = label_encoding(df_procesado[col])
                    info_preprocesamiento['transformaciones_realizadas'].append({
                        'columna': col,
                        'tipo': 'Label Encoding',
                        'num_categorias': num_unicos,
                        'mapping': str(mapping)
                    })
                    info_preprocesamiento['columnas_codificadas'].append({
                        'columna': col,
                        'tipo': 'Label Encoding',
                        'mapping': mapping
                    })
    
    # Verificar tipos tras preprocesamiento
    for col in columnas_entrada + columnas_salida:
        if col in df_procesado.columns:
            tipo = df_procesado[col].dtype
            info_preprocesamiento['tipos_tras_preprocesamiento'][col] = str(tipo)
    
    # Verificar si necesita normalización (rango de escalas)
    if columnas_entrada:
        X = df_procesado[columnas_entrada].values
        rangos = []
        
        for i in range(X.shape[1]):
            col_data = X[:, i]
            min_val = np.min(col_data)
            max_val = np.max(col_data)
            rango = max_val - min_val
            
            if rango > 0:
                rangos.append(rango)
        
        if len(rangos) > 0:
            max_rango = max(rangos)
            min_rango = min(rangos)
            
            # Si hay un rango más de 10x mayor que otro, necesita normalización
            if max_rango / min_rango > 10 or max_rango > 1000:
                info_preprocesamiento['necesita_normalizacion'] = True
                info_preprocesamiento['escalas_diferentes'] = True
                info_preprocesamiento['transformaciones_realizadas'].append({
                    'tipo': 'Normalización recomendada',
                    'razon': f'Rangos de valores muy diferentes (min: {min_rango:.2f}, max: {max_rango:.2f})'
                })
    
    # Crear DataFrame de comparación con original y transformado
    df_comparacion = df_original.copy()
    
    # Agregar columnas transformadas al lado de las originales
    for item in info_preprocesamiento['columnas_codificadas']:
        col_original = item['columna']
        if col_original in df_original.columns:
            # Agregar columna con valores transformados
            df_comparacion[f'{col_original}_transformed'] = df_procesado[col_original]
            info_preprocesamiento['datos_comparacion'].append({
                'columna_original': col_original,
                'columna_transformada': f'{col_original}_transformed',
                'mapping': item['mapping']
            })
    
    return info_preprocesamiento, df_procesado, df_comparacion


class RBFNet:
    """
    Implementación de Red Neuronal de Función de Base Radial (RBF)
    """
    
    def __init__(self, num_centros=3, error_aproximacion=0.1):
        """
        Inicializa la red RBF
        
        Args:
            num_centros (int): Número de centros radiales (neuronas ocultas)
            error_aproximacion (float): Error de aproximación óptimo
        """
        self.num_centros = num_centros
        self.error_aproximacion = error_aproximacion
        self.centros = None
        self.pesos = None  # Vector W completo incluye W0 (umbral)
        self.W0 = None  # Umbral (peso del sesgo)
        self.W1_n = None  # Pesos de los centros radiales
        self.entrenado = False
        
    def _funcion_activacion(self, d):
        """
        Función de activación radial FA(d) = d^2 * ln(d)
        
        Args:
            d (float o np.ndarray): Distancia
            
        Returns:
            float o np.ndarray: Valor de la función de activación
        """
        # Convertir a array si es necesario
        d = np.asarray(d, dtype=float)
        
        # Manejar caso d = 0 usando epsilon pequeño
        epsilon = 1e-10
        d_safe = np.where(d == 0, epsilon, d)
        
        # Calcular FA(d) = d^2 * ln(d)
        resultado = np.where(
            d == 0,
            0.0,  # Si d = 0, retornar 0
            (d_safe ** 2) * np.log(d_safe)
        )
        
        return resultado
    
    def _calcular_distancias(self, X, centros):
        """
        Calcula la matriz de distancias euclidianas
        
        Args:
            X (np.ndarray): Matriz de datos de entrada (N patrones x M características)
            centros (np.ndarray): Matriz de centros radiales (K centros x M características)
            
        Returns:
            np.ndarray: Matriz de distancias (N x K)
        """
        # Validar que sean arrays
        X = np.asarray(X, dtype=float)
        centros = np.asarray(centros, dtype=float)
        
        # Calcular distancias euclidianas
        # D_pj = sqrt((X_p - R_j)^2)
        # Usando broadcasting para calcular todas las distancias a la vez
        distancias = np.sqrt(np.sum((X[:, np.newaxis, :] - centros[np.newaxis, :, :]) ** 2, axis=2))
        
        return distancias
    
    def _calcular_activaciones(self, distancias):
        """
        Calcula las activaciones usando la función radial
        
        Args:
            distancias (np.ndarray): Matriz de distancias
            
        Returns:
            np.ndarray: Matriz de activaciones
        """
        # Aplicar función de activación a todas las distancias
        activaciones = self._funcion_activacion(distancias)
        
        return activaciones
    
    def _construir_matriz_interpolacion(self, activaciones):
        """
        Construye la matriz A agregando columna de 1s para el umbral
        
        Args:
            activaciones (np.ndarray): Matriz de activaciones (N x K)
            
        Returns:
            np.ndarray: Matriz A (N x (K+1))
        """
        num_patrones = activaciones.shape[0]
        
        # Agregar columna de 1s al inicio (para el umbral W0)
        A = np.column_stack([np.ones(num_patrones), activaciones])
        
        return A
    
    def fit(self, X, y):
        """
        Entrena la red RBF
        
        Args:
            X (np.ndarray): Matriz de datos de entrada
            y (np.ndarray): Vector de salidas deseadas
            
        Returns:
            dict: Información del entrenamiento
        """
        # Validar entrada
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"El número de patrones en X ({X.shape[0]}) no coincide con el de y ({y.shape[0]})")
        
        num_patrones, num_caracteristicas = X.shape
        
        # Inicializar centros radiales aleatoriamente dentro del rango [min(X), max(X)]
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        # Verificar que no haya valores NaN o infinitos
        if np.any(np.isnan(min_vals)) or np.any(np.isnan(max_vals)):
            raise ValueError("Los datos contienen valores NaN. Por favor, limpia tus datos.")
        
        if np.any(np.isinf(min_vals)) or np.any(np.isinf(max_vals)):
            raise ValueError("Los datos contienen valores infinitos. Por favor, limpia tus datos.")
        
        # Si min == max (columna constante), expandir ligeramente el rango
        rango_constante = (max_vals == min_vals)
        if np.any(rango_constante):
            # Expandir el rango: usar un valor absoluto mínimo o 10% del valor
            valor_expansion = np.maximum(np.abs(min_vals[rango_constante]) * 0.1, np.ones_like(min_vals[rango_constante]) * 0.1)
            min_vals[rango_constante] = min_vals[rango_constante] - valor_expansion
            max_vals[rango_constante] = max_vals[rango_constante] + valor_expansion
        
        # Si min > max (puede pasar con rangos muy pequeños), ajustar
        rangos_invalidos = max_vals < min_vals
        if np.any(rangos_invalidos):
            min_vals[rangos_invalidos], max_vals[rangos_invalidos] = max_vals[rangos_invalidos], min_vals[rangos_invalidos]
        
        # Si el rango es muy pequeño, expandir para evitar problemas numéricos
        rangos = max_vals - min_vals
        rangos_muy_pequenos = rangos < 1e-6
        if np.any(rangos_muy_pequenos):
            min_vals[rangos_muy_pequenos] -= 0.5
            max_vals[rangos_muy_pequenos] += 0.5
        
        np.random.seed(42)  # Para reproducibilidad
        self.centros = np.random.uniform(
            low=min_vals,
            high=max_vals,
            size=(self.num_centros, num_caracteristicas)
        )
        
        # 1. Calcular distancias
        distancias = self._calcular_distancias(X, self.centros)
        
        # 2. Calcular activaciones
        activaciones = self._calcular_activaciones(distancias)
        
        # 3. Construir matriz de interpolación A
        A = self._construir_matriz_interpolacion(activaciones)
        
        # 4. Resolver W = (A^T A)^(-1) A^T Y
        # Usar mínimos cuadrados con lstsq para evitar problemas con matrices singulares
        try:
            # lstsq retorna la solución de mínimos cuadrados
            W, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            
            # Guardar pesos
            self.pesos = W
            self.W0 = W[0]  # Umbral
            self.W1_n = W[1:]  # Pesos de los centros
            
            self.entrenado = True
            
            # Preparar resultado
            resultado = {
                'centros': self.centros.tolist(),
                'pesos': W.tolist(),
                'umbral': float(self.W0),
                'pesos_centros': self.W1_n.tolist(),
                'residuos': residuals.tolist() if len(residuals) > 0 else [0.0],
                'num_centros': self.num_centros
            }
            
            return resultado
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error al resolver el sistema de ecuaciones: {str(e)}")
    
    def predict(self, X):
        """
        Realiza predicciones con la red RBF
        
        Args:
            X (np.ndarray): Matriz de datos de entrada
            
        Returns:
            np.ndarray: Vector de predicciones
        """
        if not self.entrenado:
            raise ValueError("La red no ha sido entrenada. Llama a fit() primero.")
        
        # Validar entrada
        X = np.asarray(X, dtype=float)
        
        # 1. Calcular distancias
        distancias = self._calcular_distancias(X, self.centros)
        
        # 2. Calcular activaciones
        activaciones = self._calcular_activaciones(distancias)
        
        # 3. Construir matriz A
        A = self._construir_matriz_interpolacion(activaciones)
        
        # 4. Calcular Yr = A × W
        y_predicho = A.dot(self.pesos)
        
        return y_predicho

