"""
Utilidades para preprocesamiento de datos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Crear placeholders para evitar errores de importación
    StandardScaler = None
    MinMaxScaler = None
    LabelEncoder = None


def convertir_a_tipos_nativos(obj: Any) -> Any:
    """
    Convierte objetos de NumPy y pandas a tipos nativos de Python para serialización JSON
    
    Args:
        obj: Objeto a convertir
        
    Returns:
        Objeto convertido a tipos nativos de Python
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_tipos_nativos(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_tipos_nativos(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convertir_a_tipos_nativos(item) for item in obj)
    else:
        return obj


def detectar_separador_csv(archivo) -> str:
    """
    Detecta automáticamente el separador de un archivo CSV
    
    Args:
        archivo: Archivo CSV
        
    Returns:
        Separador detectado
    """
    separadores = [',', ';', '|', '\t', ' ', ':', '~']
    
    archivo.seek(0)
    try:
        sample = archivo.read(1024)
        if isinstance(sample, bytes):
            sample = sample.decode('utf-8', errors='ignore')
    except:
        archivo.seek(0)
        try:
            sample = archivo.read(1024).decode('latin-1', errors='ignore')
        except:
            sample = archivo.read(1024).decode('utf-8', errors='ignore')
    
    archivo.seek(0)
    
    separador_counts = {}
    for sep in separadores:
        separador_counts[sep] = sample.count(sep)
    
    mejor_separador = max(separador_counts, key=separador_counts.get)
    
    if separador_counts[mejor_separador] == 0:
        mejor_separador = ','
    
    return mejor_separador


def leer_csv_auto(archivo) -> pd.DataFrame:
    """
    Lee un archivo CSV detectando automáticamente el separador
    
    Args:
        archivo: Archivo CSV
        
    Returns:
        DataFrame con los datos
    """
    separador = detectar_separador_csv(archivo)
    archivo.seek(0)
    
    try:
        df = pd.read_csv(archivo, sep=separador, encoding='utf-8')
    except UnicodeDecodeError:
        archivo.seek(0)
        try:
            df = pd.read_csv(archivo, sep=separador, encoding='latin-1')
        except:
            archivo.seek(0)
            df = pd.read_csv(archivo, sep=separador, encoding='utf-8', errors='ignore')
    
    return df


def normalizar_caracteristicas(
    X: np.ndarray,
    metodo: str = 'standard',
    scaler: Optional[Any] = None
) -> Tuple[np.ndarray, Any]:
    """
    Normaliza las características numéricas
    
    Args:
        X: Datos de entrada
        metodo: Método de normalización ('standard', 'minmax')
        scaler: Scaler pre-entrenado (opcional)
        
    Returns:
        Tupla (X_normalizado, scaler)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn no está instalado. Por favor, instálalo usando: "
            "pip install scikit-learn"
        )
    
    if scaler is None:
        if metodo == 'standard':
            scaler = StandardScaler()
        elif metodo == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método de normalización '{metodo}' no soportado")
        
        X_normalizado = scaler.fit_transform(X)
    else:
        X_normalizado = scaler.transform(X)
    
    return X_normalizado, scaler


def one_hot_encoding(df: pd.DataFrame, columnas: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Aplica one-hot encoding a columnas categóricas
    
    Args:
        df: DataFrame con los datos
        columnas: Lista de nombres de columnas a codificar
        
    Returns:
        Tupla (df_codificado, mapeo_categorias)
    """
    df_resultado = df.copy()
    mapeo_categorias = {}
    
    for col in columnas:
        if col in df_resultado.columns:
            # Obtener categorías únicas
            categorias = df_resultado[col].unique()
            mapeo_categorias[col] = [str(cat) for cat in categorias]
            
            # Aplicar one-hot encoding
            dummies = pd.get_dummies(df_resultado[col], prefix=col)
            df_resultado = pd.concat([df_resultado, dummies], axis=1)
            df_resultado = df_resultado.drop(columns=[col])
    
    return df_resultado, mapeo_categorias


def label_encoding(df: pd.DataFrame, columnas: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Aplica label encoding a columnas categóricas
    
    Args:
        df: DataFrame con los datos
        columnas: Lista de nombres de columnas a codificar
        
    Returns:
        Tupla (df_codificado, mapeo_labels)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn no está instalado. Por favor, instálalo usando: "
            "pip install scikit-learn"
        )
    
    df_resultado = df.copy()
    mapeo_labels = {}
    
    for col in columnas:
        if col in df_resultado.columns:
            encoder = LabelEncoder()
            df_resultado[col] = encoder.fit_transform(df_resultado[col].astype(str))
            
            # Guardar mapeo
            mapeo_labels[col] = {
                label: int(encoded) for label, encoded in 
                zip(encoder.classes_, encoder.transform(encoder.classes_))
            }
    
    return df_resultado, mapeo_labels


def manejar_valores_faltantes(
    df: pd.DataFrame,
    estrategia: str = 'media',
    columnas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Maneja valores faltantes en el DataFrame
    
    Args:
        df: DataFrame con los datos
        estrategia: Estrategia para manejar valores faltantes ('media', 'mediana', 'moda', 'eliminar')
        columnas: Lista de columnas a procesar (None para todas)
        
    Returns:
        DataFrame sin valores faltantes
    """
    df_resultado = df.copy()
    
    if columnas is None:
        columnas = df_resultado.columns.tolist()
    
    for col in columnas:
        if col in df_resultado.columns and df_resultado[col].isna().any():
            if estrategia == 'media':
                if pd.api.types.is_numeric_dtype(df_resultado[col]):
                    df_resultado[col].fillna(df_resultado[col].mean(), inplace=True)
            elif estrategia == 'mediana':
                if pd.api.types.is_numeric_dtype(df_resultado[col]):
                    df_resultado[col].fillna(df_resultado[col].median(), inplace=True)
            elif estrategia == 'moda':
                df_resultado[col].fillna(df_resultado[col].mode()[0] if not df_resultado[col].mode().empty else 0, inplace=True)
            elif estrategia == 'eliminar':
                df_resultado = df_resultado.dropna(subset=[col])
    
    return df_resultado


def preprocesar_datos_estudiantes(
    df: pd.DataFrame,
    columnas_entrada: List[str],
    columna_salida: str,
    normalizar: bool = True,
    metodo_normalizacion: str = 'standard',
    usar_one_hot: bool = True,
    manejar_faltantes: bool = True,
    estrategia_faltantes: str = 'media'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesa datos de estudiantes para el entrenamiento del MLP
    
    Args:
        df: DataFrame con los datos de estudiantes
        columnas_entrada: Lista de columnas a usar como características
        columna_salida: Nombre de la columna objetivo
        normalizar: Si es True, normaliza las características
        metodo_normalizacion: Método de normalización ('standard', 'minmax')
        usar_one_hot: Si es True, usa one-hot encoding para categóricas
        manejar_faltantes: Si es True, maneja valores faltantes
        estrategia_faltantes: Estrategia para manejar valores faltantes
        
    Returns:
        Tupla (X, y, info_preprocesamiento)
    """
    df_procesado = df.copy()
    info_preprocesamiento = {
        'transformaciones': [],
        'mapeo_categorias': {},
        'mapeo_labels': {},
        'scaler': None,
        'columnas_originales': columnas_entrada.copy(),
        'columnas_finales': []
    }
    
    # Manejar valores faltantes
    if manejar_faltantes:
        df_procesado = manejar_valores_faltantes(
            df_procesado,
            estrategia=estrategia_faltantes,
            columnas=columnas_entrada + [columna_salida]
        )
        info_preprocesamiento['transformaciones'].append(f'Manejo de valores faltantes ({estrategia_faltantes})')
    
    # Identificar columnas categóricas y numéricas
    columnas_categoricas = []
    columnas_numericas = []
    
    for col in columnas_entrada:
        if col in df_procesado.columns:
            if pd.api.types.is_numeric_dtype(df_procesado[col]):
                columnas_numericas.append(col)
            else:
                columnas_categoricas.append(col)
    
    # Aplicar codificación a columnas categóricas
    if usar_one_hot and columnas_categoricas:
        df_procesado, mapeo_categorias = one_hot_encoding(df_procesado, columnas_categoricas)
        info_preprocesamiento['mapeo_categorias'] = mapeo_categorias
        info_preprocesamiento['transformaciones'].append('One-hot encoding aplicado')
    
    elif columnas_categoricas:
        df_procesado, mapeo_labels = label_encoding(df_procesado, columnas_categoricas)
        info_preprocesamiento['mapeo_labels'] = mapeo_labels
        info_preprocesamiento['transformaciones'].append('Label encoding aplicado')
    
    # Obtener columnas finales (pueden haber cambiado después del one-hot encoding)
    columnas_finales = [col for col in df_procesado.columns if col != columna_salida]
    info_preprocesamiento['columnas_finales'] = columnas_finales
    
    # Separar características y objetivo
    X = df_procesado[columnas_finales].values.astype(float)
    y = df_procesado[columna_salida].values.astype(float)
    
    # Normalizar características
    if normalizar:
        X, scaler = normalizar_caracteristicas(X, metodo=metodo_normalizacion)
        info_preprocesamiento['scaler'] = scaler
        info_preprocesamiento['transformaciones'].append(f'Normalización ({metodo_normalizacion})')
    
    return X, y, info_preprocesamiento


def convertir_estudiante_a_caracteristicas(
    estudiante,
    columnas_entrada: List[str],
    info_preprocesamiento: Dict[str, Any]
) -> np.ndarray:
    """
    Convierte un objeto Estudiante a un vector de características
    
    Args:
        estudiante: Objeto Estudiante
        columnas_entrada: Lista de columnas de entrada originales
        info_preprocesamiento: Información de preprocesamiento
        
    Returns:
        Vector de características normalizado
    """
    # Crear diccionario con las características del estudiante
    datos = {
        'edad': estudiante.edad,
        'sexo': estudiante.sexo,
        'direccion': estudiante.direccion,
        'tamano_familia': estudiante.tamano_familia,
        'estado_padres': estudiante.estado_padres,
        'educacion_madre': estudiante.educacion_madre,
        'educacion_padre': estudiante.educacion_padre,
        'trabajo_madre': estudiante.trabajo_madre,
        'trabajo_padre': estudiante.trabajo_padre,
        'tiempo_viaje': estudiante.tiempo_viaje,
        'tiempo_estudio': estudiante.tiempo_estudio,
        'fallos_previos': estudiante.fallos_previos,
        'apoyo_escuela': 1 if estudiante.apoyo_escuela else 0,
        'apoyo_familia': 1 if estudiante.apoyo_familia else 0,
        'clases_pagadas': 1 if estudiante.clases_pagadas else 0,
        'actividades_extra': 1 if estudiante.actividades_extra else 0,
        'guarderia': 1 if estudiante.guarderia else 0,
        'quiere_superior': 1 if estudiante.quiere_superior else 0,
        'internet': 1 if estudiante.internet else 0,
        'relacion_romantica': 1 if estudiante.relacion_romantica else 0,
        'relacion_familiar': estudiante.relacion_familiar,
        'tiempo_libre': estudiante.tiempo_libre,
        'salidas': estudiante.salidas,
        'alcohol_semana': estudiante.alcohol_semana,
        'alcohol_fin_semana': estudiante.alcohol_fin_semana,
        'salud': estudiante.salud,
        'ausencias': estudiante.ausencias,
    }
    
    # Agregar campos académicos universitarios si están disponibles y se necesitan
    if 'semestre_actual' in columnas_entrada:
        datos['semestre_actual'] = estudiante.semestre_actual if estudiante.semestre_actual else 1
    if 'puntaje_icfes_global' in columnas_entrada:
        datos['puntaje_icfes_global'] = estudiante.puntaje_icfes_global if estudiante.puntaje_icfes_global else None
    if 'estrato' in columnas_entrada:
        datos['estrato'] = estudiante.estrato if estudiante.estrato else None
    if 'programa_academico' in columnas_entrada:
        datos['programa_academico'] = estudiante.programa_academico if estudiante.programa_academico else ''
    if 'trabaja_actualmente' in columnas_entrada:
        datos['trabaja_actualmente'] = 1 if estudiante.trabaja_actualmente else 0
    if 'horas_trabajo_sem' in columnas_entrada:
        datos['horas_trabajo_sem'] = estudiante.horas_trabajo_sem if estudiante.horas_trabajo_sem else 0
    if 'promedio_semestre_anterior' in columnas_entrada:
        datos['promedio_semestre_anterior'] = estudiante.promedio_semestre_anterior if estudiante.promedio_semestre_anterior else None
    if 'promedio_acumulado' in columnas_entrada:
        datos['promedio_acumulado'] = estudiante.promedio_acumulado if estudiante.promedio_acumulado else None
    
    # Crear DataFrame con una sola fila, solo con las columnas de entrada
    datos_filtrados = {col: datos.get(col, 0) for col in columnas_entrada}
    df = pd.DataFrame([datos_filtrados])
    
    # Aplicar las mismas transformaciones que en el preprocesamiento
    # One-hot encoding
    if info_preprocesamiento.get('mapeo_categorias'):
        for col, categorias in info_preprocesamiento['mapeo_categorias'].items():
            if col in df.columns:
                valor_actual = str(df[col].iloc[0])
                # Crear columnas one-hot para todas las categorías
                for categoria in categorias:
                    df[f'{col}_{categoria}'] = 1 if valor_actual == str(categoria) else 0
                # Eliminar la columna original
                df = df.drop(columns=[col])
    
    # Label encoding
    if info_preprocesamiento.get('mapeo_labels'):
        for col, mapeo in info_preprocesamiento['mapeo_labels'].items():
            if col in df.columns:
                valor = str(df[col].iloc[0])
                if valor in mapeo:
                    df[col] = mapeo[valor]
                else:
                    # Usar el primer valor del mapeo como valor por defecto
                    df[col] = list(mapeo.values())[0] if mapeo else 0
    
    # Obtener columnas finales (después de one-hot encoding)
    columnas_finales = info_preprocesamiento.get('columnas_finales', columnas_entrada)
    
    # Asegurar que todas las columnas finales estén en el DataFrame
    # Agregar columnas faltantes con valor 0
    for col in columnas_finales:
        if col not in df.columns:
            df[col] = 0
    
    # Seleccionar solo las columnas finales en el mismo orden
    X = df[columnas_finales].values.astype(float)
    
    # Normalizar usando el scaler guardado
    if info_preprocesamiento.get('scaler'):
        scaler = info_preprocesamiento['scaler']
        X = scaler.transform(X)
    elif info_preprocesamiento.get('scaler_mean') and info_preprocesamiento.get('scaler_scale'):
        # Normalizar manualmente usando mean y scale guardados
        mean = np.array(info_preprocesamiento['scaler_mean'])
        scale = np.array(info_preprocesamiento['scaler_scale'])
        X = (X - mean) / scale
    
    return X

