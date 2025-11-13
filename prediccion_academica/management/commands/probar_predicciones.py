"""
Comando de Django para probar predicciones con datos reales
"""

from django.core.management.base import BaseCommand
from prediccion_academica.models import Estudiante, EntrenamientoMLP, PrediccionRendimiento
from prediccion_academica.mlp_engine import MLP
from prediccion_academica.utils import convertir_estudiante_a_caracteristicas
from prediccion_academica.alertas import generar_alertas_estudiante
import numpy as np


class Command(BaseCommand):
    help = 'Prueba predicciones con datos reales del dataset'

    def add_arguments(self, parser):
        parser.add_argument(
            '--entrenamiento-id',
            type=int,
            help='ID del entrenamiento a usar para las predicciones',
        )
        parser.add_argument(
            '--estudiante-id',
            type=int,
            help='ID del estudiante específico a predecir (opcional)',
        )
        parser.add_argument(
            '--limite',
            type=int,
            default=10,
            help='Número máximo de estudiantes a probar (default: 10)',
        )

    def handle(self, *args, **options):
        # Obtener el entrenamiento más reciente si no se especifica uno
        if options['entrenamiento_id']:
            entrenamiento = EntrenamientoMLP.objects.get(id=options['entrenamiento_id'])
        else:
            entrenamiento = EntrenamientoMLP.objects.order_by('-fecha_creacion').first()
        
        if not entrenamiento:
            self.stdout.write(
                self.style.ERROR('✗ No hay entrenamientos disponibles. Por favor, entrena un modelo primero.')
            )
            return
        
        self.stdout.write(f'Usando entrenamiento: {entrenamiento.nombre}')
        self.stdout.write(f'Precisión entrenamiento: {entrenamiento.precision_entrenamiento:.4f}')
        self.stdout.write(f'Precisión validación: {entrenamiento.precision_validacion:.4f}')
        self.stdout.write('')
        
        # Obtener estudiantes para probar
        if options['estudiante_id']:
            estudiantes = Estudiante.objects.filter(id=options['estudiante_id'])
        else:
            # Filtrar estudiantes que tengan la columna de salida
            if entrenamiento.columna_salida == 'G1':
                estudiantes = Estudiante.objects.exclude(calificacion_g1__isnull=True)[:options['limite']]
            elif entrenamiento.columna_salida == 'G2':
                estudiantes = Estudiante.objects.exclude(calificacion_g2__isnull=True)[:options['limite']]
            elif entrenamiento.columna_salida == 'G3':
                estudiantes = Estudiante.objects.exclude(calificacion_g3__isnull=True)[:options['limite']]
            else:
                estudiantes = Estudiante.objects.all()[:options['limite']]
        
        if estudiantes.count() == 0:
            self.stdout.write(
                self.style.ERROR('✗ No hay estudiantes disponibles para probar.')
            )
            return
        
        self.stdout.write(f'Probando predicciones para {estudiantes.count()} estudiantes...')
        self.stdout.write('')
        
        # Obtener información de preprocesamiento
        info_preprocesamiento = entrenamiento.info_preprocesamiento or {}
        if entrenamiento.scaler_mean and entrenamiento.scaler_scale:
            info_preprocesamiento['scaler_mean'] = entrenamiento.scaler_mean
            info_preprocesamiento['scaler_scale'] = entrenamiento.scaler_scale
        
        # Obtener el número de características después del preprocesamiento
        if entrenamiento.num_caracteristicas_finales:
            n_entradas = entrenamiento.num_caracteristicas_finales
        else:
            columnas_finales = info_preprocesamiento.get('columnas_finales', entrenamiento.columnas_entrada)
            n_entradas = len(columnas_finales) if columnas_finales else len(entrenamiento.columnas_entrada)
        
        # Reconstruir el MLP
        arquitectura = [n_entradas] + entrenamiento.neuronas_por_capa + [1]
        mlp = MLP(
            arquitectura=arquitectura,
            tasa_aprendizaje=entrenamiento.tasa_aprendizaje,
            funcion_activacion=entrenamiento.funcion_activacion
        )
        mlp.cargar_pesos(entrenamiento.pesos_capas, entrenamiento.sesgos_capas)
        
        # Realizar predicciones
        errores = []
        predicciones_correctas = 0
        total_predicciones = 0
        
        for estudiante in estudiantes:
            try:
                # Convertir estudiante a características
                X = convertir_estudiante_a_caracteristicas(
                    estudiante,
                    entrenamiento.columnas_entrada,
                    info_preprocesamiento
                )
                
                # Hacer predicción
                prediccion = mlp.predecir(X)[0]
                
                # Obtener calificación real
                calificacion_real = getattr(estudiante, f'calificacion_{entrenamiento.columna_salida.lower()}', None)
                
                if calificacion_real is not None:
                    error = abs(prediccion - calificacion_real)
                    errores.append(error)
                    
                    # Crear o actualizar predicción
                    prediccion_obj, creado = PrediccionRendimiento.objects.get_or_create(
                        estudiante=estudiante,
                        entrenamiento=entrenamiento,
                        defaults={
                            'calificacion_predicha': float(prediccion),
                            'calificacion_real': calificacion_real,
                            'caracteristicas_usadas': {col: getattr(estudiante, col, None) for col in entrenamiento.columnas_entrada}
                        }
                    )
                    
                    if not creado:
                        prediccion_obj.calificacion_predicha = float(prediccion)
                        prediccion_obj.calificacion_real = calificacion_real
                        prediccion_obj.calcular_error()
                    
                    # Generar alertas
                    try:
                        generar_alertas_estudiante(estudiante, prediccion_obj)
                    except Exception as e:
                        pass  # Continuar aunque haya error en alertas
                    
                    # Verificar si la predicción es correcta (dentro de 2 puntos)
                    if error <= 2.0:
                        predicciones_correctas += 1
                    
                    total_predicciones += 1
                    
                    # Mostrar resultado
                    estado = '✓' if error <= 2.0 else '✗'
                    self.stdout.write(
                        f'{estado} {estudiante.nombre} {estudiante.apellido}: '
                        f'Predicción: {prediccion:.2f}, Real: {calificacion_real:.2f}, Error: {error:.2f}'
                    )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Error al predecir para {estudiante.nombre}: {str(e)}')
                )
        
        # Mostrar estadísticas
        self.stdout.write('')
        self.stdout.write('=' * 60)
        self.stdout.write('ESTADÍSTICAS DE PREDICCIÓN')
        self.stdout.write('=' * 60)
        self.stdout.write(f'Total de predicciones: {total_predicciones}')
        self.stdout.write(f'Predicciones correctas (error ≤ 2.0): {predicciones_correctas}')
        self.stdout.write(f'Precisión: {(predicciones_correctas / total_predicciones * 100):.2f}%' if total_predicciones > 0 else 'Precisión: N/A')
        
        if errores:
            error_promedio = np.mean(errores)
            error_mediano = np.median(errores)
            error_max = np.max(errores)
            error_min = np.min(errores)
            
            self.stdout.write(f'Error promedio: {error_promedio:.2f}')
            self.stdout.write(f'Error mediano: {error_mediano:.2f}')
            self.stdout.write(f'Error máximo: {error_max:.2f}')
            self.stdout.write(f'Error mínimo: {error_min:.2f}')
        
        self.stdout.write('=' * 60)

