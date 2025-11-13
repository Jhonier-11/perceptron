"""
Comando de Django para generar alertas para todos los estudiantes
"""

from django.core.management.base import BaseCommand
from prediccion_academica.alertas import generar_alertas_todos_estudiantes, limpiar_alertas_antiguas


class Command(BaseCommand):
    help = 'Genera alertas automáticas para todos los estudiantes en riesgo'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limpiar',
            action='store_true',
            help='Elimina alertas antiguas que ya fueron vistas',
        )
        parser.add_argument(
            '--dias',
            type=int,
            default=90,
            help='Número de días de antigüedad para limpiar alertas (default: 90)',
        )

    def handle(self, *args, **options):
        self.stdout.write('Generando alertas para todos los estudiantes...')
        
        try:
            total_alertas = generar_alertas_todos_estudiantes()
            self.stdout.write(
                self.style.SUCCESS(f'✓ Se generaron {total_alertas} alertas exitosamente.')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Error al generar alertas: {str(e)}')
            )
            return
        
        if options['limpiar']:
            self.stdout.write('Limpiando alertas antiguas...')
            try:
                eliminadas = limpiar_alertas_antiguas(options['dias'])
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Se eliminaron {eliminadas} alertas antiguas.')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Error al limpiar alertas: {str(e)}')
                )

