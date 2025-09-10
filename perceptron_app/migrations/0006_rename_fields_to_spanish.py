# Generated manually for renaming fields to Spanish

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('perceptron_app', '0005_alter_perceptrontraining_max_error'),
    ]

    operations = [
        # Rename PerceptronTraining fields
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='name',
            new_name='nombre',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='created_at',
            new_name='fecha_creacion',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='learning_rate',
            new_name='tasa_aprendizaje',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='epochs',
            new_name='iteraciones',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='max_error',
            new_name='error_maximo',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='input_columns',
            new_name='columnas_entrada',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='output_columns',
            new_name='columnas_salida',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='final_weights',
            new_name='pesos_finales',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='final_bias',
            new_name='sesgo_final',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='accuracy',
            new_name='precision',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='training_errors',
            new_name='errores_entrenamiento',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='weight_evolution',
            new_name='evolucion_pesos',
        ),
        migrations.RenameField(
            model_name='perceptrontraining',
            old_name='data_file',
            new_name='archivo_datos',
        ),
        # Rename Prediction fields
        migrations.RenameField(
            model_name='prediction',
            old_name='created_at',
            new_name='fecha_prediccion',
        ),
        migrations.RenameField(
            model_name='prediction',
            old_name='input_values',
            new_name='valores_entrada',
        ),
        migrations.RenameField(
            model_name='prediction',
            old_name='predicted_output',
            new_name='salida_predicha',
        ),
        # Rename the ForeignKey field
        migrations.RenameField(
            model_name='prediction',
            old_name='training',
            new_name='entrenamiento',
        ),
    ]
