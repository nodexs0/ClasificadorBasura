import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import os
from tflite_support import metadata
import math  # Importa math para usar math.ceil

# Bloque Squeeze-and-Excitation (SE)
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1,1,filters))(se)  # Reshape para multiplicar por canales
    x = Multiply()([input_tensor, se])
    return x

# 1. Cargar MobileNetV2 preentrenado en ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas base
base_model.trainable = False

# 2. Agregar bloque SE y capas personalizadas con regularización y dropout
x = base_model.output
x = se_block(x)  # Atención SE
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)  # Dropout para reducir el sobreajuste
predictions = Dense(7, activation='softmax')(x)

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compilar el modelo con Adam
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Preprocesamiento y aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Asegurar que los datos se mezclen
)

validation_generator = train_datagen.flow_from_directory(
    'val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # No mezclar en validación para mantener el orden
)

# Calcula steps_per_epoch CORRECTO con math.ceil
train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
val_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

# Calcular los pesos de clase para manejar clases desbalanceadas
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# 5. Callbacks para control de entrenamiento
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('Modelos/best_model.h5', monitor='val_loss', save_best_only=True)
]

# 6. Entrenamiento inicial
model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps= val_steps,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# 7. Fine-tuning avanzado
base_model.trainable = True
for layer in base_model.layers[:100]:  # Congelar las primeras 100 capas
    layer.trainable = False

# Compilar nuevamente con un learning rate más bajo
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento adicional
model.fit(
    train_generator,
    steps_per_epoch= train_steps,
    validation_data=validation_generator,
    validation_steps= val_steps,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights
)

# 8. Evaluar en datos de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# 9. Generar reporte de clasificación y matriz de confusión
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 10. Guardar el modelo en formato H5
model.save('Modelos/recycling_classifier_model.h5')

# 11. Crear un modelo mejorado con normalización incorporada
input_tensor = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32, name='input_image')
x = tf.keras.layers.Lambda(lambda x: x / 255.0)(input_tensor)  # Normalización incorporada
output_tensor = model(x)
full_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

# 12. Convertir a TFLite con metadatos
converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
tflite_model = converter.convert()

# Obtener etiquetas ordenadas
sorted_labels = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda item: item[1])]

# Guardar modelo con metadatos
try:
    # Crear archivo de etiquetas en memoria
    label_file = metadata.FloatTxtFile.create(sorted_labels, "labels.txt")
    
    # Configurar metadatos de entrada
    input_metadata = metadata.InputMetadata()
    input_metadata.name = "image"
    input_metadata.description = "Input image to be classified. 224x224 RGB."
    input_metadata.content = metadata.Content()
    input_metadata.content.contentProperties = metadata.ImageProperties()
    input_metadata.content.contentProperties.colorSpace = metadata.ColorSpaceType.RGB
    input_metadata.content.contentPropertiesType = metadata.ContentProperties.ImageProperties
    
    # Configurar metadatos de salida
    output_metadata = metadata.OutputMetadata()
    output_metadata.name = "probability"
    output_metadata.description = "Probabilities of the classes"
    output_metadata.associatedFiles = [label_file]
    
    # Crear escritor y agregar metadatos
    writer = metadata.MetadataWriter.create_from_tflite_model(tflite_model)
    writer = writer.with_input_metadata(input_metadata, 0)
    writer = writer.with_output_metadata(output_metadata, 0)
    writer = writer.with_author("node")
    writer = writer.with_description("Clasificador de residuos reciclables")
    
    # Generar modelo con metadatos
    tflite_model_with_metadata = writer.populate()
    
    # Guardar modelo final
    with open('Modelos/recycling_classifier_model.tflite', 'wb') as f:
        f.write(tflite_model_with_metadata)
        
except ImportError:
    print("Advertencia: tflite_support no instalado. Guardando sin metadatos.")
    with open('Modelos/recycling_classifier_model.tflite', 'wb') as f:
        f.write(tflite_model)
