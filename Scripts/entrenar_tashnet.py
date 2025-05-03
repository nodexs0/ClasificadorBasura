import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Cargar MobileNetV2 preentrenado en ImageNet (sin la capa de salida)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas base para que no se entrenen
base_model.trainable = False

# 2. Agregar capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reducir la dimensión espacial
x = Dense(1024, activation='relu')(x)  # Capa densa de 1024 neuronas
predictions = Dense(6, activation='softmax')(x)  # Capa de salida para 6 clases

# 3. Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Ver el resumen del modelo
model.summary()

# 5. Preparar los datos de entrenamiento, validación y prueba
data_dir = 'C:/Users/nodex/OneDrive/Escritorio/Proyecto_2do_parcial_DL/Datasets/TrashNet/dataset-resized'

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 6. Definir callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# 7. Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
    callbacks=callbacks
)

# 8. Descongelar algunas capas para entrenamiento adicional
base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5,
    callbacks=callbacks
)

# 9. Evaluación del modelo
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# 10. Generar reporte de clasificación y matriz de confusión
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 11. Guardar el modelo
model.save('recycling_classifier_model.h5')

# 12. Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('recycling_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)
