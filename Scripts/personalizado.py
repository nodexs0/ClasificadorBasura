import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json

# Definir la arquitectura personalizada
def create_model(input_shape, num_classes, scale=1.0):
    model = Sequential()

    # Bloque 1
    model.add(Conv2D(int(96 * scale), (11, 11), strides=(4, 4), padding='same', 
                     activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Bloque 2
    model.add(Conv2D(int(256 * scale), (5, 5), strides=(1, 1), padding='same', 
                     activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Bloque 3
    model.add(Conv2D(int(384 * scale), (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(int(384 * scale), (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(int(256 * scale), (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Flatten y Fully Connected
    model.add(Flatten())
    model.add(Dense(int(4096 * scale), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(int(4096 * scale), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Parámetros
input_shape = (224, 224, 3)
num_classes = 7
scale = 1.0

# Crear el modelo
model = create_model(input_shape, num_classes, scale)

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preprocesamiento y aumento de datos
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
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Calcular los pesos de clase
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Callbacks avanzados
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('Modelos/best_model_custom.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Entrenamiento inicial con historial guardado
history_initial = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# Guardar el historial inicial
with open('Modelos/history_initial.json', 'w') as f:
    json.dump(history_initial.history, f)

# Fine-tuning: Descongelar todas las capas
for layer in model.layers:
    layer.trainable = True

# Recompilar con un learning rate más bajo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento avanzado con historial guardado
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights
)

# Guardar el historial de fine-tuning
with open('Modelos/history_finetune.json', 'w') as f:
    json.dump(history_finetune.history, f)

# Evaluación final
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

# Generar reporte de clasificación y matriz de confusión
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Guardar el modelo final en formato h5
model.save('Modelos/recycling_classifier_model.h5')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open('Modelos/recycling_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Verificar carga del modelo TFLite
interpreter = tf.lite.Interpreter(model_path='Modelos/recycling_classifier_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo TFLite cargado correctamente.")
print("Detalles de entrada:", input_details)
print("Detalles de salida:", output_details)
