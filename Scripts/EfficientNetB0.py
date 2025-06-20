import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Centralización de hiperparámetros
params = {
    'input_shape': (224, 224, 3),
    'batch_size': 32,
    'learning_rate': 0.001,
    'fine_tune_learning_rate': 1e-4,
    'epochs_initial': 20,
    'epochs_fine_tune': 15,
    'dropout_rate': 0.5,
    'l2_reg': 0.01,
    'num_classes': 7
}

def create_model(params):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=params['input_shape'])
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(x)
    x = Dropout(params['dropout_rate'])(x)
    predictions = Dense(params['num_classes'], activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def preprocess_data(params):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        'train',
        target_size=params['input_shape'][:2],
        batch_size=params['batch_size'],
        class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        'val',
        target_size=params['input_shape'][:2],
        batch_size=params['batch_size'],
        class_mode='categorical'
    )
    return train_gen, val_gen

def create_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('Modelos/best_model.h5', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

# Configurar modelo
model = create_model(params)
model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocesar datos
train_gen, val_gen = preprocess_data(params)

# Calcular pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights = dict(enumerate(class_weights))

# Entrenamiento inicial
model.fit(train_gen, validation_data=val_gen, epochs=params['epochs_initial'],
          class_weight=class_weights, callbacks=create_callbacks())

# Fine-tuning
model.trainable = True
for layer in model.layers[:100]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=params['fine_tune_learning_rate']),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=params['epochs_fine_tune'],
          class_weight=class_weights, callbacks=create_callbacks())

# Evaluar en datos de prueba
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'test',
    target_size=params['input_shape'][:2],
    batch_size=params['batch_size'],
    class_mode='categorical',
    shuffle=False
)
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Generar reporte de clasificación y matriz de confusión
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Guardar el modelo en formato SavedModel
model.save('Modelos/recycling_classifier_model')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('Modelos/recycling_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)
