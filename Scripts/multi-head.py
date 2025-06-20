import math
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Dropout, Conv2D, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Multi-Head Attention Layer
class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, attention_dim):
        super(MultiHeadAttentionBlock, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense = tf.keras.layers.Dense(attention_dim, activation='relu')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        out1 = self.norm1(inputs + attn_output)  # Residual connection + normalization
        out2 = self.dense(out1)
        return self.norm2(out1 + out2)  # Residual connection + normalization

# Función para graficar y guardar curvas de pérdida y precisión
def plot_training_curves(history, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Pérdida
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))

    # Precisión
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))

# Función para graficar y guardar la matriz de confusión
def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path)

# 1. Base MobileNetV2 preentrenada
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 2. Añadir bloque Multi-Head sobre las características
x = base_model.output  # Shape (None, 7, 7, 1280)
x = Conv2D(64, (1, 1), activation='relu')(x)  # Reducir canales a 64
x = Reshape((-1, 64))(x)  # Convertir a secuencia: (None, 49, 64)
x = MultiHeadAttentionBlock(num_heads=4, attention_dim=64)(x)  # Multi-Head
x = Dropout(0.3)(x)  # Dropout para reducir el sobreajuste

# 3. Capas densas finales con regularización y dropout
x = GlobalAveragePooling1D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.3)(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compilar modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Preparar aumentos y generadores
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
    'organized_dataset_trashnet/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'organized_dataset_trashnet/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 6. Calcular pesos de clase para desbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# 7. Entrenamiento inicial
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('Modelos/best_model_attention.h5', monitor='val_loss', save_best_only=True)
]

steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

# Directorio para resultados
output_dir = 'Modelos/Resultados'

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# Guardar curvas de entrenamiento
plot_training_curves(history, output_dir)

# 8. Fine-tuning avanzado
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_ft = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights
)

# Guardar curvas de fine-tuning
plot_training_curves(history_ft, output_dir)

# 9. Evaluación en test y guardar matriz de confusión
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'organized_dataset_trashnet/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Guardar la matriz de confusión
save_confusion_matrix(y_true, y_pred, list(test_generator.class_indices.keys()), os.path.join(output_dir, 'confusion_matrix.png'))

# 10. Guardar modelo sin metadata
model.save('Modelos/recycling_classifier_attention.h5')
