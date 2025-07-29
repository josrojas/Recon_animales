from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import os

# Parámetros
IMG_SIZE = (224, 224)  # MobileNetV2 requiere al menos 96x96
BATCH_SIZE = 32
EPOCHS = 20

# Preprocesamiento
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    'data/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    'data/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Arquitectura con MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

num_classes = train_data.num_classes

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Entrenamiento
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop]
)

# Guardar modelo
os.makedirs('model', exist_ok=True)
model.save('model/model.h5')
print("✅ Modelo con MobileNetV2 guardado correctamente.")
print("Clases detectadas:", train_data.class_indices)
