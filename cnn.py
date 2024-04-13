import sys
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

K.clear_session()

data_entrenamiento = 'CNN/data/entrenamiento'
data_validacion = 'CNN/data/validacion'

epocas = 20
longitud, altura = 150, 150
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 9
lr = 0.0004

entrenamiento_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

cnn = Sequential()
cnn.add(Conv2D(filtrosConv1, tamano_filtro1, padding="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv2, tamano_filtro2, padding="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

optimizer = Adam(learning_rate=lr)

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

# Cambio de fit_generator a fit
cnn.fit(
    entrenamiento_generador,
    epochs=epocas,
    validation_data=validacion_generador
)

target_dir = 'CNN/modelo'
os.makedirs(target_dir, exist_ok=True)
cnn.save(os.path.join(target_dir, 'modelo.keras'))
cnn.save_weights(os.path.join(target_dir, 'pesos.weights.h5'))
