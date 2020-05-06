from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy

# obtenemos corpus entrenamiento y test la primera vez desde Internet 
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# las siguientes veces, lo podemos cargar de esta forma porque lo tenemos en ~/.keras/datasets 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('mnist.npz')

print("CARACTERÍSTICAS (x)")
print("  dimensión: ",x_train.ndim)
print("  forma: ",x_train.shape)
print("  tipo: ",x_train.dtype)
print("ETIQUETAS (y)")
print("  dimensión: ",y_train.ndim)
print("  forma: ",y_train.shape)
print("  tipo: ",y_train.dtype)

print(x_test.shape)
print(y_test.shape)

x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train, num_classes=10)

model = Sequential([
    Flatten(input_shape=(28, 28)),      # Capa entrada
    Dense(10, activation=tf.nn.softmax) # Capa salida
])

model.summary()

#entrenamos
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=50,
          epochs=10)

#guardamos el modelo
model.save('number_prediction_model') 