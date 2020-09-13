# https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense
import numpy as np
import os

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]


lenet_5_model = Sequential([
    Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'), #C1
    AveragePooling2D(), #S2
    Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
    AveragePooling2D(), #S4
    Flatten(), #Flatten
    Dense(120, activation='tanh'), #C5
    Dense(84, activation='tanh'), #F6
    Dense(10, activation='softmax') #Output layer
])

lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

lenet_5_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))
mobilenet_save_path = os.path.join('.', 'save')

tf.saved_model.save(lenet_5_model, mobilenet_save_path) # https://www.tensorflow.org/guide/saved_model

print ('Testing')
lenet_5_model.evaluate(test_x, test_y)
