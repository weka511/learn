# https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x                              = train_x / 255.0
test_x                               = test_x / 255.0
train_x                              = tf.expand_dims(train_x, 3)
test_x                               = tf.expand_dims(test_x, 3)
val_x                                = train_x[:5000]
val_y                                = train_y[:5000]
# https://www.tensorflow.org/tutorials/keras/save_and_load
mobilenet_save_path = os.path.join('.', 'save')
#loaded = tf.saved_model.load(mobilenet_save_path)
new_model = tf.keras.models.load_model(mobilenet_save_path)

# Check its architecture
new_model.summary()

#print(list(loaded.signatures.keys()))  # ["serving_default"]

print ('Testing')
new_model.evaluate(test_x, test_y)
