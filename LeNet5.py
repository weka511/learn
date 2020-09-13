# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

# https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser('Convolutional Neural Net based on LeNet-5')
parser.add_argument('--file',default='LeNet5')
parser.add_argument('--path',default='.')
parser.add_argument('action',choices=['train','test'])
parser.add_argument('--epochs',type=int,default=5)
args = parser.parse_args()

if args.action == 'train':
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x                              = train_x / 255.0
    test_x                               = test_x / 255.0
    train_x                              = tf.expand_dims(train_x, 3)
    test_x                               = tf.expand_dims(test_x, 3)
    val_x                                = train_x[:5000]
    val_y                                = train_y[:5000]
       
    model = Sequential([
        Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'), #C1
        AveragePooling2D(), #S2
        Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
        AveragePooling2D(), #S4
        Flatten(), #Flatten
        Dense(120, activation='tanh'), #C5
        Dense(84, activation='tanh'), #F6
        Dense(10, activation='softmax') #Output layer
    ])
    
    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    model.fit(train_x, train_y, epochs=args.epochs, validation_data=(val_x, val_y))
    save_path = os.path.join(args.path,args.file)
    
    tf.saved_model.save(model, save_path) # https://www.tensorflow.org/guide/saved_model
    print (f'Saved to {save_path}')

if args.action == 'test':
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x                              = train_x / 255.0
    test_x                               = test_x / 255.0
    train_x                              = tf.expand_dims(train_x, 3)
    test_x                               = tf.expand_dims(test_x, 3)
    val_x                                = train_x[:5000]
    val_y                                = train_y[:5000]
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    load_path = os.path.join(args.path,args.file)
    new_model = tf.keras.models.load_model(load_path)
    print (f'Loaded {load_path}')
    print ('Check its architecture')
    new_model.summary()
    
    print ('Testing')
    new_model.evaluate(test_x, test_y)
    