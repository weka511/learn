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

# https://www.tensorflow.org/tutorials/keras/save_and_load

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# create_data
#
# Read training and test data

def create_data():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x                              = train_x / 255.0
    test_x                               = test_x / 255.0
    train_x                              = tf.expand_dims(train_x, 3)
    test_x                               = tf.expand_dims(test_x, 3)
    val_x                                = train_x[:5000]
    val_y                                = train_y[:5000]
    return (train_x, train_y), (test_x, test_y), (val_x,val_y)

# create_model
#
# Create Neuaral Network model and compile it
#
# Parameters:
#     train_x    Used to assign size to first layer

def create_model(train_x):
    product = Sequential([
        Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'), #C1
        AveragePooling2D(), #S2
        Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
        AveragePooling2D(), #S4
        Flatten(), #Flatten
        Dense(120, activation='tanh'), #C5
        Dense(84, activation='tanh'), #F6
        Dense(10, activation='softmax') #Output layer
    ])
    product.compile(optimizer='adam',
                   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return product

if __name__=='__main__':
    import argparse
    import os
    
    print(f'Using Tensorflow {tf.version.VERSION}')    
    path   = os.path.join(os.getenv('APPDATA'),'LeNet5')
    parser = argparse.ArgumentParser('Convolutional Neural Net based on LeNet-5')
    parser.add_argument('action',   choices=['train','test', 'continue'],       help = 'Train or test')
    parser.add_argument('--checkpoint',                       default='checkpoint', help = 'Name of file to save and restore network')
    parser.add_argument('--path',                             default=path,     help = 'Path of file used to save and restore network')
    parser.add_argument('--epochs', type=int,                 default=5,       help = 'Number of epochs for training')
    args        = parser.parse_args()
    
    checkpoint  = os.path.join(args.path, args.checkpoint, 'cp-{epoch:04d}.ckpt')
    
    cp_callback = ModelCheckpoint(filepath   = checkpoint,
                                 save_weights_only = True,
                                 save_freq         = 'epoch',
                                 verbose           = 1)
    
    if args.action == 'train':
        (train_x, train_y), (test_x, test_y), (val_x,val_y) = create_data()
        
        model = create_model(train_x)
        
        model.save_weights(checkpoint.format(epoch=0))
        
        model.fit(train_x, train_y,
                  epochs          = args.epochs,
                  validation_data = (val_x, val_y),
                  callbacks       = [cp_callback])
  
    
    if args.action == 'test':
        latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint))
        print (f'Latest checkpoint: {latest}')
        _, (test_x, test_y), _ = create_data()

        model              = create_model(test_x)
       
        model.load_weights(latest) 
        print (f'Loaded {latest}')
        
        loss,accuracy = model.evaluate(test_x, test_y, verbose=2)
        print (f'Tested: loss={loss}, accuracy={accuracy}')
       
    if args.action=='continue':
        latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint))
        print (f'Latest checkpoint: {latest}')        
        (train_x, train_y), (test_x, test_y), (val_x,val_y) = create_data()
        
        model              = create_model(train_x)
    
        model.load_weights(latest) 
        print (f'Loaded {latest}')
 
        model.fit(train_x, train_y,
                  epochs=args.epochs,
                  validation_data=(val_x, val_y),
                  callbacks=[cp_callback])        
   