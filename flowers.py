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
#
# This program is a template for construction other Tensorflow scripts.

# https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

# https://www.tensorflow.org/tutorials/keras/save_and_load

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir    = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir    = pathlib.Path(data_dir)

batch_size  = 32
img_height  = 180
img_width   = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create_model
#
# Create Neuaral Network model and compile it
#
# Parameters:
#     train_x    Used to assign size to first layer

def create_model(num_classes = 5):
    product = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),
      Conv2D(32, 3, activation='relu'),
      MaxPooling2D(),
      Conv2D(32, 3, activation='relu'),
      MaxPooling2D(),
      Conv2D(32, 3, activation='relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(num_classes)
    ])    

    product.compile(optimizer='adam',
                   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return product

#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
    #for i in range(9):
        #ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(images[i].numpy().astype("uint8"))
        #plt.title(class_names[labels[i]])
        #plt.axis("off")

    #for image_batch, labels_batch in train_ds:
        #print(image_batch.shape)
        #print(labels_batch.shape)
        #break
#plt.show()

# scheduler
#
# Controls learning rate
#
# snarfed from https://medium.com/ydata-ai/how-to-use-tensorflow-callbacks-f54f9bb6db25

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))
    
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
    
    csv_logger_callback  = CSVLogger('training.txt')
    
    learning_rate_callback = LearningRateScheduler(scheduler)
    
    if args.action == 'train':
        #(train_x, train_y), (test_x, test_y), (val_x,val_y) = create_data()
        
        model = create_model()#train_x)
        
        #model.save_weights(checkpoint.format(epoch=0))
        history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=args.epochs,
          callbacks       = [cp_callback,csv_logger_callback,learning_rate_callback]          
        )        
        #history = model.fit(train_x, train_y,
                            #epochs          = args.epochs,
                            #validation_data = (val_x, val_y),
                            #callbacks       = [cp_callback,csv_logger_callback,learning_rate_callback])

        #print(f'loss={history.history["loss"]},accuracy={history.history["sparse_categorical_accuracy"]}')
        #print (f'validation loss={history.history["val_loss"]},accuracy={history.history["val_sparse_categorical_accuracy"]}')
        # loss, sparse_categorical_accuracy, val_loss, val_sparse_categorical_accuracy

  
    
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
        csv_logger_callback  = CSVLogger('training.txt',append=True)
        model.fit(train_x, train_y,
                  epochs=args.epochs,
                  validation_data=(val_x, val_y),
                  callbacks=[cp_callback,csv_logger_callback])        
   