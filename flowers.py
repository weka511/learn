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
# This program is a template for constructing other Tensorflow scripts.

# https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

# https://www.tensorflow.org/tutorials/keras/save_and_load

# https://www.tensorflow.org/tutorials/load_data/images

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,regularizers
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense,MaxPooling2D,Dropout,SpatialDropout2D,Softmax
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
import tensorflow_datasets as tfds
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import argparse
import sys
import matplotlib.pyplot as plt

def create_data(dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                fname       = 'flower_photos',
                batch_size  = 32,
                img_height  = 180,
                img_width   = 180):
    
    data_dir  = pathlib.Path(get_file(origin=dataset_url,
                                      fname=fname,
                                      untar=True))

    train_ds = image_dataset_from_directory(data_dir,
        validation_split = 0.2,
        subset           = 'training',
        seed             = 123,
        image_size       = (img_height, img_width),
        batch_size       = batch_size)
    
    val_ds = image_dataset_from_directory(data_dir,
        validation_split = 0.2,
        subset           = "validation",
        seed             = 123,
        image_size       = (img_height, img_width),
        batch_size       = batch_size)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    return (train_ds.cache().prefetch(buffer_size=AUTOTUNE),
            val_ds.cache().prefetch(buffer_size=AUTOTUNE),
            train_ds.class_names)

# create_model
#
# Create Neural Network model and compile it
#
# Parameters:
#     train_x    Used to assign size to first layer

def create_model(num_classes = 5):
    product = Sequential([
      Rescaling(1./255),
      Conv2D(32, 3, activation='relu'),
      MaxPooling2D(),
      Conv2D(32, 3, activation='relu'),
      MaxPooling2D(),
      Conv2D(32, 3, activation='relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.0001)),
      #Dropout(0.2),
      Dense(num_classes)
    ])    

    product.compile(optimizer = 'adam',
                   loss       = SparseCategoricalCrossentropy(from_logits=True),
                   metrics    = [SparseCategoricalAccuracy()])

    return product


# scheduler
#
# Controls learning rate
#
# snarfed from https://medium.com/ydata-ai/how-to-use-tensorflow-callbacks-f54f9bb6db25

def scheduler(epoch,mult=0.001,steps=10,decay=0.1):
    return mult if epoch < steps else mult * tf.math.exp(decay * (steps - epoch))


def parse_args():
    parser   = argparse.ArgumentParser('Trining Convolutional Neural Net')
    parser.add_argument('action',   
                        choices = ['train',
                                   'test', 
                                   'continue',
                                   'show'],
                        help    = 'Train or test')
    parser.add_argument('--checkpoint',
                        default = 'checkpoint',
                        help    = 'Name of file to save and restore network')
    parser.add_argument('--path',
                        default = path,
                        help    = 'Path of file used to save and restore network')
    parser.add_argument('--epochs',
                        type    = int,
                        default = 5, 
                        help    = 'Number of epochs for training')
    parser.add_argument('--logfile',
                        default = basename,
                        help    = 'Name of logfile')
    parser.add_argument('--plotfiles',
                        default = basename,
                        help    = 'Name of Plots')    
    parser.add_argument('--images', 
                        default = './figs', 
                        help    = 'Path for storing images')
    parser.add_argument('--show',
                        action  = 'store_true', 
                        default = False,
                        help    = 'Display plots')
    return parser.parse_args() 
    
def predictions(model,val_ds):
    probability_model = Sequential([model,Softmax()])
    for images_batch,labels_batch in val_ds:
        for probabilities,image,label in zip(probability_model.predict(images_batch),images_batch,labels_batch):
            choice = np.argmax(probabilities)
            yield choice,image,(int)(label),probabilities
            
if __name__=='__main__':
    from matplotlib import rc
    rc('text', usetex=True)
    print(f'Using Tensorflow {tf.version.VERSION}')
    basename               = os.path.basename(__file__).split('.')[0]
    path                   = os.path.join(os.getenv('APPDATA'),basename)
    args                   = parse_args() 
    
    checkpoint             = os.path.join(args.path, args.checkpoint, 'cp-{epoch:04d}.ckpt')
    checkpoint_callback    = ModelCheckpoint(filepath       = checkpoint,
                                          save_weights_only = True,
                                          save_freq         = 'epoch',
                                          verbose           = 1)
    
    logfile                = args.logfile if len (args.logfile.split('.'))>1 else args.logfile + '.txt'
    
    csv_logger_callback    = CSVLogger(logfile)
    
    learning_rate_callback = LearningRateScheduler(scheduler)
    
    if args.action=='show':
        train_ds,_,class_names = create_data()
        n                      = 4
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(n*n):
                ax = plt.subplot(n, n, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        
        plt.show()
        sys.exit()
        
    if args.action == 'train':
        train_ds,val_ds,_ = create_data()
        
        model             = create_model()
        
        model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = args.epochs,
            callbacks       = [checkpoint_callback,
                               csv_logger_callback,
                               learning_rate_callback]          
        )        
  
        sys.exit()
        
    if args.action == 'test':
        latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint))
        print (f'Latest checkpoint: {latest}')
        _,val_ds,class_names = create_data()

        model                = create_model()
       
        model.load_weights(latest) 
        print (f'Loaded {latest}')
        
        n_rows       = 4
        n_columns    = 4
        image_number = 0
        n_trials     = 0
        n_mismatches = 0
        n_figures    = 0
        
        for predicted,image,expected,probabilities in predictions(model,val_ds):
            n_trials += 1
            if predicted == expected: continue
            n_mismatches += 1
            if image_number==0:
                plt.figure(figsize=(15,10))
            image_number += 1
            
            plt.subplot(n_rows,2*n_columns,2*image_number-1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])       
            plt.imshow(image/255.0, cmap=plt.cm.binary)
            plt.xlabel(rf'{class_names[expected]}$\ne${class_names[predicted]}')
            
            plt.subplot(n_rows,2*n_columns,2*image_number)
            plt.grid(False)
            plt.xticks(range(len(probabilities)))
            plt.yticks([])            
            barplot = plt.bar(range(len(probabilities)),probabilities)
            plt.ylim([0, 1])
            barplot[predicted].set_color('red')
            barplot[expected].set_color('blue')
            
            if image_number==n_rows*n_columns:
                plt.savefig(os.path.join(args.images,f'{args.plotfiles}{n_figures:04d}'))
                if not args.show:
                    plt.close()
                n_figures += 1
                image_number = 0
                
        if image_number>0:
            plt.savefig(os.path.join(args.images,f'{args.plotfiles}{n_figures:04d}'))
        print (f'{n_mismatches} mismatches in {n_trials} trials')        
        if args.show:
            plt.show()
            
        sys.exit()
        
    if args.action=='continue':
        latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint))
        print (f'Latest checkpoint: {latest}') 
        train_ds,val_ds,_ = create_data()

        
        model              = create_model()
    
        model.load_weights(latest) 
        print (f'Loaded {latest}')
        csv_logger_callback  = CSVLogger(logfile,append=True)
        model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = args.epochs,
            callbacks       = [checkpoint_callback,
                               csv_logger_callback,
                               learning_rate_callback]   )        
 
        sys.exit()
   