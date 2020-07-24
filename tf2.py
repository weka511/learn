# snarfed from https://www.tensorflow.org/tutorials/quickstart/beginner
# and https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
# Final vesion borrws from https://www.tensorflow.org/tutorials/images/cnn
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers            import Convolution2D, MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax
from tensorflow.nn           import relu
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test                      = x_train / 255.0, x_test / 255.0
x_train                              = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test                               = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape                          = (28, 28, 1)
# Build model
model = Sequential([
  Convolution2D(32, 3, 3, activation=relu, input_shape=input_shape,padding='same'),
  MaxPooling2D(pool_size=(2,2)),
  Convolution2D(64, (3, 3), activation=relu,padding='same'),
  MaxPooling2D(pool_size=(2,2)),
  Convolution2D(64, (3, 3), activation=relu,padding='same'),
  Flatten(), 
  Dense(64, activation=relu), 
  Dropout(0.25),  
  Dense(10)
])
model.summary()


model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating
# over the entire dataset for a given number of "epochs".
history = model.fit(x_train, y_train, 
                    epochs=25, 
                    validation_data=(x_test,y_test))

#  check the model's performance

probability_model = Sequential([ model,  Softmax()])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([min(min(history.history['accuracy']),
              min(history.history['val_accuracy'])), 
          max(max(history.history['accuracy']),
              max(history.history['val_accuracy']))          ])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print (f'Accuracy={test_acc}')   
    
plt.show()
