# snarfed from https://www.tensorflow.org/tutorials/quickstart/beginner
# and https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print (x_train.shape) # (60000, 28, 28)
print (y_train.shape) #(60000, )
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print (x_train.shape) # (60000, 28, 28)
print (y_train.shape) #(60000, )

input_shape = (28, 28, 1)
# Build model
model = Sequential([
  Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape,padding='same'),
  MaxPooling2D(pool_size=(2,2)),
  Convolution2D(64, (3, 3), activation='relu',padding='same'),
  MaxPooling2D(pool_size=(2,2)),
  Convolution2D(64, (3, 3), activation='relu',padding='same'),
  Flatten(), # Flattens the input. Does not affect the batch size.
  Dense(64, activation=tf.nn.relu), # Just your regular densely-connected NN layer.
  #Dropout(0.2),                  # The Dropout layer randomly sets input units to 0 with a
                                                 ##frequency of rate at each step during training time,
                                                 ## which helps prevent overfitting. Inputs not set to 0 
                                                 ## are scaled up by 1/(1 - rate) such that the sum over
                                                 ## all inputs is unchanged.
 Dense(10)
])
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating
# over the entire dataset for a given number of "epochs".
model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test,y_test))

#  check the models performance

model.evaluate(x_test,  y_test, verbose=2)

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print (probability_model(x_test[:5]))
for i in range(5):
    plt.figure()
    plt.imshow(x_test[i])
plt.show()
