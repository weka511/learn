# snarfed from https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf
import matplotlib.pyplot as plt

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flattens the input. Does not affect the batch size.
  tf.keras.layers.Dense(128, activation='relu'), # Just your regular densely-connected NN layer.
  tf.keras.layers.Dropout(0.2),                  # The Dropout layer randomly sets input units to 0 with a
                                                 #frequency of rate at each step during training time,
                                                 # which helps prevent overfitting. Inputs not set to 0 
                                                 # are scaled up by 1/(1 - rate) such that the sum over
                                                 # all inputs is unchanged.
  tf.keras.layers.Dense(10)
])

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(x_train[:1]).numpy()
# The tf.nn.softmax function converts these logits to "probabilities" for each class: 
tf.nn.softmax(predictions).numpy()
# The losses.SparseCategoricalCrossentropy loss takes a vector of logits 
# and a True index and returns a scalar loss for each example. This loss is equal to the negative
# log probability of the true class: It is zero if the model is sure of the correct class.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print (loss_fn(y_train[:1], predictions).numpy())

# We specify the training configuration (optimizer, loss, metrics):
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# T rain the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating
# over the entire dataset for a given number of "epochs".
model.fit(x_train, y_train, epochs=5)

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
