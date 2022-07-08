import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from datetime import datetime
def print_current_time(start_time=None):
    now = datetime.now()
    print('Time: ' + now.strftime("%H:%M:%S"))

    if start_time != None:
        diff_time = now - start_time
        print('Elapsed Time: ' + str(diff_time))

    return now

# Print start time
s_time = print_current_time()

print(f"tensor flow version : {tf.__version__}")
print(f"keras version : {keras.__version__}")

# Load mnist dataset
fasion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fasion_mnist.load_data()

# Split the full train data into train and validation
# Also, apply a simple scaling to input X, to have values between 0.0 and 1.0
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Build a model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Plot the train history
#pd.DataFrame(history.history).plot(figsize=(8, 5))
#plt.grid(True)
#plt.gca().set_ylim(0, 1)
#plt.show()

# Evaluate the model
model.evaluate(X_test, y_test)

# Generate predicts
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=1)
print(y_pred)

# Print the very end time
e_time = print_current_time(s_time)
