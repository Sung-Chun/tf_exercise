import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# Standardize the input data by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Build a model
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(loss="mean_squared_error", optimizer="sgd")

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Plot the train history
#pd.DataFrame(history.history).plot(figsize=(8, 5))
#plt.grid(True)
#plt.gca().set_ylim(0, 1)
#plt.show()

# Evaluate the model
mse_test = model.evaluate(X_test, y_test)

# Generate predicts
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)

# Print end time
e_time = print_current_time(s_time)
