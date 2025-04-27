import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Implementing a simple Neural Network Regression model for the Subway Prediction Data using TensorFlow/Keras.
# Andrew Chung, hc893, 4/27/2025

# import data
data = pd.read_csv("subwaydata.csv").iloc[:, 1:]
X = data.iloc[:, 3:]
y = data['ridership'].to_numpy()/1000 # 1K scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 893)
y_train, y_test = np.log(y_train), np.log(y_test)

# standard scaling before NN implementation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialize neural net -- 24->12->1 architecture
net = keras.Sequential([
  layers.Dense(24, activation = 'relu', input_shape = (X_train.shape[1], )),
  layers.Dense(12, activation = 'relu'), layers.Dense(1)
])

# compile net (Adam Optimizer)
net.compile(
  optimizer = keras.optimizers.Adam(learning_rate = 0.001),
  loss = 'mse', metrics = ['mse']
)

# train neural net
history = net.fit(
  X_train, y_train,
  validation_split = 0.2, 
  epochs = 300, batch_size = 16,
  # define early stop (monitor for Loss function)
  callbacks = [keras.callbacks.EarlyStopping(
    monitor = 'val_loss', patience = 10, restore_best_weights = True
  )],
  verbose = 0
)

# Predict
y_pred = net.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: {}".format(mse))
print("R²: {}".format(r2))
