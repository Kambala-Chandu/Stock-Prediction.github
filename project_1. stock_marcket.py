# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate or import historical stock price data
# For this example, let's create some random data
np.random.seed(42)
num_data_points = 100
stock_price_data = np.random.randint(100, 200, size=num_data_points).astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
stock_price_data = scaler.fit_transform(stock_price_data.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
X_train, y_train = create_dataset(stock_price_data, look_back)

# Reshape the data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)

# Plot the original and predicted stock prices
plt.plot(scaler.inverse_transform(stock_price_data), label="Original Stock Price")
plt.plot(range(look_back, num_data_points), train_predict, label="Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction using LSTM")
plt.legend()
plt.show()
