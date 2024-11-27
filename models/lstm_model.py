# models/lstm_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def preprocess_lstm_data(df, look_back=5):
    """Preprocess sleep data for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['sleep_hours']])

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM input format
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build the LSTM model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X, y, epochs=20, batch_size=16):
    """Train the LSTM model."""
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

def forecast_lstm(model, X, scaler):
    """Forecast sleep hours with the trained LSTM model."""
    predicted_sleep = model.predict(X[-1].reshape(1, X.shape[1], 1))
    predicted_sleep = scaler.inverse_transform(predicted_sleep)
    return predicted_sleep[0][0]
