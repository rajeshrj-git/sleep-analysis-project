from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data_for_lstm(df, features, target):
    X = df[features].values
    y = df[target].values
    # Fit scaler only on the target variable (health condition)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))  # Scaling only target variable
    X_scaled = X  # X remains unchanged for now
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for LSTM
    return X_scaled, y_scaled, scaler


# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_lstm_model(X, y, model):
    model.fit(X, y, epochs=50, batch_size=32,verbose = 0)

def predict_health_condition(model, X, scaler):
    # Predict health condition for each time step
    predicted_health_condition = model.predict(X)

    # Reshape predicted values to be in the correct shape for inverse transformation
    predicted_health_condition_reshaped = predicted_health_condition.reshape(-1, 1)

    # Inverse transform the predicted health condition (target variable)
    predicted_health_condition_original = scaler.inverse_transform(predicted_health_condition_reshaped)

    return predicted_health_condition_original