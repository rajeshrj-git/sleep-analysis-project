# run_analysis.py
import pandas as pd
from utils.data_preprocessing import load_data, clean_data, save_cleaned_data
from models.arima_model import train_arima_model, forecast_arima, plot_arima_forecast
from models.lstm_model import preprocess_lstm_data, build_lstm_model, train_lstm_model, forecast_lstm
from utils.anomaly_detection import detect_anomalies
from utils.recommendation import generate_recommendation

def run_analysis():
    # Load and clean the data
    df = load_data('data/raw_data.json')
    df = clean_data(df)
    save_cleaned_data(df, 'data/cleaned_data.csv')

    # Train ARIMA model and forecast sleep
    arima_model = train_arima_model(df)
    arima_forecast = forecast_arima(arima_model, steps=5)

    # Plot the ARIMA forecast
    steps = 5  # Define the number of forecasted steps
    plot_arima_forecast(df, arima_forecast, steps)  # Pass steps here

    # Prepare LSTM data and train the model
    X, y, scaler = preprocess_lstm_data(df)
    lstm_model = build_lstm_model(input_shape=(X.shape[1], 1))
    train_lstm_model(lstm_model, X, y)
    lstm_forecast = forecast_lstm(lstm_model, X, scaler)

    # Detect anomalies in the sleep data
    anomalies = detect_anomalies(df)
    print(f"Detected anomalies: {anomalies}")

    # Generate recommendations
    recommendation = generate_recommendation(df['sleep_hours'])
    print(f"Recommendation: {recommendation}")
