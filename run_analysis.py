import os
import sys
import warnings
import numpy as np
import pandas as pd
from utils.data_preprocessing import load_data, save_cleaned_data
from utils.feature_engineering import feature_engineering, analyze_sleep_trend
from utils.recommendation import generate_general_recommendation
from models.lstm_model import prepare_data_for_lstm, build_lstm_model, train_lstm_model, predict_health_condition


def run_analysis():
    # Load and preprocess data
    df = load_data('data/raw_data.json')
    df = feature_engineering(df)
    
    # If health_condition doesn't exist, create a basic health condition column
    if 'health_condition' not in df.columns:
        df['health_condition'] = df['sleep_hours'] * 0.5 + df['hrv'] * 0.3 

    # Calculate Average Sleep Duration (Last 7 Days)
    avg_sleep_duration = df['sleep_hours'].tail(7).mean()
    print(f"Average Sleep Duration (Last 7 Days): {avg_sleep_duration:.2f} hours")
    
    # Generate General Recommendation
    general_recommendation = generate_general_recommendation(avg_sleep_duration)
    print(f"General Recommendation: {general_recommendation}")
    
    # Analyze Sleep Trend
    sleep_trend_analysis = analyze_sleep_trend(df)
    print(f"Sleep Trend Analysis: {sleep_trend_analysis}")
    
    # Select the features and target for LSTM model
    features = ['sleep_hours', 'steps', 'hrv', 'sleep_7_day_avg', 'steps_7_day_avg', 'hrv_7_day_avg', 'sleep_trend']
    target = 'health_condition'  # Assuming this is the target we want to predict (e.g., health condition in percentage)
    
    # Prepare the data for LSTM model
    X, y, scaler = prepare_data_for_lstm(df, features, target)

    # Build and train the LSTM model
    lstm_model = build_lstm_model(input_shape=(X.shape[1], 1))
    train_lstm_model(X, y, lstm_model)

    # Predict health condition after 3 months
    predicted_health_condition = predict_health_condition(lstm_model, X, scaler)

    # Get the last predicted health condition value (after 3 months)
    predicted_health_condition_after_3_months = predicted_health_condition[-1][0]  # Assuming 1D target

    # Ensure that improvement is a scalar value
    current_health_condition = y[-1]  # Current health condition from the data
    improvement = predicted_health_condition_after_3_months - current_health_condition
    if isinstance(improvement, np.ndarray):
        improvement = improvement.item()  # Convert to scalar if it's a numpy array

    # Calculate and print improvement or decline
    if predicted_health_condition_after_3_months > current_health_condition:
        print(f"If you maintain this sleep, your health will improve by {improvement:.2f}% over the next 3 months.")
    elif predicted_health_condition_after_3_months < current_health_condition:
        decline = current_health_condition - predicted_health_condition_after_3_months
        if isinstance(decline, np.ndarray):
            decline = decline.item()  # Convert to scalar if it's a numpy array
        print(f"If you maintain this sleep, your health will decline by {decline:.2f}% over the next 3 months.")
    else:
        print("Your health is expected to remain the same over the next 3 months based on your current sleep.")


if __name__ == "__main__":
    run_analysis()






