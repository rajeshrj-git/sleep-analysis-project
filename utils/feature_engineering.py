import pandas as pd

# Feature engineering to calculate rolling averages and trends
def feature_engineering(df):
    df['sleep_7_day_avg'] = df['sleep_hours'].rolling(window=7).mean()
    df['steps_7_day_avg'] = df['steps'].rolling(window=7).mean()
    df['hrv_7_day_avg'] = df['hrv'].rolling(window=7).mean()
    df['sleep_trend'] = df['sleep_hours'].diff()  # The difference in sleep hours from the previous day
    df = df.dropna()  # Remove rows with NaN values from rolling averages
    return df

# Analyze sleep trend
def analyze_sleep_trend(df):
    if df['sleep_trend'].mean() > 0:
        return "Your sleep hours are improving over time. Keep up the good work!"
    elif df['sleep_trend'].mean() < 0:
        return "Your sleep hours are declining over time. Consider improving your sleep habits."
    else:
        return "Your sleep hours are stable. Maintain your current sleep routine."
