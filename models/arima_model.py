import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def train_arima_model(df, order=(1, 1, 1)):
    """Train an ARIMA model on the sleep data."""
    
    # Ensure 'date' column is a datetime type and set it as the index
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)
    
    # Ensure there is enough data for ARIMA (generally at least 10-20 data points)
    if len(df) < 5:
        raise ValueError("Not enough data points for ARIMA. Minimum 5 data points required.")
    
    # Fit ARIMA model
    model = ARIMA(df['sleep_hours'], order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=5):
    """Forecast sleep hours for the next 'steps' days."""
    forecast = model_fit.forecast(steps=steps)
    return forecast

import matplotlib.pyplot as plt
import pandas as pd

def plot_arima_forecast(df, forecast, steps):
    """Plot the actual sleep data and ARIMA forecast."""
    plt.plot(df.index, df['sleep_hours'], label="Observed Sleep Hours")
    
    # Generate forecasted dates based on the length of the forecast
    forecast_dates = pd.date_range(df.index[-1], periods=steps + 1, freq='D')[1:]
    
    # Plot the forecasted data
    plt.plot(forecast_dates, forecast, label="Forecasted Sleep Hours", linestyle='--')
    
    plt.title("Sleep Duration with ARIMA Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Sleep Hours")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
