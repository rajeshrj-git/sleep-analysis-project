import pandas as pd

# Function to analyze the sleep trend over the last few days
def analyze_sleep_trend(df):
    """
    Analyzes the trend of sleep hours in the last 7 days and provides a recommendation based on that trend.
    Returns a string message indicating the trend and advice.
    """
    # Calculate the trend using the difference in sleep hours (i.e., how sleep hours have changed day-to-day)
    sleep_trend = df['sleep_trend'].tail(7)  # Use the last 7 days of sleep trend
    
    # Check if the trend is positive, negative, or stable
    if sleep_trend.mean() > 0:
        return "Your sleep hours are improving over time. Keep up the good work!"
    elif sleep_trend.mean() < 0:
        return "Your sleep hours are declining over time. Consider improving your sleep habits."
    else:
        return "Your sleep hours are stable. Maintain your current sleep routine."

# Optional: Function to visualize sleep trend (can be expanded if needed)
def plot_sleep_trend(df):
    """
    Visualizes the sleep trend over time (for the last 30 days).
    """
    import matplotlib.pyplot as plt
    
    # Plot sleep hours for the last 30 days
    df['sleep_hours'].tail(30).plot(kind='line', title='Sleep Hours Trend (Last 30 Days)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Sleep Hours')
    plt.show()
