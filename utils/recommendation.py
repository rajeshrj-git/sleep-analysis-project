# utils/recommendation.py

import numpy as np

def generate_recommendation(sleep_data):
    """Generate a sleep recommendation based on the analysis."""
    avg_sleep = np.mean(sleep_data)
    if avg_sleep < 6.5:
        return "Your average sleep time is below 6.5 hours. Consider a consistent bedtime routine for better health."
    elif avg_sleep >= 7.5:
        return "Great! You're getting enough sleep. Keep maintaining your current habits."
    else:
        return "Your sleep is within a healthy range. Make sure to monitor it regularly."
