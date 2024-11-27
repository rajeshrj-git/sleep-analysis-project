# Generate general recommendation based on average sleep duration
def generate_general_recommendation(avg_sleep_duration):
    if avg_sleep_duration >= 7:
        return "Your average sleep duration is within a healthy range. Maintain consistency for optimal health."
    else:
        return "Consider increasing your sleep duration for optimal health."
