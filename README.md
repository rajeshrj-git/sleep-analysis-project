# Sleep Analysis Project

This project performs sleep analysis based on user data, including metrics like steps, heart rate, sleep hours, and heart rate variability (HRV). It uses ARIMA and LSTM models to forecast sleep trends and detect anomalies. Additionally, the project provides personalized recommendations based on the analysis.

## Features

- **Data Preprocessing**: Cleans and prepares the raw sleep data.
- **ARIMA Model**: Trains an ARIMA model to forecast sleep hours for the upcoming days.
- **LSTM Model**: Builds and trains an LSTM model for time series prediction of sleep data.
- **Anomaly Detection**: Detects anomalies in the sleep data using z-scores.
- **Recommendations**: Provides personalized recommendations based on the user’s sleep data.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.7 or higher
- Pip (Python package installer)

## Installation

### 1. Clone the Repository
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/sleep-analysis-project.git



### Run the Application
cd sleep-analysis-project
pip install -r requirements.txt
python api/app.py

