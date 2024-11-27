# utils/data_preprocessing.py

import pandas as pd
import numpy as np
from scipy.stats import zscore
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Assuming data['metrics'] contains the actual data you're interested in
    df = pd.DataFrame(data['metrics'])
    
    # Check the DataFrame structure
    print(df.head())
    return df
def clean_data(df):
    """Handle missing data, outliers, and other preprocessing tasks."""
    # Fill missing values in sleep_hours with mean
    df['sleep_hours'].fillna(df['sleep_hours'].mean(), inplace=True)

    # Remove outliers using Z-score (threshold = 3)
    df['z_score'] = zscore(df['sleep_hours'])
    df = df[df['z_score'].abs() <= 3]
    
    # Return cleaned DataFrame
    return df

def save_cleaned_data(df, output_file):
    """Save the cleaned DataFrame to a CSV file."""
    df.to_csv(output_file, index=False)
