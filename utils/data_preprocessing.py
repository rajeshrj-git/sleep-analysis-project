import pandas as pd
import json
from pandas import json_normalize

# Load data from JSON
# def load_data(file_path):
#     return pd.read_json(file_path)
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Flatten the 'metrics' part of the JSON data
    df = json_normalize(data['metrics'])
    
    # Optionally, return the full DataFrame with the 'user_id' included if needed
    df['user_id'] = data['user_id']
    
    return df
# Save cleaned data
def save_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)
