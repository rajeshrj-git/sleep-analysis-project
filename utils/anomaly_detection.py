# utils/anomaly_detection.py

from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    """Detect anomalies in sleep hours using Isolation Forest."""
    clf = IsolationForest(contamination=0.1)
    df['anomaly'] = clf.fit_predict(df[['sleep_hours']])
    anomalies = df[df['anomaly'] == -1]
    return anomalies
