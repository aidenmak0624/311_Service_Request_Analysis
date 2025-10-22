import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data in chunks
def load_data(file_path, chunk_size):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    data = pd.concat(chunks, ignore_index=True)
    return data

# Filter data for specific conditions
def filter_data(df):
    #df = df[(df['Subject'] == 'Service Request')]
    df = df[df['Type'] == 'Bus Schedule']
    df['Open Date'] = pd.to_datetime(df['Open Date'], errors='coerce', format="%m/%d/%Y %I:%M:%S %p")
    df = df.dropna(subset=['Open Date'])
    return df

# Group by date to get daily case counts
def aggregate_daily_cases(df):
    daily_cases = df.groupby(df['Open Date'].dt.date).size().to_frame(name='case_count').reset_index()
    daily_cases['date'] = pd.to_datetime(daily_cases['Open Date'])
    daily_cases.set_index('date', inplace=True)
    return daily_cases

# Detect anomalies based on residuals
def detect_anomalies(residuals):
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    threshold_upper = mean_residual + 3 * std_residual
    threshold_lower = mean_residual - 3 * std_residual
    anomalies = residuals[(residuals > threshold_upper) | (residuals < threshold_lower)]
    return anomalies

# Plot results
def plot_results(daily_cases, decomposition, anomalies):
    # Seasonal decomposition plot
    decomposition.plot()
    plt.show()

    # Anomalies plot
    plt.figure(figsize=(10, 6))
    plt.plot(daily_cases.index, daily_cases['case_count'], label='Case Count')
    plt.scatter(anomalies.index, daily_cases.loc[anomalies.index, 'case_count'], color='red', label='Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Case Count')
    plt.title('Case Count (Bus Schedule) with Anomalies Detected')
    plt.legend()
    plt.show()

    # Observed vs Reconstructed
    estimate = decomposition.trend + decomposition.seasonal
    plt.figure(figsize=(12, 4))
    plt.plot(daily_cases['case_count'], label='Observed')
    plt.plot(estimate, label='Reconstructed', linestyle='--')
    plt.legend()
    plt.show()

# Main analysis
file_path = '/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv'
chunk_size = 200000
df = load_data(file_path, chunk_size)
df = filter_data(df)
daily_cases = aggregate_daily_cases(df)

# Perform seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(daily_cases['case_count'], model='additive', period=365)
residuals = decomposition.resid.dropna()

# Detect and save anomalies
anomalies = detect_anomalies(residuals)
anomalies_data = daily_cases[daily_cases.index.isin(anomalies.index)]
anomalies_data.to_csv('service_request_outliers_bus_schedule.csv', index=True)

# Plot results
plot_results(daily_cases, decomposition, anomalies)
