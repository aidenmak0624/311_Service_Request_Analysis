import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset
#df = pd.read_csv("sample_1000_records.csv")
chunk_size = 200000  # Adjust chunk size based on available memory
chunks = pd.read_csv('/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv', chunksize=chunk_size)
data_chunks = []
for chunk in chunks:
    data_chunks.append(chunk)
df = pd.concat(data_chunks, ignore_index=True)

#Data filtration
df = df[df['Subject'] == 'Service Request']
df['Open Date'] = pd.to_datetime(df['Open Date'], errors='coerce',format="%m/%d/%Y %I:%M:%S %p")
df = df.dropna(subset=['Open Date'])

# Group by date to get daily case counts
#df['date'] = df['Open Date'].dt.date
#daily_cases = df.groupby('date').size().to_frame(name='case_count').reset_index()
daily_cases = df.groupby(df['Open Date'].dt.date).size().to_frame(name='case_count').reset_index()
# Ensure date is in datetime format for decomposition
daily_cases['date'] = pd.to_datetime(daily_cases['Open Date'])
daily_cases.set_index('date', inplace=True)

# Perform seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(daily_cases['case_count'], model='additive', period=365)  # Weekly seasonality

# Plot the decomposition
decomposition.plot()
plt.show()

# Detect anomalies based on residuals
residuals = decomposition.resid.dropna()  # Drop NaN values in residuals
#anomaly_threshold = residual.std() * 3  # Define anomaly as 3 standard deviations from the mean
#anomalies = residual[abs(residual) > anomaly_threshold]
#lower = decomposition.resid.mean()-anomaly_threshold
#higher = decomposition.resid.mean()+anomaly_threshold
mean_residual = residuals.mean()
std_residual = residuals.std()
threshold_upper = mean_residual + 3 * std_residual
threshold_lower = mean_residual - 3 * std_residual
anomalies = residuals[(residuals > threshold_upper) | (residuals < threshold_lower)]

# Plot the results with anomalies highlighted
plt.figure(figsize=(10, 6))
plt.plot(daily_cases.index, daily_cases['case_count'], label='Case Count')
plt.scatter(anomalies.index, daily_cases.loc[anomalies.index, 'case_count'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Case Count')
plt.title('Case Count with Anomalies Detected via Seasonal Decomposition')
plt.legend()
plt.show()


estimate = decomposition.trend+decomposition.seasonal
plt.figure(figsize=(12,4))
plt.plot(daily_cases['case_count'])
plt.plot(estimate)
plt.show()


# Filter anomalies and save to CSV
#anomalies_data = df.loc[anomalies.index]
#anomalies_data.to_csv('service_request_outliers.csv', index=True)