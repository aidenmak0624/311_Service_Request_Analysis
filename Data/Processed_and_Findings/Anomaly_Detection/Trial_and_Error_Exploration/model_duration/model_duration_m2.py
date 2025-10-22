import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Input Data
chunk_size = 50000  # Adjust chunk size based on available memory
chunks = pd.read_csv('/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv', chunksize=chunk_size)
data_chunks = []
for chunk in chunks:
    data_chunks.append(chunk)
df = pd.concat(data_chunks, ignore_index=True)

# Convert dates and calculate case duration
df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Closed Date'] = pd.to_datetime(df['Closed Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['Open Date', 'Closed Date'])
df['Case Duration (hours)'] = (df['Closed Date'] - df['Open Date']).dt.total_seconds() / 3600
df = df[df['Case Duration (hours)'] >= 0]

# Find unique types and initialize anomaly tracking
unique_types = df['Type'].unique()
anomalies_summary = {}
anomaly_counts = {}

for unique_type in unique_types:
    df_type = df[df['Type'] == unique_type]

    # Ensure type-specific durations are sufficient for analysis
    if len(df_type) < 365:  # Decomposition needs a full year's data
        anomaly_counts[unique_type] = 0
        continue

    # Sort by 'Open Date' to ensure proper time series
    df_type = df_type.sort_values(by='Open Date')

    try:
        # Perform seasonal decomposition
        decomposition = sm.tsa.seasonal_decompose(
            df_type['Case Duration (hours)'], model='additive', period=365
        )

        # Detect anomalies based on residuals
        residuals = decomposition.resid.dropna()
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        threshold_upper = mean_residual + 3 * std_residual
        threshold_lower = mean_residual - 3 * std_residual
        anomalies = residuals[(residuals > threshold_upper) | (residuals < threshold_lower)]

        # Store anomalies for this type
        anomalies_summary[unique_type] = anomalies
        anomaly_counts[unique_type] = len(anomalies)

    except ValueError as e:
        anomaly_counts[unique_type] = 0

# Convert anomaly counts to a DataFrame for visualization
anomaly_counts_df = pd.DataFrame(list(anomaly_counts.items()), columns=['Type', 'Anomaly Count'])
anomaly_counts_df = anomaly_counts_df.sort_values(by='Anomaly Count', ascending=False)

# Plot anomaly counts
anomaly_counts_df.plot(kind='bar', x='Type', y='Anomaly Count', title='Anomaly Counts by Type')
plt.figure(figsize=(30, 10))
plt.xlabel('Type')
plt.ylabel('Anomaly Count')
plt.show()
