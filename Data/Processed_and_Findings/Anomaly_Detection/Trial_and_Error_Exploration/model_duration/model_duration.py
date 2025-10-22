import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


#Input Data
chunk_size = 50000  # Adjust chunk size based on available memory
chunks = pd.read_csv('/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv', chunksize=chunk_size)
data_chunks = []
for chunk in chunks:
    data_chunks.append(chunk)
df = pd.concat(data_chunks, ignore_index=True)




#Find uniques by Type
unique_types = df['Type'].unique()
print(unique_types)

#Find case_count by Type
type_counts = df.groupby('Type').size().reset_index(name='count').sort_values(by='count', ascending=False)
print(type_counts)

# Bar chart of types
type_counts.plot(kind='bar', x='Type', y='count', title='Count by Type')
plt.xlabel('Reason')
plt.ylabel('Count')
plt.show()




df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Closed Date'] = pd.to_datetime(df['Closed Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['Open Date', 'Closed Date'])
df['Case Duration (hours)'] = (df['Closed Date'] - df['Open Date']).dt.total_seconds() / 3600
df = df[df['Case Duration (hours)'] >= 0]
durations = df['Case Duration (hours)'].dropna()

#Find anomalies among different Types
anomalies_summary = {}
anomaly_counts = {}

for unique_type in unique_types:
    df_type = df[df['Type']== unique_type]





    try:
        # Perform seasonal decomposition
        decomposition = sm.tsa.seasonal_decompose(
            durations, model='additive', period=365
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

        #print(f"Anomalies for {unique_type}:")
        #print(anomalies)

    except ValueError as e:
        #print(f"Decomposition failed for {unique_type}: {e}")
        anomaly_counts[unique_type] = 0