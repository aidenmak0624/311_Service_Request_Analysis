#This program aims to use isolation tree to detect the anomalies for duration
#Assuming Duration is stationary data that does not affect by time.


#OR we can actually see are there any trend?

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sodapy import Socrata


#df = pd.read_csv("/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv")
chunk_size = 100000  # Adjust chunk size based on available memory
chunks = pd.read_csv('/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv', chunksize=chunk_size)

# Initialize an empty list to accumulate processed chunks
data_chunks = []

# Process each chunk
for chunk in chunks:
    # Apply any necessary preprocessing on each chunk here
    data_chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
df = pd.concat(data_chunks, ignore_index=True)

df['Open Date'] = pd.to_datetime(df['Open Date'], errors='coerce')
df['Closed Date'] = pd.to_datetime(df['Closed Date'], errors='coerce')
df = df.dropna(subset=['Open Date', 'Closed Date'])
df['Case Duration (hours)'] = (df['Closed Date'] - df['Open Date']).dt.total_seconds() / 3600
df = df[df['Case Duration (hours)'] >= 0]
durations = df['Case Duration (hours)']



# A histrogram to see 
#plt.figure(figsize=(10, 6))
#plt.hist(df['Case Duration (hours)'].dropna(), bins=50, edgecolor='black', color='skyblue')  # Drop NaN values if any
#plt.title('Histogram of Service Request Duration')
#plt.xlabel('Duration (hours)')
#plt.ylabel('Frequency')
#plt.grid(True)
#plt.show()
X = df[['Case Duration (hours)']].dropna() 
iso_forest_duration = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = iso_forest_duration.fit_predict(X)

df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Step 4: Visualize the anomalies in the duration
plt.figure(figsize=(10, 6))
#plt.hist(df['Case Duration (hours)'], bins=50, edgecolor='black', color='skyblue', alpha=0.7, label='Duration Distribution')
plt.scatter(df['Case Duration (hours)'][df['Anomaly'] == 1], 
            [0] * df['Anomaly'].sum(),  # Plot anomalies at y = 0
            color='red', label='Anomalies', zorder=5)
plt.title('Anomalies in Service Request Duration')
plt.xlabel('Duration (hours)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
# Optional: Print the detected anomalies
outliers = df[df['Anomaly'] == 1]  # Filter rows where anomalies are detected
# Save the anomalies to a CSV file
outliers.to_csv('service_request_outliers.csv', index=False)