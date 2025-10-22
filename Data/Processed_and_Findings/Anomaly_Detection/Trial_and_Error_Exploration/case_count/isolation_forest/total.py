import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest

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



# Prepare data for anomaly detection
case_data = daily_cases[['case_count']].values  # Only use 'case_count' for Isolation Forest

# Configure and train the Isolation Forest model
clf = IsolationForest(contamination=0.01, random_state=42)
anomalies = clf.fit_predict(case_data)  # -1 indicates anomaly, 1 indicates normal
daily_cases['anomaly'] = anomalies  # Add anomaly labels to the DataFrame

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(daily_cases['date'], daily_cases['case_count'], label='Case Count')
plt.scatter(
    daily_cases.loc[daily_cases['anomaly'] == -1, 'date'],
    daily_cases.loc[daily_cases['anomaly'] == -1, 'case_count'],
    color='red', label='Anomaly'
)
plt.xlabel('Date')
plt.ylabel('Case Count')
plt.title('Case Count with Anomalies Detected')
plt.legend()
plt.show()
