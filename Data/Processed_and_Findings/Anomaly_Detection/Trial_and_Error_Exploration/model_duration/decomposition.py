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
df = df[df["Subject"] == "Service Request"]
df = df[df["Type"] == "Water Clean Up After Repairs"]
df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Closed Date'] = pd.to_datetime(df['Closed Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['Open Date', 'Closed Date'])
#calculate duration
df['Case Duration (hours)'] = (df['Closed Date'] - df['Open Date']).dt.total_seconds() / 3600
df = df[df['Case Duration (hours)'] >= 0]

df['Date'] = df['Open Date'].dt.date  # Extract date component
daily_mean = df.groupby('Date')['Case Duration (hours)'].mean()
#2008 0101 3 hrs

plt.figure(figsize=(12, 6))
plt.plot(daily_mean.index, daily_mean.values, marker='o', linestyle='-', color='b')
plt.title('Mean Case Duration Per Day', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Mean Case Duration (hours)', fontsize=14)
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



decomposition = sm.tsa.seasonal_decompose(
            daily_mean, model='additive', period=365
        )
decomposition.plot()

plt.show()

estimate = decomposition.trend+decomposition.seasonal

plt.figure(figsize=(12, 6))
plt.plot(decomposition.observed, label='Original', color='blue', linestyle='-', marker='o')
plt.plot(estimate, label='Trend Estimate', color='red', linestyle='--', marker='x')
plt.title('Original Time Series and Estimated Trend', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
      

2020 01 01     number of case 50000 - duration 4000 mean 4000/50000
2020 01 02      2000        duration 3000
2020 01 03      1000        duration 1000
seasonal_decompose
anomalies for which date we got higher mean duration
    
    compare with other types

    analysis which type got more anomlies(higher mean duratin for date.)
one type could be 1 days
other type have 2 days