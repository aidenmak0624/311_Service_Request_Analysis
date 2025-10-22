import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

chunk_size = 200000  # Adjust chunk size based on available memory
chunks = pd.read_csv('/Users/chinweimak/Documents/DataMining/Explore2/sample_1000_records.csv', chunksize=chunk_size)
data_chunks = []
for chunk in chunks:
    data_chunks.append(chunk)
df = pd.concat(data_chunks, ignore_index=True)

df = df[df['Subject'] == 'Service Request']
df['Open Date'] = pd.to_datetime(df['Open Date'], errors='coerce',format="%m/%d/%Y %I:%M:%S %p")
df = df.dropna(subset=['Open Date'])

#Find uniques by Type
unique_types = df['Type'].unique()
print(unique_types)

#Find case_count by Type
type_counts = df.groupby('Type').size().reset_index(name='count').sort_values(by='count', ascending=False)
print(type_counts)

# Bar chart of types
type_counts.plot(kind='bar', x='Type', y='count', title='Count by Reason')
plt.xlabel('Reason')
plt.ylabel('Count')
plt.show()


#Find anomalies among different Types

for unique_type in unique_types:
    df_type = df[df['Type']== unique_type]

    daily_cases_type = df_type.groupby(df['Open Date'].dt.date).size().to_frame(name='case_count').reset_index()
    # Aggregate data by week
    daily_cases_type['week'] = daily_cases_type.index.to_period('W').start_time
    weekly_cases = daily_cases_type.groupby('week').sum()

    # Check if enough points for decomposition
    if len(weekly_cases) >= 104:  # 2 years of weekly data (52 weeks * 2)
        decomposition = sm.tsa.seasonal_decompose(
            weekly_cases['case_count'], model='additive', period=52
        )
        decomposition.plot()
        plt.show()
    else:
        print("Insufficient data for weekly seasonal decomposition.")


