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

unique_types = df['Reason'].unique()
print(unique_types)
type_counts = df.groupby('Reason').size().reset_index(name='count')
print(type_counts)

categorized_df = pd.DataFrame(type_counts)

pivot_table = pd.pivot_table(df, values='Case ID', index='Reason', columns='Neighbourhood', aggfunc='count', fill_value=0)
print(pivot_table)

# Bar chart of types
type_counts.plot(kind='bar', x='Reason', y='count', title='Count by Reason')
plt.xlabel('Reason')
plt.ylabel('Count')
plt.show()

type_counts_sorted = type_counts.sort_values(by='count', ascending=False)

# Display the sorted DataFrame
print(type_counts_sorted)

type_counts_sorted.to_csv('type_counts_sorted.csv', index=False)