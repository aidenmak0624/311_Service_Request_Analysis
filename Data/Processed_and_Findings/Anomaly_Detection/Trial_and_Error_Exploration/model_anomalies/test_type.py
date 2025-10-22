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

unique_types = df['Type'].unique()
print(unique_types)
type_counts = df.groupby('Type').size().reset_index(name='count')
print(type_counts)