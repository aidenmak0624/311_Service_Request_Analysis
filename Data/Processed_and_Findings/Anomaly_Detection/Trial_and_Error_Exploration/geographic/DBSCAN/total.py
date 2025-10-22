import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the dataset
chunk_size = 50000  # Adjust chunk size based on available memory
chunks = pd.read_csv('/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv', chunksize=chunk_size)

# Initialize an empty list to accumulate processed chunks
data_chunks = []

# Process each chunk
for chunk in chunks:
    data_chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
df = pd.concat(data_chunks, ignore_index=True)

# Check the available columns to verify 'geometry' is present
print("Columns in the dataset:", df.columns)

# Convert 'Open Date' to datetime format
df['Open Date'] = pd.to_datetime(df['Open Date'], errors='coerce')
df = df[df['Open Date'].dt.year == 2020]
df = df.dropna(subset=['Open Date'])

# Check if the 'geometry' column has any missing or malformed data
print("Number of rows with missing geometry:", df['Geometry'].isna().sum())

# Filter rows with valid geometry (remove rows with missing or malformed geometry)
df = df.dropna(subset=['Geometry'])

# Ensure 'Geometry' is treated as a string
df['Geometry'] = df['Geometry'].astype(str)

# Extract longitude and latitude using string manipulation
df[['Longitude', 'Latitude']] = df['Geometry'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)', expand=True)

# Convert extracted values to numeric (they may be strings initially)
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')

# Check for rows with missing longitude or latitude
print(f"Rows with missing longitude or latitude: {df[['Longitude', 'Latitude']].isna().sum()}")

# Drop rows with missing longitude or latitude
df = df.dropna(subset=['Longitude', 'Latitude'])

# Check that we have valid coordinate data
print("Number of valid rows after filtering for coordinates:", len(df))

# Group by neighborhood and aggregate the case count per neighborhood
neighborhood_case_counts = df.groupby('Neighbourhood').size().reset_index(name='case_count')
df = df.merge(neighborhood_case_counts, on='Neighbourhood')

# Prepare data for DBSCAN
coords = df[['Latitude', 'Longitude']].values
if coords.shape[0] > 0:
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    # Apply DBSCAN
    epsilon = 0.1  # Adjust this based on scale; smaller values may be better for finer granularity
    min_samples = 3  # Minimum number of points to form a cluster
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(scaled_coords)

    # Add DBSCAN labels to the dataframe
    df['cluster'] = db.labels_

    # Identify anomalies (points labeled as -1 by DBSCAN are considered anomalies)
    anomalies = df[df['cluster'] == -1]

    # Plot clusters and anomalies
    plt.figure(figsize=(10, 8))

    # Plot clustered points (labeled 0 or other positive labels) 
    plt.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='viridis', marker='o', s=20, label='Clustered Points')

    # Plot anomalies (labeled -1)
    plt.scatter(anomalies['Longitude'], anomalies['Latitude'], c='red', marker='x', s=50, label='Anomalies')

    # Adding labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Clusters and Anomalies Detected by DBSCAN')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
else:
    print("No valid coordinates available for clustering.")
