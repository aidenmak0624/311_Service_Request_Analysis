import pandas as pd

def count_cases_by_date(csv_file):#target_date):
    # Load the dataset
    
    chunk_size = 50000  # Adjust chunk size based on available memory
    chunks = pd.read_csv(csv_file, chunksize=chunk_size)

    data_chunks = []

# Process each chunk
    for chunk in chunks:
    # Apply any necessary preprocessing on each chunk here
        data_chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
    df = pd.concat(data_chunks, ignore_index=True)

    # Convert the 'Open Date' to datetime format
    df['Open Date'] = pd.to_datetime(df['Open Date'], format="%m/%d/%Y %I:%M:%S %p")

    # Filter the rows based on the target date
    #filtered_cases = df[df['Open Date'].dt.date == pd.to_datetime(target_date).date()]
    #filtered_cases = df[(df['Type']=='Bus Schedule')& (df['Subject'] == 'Service Request')]
    filtered_cases = df[df['Subject'] == 'Service Request']
    filtered_cases = df[df['Type'] == 'Bus Schedule']
    # Count the number of cases for the specified date
    case_count = filtered_cases.shape[0]  # Number of rows
    print(f"Number of cases on : {case_count}")
    return case_count

# Usage example
csv_file = '/Users/chinweimak/Documents/DataMining/DataMiningProjectSelfExplore/311_Requests_20241002.csv.download/311_Requests_20241002.csv'  # Replace with the path to your CSV file
#target_date = '2012-08-29'  # The date you want to count cases for
count_cases_by_date(csv_file)#target_date)
