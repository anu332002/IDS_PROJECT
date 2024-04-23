import pandas as pd

# Sample network traffic data
sample_data = {
    'Source_IP': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
    'Destination_IP': ['10.0.0.1', '10.0.0.2', '10.0.0.3'],
    'Protocol': ['TCP', 'UDP', 'TCP'],
    'Port': [80, 443, 22],
    'Bytes': [1000, 500, 1500]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(sample_data)

# Write the DataFrame to CSV file
df.to_csv('data/network_traffic.csv', index=False)
print("Sample network traffic data written to network_traffic.csv")
