import numpy as np
import pandas as pd

# Define sample data size
num_samples = 1000

# Generate random features
np.random.seed(0)  # For reproducibility
features = {
    'Bytes': np.random.randint(1, 1000, num_samples),
    'Destination_IP': np.random.choice(['192.168.1.2', '192.168.1.3'], num_samples),
    'Port_443': np.random.randint(0, 2, num_samples),  # Binary port presence
    'Port_80': np.random.randint(0, 2, num_samples),   # Binary port presence
    'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples)
}

# Create a DataFrame from the features
df = pd.DataFrame(features)

# Create additional features that were seen during model fitting
df['Port_10.0.0.2'] = np.random.randint(0, 2, num_samples)  # Binary port presence
df['Port_10.0.0.3'] = np.random.randint(0, 2, num_samples)  # Binary port presence
df['Protocol_192.168.1.2'] = np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples)
df['Protocol_192.168.1.3'] = np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples)

# Generate target labels (binary classification)
df['Intrusion'] = np.random.randint(0, 2, num_samples)

# Save the generated dataset to a CSV file
df.to_csv('data\preprocessed_data.csv', index=False)

print("Sample dataset generated and saved.")
