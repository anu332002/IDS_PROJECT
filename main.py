from scapy.all import sniff, IP
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src import intrusion_detection
import src.machine_learning as machine_learning

def process_packet(packet, trained_model, scaler, feature_columns):
    try:
        # Extract relevant information from the packet using Scapy
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst

        # Create a DataFrame with the packet information
        packet_data = {
            'Source_IP': [src_ip],
            'Destination_IP': [dst_ip],
            # Add more features as needed based on your model
        }
        df = pd.DataFrame(packet_data, columns=feature_columns)

        # Scale the features using the same scaler used during training
        scaled_features = scaler.transform(df)

        # Perform intrusion detection using the machine learning model
        prediction = trained_model.predict(scaled_features)
            
        # Example: Print intrusion detection result if an attack is detected
        if prediction == 1:  # Assuming 1 represents an attack in your model
            print(f"INTRUSION DETECTED: Source IP: {src_ip}, Destination IP: {dst_ip}")
    except Exception as e:
        print(f"Error processing packet: {e}")

def main():
    trained_model = machine_learning.loaded_model
    if trained_model is None:
        print("Error loading the trained model. Exiting.")
        return

    # Load the scaler and feature columns used during training
    scaler = machine_learning.loaded_scaler
    feature_columns = machine_learning.feature_columns

    if scaler is None or feature_columns is None:
        print("Error loading scaler or feature columns. Exiting.")
        return

    # Start capturing packets and process them in real-time using Scapy
    sniff(filter="ip", prn=lambda x: process_packet(x, trained_model, scaler, feature_columns), iface="Wi-Fi")

if __name__ == "__main__":
    main()
