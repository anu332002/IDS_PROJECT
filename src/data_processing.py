import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data():
    # Load the synthetic dataset
    df = pd.read_csv('data/synthetic_network_traffic.csv')

    # Perform one-hot encoding for categorical variables
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[['Source_IP', 'Destination_IP', 'Protocol']])
    
    # Manually create feature names based on categorical variable values
    categories = encoder.categories_
    encoded_feature_names = []
    for i, cat_list in enumerate(categories):
        for cat_value in cat_list[1:]:  # Skip the first category
            # Adjust the indexing to stay within bounds of the DataFrame
            col_index = len(df.columns) - len(categories) + i  # Index of the column in the original DataFrame
            encoded_feature_names.append(f'{df.columns[col_index]}_{cat_value}')

    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Combine encoded features with numerical features
    numerical_features = df[['Port', 'Intrusion']]
    df_final = pd.concat([df_encoded, numerical_features], axis=1)

    # Split the data into features (X) and target (y)
    X = df_final.drop(['Intrusion'], axis=1)
    y = df_final['Intrusion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data to CSV (optional)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False, header=True)
    y_test.to_csv('data/y_test.csv', index=False, header=True)

    print("Preprocessing completed and data split into training and testing sets.")

if __name__ == "__main__":
    preprocess_data()
