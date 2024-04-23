import pandas as pd
def detect_intrusion(trained_model, features):
    # Create a DataFrame from the features dictionary
    features_df = pd.DataFrame(features)

    # Perform prediction using the trained model
    prediction = trained_model.predict(features_df)

    return prediction
