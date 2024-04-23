from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib

# Load the preprocessed training and testing data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv', header=None, skiprows=1)[0]
y_train = y_train.values.ravel()  # Convert DataFrame to 1D array (if needed)
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv', header=None, skiprows=1)[0]  # Remove 'squeeze=True'
y_test = y_test.values.ravel()  # Convert DataFrame to 1D array (if needed)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Save the trained model to a file
joblib.dump(rf_classifier, 'models/trained_model.pkl')

# Load the trained model from file
loaded_model = joblib.load('models/trained_model.pkl')

# Function to load the model
def load_model(model_path):
    try:
        # Load the trained model
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Example usage:
# loaded_model = load_model('models/trained_model.pkl')
