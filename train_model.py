import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load your fish dataset
fish_data = pd.read_csv('C:/Users/mehar/Downloads/fish_species_prediction/fish.csv')

# Assume X contains features and y contains target labels
X = fish_data.drop(columns=['Species'])
y = fish_data['Species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'fish_species_model.pkl')
