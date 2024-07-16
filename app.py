from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('fish_species_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Feature names
        feature_names = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
        
        # Get input values from form
        input_values = [float(request.form[feature]) for feature in feature_names]
        
        # Convert to DataFrame to maintain feature names
        input_df = pd.DataFrame([input_values], columns=feature_names)
        
        # Predict class
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', prediction_text=f'Predicted species: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
