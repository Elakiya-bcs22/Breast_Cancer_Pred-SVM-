
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load column names
df = pd.read_csv('data.csv')
feature_names = df.columns[:-1]  # exclude target

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[f]) for f in feature_names]
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    result = "Benign (No Cancer)" if prediction == 1 else "Malignant (Cancer)"
    return render_template('index.html', features=feature_names, result=result)

if __name__ == '__main__':
    app.run(debug=True)