# churn_app.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template_string
import os
import pickle

# Load and preprocess dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
if not os.path.exists("churn_model.pkl"):
    pickle.dump(model, open("churn_model.pkl", "wb"))

# Flask app setup
app = Flask(__name__)
model = pickle.load(open("churn_model.pkl", "rb"))

# Simple HTML template
html_form = '''
<!DOCTYPE html>
<html>
<head><title>Churn Prediction</title></head>
<body>
    <h2>Customer Churn Prediction</h2>
    <form action="/predict" method="post">
        {% for col in columns %}
        <label>{{ col }}:</label><br>
        <input type="text" name="{{ col }}" required><br><br>
        {% endfor %}
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template_string(html_form, columns=X.columns, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[col]) for col in X.columns]
        prediction = model.predict(np.array(values).reshape(1, -1))[0]
        result = "Churn" if prediction == 1 else "Not Churn"
        return render_template_string(html_form, columns=X.columns, prediction=result)
    except:
        return render_template_string(html_form, columns=X.columns, prediction="Invalid Input")

if __name__ == '__main__':
    app.run(debug=True)
