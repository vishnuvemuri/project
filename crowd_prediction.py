import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re
import joblib  # For saving and loading the model
import sys

API_KEY = 'yi8U3ni7qxsREArm1ME1ZyMr9lU5liRl'  # Replace with your actual API key

# Load the dataset
df = pd.read_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\dataset for crowd prediction.csv", encoding="ISO-8859-1")

# Ensure a valid Date column exists
df['Date'] = [datetime.now().date() - timedelta(days=i) for i in range(len(df))]  # Generate synthetic past dates

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)  # Remove invalid dates

# Convert categorical features into numerical values
df['Category'] = df['Category'].astype('category').cat.codes
df['Location'] = df['Location'].astype('category').cat.codes
df['Required Time'] = df['Required Time'].str.extract('(\d+)').astype(float)

# Feature Engineering
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Hour'] = np.random.randint(8, 20, size=len(df))  # Generate random operational hours

# Preparing Data for ML Model
features = ['DayOfWeek', 'Hour', 'Category', 'Location', 'Required Time']

df['CrowdLevel'] = np.random.randint(0, 3, size=len(df))  # Generate synthetic numerical crowd levels (0=Low, 1=Moderate, 2=High)

target = 'CrowdLevel'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVR model with adjusted hyperparameters
model = SVR(kernel='rbf', C=500, gamma=0.05, epsilon=0.2)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'crowd_prediction.pkl')

# Evaluate Model
predictions = model.predict(X_test)
predictions = np.round(predictions).astype(int)  # Convert back to categorical values
predictions = np.clip(predictions, 0, 2)  # Ensure values stay within range (Low=0, Moderate=1, High=2)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Function to load and use the saved model
def predict_crowd(museum_name, date_input, time_str):
    model = joblib.load('crowd_prediction.pkl')
    selected_datetime = datetime.strptime(f"{date_input} {time_str}", "%d-%m-%Y %H:%M")
    
    if selected_datetime < datetime.now():
        return "Past times cannot be predicted."
    
    museum = df[df['Name'].str.lower().str.strip() == museum_name.lower().strip()]
    if museum.empty:
        return "Museum not found."
    
    input_features = np.array([[selected_datetime.weekday(), selected_datetime.hour, museum.iloc[0]['Category'], museum.iloc[0]['Location'], museum.iloc[0]['Required Time']]])
    predicted_crowd_numeric = int(round(model.predict(input_features)[0]))
    predicted_crowd_numeric = np.clip(predicted_crowd_numeric, 0, 2)
    
    crowd_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
    return f"Predicted crowd level at {museum_name} on {date_input} at {time_str} is {crowd_mapping[predicted_crowd_numeric]}."
