import pandas as pd
import numpy as np
import mysql.connector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# MySQL Connection Setup
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='Mohana@04',
        database='museum',
    )

# Fetch museum dynamic pricing settings
def get_museum_pricing(museum_name):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT pricing_factor, factor_status FROM museum_pricing WHERE museum_name = %s"
    cursor.execute(query, (museum_name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

# Simulated dataset creation
def load_dataset():
    data = pd.DataFrame({
        'base_price': np.random.randint(100, 500, 500),
        'hour': np.random.randint(8, 22, 500),
        'weekday': np.random.randint(0, 7, 500),
        'public_holiday': np.random.choice([0, 1], 500, p=[0.8, 0.2]),
        'special_event': np.random.choice([0, 1], 500, p=[0.9, 0.1]),
    })

    # Apply multipliers
    data['time_factor'] = np.where((data['hour'] >= 10) & (data['hour'] <= 19), 1.5, 1.0)
    data['day_factor'] = np.where(data['weekday'] >= 5, 1.3, 1.0)
    data['event_factor'] = np.where(data['public_holiday'] == 1, 1.4, np.where(data['special_event'] == 1, 1.2, 1.0))

    # Calculate final price
    data['final_price'] = data['base_price'] * data['time_factor'] * data['day_factor'] * data['event_factor']
    
    return data

# Train ML Model
def train_model():
    data = load_dataset()
    feature_columns = ['base_price', 'hour', 'weekday', 'public_holiday', 'special_event']
    X = data[feature_columns]
    y = data['final_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Save model
    joblib.dump(model, "dynamic_pricing_model.pkl")
    print("Model saved as dynamic_pricing_model.pkl")

# Predict ticket price with dynamic pricing check
def predict_price(museum_name, base_price, hour, weekday, public_holiday, special_event):
    model = joblib.load("dynamic_pricing_model.pkl")

    # Get museum pricing settings from database
    museum_pricing = get_museum_pricing(museum_name)

    if museum_pricing and museum_pricing["factor_status"]:  # Apply dynamic pricing if enabled
        pricing_factor = museum_pricing["pricing_factor"]
    else:
        pricing_factor = 1.0  # Default normal price

    input_df = pd.DataFrame([[base_price, hour, weekday, public_holiday, special_event]],
                            columns=['base_price', 'hour', 'weekday', 'public_holiday', 'special_event'])

    predicted_price = model.predict(input_df)[0] * pricing_factor
    return round(predicted_price, 2)

# Run Training
if __name__ == "__main__":
    train_model()

    # Example prediction
    museum = "National Museum"
    print(f"Predicted Price for {museum}: ", predict_price(museum, 300, 11, 6, 1, 0))