import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.set_page_config(page_title="Weather Prediction App", layout="wide")

# Title of the app
st.title("Weather Prediction for the Next 21 Days")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data.csv")  # Replace with your actual dataset path
    return df

df = load_data()

# Remove 'date' and 'cloud_cover' columns if they exist
df = df.drop(['date', 'cloud_cover'], axis=1, errors='ignore')

# Encode categorical column ('rain_or_not') if not already encoded
df['rain_or_not'] = df['rain_or_not'].map({'Rain': 1, 'No Rain': 0})  # Convert to binary (1 or 0)

# Handle missing values: Remove rows with NaN in target columns
df = df.dropna(subset=['avg_temperature', 'humidity', 'avg_wind_speed', 'pressure', 'rain_or_not'])

# Parameters for regressor and classifier
best_regressor_params = {
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'criterion': 'gini'
}

best_classifier_params = {
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'criterion': 'gini'
}

# Set N to the number of rows in the dataset
N = 21  # Set N as 21 for predicting the next 21 days

# Function to train RandomForestRegressor for numerical prediction
def train_regressor(target_col):
    X, y = [], []
    for i in range(N, len(df)):
        X.append(df[target_col].iloc[i-N:i].values)
        y.append(df[target_col].iloc[i])

    X = np.array(X)
    y = np.array(y)

    model = RandomForestRegressor(
        n_estimators=best_regressor_params['n_estimators'],
        max_depth=best_regressor_params['max_depth'],
        min_samples_split=best_regressor_params['min_samples_split'],
        random_state=42
    )
    model.fit(X, y)
    future_predictions = model.predict(X[-21:])
    
    return future_predictions

# Function to train RandomForestClassifier for rain prediction
def train_classifier():
    X, y = [], []
    for i in range(N, len(df)):
        X.append(df.drop(columns=['rain_or_not']).iloc[i-N:i].values.flatten())
        y.append(df['rain_or_not'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    classifier = RandomForestClassifier(
        n_estimators=best_classifier_params['n_estimators'],
        max_depth=best_classifier_params['max_depth'],
        min_samples_split=best_classifier_params['min_samples_split'],
        random_state=42
    )
    classifier.fit(X, y)
    rain_predictions = classifier.predict(X[-21:])
    rain_probabilities = classifier.predict_proba(X[-21:])[:, 1]

    return rain_predictions, rain_probabilities

# Predict numerical values for the next 21 days
temperature_predictions = train_regressor('avg_temperature')
humidity_predictions = train_regressor('humidity')
wind_speed_predictions = train_regressor('avg_wind_speed')
pressure_predictions = train_regressor('pressure')

# Predict rain or not for the next 21 days
rain_predictions, rain_probabilities = train_classifier()

# Display the predictions in a nice format on Streamlit

st.subheader("🔹 Predicted Temperature, Humidity, Wind Speed & Pressure for the Next 21 Days")

# Create a dataframe to show all numerical predictions
prediction_df = pd.DataFrame({
    'Day': [f"Day {i}" for i in range(1, 22)],  # Day 1, Day 2, ..., Day 21
    'Temperature (°C)': temperature_predictions,
    'Humidity (%)': humidity_predictions,
    'Wind Speed (m/s)': wind_speed_predictions,
    'Pressure (hPa)': pressure_predictions
})

st.dataframe(prediction_df)


# Show Rain Prediction
st.subheader("🔹 Rain Prediction (Rain or No Rain) for the Next 21 Days")

rain_df = pd.DataFrame({
    'Day': [f"Day {i}" for i in range(1, 22)],  # Day 1, Day 2, ..., Day 21
    'Rain Probability': rain_probabilities,
    'Predicted Rain': ['Rain' if x == 1 else 'No Rain' for x in rain_predictions]
})

st.dataframe(rain_df)

# Plot rain probabilities
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(rain_df['Day'], rain_df['Rain Probability'], color='skyblue')
ax.set_title("Rain Probability for Next 21 Days")
ax.set_xlabel("Day")
ax.set_ylabel("Rain Probability")

# Rotate the x-axis labels for better alignment
plt.xticks(rotation=45, ha='right')

st.pyplot(fig)
