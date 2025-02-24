
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import folium
import streamlit as st
import requests
from streamlit_folium import folium_static

# Set random seed
np.random.seed(42)

# Generate synthetic data
dates = pd.date_range(start="2017-01-01", end="2024-12-01", freq="ME")
locations = ["Kanyama", "Matero", "Lusaka_District", "Chawama"]
rainfall = [np.random.uniform(0, 50) if month in [5, 6, 7, 8, 9] else np.random.uniform(150, 300) for month in dates.month]
temp = [np.random.uniform(15, 25) if month in [5, 6, 7, 8, 9] else np.random.uniform(25, 35) for month in dates.month]
cases = [np.random.randint(0, 100) if r < 100 and t < 20 else np.random.randint(200, 2000) for r, t in zip(rainfall, temp)]
deaths = [int(c * np.random.uniform(0.02, 0.05)) for c in cases]
pop_density = np.random.uniform(5000, 7000, len(dates))
sanitation = np.random.uniform(0.4, 0.6, len(dates))

# Add outbreak spikes
for i, date in enumerate(dates):
    if date.year == 2017 and date.month in [11, 12]:
        cases[i] = np.random.randint(1000, 2500)
    elif date.year == 2023 and date.month in [10, 11, 12]:
        cases[i] = np.random.randint(1000, 3000)
    elif date.year == 2024 and date.month in [1, 2, 3]:
        cases[i] = np.random.randint(500, 2000)

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Cholera_Cases": cases,
    "Deaths": deaths,
    "Rainfall_mm": rainfall,
    "Temperature_C": temp,
    "Population_Density": pop_density,
    "Sanitation_Level": sanitation,
    "Location": np.random.choice(locations, len(dates))
})

# Train model
data["Outbreak"] = (data["Cholera_Cases"] > 200).astype(int)
X = data[["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"]]
y = data["Outbreak"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fetch NASA POWER weather data
def get_nasa_weather():
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": 28.2833,
        "latitude": -15.4167,
        "start": "202401",
        "end": "202402",
        "format": "JSON"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        weather = response.json()["properties"]["parameter"]
        latest_rain = weather["PRECTOTCORR"]["202402"] * 30
        latest_temp = weather["T2M"]["202402"]
        return latest_rain, latest_temp
    except Exception as e:
        st.write(f"NASA API failed: {e}. Using defaults.")
        return 200, 30

# Streamlit app
st.title("Lusaka Cholera Prediction System")
st.write("Enter population density and sanitation once; weather is fetched automatically.")

# One-time inputs
if "density" not in st.session_state:
    st.session_state.density = st.slider("Population Density (per sq km)", 5000.0, 7000.0, 6500.0)
    st.session_state.sanitation = st.slider("Sanitation Level (0-1)", 0.0, 1.0, 0.5)
density_input = st.session_state.density
sanitation_input = st.session_state.sanitation

location_input = st.selectbox("Location", locations)
rainfall_auto, temp_auto = get_nasa_weather()
st.write(f"Latest Weather (NASA, Feb 2024): Rainfall={rainfall_auto:.2f}mm, Temp={temp_auto:.2f}°C")

# Predict
sample = pd.DataFrame([[rainfall_auto, temp_auto, density_input, sanitation_input]], 
                      columns=["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"])
risk = model.predict_proba(sample)[0][1] * 100
st.write(f"Predicted Outbreak Risk: {risk:.2f}%")

# SMS alert simulation
if risk > 50:
    alert_msg = f"ALERT: High cholera risk in {location_input} - {risk:.2f}%"
    st.write(alert_msg)
else:
    st.write(f"Low risk: {risk:.2f}%")

# Risk map
m = folium.Map(location=[-15.4167, 28.2833], zoom_start=12)
loc_coords = {
    "Kanyama": [-15.45, 28.25],
    "Matero": [-15.40, 28.20],
    "Lusaka_District": [-15.42, 28.28],
    "Chawama": [-15.43, 28.27]
}
for loc, coords in loc_coords.items():
    loc_sample = pd.DataFrame([[rainfall_auto, temp_auto, density_input, sanitation_input]], 
                              columns=["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"])
    loc_risk = model.predict_proba(loc_sample)[0][1] * 100
    folium.Marker(coords, popup=f"{loc}: {loc_risk:.2f}%").add_to(m)
folium_static(m)
