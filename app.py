import streamlit as st
import numpy as np
import joblib
import pickle as pkl

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# with open('model.pkl','rb') as f:
#     model = pkl.load(f)

# with open('scaler.pkl', 'rb') as f:
#     scaler = pkl.load(f)

st.title("🌾 Crop Recommendation System")
st.write("Enter your soil and weather conditions to get the best crop recommendation.")

# Input Fields
nitrogen = st.number_input("Nitrogen (N)", min_value=0.0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0)
potassium = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
pH = st.number_input("pH Value", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Prediction
if st.button("Recommend Crop"):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.success(f"✅ Recommended Crop: {prediction[0]}")
