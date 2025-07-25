import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Rain Prediction", layout="wide")

# Load model
rain = joblib.load("Rain.joblib")

st.title("🌦️ Rain Tomorrow Prediction App")

# Input form
with st.form("weather_form"):
    st.subheader("Enter today's weather conditions:")

    col1, col2 = st.columns(2)
    with col1:
        Location = st.selectbox("Location",option=sorted(df["Location"].dropna().unique) )
        WindGustDir = st.selectbox("Wind Gust Direction", options=['N', 'S', 'E', 'W', 'NW', 'SE'])
        RainToday = st.selectbox("Did it rain today?", options=['Yes', 'No'])
        MinTemp = st.number_input("Minimum Temperature", value=10.0)
        MaxTemp = st.number_input("Maximum Temperature", value=20.0)
        Rainfall = st.number_input("Rainfall (mm)", value=0.0)

    with col2:
        WindSpeed9am = st.number_input("Wind Speed at 9am", value=5.0)
        WindSpeed3pm = st.number_input("Wind Speed at 3pm", value=10.0)
        Humidity9am = st.number_input("Humidity at 9am (%)", value=60.0)
        Humidity3pm = st.number_input("Humidity at 3pm (%)", value=65.0)
        Pressure9am = st.number_input("Pressure at 9am (hPa)", value=1015.0)
        Pressure3pm = st.number_input("Pressure at 3pm (hPa)", value=1012.0)
        Cloud9am = st.number_input("Cloud at 9am (0-9)", value=5.0)
        Cloud3pm = st.number_input("Cloud at 3pm (0-9)", value=6.0)
        Temp9am = st.number_input("Temperature at 9am", value=15.0)
        Temp3pm = st.number_input("Temperature at 3pm", value=19.0)

    submitted = st.form_submit_button("Predict")

# Make prediction
if submitted:
    input_data = pd.DataFrame([{
        'Location': Location,
        'WindGustDir': WindGustDir,
        'RainToday': RainToday,
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'WindSpeed9am': WindSpeed9am,
        'WindSpeed3pm': WindSpeed3pm,
        'Humidity9am': Humidity9am,
        'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am,
        'Pressure3pm': Pressure3pm,
        'Cloud9am': Cloud9am,
        'Cloud3pm': Cloud3pm,
        'Temp9am': Temp9am,
        'Temp3pm': Temp3pm
    }])

    pred = rain['model'].predict(input_data)[0]
    prob = rain['model'].predict_proba(input_data)[0][pred]

    st.success(f"Prediction: {'🌧️ Rain' if pred == 1 else '☀️ No Rain'}")
    st.info(f"Confidence: {prob:.2%}")
