import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Rain Prediction", layout="wide")

# Load model and dataset
df = pd.read_csv("weatherAUS.csv")
rain = joblib.load("Rain.joblib")

st.title("🌦️ Rain Tomorrow Prediction App")

# === Sidebar Input Form ===
st.sidebar.header("Input Today's Weather Conditions")

Location = st.sidebar.selectbox("Location", options=sorted(df["Location"].dropna().unique()))
WindGustDir = st.sidebar.selectbox("Wind Gust Direction", options=sorted(df["WindGustDir"].dropna().unique()))
RainToday = st.sidebar.selectbox("Did it rain today?", options=["Yes", "No"])

MinTemp = st.sidebar.number_input("Minimum Temperature", value=10.0)
MaxTemp = st.sidebar.number_input("Maximum Temperature", value=20.0)
Rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)
WindSpeed9am = st.sidebar.number_input("Wind Speed at 9am", value=5.0)
WindSpeed3pm = st.sidebar.number_input("Wind Speed at 3pm", value=10.0)
Humidity9am = st.sidebar.number_input("Humidity at 9am (%)", value=60.0)
Humidity3pm = st.sidebar.number_input("Humidity at 3pm (%)", value=65.0)
Pressure9am = st.sidebar.number_input("Pressure at 9am (hPa)", value=1015.0)
Pressure3pm = st.sidebar.number_input("Pressure at 3pm (hPa)", value=1012.0)
Cloud9am = st.sidebar.number_input("Cloud at 9am (0-9)", value=5.0)
Cloud3pm = st.sidebar.number_input("Cloud at 3pm (0-9)", value=6.0)
Temp9am = st.sidebar.number_input("Temperature at 9am", value=15.0)
Temp3pm = st.sidebar.number_input("Temperature at 3pm", value=19.0)

# Prediction Button
if st.button("Predict Rain Tomorrow"):
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

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Confidence of Rain"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "blue"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 100], 'color': "lightblue"}]
              }))
    st.plotly_chart(fig)

# === Optional Feature: Rainfall Trend Chart ===
st.subheader("📈 Historical Rainfall Trends")
if st.checkbox("Show rainfall trends by location"):
    df['Date'] = pd.to_datetime(df['Date'])
    locs = st.multiselect("Select Location(s):", df['Location'].unique(), default=["Sydney"])
    filtered = df[df['Location'].isin(locs)]
    fig = px.line(filtered, x="Date", y="Rainfall", color="Location", title="Rainfall Over Time")
    st.plotly_chart(fig, use_container_width=True)

# === Optional Feature: Feature Importance ===
st.subheader("🔍 What Features Matter Most?")
if st.checkbox("Show feature importance"):
    model = rain['model'].named_steps['classifier']
    preprocessor = rain['model'].named_steps['preprocessor']
    ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = ohe.get_feature_names_out(rain['categorical_cols'])
    all_features = list(rain['numerical_cols']) + list(cat_features)

    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Most Important Features")
    st.pyplot(fig)
