import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
st.set_page_config(page_title="Rain Prediction", layout="wide")

# Load model and dataset
@st.cache_data
def load_data():
    file_path="weatherAUS.csv"
    if not os.path.exists(file_path):
        st.error(f"File not Found:{file_path}.Please let's make sure it' in the same folder")
        st.stop()
    return pd.read_csv(file_path)

@st.cache_resource
def load_model():
    file_Rain="RainDecision.joblib"
    if not os.path.exists(file_Rain):
        st.error(f"File not Found: {file_Rain}.Please make sure that the joblib is correct")
    return joblib.load(file_Rain)

# === File upload or default ===
st.sidebar.header("📁 Upload Your Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload your CSV (1 row or full dataset)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in rain['inputs']):
            st.error("❌ Uploaded file is missing required columns.")
            st.stop()
        st.success("✅ Custom dataset loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()
else:
    df = load_data()


df = load_data()
rain = load_model()

st.title("🌦️ GROUP 3 WEATHER FORCAST APPLICATION USING DECISION TREE(RAIN TOMORROW)")

# === Sidebar Input Form ===
st.sidebar.header("Input Today's Weather Conditions")

Location = st.sidebar.selectbox("Location", options=sorted(df["Location"].dropna().unique()))
RainToday = st.sidebar.selectbox("Did it rain today?", options=["Yes", "No"])

MinTemp = st.sidebar.number_input("Minimum Temperature", value=10.0)

MaxTemp = st.sidebar.number_input("Maximum Temperature", value=20.0)
Rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)
Evaporation = st.sidebar.number_input("Evaporation", value=5.0)
Sunshine = st.sidebar.number_input("Sunshine", value=5.0)
WindGustDir = st.sidebar.selectbox("Wind Gust Direction", options=sorted(df["WindGustDir"].dropna().unique()))
WindGustSpeed= st.sidebar.number_input("Wind Gust Speed", value=10.0)
WindDir9am = st.sidebar.selectbox("Wind Direction at 9am", options=sorted(df["WindDir9am"].dropna().unique()))
WindDir3pm = st.sidebar.selectbox("Wind Direction at 3pm", options=sorted(df["WindDir3pm"].dropna().unique()))
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
        'WindGustSpeed':WindGustSpeed,
        'WindDir9am':WindDir9am,
        'WindDir3pm':WindDir3pm,
        'RainToday': RainToday,
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'Evaporation': Evaporation,
        'Sunshine': Sunshine,
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

    # Preprocessing steps using Rain dictionary
    input_data[rain['numerical_cols']] = rain['imputer'].transform(input_data[rain['numerical_cols']])
    input_data[rain['numerical_cols']] = rain['scaler'].transform(input_data[rain['numerical_cols']])
    encoded = rain['encoder'].transform(input_data[rain['categorical_cols']])
    encoded_df = pd.DataFrame(encoded if isinstance(encoded, np.ndarray) else encoded.toarray(), columns=rain['encoded_cols'])
    X_input = pd.concat([input_data[rain['numerical_cols']].reset_index(drop=True), encoded_df], axis=1)

    pred = rain['model'].predict(X_input)[0]
    prob = rain['model'].predict_proba(X_input)[0][list(rain['model'].classes_).index(pred)]

    st.success(f"Prediction: {'🌧️ Rain' if pred == 'Yes' else '☀️ No Rain'}")
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
    csv = input_data.assign(Prediction=pred).to_csv(index=False)
    st.download_button("Download Result", csv, "prediction.csv", "text/csv")

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
    model = rain['model']
    importances = model.feature_importances_
    all_features = list(rain['numerical_cols']) + list(rain['encoded_cols'])

    imp_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Most Important Features")
    st.pyplot(fig)
