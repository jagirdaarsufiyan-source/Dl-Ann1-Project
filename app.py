import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart ECG Predictor",
    page_icon="❤️",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS Styling
# --------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Encoders
# --------------------------------------------------
model = tf.keras.models.load_model("model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_sex.pkl", "rb") as f:
    label_encoder_sex = pickle.load(f)

with open("label_encoder_st.pkl", "rb") as f:
    onehot_encoder_st = pickle.load(f)

# --------------------------------------------------
# Title Section
# --------------------------------------------------
st.markdown('<div class="main-title">❤️ Heart ECG Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter patient details to predict ECG result</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Input Section (Card Style Layout)
# --------------------------------------------------

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=1, max_value=120)
        Sex = st.selectbox("Sex", ["M", "F"])
        RestingBP = st.number_input("Resting Blood Pressure")
        Cholesterol = st.number_input("Cholesterol")
        FastingBS = st.selectbox("Fasting Blood Sugar", [0, 1])

    with col2:
        MaxHR = st.number_input("Max Heart Rate")
        Oldpeak = st.number_input("Oldpeak", format="%.2f")
        HeartDisease = st.selectbox("Heart Disease", [0, 1])
        ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔍 Predict ECG"):

    # Encode Sex
    Sex_encoded = label_encoder_sex.transform([Sex])[0]

    # Create DataFrame
    input_data = pd.DataFrame({
        "Age": [Age],
        "Sex": [Sex_encoded],
        "RestingBP": [RestingBP],
        "Cholesterol": [Cholesterol],
        "FastingBS": [FastingBS],
        "MaxHR": [MaxHR],
        "Oldpeak": [Oldpeak],
        "HeartDisease": [HeartDisease],
        "ST_Slope": [ST_Slope]
    })

    # One Hot Encode ST_Slope
    st_encoded = onehot_encoder_st.transform(input_data[["ST_Slope"]]).toarray()
    st_encoded_df = pd.DataFrame(
        st_encoded,
        columns=onehot_encoder_st.get_feature_names_out(["ST_Slope"])
    )

    input_data = pd.concat(
        [input_data.drop("ST_Slope", axis=1), st_encoded_df],
        axis=1
    )

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_value = prediction[0][0]

    # Display Result
    if prediction_value > 0.5:
        st.markdown(
            '<div class="result-box" style="background-color:#ff4b4b;">⚠️ ECG Result: Abnormal (1)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#00c853;">✅ ECG Result: Normal (0)</div>',
            unsafe_allow_html=True
        )
