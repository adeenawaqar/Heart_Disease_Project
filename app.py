import streamlit as st
import numpy as np
import joblib

# ===========================
# Load Model & Scaler
# ===========================

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===========================
# Title
# ===========================

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient details below and click Predict.")

# ===========================
# User Inputs
# ===========================

Age = st.number_input("Age", min_value=1, max_value=120, value=30)

Sex = st.selectbox("Sex", [0, 1])

Chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])

BP = st.number_input("Resting Blood Pressure", value=120)

Cholesterol = st.number_input("Cholesterol", value=200)

FBS_over_120 = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

EKG_results = st.selectbox("EKG Results", [0, 1, 2])

Max_HR = st.number_input("Maximum Heart Rate", value=150)

Exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])

ST_depression = st.number_input("ST Depression", value=1.0)

Slope_ST = st.selectbox("Slope of ST Segment", [1, 2, 3])

Number_of_vessels_fluro = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

Thallium = st.selectbox("Thallium", [3, 6, 7])

# ===========================
# Prediction
# ===========================

if st.button("Predict"):

    features = np.array([[
        Age,
        Sex,
        Chest_pain_type,
        BP,
        Cholesterol,
        FBS_over_120,
        EKG_results,
        Max_HR,
        Exercise_angina,
        ST_depression,
        Slope_ST,
        Number_of_vessels_fluro,
        Thallium
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
