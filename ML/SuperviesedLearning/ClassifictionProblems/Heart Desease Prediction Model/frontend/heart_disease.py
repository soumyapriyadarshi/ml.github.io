import streamlit as st
import pandas as pd
import joblib as jb

# Unpikiling the model and scaler and columns or Loading the model, scaler and columns
model = jb.load("KNN_heart.pkl")
scaler = jb.load("scaler.pkl")
expected_columns = jb.load("columns.pkl")

print(expected_columns)

# UI
st.title("Heart Stroke Prediction")
st.markdown("Fill up the following details")

age = st.number_input("Age", min_value = 18, max_value = 100, value = 35)
sex = st.select_slider("Sex", ["M","F"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value = 80, max_value = 200, value = 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value = 100, max_value = 600, value = 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", min_value = 60, max_value = 220, value = 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y","N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Flat": 1 if slope == "Flat" else 0,
        "ST_Slope_Up": 1 if slope == "Up" else 0,
    }

    input_df = pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.write("Prediction:", prediction)
    st.write("Probability:", model.predict_proba(scaled_input))

    if prediction == 1:
        st.error("High Risk of Heart Disease")

    else:
        st.success("Low Risk of Heart Disease")