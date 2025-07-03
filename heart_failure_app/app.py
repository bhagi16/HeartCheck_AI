# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

# Set page config
st.set_page_config(page_title="HeartCheck AI", layout="centered")

# Title & Description
st.title("💓 Heart Failure Prediction App")
st.markdown("""
This app predicts the **risk of heart failure** based on clinical input parameters.

Please enter the patient details below:
""")

# Load the trained Keras model

MODEL_PATH = "model/heart_model.keras"

if os.path.exists(MODEL_PATH):
    model = load_model("model/heart_model.keras")

else:
    st.error("❌ Model file not found. Make sure 'heart_model.pkl' is in the 'model/' folder.")
    st.stop()

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    anaemia = st.selectbox("Anaemia", ("No", "Yes"))
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=250)
    diabetes = st.selectbox("Diabetes", ("No", "Yes"))
    ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=35)
    high_blood_pressure = st.selectbox("High Blood Pressure", ("No", "Yes"))
    platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=25000, max_value=900000, value=265000)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=1.0)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=110, max_value=150, value=137)
    sex = st.selectbox("Sex", ("Female", "Male"))
    smoking = st.selectbox("Smoking", ("No", "Yes"))
    time = st.number_input("Follow-up Time (in days)", min_value=1, max_value=300, value=130)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input as a NumPy array
    input_data = np.array([[
        age,
        1 if anaemia == "Yes" else 0,
        creatinine_phosphokinase,
        1 if diabetes == "Yes" else 0,
        ejection_fraction,
        1 if high_blood_pressure == "Yes" else 0,
        platelets,
        serum_creatinine,
        serum_sodium,
        1 if sex == "Male" else 0,
        1 if smoking == "Yes" else 0,
        time  
    ]])

    # Predict
    prediction = model.predict(input_data)
    st.write("🔍 Raw prediction:", prediction)

    # Some Keras models output probabilities; some output labels
    # If your model outputs a probability:
    if prediction.shape[-1] == 1:
        prob = prediction[0][0]
        label = 1 if prob >= 0.5 else 0
    else:
        label = np.argmax(prediction[0])
        prob = prediction[0][label]

    # Display result
    st.markdown("### 🧠 Prediction Result:")
    st.success(f"{'🔴 High Risk' if label == 1 else '🟢 Low Risk'} of Heart Failure (Confidence: {prob:.2f})")

