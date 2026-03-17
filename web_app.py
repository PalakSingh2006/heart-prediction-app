import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load('heart_model.pkl')

st.set_page_config(page_title="Heart Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")

st.markdown("### Enter Patient Details")

# Inputs
age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
cp = ["ATA","NAP","ASY","TA"].index(cp)

bp = st.number_input("Resting BP", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)

fbs = st.selectbox("Fasting Blood Sugar", [0,1])

ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
ecg = ["Normal","ST","LVH"].index(ecg)

hr = st.number_input("Max Heart Rate", 60, 220)

angina = st.selectbox("Exercise Angina", ["No", "Yes"])
angina = 1 if angina == "Yes" else 0

oldpeak = st.number_input("Oldpeak", 0.0, 6.0)

slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
slope = ["Up","Flat","Down"].index(slope)

# Prediction
if st.button("Predict"):
    data = np.array([[age, sex, cp, bp, chol, fbs, ecg, hr, angina, oldpeak, slope]])

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected\n\nProbability: {prob*100:.2f}%")
    else:
        st.success(f"✅ No Heart Disease\n\nProbability: {(1-prob)*100:.2f}%")

    # Graph
    fig, ax = plt.subplots()
    ax.bar(["No Disease", "Disease"], [1-prob, prob])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)