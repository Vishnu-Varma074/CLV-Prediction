import streamlit as st
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="CLV Prediction", layout="centered")
st.title("📊 Customer Lifetime Value (CLV) Prediction")

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
@st.cache_resource
def load_model():
    with open("rf_clv_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("🎯 Enter Customer Details")

recency = st.number_input("Recency (days)", value=30)
frequency = st.number_input("Frequency (transactions)", value=5)
monetary = st.number_input("Monetary (total spend)", value=1000)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict CLV"):
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"💰 Predicted CLV: ₹ {round(prediction, 2)}")
