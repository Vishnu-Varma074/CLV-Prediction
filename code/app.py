'''import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="CLV Prediction", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) Prediction")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/online_retail_II.csv")

st.subheader("🔍 Raw Data")
st.dataframe(df.head())

# -------------------------------
# PREPROCESSING
# -------------------------------
df = df[df["Customer ID"].notna()]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["Price"]

today = pd.Timestamp.today()

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (today - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]
rfm["clv"] = rfm["Monetary"]

# -------------------------------
# FEATURES
# -------------------------------
X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["clv"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODELS
# -------------------------------
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate(model):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return mae, rmse

lr_mae, lr_rmse = evaluate(lr)
rf_mae, rf_rmse = evaluate(rf)

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Linear Regression MAE", round(lr_mae, 2))
    st.metric("Linear Regression RMSE", round(lr_rmse, 2))

with col2:
    st.metric("Random Forest MAE", round(rf_mae, 2))
    st.metric("Random Forest RMSE", round(rf_rmse, 2))

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("🔥 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.subheader("🎯 Predict CLV for New Customer")

recency = st.number_input("Recency (days)", value=30)
frequency = st.number_input("Frequency (transactions)", value=5)
monetary = st.number_input("Monetary (total spend)", value=1000)

if st.button("Predict CLV"):
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    
    prediction = rf.predict(input_scaled)[0]
    
    st.success(f"Predicted CLV: {round(prediction, 2)}")'''

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