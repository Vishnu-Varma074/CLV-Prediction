# ================================
# Customer Lifetime Value (CLV) Prediction
# ================================

# ---------- Imports ----------
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Load Data ----------
df = pd.read_csv("../data/online_retail_II.csv")

# ---------- Data Cleaning ----------
df = df[df["Customer ID"].notna()]
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# ---------- Feature Engineering ----------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["Price"]

# ---------- TIME-BASED SPLIT ----------
split_date = df["InvoiceDate"].quantile(0.7)

past_df = df[df["InvoiceDate"] <= split_date]
future_df = df[df["InvoiceDate"] > split_date]

# ---------- CREATE RFM (PAST DATA) ----------
today = past_df["InvoiceDate"].max()

rfm = past_df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (today - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]

# ---------- CREATE TRUE CLV (FUTURE DATA) ----------
future_clv = future_df.groupby("Customer ID")["TotalPrice"].sum().reset_index()
future_clv.columns = ["customer_id", "clv"]

# ---------- MERGE ----------
rfm = rfm.merge(future_clv, on="customer_id", how="left")
rfm["clv"] = rfm["clv"].fillna(0)

# ---------- FEATURES & TARGET ----------
X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["clv"]

# ---------- TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- SCALING (CORRECT WAY) ----------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- MODEL ----------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ---------- EVALUATION ----------
preds = rf_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("MAE:", mae)
print("RMSE:", rmse)

# ---------- FEATURE IMPORTANCE ----------
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)

# ---------- SAVE MODEL ----------
with open("rf_clv_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model and scaler saved successfully!")
