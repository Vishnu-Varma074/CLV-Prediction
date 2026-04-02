# ================================
# Customer Lifetime Value (CLV) Prediction – Backend Only
# ================================

# ---------- Imports ----------
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------- Load Data ----------
df = pd.read_excel(
    "online_retail_II.xlsx",
    sheet_name="Year 2010-2011",
    engine="openpyxl"
)

# Remove rows with missing Customer ID
df = df[df["Customer ID"].notna()]

# Remove cancelled transactions (Invoice starting with 'C')
df = df[~df["Invoice"].astype(str).str.startswith("C")]

# Remove negative or zero quantities
df = df[df["Quantity"] > 0]

# Remove zero or negative prices
df = df[df["Price"] > 0]

# ---------- Feature Engineering (RFM) ----------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["Price"]

today = pd.Timestamp.today()

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (today - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.rename(columns={
    "Customer ID": "customer_id",
    "InvoiceDate": "Recency",
    "Invoice": "Frequency",
    "TotalPrice": "Monetary"
}, inplace=True)

rfm["clv"] = rfm["Monetary"]

df = rfm.copy()


# ---------- Data Preprocessing ----------
# Drop non-required columns
df.drop(columns=["customer_id"], inplace=True)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(columns=["clv"])
y = df["clv"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ---------- Model Training ----------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# ---------- Model Evaluation ----------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return mae, rmse


lr_mae, lr_rmse = evaluate(lr_model, X_test, y_test)
rf_mae, rf_rmse = evaluate(rf_model, X_test, y_test)

print("Linear Regression MAE:", lr_mae)
print("Linear Regression RMSE:", lr_rmse)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)


# ---------- Cross Validation ----------
cv_scores = cross_val_score(
    rf_model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error"
)

print("Cross Validation MAE:", -cv_scores.mean())


# ---------- Feature Importance ----------
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(feature_importance)