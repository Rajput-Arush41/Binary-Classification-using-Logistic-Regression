"""
Streamlit deployment app for the Telco Customer Churn Logistic Regression model.

Run locally with:
    streamlit run streamlit_app.py

Expects the trained model artifact at ../outputs/logistic_regression_model.pkl
(adjust MODEL_PATH below if you move files around).
"""
import streamlit as st
import pandas as pd
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "logistic_regression_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["scaler"], artifact["feature_columns"], artifact["numerical_features"]

model, scaler, feature_columns, numerical_features = load_model()

st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate their probability of churning, using a Logistic Regression model trained on the Telco Customer Churn dataset.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure) * 70.0)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    raw = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }])

    categorical_cols = [c for c in raw.columns if c not in numerical_features]
    encoded = pd.get_dummies(raw, columns=categorical_cols)

    # Align to the exact training feature schema (fills missing dummy columns with 0)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[numerical_features] = scaler.transform(encoded[numerical_features])

    proba = model.predict_proba(encoded)[0, 1]
    pred = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"

    st.subheader("Result")
    st.metric("Churn Probability", f"{proba*100:.1f}%")
    if proba >= 0.5:
        st.error(f"⚠️ {pred}")
    else:
        st.success(f"✅ {pred}")
    st.progress(min(int(proba * 100), 100))
