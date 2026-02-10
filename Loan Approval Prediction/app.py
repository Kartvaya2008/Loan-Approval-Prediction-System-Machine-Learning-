import streamlit as st
import pandas as pd
import pickle as pk
import os

# -----------------------------
# Load model safely (cloud-safe)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pk.load(open(model_path, "rb"))
scaler = pk.load(open(scaler_path, "rb"))

# -----------------------------
# UI
# -----------------------------
st.title("🏦 Loan Prediction App")

no_of_dep = st.slider("Choose No of dependents", 0, 5)
grad = st.selectbox("Choose Education", ["Graduated", "Not Graduated"])
self_emp = st.selectbox("Self Employed ?", ["Yes", "No"])
annual_income = st.slider("Choose Annual Income", 0, 10_000_000)
loan_amount = st.slider("Choose Loan Amount", 0, 10_000_000)
loan_dur = st.slider("Choose Loan Duration (years)", 0, 20)
cibil = st.slider("Choose Cibil Score", 0, 1000)
assets = st.slider("Choose Assets", 0, 10_000_000)

# -----------------------------
# Encode inputs
# -----------------------------
grad_s = 0 if grad == "Graduated" else 1
emp_s = 1 if self_emp == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    pred_data = pd.DataFrame(
        [[
            no_of_dep,
            grad_s,
            emp_s,
            annual_income,
            loan_amount,
            loan_dur,
            cibil,
            assets
        ]],
        columns=[
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "Assets"
        ],
    )

    pred_scaled = scaler.transform(pred_data)
    prediction = model.predict(pred_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan is Approved")
    else:
        st.error("❌ Loan is Rejected")
