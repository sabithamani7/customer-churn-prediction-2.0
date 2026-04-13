import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction System")

st.write("Enter Customer Details")

# ==============================
# INPUT FIELDS (8 FEATURES)
# ==============================

tenure = st.number_input("Tenure (Months)", min_value=0)

monthly_charges = st.number_input("Monthly Charges")

total_charges = st.number_input("Total Charges")

contract = st.selectbox("Contract Type", ["Month-to-Month","One Year","Two Year"])

internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

online_security = st.selectbox("Online Security", ["Yes","No"])

tech_support = st.selectbox("Tech Support", ["Yes","No"])

paperless_billing = st.selectbox("Paperless Billing", ["Yes","No"])

# ==============================
# ENCODING
# ==============================

contract_map = {
    "Month-to-Month":0,
    "One Year":1,
    "Two Year":2
}

internet_map = {
    "DSL":0,
    "Fiber optic":1,
    "No":2
}

yes_no = {
    "No":0,
    "Yes":1
}

contract = contract_map[contract]
internet_service = internet_map[internet_service]
online_security = yes_no[online_security]
tech_support = yes_no[tech_support]
paperless_billing = yes_no[paperless_billing]

# ==============================
# PREDICTION
# ==============================

if st.button("Predict Churn"):

    input_data = np.array([[tenure,
                            monthly_charges,
                            total_charges,
                            contract,
                            internet_service,
                            online_security,
                            tech_support,
                            paperless_billing]])

    # Scale input
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer will Stay")