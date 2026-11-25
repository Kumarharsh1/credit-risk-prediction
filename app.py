import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configure App Page
st.set_page_config(
    page_title="Credit Risk & Loan Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Prediction Platform (Real ML Model)")
st.write("Provide the applicant parameters to trigger predictions from our trained Random Forest model.")

# Load the actual trained pipeline
@st.cache_resource
def load_artifacts():
    with open('credit_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

try:
    model, model_features = load_artifacts()
    st.success("✅ Pre-trained Random Forest model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# Inputs
st.sidebar.header("Applicant Profile")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender (0=Female, 1=Male, 2=Other)", [0, 1, 2])
marital_status = st.sidebar.selectbox("Marital Status (0=Divorced, 1=Married, 2=Single, 3=Widowed)", [0, 1, 2, 3])
city_tier = st.sidebar.selectbox("City Tier (0=Tier 1, 1=Tier 2, 2=Tier 3)", [0, 1, 2])
state = st.sidebar.slider("State (Encoded Index)", 0, 30, 10)
employment_type = st.sidebar.selectbox("Employment Type (Encoded)", [0, 1, 2, 3, 4])

bank_account_vintage_years = st.sidebar.slider("Bank Account Vintage (Years)", 0.0, 20.0, 3.0, 0.1)
monthly_income_inr = st.sidebar.number_input("Monthly Income (INR)", value=50000)
existing_loans_count = st.sidebar.number_input("Existing Loans Count", value=1)
existing_emi_monthly_inr = st.sidebar.number_input("Existing Monthly EMI (INR)", value=4000)
credit_utilization_ratio_pct = st.sidebar.slider("Credit Utilization (%)", 0.0, 100.0, 30.0)
num_credit_inquiries_last_6m = st.sidebar.number_input("Inquiries Last 6M", value=0)
late_payments_last_12m = st.sidebar.number_input("Late Payments Last 12M", value=0)
loan_amount_requested_inr = st.sidebar.number_input("Loan Amount Requested (INR)", value=200000)
loan_tenure_months = st.sidebar.number_input("Loan Tenure (Months)", value=36)
loan_purpose = st.sidebar.selectbox("Loan Purpose (Encoded)", [0, 1, 2, 3, 4, 5, 6, 7])
collateral_provided = st.sidebar.selectbox("Collateral (0=No, 1=Yes)", [0, 1])
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 700)
interest_rate_offered_pct = st.sidebar.slider("Interest Rate Offered (%)", 5.0, 40.0, 14.5)
default_risk_score = st.sidebar.slider("Default Risk Score", 0.0, 10.0, 2.0)

if st.button("Predict Default & Credit Score"):
    # Map inputs directly to model features
    input_data = pd.DataFrame([[
        age, gender, marital_status, city_tier, state, employment_type,
        bank_account_vintage_years, monthly_income_inr, existing_loans_count, existing_emi_monthly_inr,
        credit_utilization_ratio_pct, num_credit_inquiries_last_6m, late_payments_last_12m, 
        loan_amount_requested_inr, loan_tenure_months, loan_purpose, collateral_provided, 
        cibil_score, interest_rate_offered_pct, default_risk_score
    ]], columns=model_features)
    
    # Real Prediction using machine learning!
    prob = model.predict_proba(input_data)[0][1]
    credit_score = int(850 - (prob * 550))
    
    if prob < 0.3:
        risk_cat = "Low Risk"
        recommendation = "✅ Approved / Eligible for Loan"
        color = "green"
    elif prob < 0.7:
        risk_cat = "Medium Risk"
        recommendation = "⚠️ Review Needed / Manual Underwriting"
        color = "orange"
    else:
        risk_cat = "High Risk"
        recommendation = "❌ Declined"
        color = "red"
        
    st.markdown("--- ")
    st.subheader("📊 Model Evaluation Output")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Generated Credit Score", f"{credit_score} / 850")
    with col2:
        st.metric("ML Predicted Default Prob", f"{prob*100:.2f}%")
    with col3:
        st.markdown(f"**Risk Status**: <span style='color:{color}; font-size:20px; font-weight:bold;'>{risk_cat}</span>", unsafe_allow_html=True)
        
    st.info(f"**Lending Recommendation**: {recommendation}")
