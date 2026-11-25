import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Credit Risk & Loan Default Prediction Platform",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk & Loan Default Prediction Platform")
st.write(
    "Evaluate applicant creditworthiness, predict default probability, "
    "generate credit scores, and assess borrower risk using a trained CatBoost model."
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():
    with open("credit_risk_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model_features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, features


try:
    model, model_features = load_artifacts()
    st.success("✅ CatBoost credit risk model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Applicant Profile")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Divorced", "Married", "Single", "Widowed"]
)

city_tier = st.sidebar.selectbox(
    "City Tier",
    ["Tier 1", "Tier 2", "Tier 3"]
)

state = st.sidebar.slider(
    "State Index",
    0,
    30,
    10
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    [
        "Salaried - PSU",
        "Salaried - Private",
        "Salaried - Government",
        "Self-Employed Professional",
        "Unemployed"
    ]
)

bank_account_vintage_years = st.sidebar.slider(
    "Bank Account Vintage (Years)",
    0.0,
    20.0,
    3.0,
    0.1
)

monthly_income_inr = st.sidebar.number_input(
    "Monthly Income (INR)",
    value=50000
)

existing_loans_count = st.sidebar.number_input(
    "Existing Loans Count",
    value=1
)

existing_emi_monthly_inr = st.sidebar.number_input(
    "Existing Monthly EMI (INR)",
    value=4000
)

credit_utilization_ratio_pct = st.sidebar.slider(
    "Credit Utilization (%)",
    0.0,
    100.0,
    30.0
)

num_credit_inquiries_last_6m = st.sidebar.number_input(
    "Credit Inquiries (Last 6 Months)",
    value=0
)

late_payments_last_12m = st.sidebar.number_input(
    "Late Payments (Last 12 Months)",
    value=0
)

loan_amount_requested_inr = st.sidebar.number_input(
    "Loan Amount Requested (INR)",
    value=200000
)

loan_tenure_months = st.sidebar.number_input(
    "Loan Tenure (Months)",
    value=36
)

loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "Business",
        "Education",
        "Home",
        "Medical",
        "Other",
        "Personal",
        "Vehicle",
        "Wedding"
    ]
)

collateral_provided = st.sidebar.selectbox(
    "Collateral Provided",
    ["No", "Yes"]
)

cibil_score = st.sidebar.slider(
    "CIBIL Score",
    300,
    900,
    700
)

interest_rate_offered_pct = st.sidebar.slider(
    "Interest Rate Offered (%)",
    5.0,
    40.0,
    14.5
)

default_risk_score = st.sidebar.slider(
    "Default Risk Score",
    0.0,
    10.0,
    2.0
)

# =========================
# ENCODING MAPS
# =========================

gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

marital_map = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2,
    "Widowed": 3
}

city_map = {
    "Tier 1": 0,
    "Tier 2": 1,
    "Tier 3": 2
}

employment_map = {
    "Salaried - PSU": 0,
    "Salaried - Private": 1,
    "Salaried - Government": 2,
    "Self-Employed Professional": 3,
    "Unemployed": 4
}

loan_purpose_map = {
    "Business": 0,
    "Education": 1,
    "Home": 2,
    "Medical": 3,
    "Other": 4,
    "Personal": 5,
    "Vehicle": 6,
    "Wedding": 7
}

collateral_map = {
    "No": 0,
    "Yes": 1
}

# Convert categories
gender = gender_map[gender]
marital_status = marital_map[marital_status]
city_tier = city_map[city_tier]
employment_type = employment_map[employment_type]
loan_purpose = loan_purpose_map[loan_purpose]
collateral_provided = collateral_map[collateral_provided]

# =========================
# PREDICTION
# =========================

if st.button("Predict Default & Credit Score"):

    input_data = pd.DataFrame([[
        age,
        gender,
        marital_status,
        city_tier,
        state,
        employment_type,
        bank_account_vintage_years,
        monthly_income_inr,
        existing_loans_count,
        existing_emi_monthly_inr,
        credit_utilization_ratio_pct,
        num_credit_inquiries_last_6m,
        late_payments_last_12m,
        loan_amount_requested_inr,
        loan_tenure_months,
        loan_purpose,
        collateral_provided,
        cibil_score,
        interest_rate_offered_pct,
        default_risk_score
    ]], columns=model_features)

    prob = model.predict_proba(input_data)[0][1]

    credit_score = int(850 - (prob * 550))

    if prob < 0.30:
        risk_cat = "Low Risk"
        recommendation = "✅ Approved / Eligible for Loan"
        color = "green"

    elif prob < 0.70:
        risk_cat = "Medium Risk"
        recommendation = "⚠️ Requires Manual Review"
        color = "orange"

    else:
        risk_cat = "High Risk"
        recommendation = "❌ High Default Risk"
        color = "red"

    st.markdown("---")

    st.subheader("📊 Credit Risk Assessment Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Credit Score",
            f"{credit_score}/850"
        )

    with col2:
        st.metric(
            "Default Probability",
            f"{prob*100:.2f}%"
        )

    with col3:
        st.markdown(
            f"<h3 style='color:{color};'>{risk_cat}</h3>",
            unsafe_allow_html=True
        )

    st.info(
        f"**Lending Recommendation:** {recommendation}"
    )
