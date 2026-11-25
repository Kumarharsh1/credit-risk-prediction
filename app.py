"""
Credit Risk & Loan Default Prediction Platform (Streamlit + CatBoost)

Run:  streamlit run app.py

Files expected next to this script:
    credit_risk_model.pkl   trained CatBoostClassifier
    model_features.pkl      list of feature names, in training order
Optional (written by train_model.py):
    encoders.json           category -> integer code used during training
    risk_thresholds.json    probability cut-offs for Low / Medium / High risk
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk & Loan Default Prediction",
    page_icon="💳",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
ENCODERS_PATH = BASE_DIR / "encoders.json"
THRESHOLDS_PATH = BASE_DIR / "risk_thresholds.json"

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# Fallback label -> code maps. They follow sklearn LabelEncoder's convention
# (alphabetical order). If train_model.py produced encoders.json, that file is
# used instead, so the app always matches the training data.
DEFAULT_ENCODERS = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "marital_status": {"Divorced": 0, "Married": 1, "Single": 2, "Widowed": 3},
    "city_tier": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2},
    "employment_type": {
        "Salaried - Government": 0,
        "Salaried - PSU": 1,
        "Salaried - Private": 2,
        "Self-Employed Professional": 3,
        "Unemployed": 4,
    },
    "loan_purpose": {
        "Business": 0, "Education": 1, "Home": 2, "Medical": 3,
        "Other": 4, "Personal": 5, "Vehicle": 6, "Wedding": 7,
    },
    "collateral_provided": {"No": 0, "Yes": 1},
}

# The model outputs LOW probabilities (defaults are rare in the training data),
# so the old 0.30 / 0.70 cut-offs labelled almost everyone "Low Risk".
# These fallbacks are used only if risk_thresholds.json is missing.
DEFAULT_THRESHOLDS = {"low_max": 0.05, "medium_max": 0.15}

# Scorecard scaling for the 300-900 score: BASE_SCORE at BASE_ODDS (good:bad),
# and the score changes by PDO points every time the odds double.
BASE_SCORE, BASE_ODDS, PDO = 600, 30, 40


# --------------------------------------------------------------------------
# LOADERS
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = list(pickle.load(f))

    encoders = {k: dict(v) for k, v in DEFAULT_ENCODERS.items()}
    if ENCODERS_PATH.exists():
        with open(ENCODERS_PATH) as f:
            encoders.update(json.load(f))

    thresholds = dict(DEFAULT_THRESHOLDS)
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            thresholds.update(json.load(f))

    return model, features, encoders, thresholds


try:
    model, model_features, encoders, thresholds = load_artifacts()
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.stop()


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def risk_score(prob: float) -> int:
    """Convert default probability to a 300-900 scorecard-style score."""
    prob = float(np.clip(prob, 1e-4, 1 - 1e-4))
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    score = offset + factor * np.log((1 - prob) / prob)
    return int(np.clip(round(score), 300, 900))


def risk_band(prob: float):
    if prob < thresholds["low_max"]:
        return "Low Risk", "✅ Eligible - standard approval process", "green"
    if prob < thresholds["medium_max"]:
        return "Medium Risk", "⚠️ Manual underwriting recommended", "orange"
    return "High Risk", "❌ High default risk - decline or require collateral", "red"


def monthly_emi(principal: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 1200.0
    if months <= 0:
        return 0.0
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def select_encoded(label, key, help_text=None):
    """Selectbox that shows readable labels and returns the model's code."""
    options = list(encoders[key].keys())
    choice = st.sidebar.selectbox(label, options, help=help_text)
    return encoders[key][choice]


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("💳 Credit Risk & Loan Default Prediction")
st.write(
    "Estimate an applicant's probability of default with a trained CatBoost "
    "model, convert it to a risk score, and get a risk band."
)
st.caption(
    "Demo / educational tool. Not a substitute for a real credit bureau score "
    "or lending policy."
)

# --------------------------------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------------------------------
st.sidebar.header("Applicant Profile")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
gender = select_encoded("Gender", "gender")
marital_status = select_encoded("Marital Status", "marital_status")
city_tier = select_encoded("City Tier", "city_tier")
state = st.sidebar.slider("State (encoded index)", 0, 30, 10)
employment_type = select_encoded("Employment Type", "employment_type")

st.sidebar.header("Financial Profile")
bank_account_vintage_years = st.sidebar.slider(
    "Bank Account Vintage (years)", 0.0, 20.0, 3.0, 0.1
)
monthly_income_inr = st.sidebar.number_input(
    "Monthly Income (INR)", min_value=1000, value=50000, step=1000
)
existing_loans_count = st.sidebar.number_input(
    "Existing Loans (count)", min_value=0, max_value=50, value=1
)
existing_emi_monthly_inr = st.sidebar.number_input(
    "Existing Monthly EMI (INR)", min_value=0, value=4000, step=500
)
credit_utilization_ratio_pct = st.sidebar.slider(
    "Credit Utilization (%)", 0.0, 100.0, 30.0
)
num_credit_inquiries_last_6m = st.sidebar.number_input(
    "Credit Inquiries (last 6 months)", min_value=0, max_value=50, value=0
)
late_payments_last_12m = st.sidebar.number_input(
    "Late Payments (last 12 months)", min_value=0, max_value=50, value=0
)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 700)

st.sidebar.header("Loan Request")
loan_amount_requested_inr = st.sidebar.number_input(
    "Loan Amount Requested (INR)", min_value=10000, value=200000, step=10000
)
loan_tenure_months = st.sidebar.number_input(
    "Loan Tenure (months)", min_value=3, max_value=360, value=36
)
loan_purpose = select_encoded("Loan Purpose", "loan_purpose")
collateral_provided = select_encoded("Collateral Provided", "collateral_provided")
interest_rate_offered_pct = st.sidebar.slider(
    "Interest Rate Offered (%)", 5.0, 40.0, 14.5
)

# Only ask for the internal risk score if the loaded model actually uses it.
default_risk_score = 2.0
if "default_risk_score" in model_features:
    default_risk_score = st.sidebar.slider(
        "Internal Default Risk Score (0-10)",
        0.0, 10.0, 2.0,
        help="Pre-computed internal score. It is the model's strongest input.",
    )

# --------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------
if st.button("Predict Default Risk", type="primary"):

    inputs = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "city_tier": city_tier,
        "state": state,
        "employment_type": employment_type,
        "bank_account_vintage_years": bank_account_vintage_years,
        "monthly_income_inr": monthly_income_inr,
        "existing_loans_count": existing_loans_count,
        "existing_emi_monthly_inr": existing_emi_monthly_inr,
        "credit_utilization_ratio_pct": credit_utilization_ratio_pct,
        "num_credit_inquiries_last_6m": num_credit_inquiries_last_6m,
        "late_payments_last_12m": late_payments_last_12m,
        "loan_amount_requested_inr": loan_amount_requested_inr,
        "loan_tenure_months": loan_tenure_months,
        "loan_purpose": loan_purpose,
        "collateral_provided": collateral_provided,
        "cibil_score": cibil_score,
        "interest_rate_offered_pct": interest_rate_offered_pct,
        "default_risk_score": default_risk_score,
    }

    missing = [c for c in model_features if c not in inputs]
    if missing:
        st.error(f"Model expects features the app does not collect: {missing}")
        st.stop()

    # Select by NAME, in the model's own order, so a column can never be
    # silently placed under the wrong feature.
    input_df = pd.DataFrame([inputs])[model_features]

    prob = float(model.predict_proba(input_df)[0][1])
    score = risk_score(prob)
    band, recommendation, color = risk_band(prob)

    st.markdown("---")
    st.subheader("📊 Assessment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score (300-900)", score)
    c2.metric("Default Probability", f"{prob * 100:.2f}%")
    c3.markdown(
        f"<h3 style='color:{color};margin-top:0'>{band}</h3>",
        unsafe_allow_html=True,
    )
    st.info(f"**Recommendation:** {recommendation}")
    st.caption(
        f"Bands: Low < {thresholds['low_max']:.0%}, "
        f"Medium < {thresholds['medium_max']:.0%}, High above. "
        "Probabilities are low overall because defaults are rare in the "
        "training data, so judge a result relative to these bands."
    )

    # Affordability check (rule-based, shown for context only)
    new_emi = monthly_emi(
        loan_amount_requested_inr, interest_rate_offered_pct, int(loan_tenure_months)
    )
    foir = (existing_emi_monthly_inr + new_emi) / monthly_income_inr
    a1, a2, a3 = st.columns(3)
    a1.metric("Estimated New EMI", f"₹{new_emi:,.0f}")
    a2.metric("Existing EMI / Income", f"{existing_emi_monthly_inr / monthly_income_inr:.0%}")
    a3.metric("Total EMI / Income", f"{foir:.0%}")
    if foir > 0.50:
        st.warning(
            "Total EMI would exceed 50% of monthly income - a common "
            "affordability limit. This check does not change the model output."
        )

    # Top drivers for this applicant (SHAP values from CatBoost)
    try:
        from catboost import Pool

        shap = model.get_feature_importance(Pool(input_df), type="ShapValues")[0][:-1]
        drivers = (
            pd.DataFrame({
                "Feature": model_features,
                "Value": input_df.iloc[0].values,
                "Effect on default risk": shap,
            })
            .assign(abs_effect=lambda d: d["Effect on default risk"].abs())
            .sort_values("abs_effect", ascending=False)
            .head(6)
            .drop(columns="abs_effect")
        )
        drivers["Direction"] = np.where(
            drivers["Effect on default risk"] > 0, "↑ raises risk", "↓ lowers risk"
        )
        st.subheader("Why this result? (top drivers)")
        st.dataframe(
            drivers.style.format({"Effect on default risk": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        pass  # explanation is optional

    with st.expander("Model input (for debugging)"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)
