import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# ── Load Models ─────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "gmsc": "gmsc_model.pkl",
        "gmsc_xgb": "gmsc_xgb_model.pkl",
        "amex": "amex_model.pkl",
        "amex_xgb": "amex_xgb_model.pkl",
    }
    for key, fname in files.items():
        try:
            with open(fname, "rb") as f:
                models[key] = pickle.load(f)
        except:
            models[key] = None
    return models

models = load_models()

def risk_label(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟠 Medium Risk"
    else:
        return "🔴 High Risk"
FEATURE_LABELS = {
    "P_2": "Recent Payment Amount",
    "D_39": "Payment Delay Score",
    "B_1": "Outstanding Balance",
    "B_2": "Credit Utilization",
    "R_1": "Risk Indicator",
    "S_3": "Spending Pattern",
    "D_41": "Recent Delay Count",
    "B_3": "Balance Change",
    "D_42": "Missed Payment Indicator",
    "D_43": "Late Payment Trend",
    "D_44": "Payment Consistency",
    "B_4": "Balance Variation",
    "D_45": "Repayment Behaviour",
    "B_5": "Credit Usage Level",
    "R_2": "Risk Pattern",
    "D_46": "Payment Stability",
    "D_47": "Default Signal",
    "D_48": "Credit Health",
    "D_49": "Financial Stress",
    "B_6": "Outstanding Dues",
}
st.title("💳 Credit Risk Predictor")
st.caption("Check if a customer is likely to default based on financial behavior")

tab1, tab2 = st.tabs(["🏦 GMSC (Basic Users)", "🏢 AMEX (Advanced Users)"])

# ═══════════════════════════════
# GMSC TAB (User Friendly)
# ═══════════════════════════════
with tab1:

    st.subheader("📊 Basic Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 30)
        income = st.number_input("Monthly Income (₹)", 1000, 100000, 5000)
        dependents = st.number_input("Number of Dependents", 0, 10, 0)

    with col2:
        debt = st.slider("Debt Ratio", 0.0, 1.0, 0.3)
        late_payments = st.number_input("Number of Late Payments", 0, 20, 0)
        credit_lines = st.number_input("Open Credit Lines", 0, 20, 5)

    st.info("💡 More late payments + high debt = higher risk")

    gmsc_input = pd.DataFrame([{
        "age": age,
        "MonthlyIncome": income,
        "DebtRatio": debt,
        "RevolvingUtilizationOfUnsecuredLines": debt,
        "NumberOfTime30-59DaysPastDueNotWorse": late_payments,
        "NumberOfTime60-89DaysPastDueNotWorse": late_payments,
        "NumberOfTimes90DaysLate": late_payments,
        "NumberOfOpenCreditLinesAndLoans": credit_lines,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfDependents": dependents,
        "debt_ratio": debt,
        "income_to_debt": income/(debt+1)
    }])
    if st.button("🔍 Check Risk (GMSC)"):
        model = models["gmsc"]

        if model:
            try:
                features = model.feature_names_in_
                for col in features:
                    if col not in gmsc_input:
                        gmsc_input[col] = 0
                gmsc_input = gmsc_input[features]

                prob = float(model.predict_proba(gmsc_input)[0][1])

                st.success("✅ Prediction Complete")

                st.metric("📊 Default Probability", f"{prob:.2%}")
                st.progress(prob)

                st.subheader("🧠 What this means")

                if prob > 0.7:
                    st.error("🔴 High risk: Customer may default")
                elif prob > 0.3:
                    st.warning("🟠 Moderate risk: Be cautious")
                else:
                    st.success("🟢 Low risk: Customer is safe")

            except Exception as e:
                st.error("Error in prediction")
                st.write(e)

# ═══════════════════════════════
# AMEX TAB (Advanced Users)
# ═══════════════════════════════
with tab2:

    st.subheader("📊 Advanced Customer Financial Data")

    model = models["amex"]

    FEATURES = list(model.feature_names_in_)

    if st.button("⚡ Fill Sample Data"):
        for f in FEATURES:
            st.session_state[f] = np.random.uniform(0, 1000)

    amex_vals = {}

    # ✅ PLACE YOUR CODE HERE 👇
    for f in FEATURES:
        label = FEATURE_LABELS.get(f, "Financial Indicator")

        amex_vals[f] = st.number_input(
            label,
            value=st.session_state.get(f, 0.0),
            key=f
        )

    amex_input = pd.DataFrame([amex_vals])

        if st.button("🔍 Check Risk (AMEX)"):

            try:
                for col in FEATURES:
                    if col not in amex_input:
                        amex_input[col] = 0

                amex_input = amex_input[FEATURES]

                prob = float(model.predict_proba(amex_input)[0][1])

                st.success("✅ Prediction Complete")

                st.metric("📊 Default Probability", f"{prob:.2%}")
                st.progress(prob)

                st.subheader("🧠 Interpretation")

                if prob > 0.7:
                    st.error("🔴 High risk: Likely to default")
                elif prob > 0.3:
                    st.warning("🟠 Moderate risk")
                else:
                    st.success("🟢 Low risk")

            except Exception as e:
                st.error("Prediction error")
                st.write(e)

# Footer
st.divider()
st.caption("💡 Built using Machine Learning (Logistic Regression + XGBoost)")
