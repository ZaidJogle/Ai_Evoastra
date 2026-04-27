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
        "amex": "amex_model.pkl",
    }
    for key, fname in files.items():
        try:
            with open(fname, "rb") as f:
                models[key] = pickle.load(f)
        except:
            models[key] = None
    return models

models = load_models()

# ── Risk Label ─────────────────────────
def risk_label(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟠 Medium Risk"
    else:
        return "🔴 High Risk"

# ── Feature Labels (AMEX UI) ───────────
FEATURE_LABELS = {
    "P_2": "Recent Payment Amount",
    "D_39": "Payment Delay Score",
    "B_1": "Outstanding Balance",
    "B_2": "Credit Utilization",
    "R_1": "Risk Indicator",
    "S_3": "Spending Pattern",
    "D_41": "Recent Delay Count",
}

# ── UI ─────────────────────────
st.title("💳 Credit Risk Predictor")
st.caption("Check if a customer is likely to default based on financial behavior")

tab1, tab2 = st.tabs(["🏦 GMSC (Basic Users)", "🏢 AMEX (Advanced Users)"])

# ═══════════════════════════════
# GMSC TAB
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
                # Align features
                for col in model.feature_names_in_:
                    if col not in gmsc_input:
                        gmsc_input[col] = 0

                gmsc_input = gmsc_input[model.feature_names_in_]

                prob = float(model.predict_proba(gmsc_input)[0][1])

                st.success("✅ Prediction Complete")
                st.metric("📊 Default Probability", f"{prob:.2%}")
                st.progress(prob)

                st.subheader("🧠 Explanation")

                reasons = []

                if debt > 0.6:
                    reasons.append("High debt ratio")

                if late_payments > 3:
                    reasons.append("Too many late payments")

                if income < 3000:
                    reasons.append("Low income")

                if credit_lines < 2:
                    reasons.append("Limited credit history")

                if dependents > 3:
                    reasons.append("High financial burden")

                if len(reasons) == 0:
                    st.success("Customer profile looks financially stable ✅")
                else:
                    st.write("📌 Key Risk Factors:")
                    for r in reasons:
                        st.write(f"• {r}")

            except Exception as e:
                st.error("Prediction error")
                st.write(e)
        else:
            st.error("Model not found")

# ═══════════════════════════════
# AMEX TAB
# ═══════════════════════════════
with tab2:

    st.subheader("📊 Advanced Customer Financial Data")

    model = models["amex"]

    if model is None:
        st.error("Model not found")
    else:
        # Reduced features (only 7 inputs)
        FEATURES = [
            "P_2",
            "D_39",
            "B_1",
            "B_2",
            "R_1",
            "S_3",
            "D_41"
        ]

        if st.button("⚡ Fill Sample Data"):
            for f in FEATURES:
                st.session_state[f] = float(np.random.uniform(0, 1000))

        amex_vals = {}

        for f in FEATURES:
            amex_vals[f] = st.number_input(
                FEATURE_LABELS.get(f, f),
                value=float(st.session_state.get(f, 0.0)),
                key=f
            )

        if st.button("🔍 Check Risk (AMEX)"):
            try:
                # Create full input with all model features
                full_input = pd.DataFrame([{}])

                for col in model.feature_names_in_:
                    if col in amex_vals:
                        full_input[col] = amex_vals[col]
                    else:
                        full_input[col] = 0

                full_input = full_input[model.feature_names_in_]

                prob = float(model.predict_proba(full_input)[0][1])

                st.success("✅ Prediction Complete")
                st.metric("📊 Default Probability", f"{prob:.2%}")
                st.progress(prob)

                st.subheader("🧠 Explanation")

                reasons = []

                if amex_vals["D_39"] > 500:
                    reasons.append("High payment delay score")

                if amex_vals["B_2"] > 0.8:
                    reasons.append("High credit utilization")

                if amex_vals["D_41"] > 2:
                    reasons.append("Frequent delays")

                if amex_vals["P_2"] < 100:
                    reasons.append("Low recent payment")

                if len(reasons) == 0:
                    st.success("Customer financial behavior looks stable ✅")
                else:
                    st.write("📌 Key Risk Factors:")
                    for r in reasons:
                        st.write(f"• {r}")

            except Exception as e:
                st.error("Prediction error")
                st.write(e)

# Footer
st.divider()
st.caption("💡 Built using Machine Learning (Logistic Regression + XGBoost)")
