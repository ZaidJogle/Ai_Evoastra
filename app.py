import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ─────────────────────────
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# ── Load models ─────────────────────────
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

# ── Risk Label ─────────────────────────
def risk_label(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟠 Medium Risk"
    else:
        return "🔴 High Risk"

# ── UI ─────────────────────────
st.title("💳 Credit Risk Predictor")

tab_gmsc, tab_amex = st.tabs(["🏦 GMSC", "🏢 AMEX"])

# ═══════════════════════════════════════
# GMSC TAB (NO CHANGE - WORKING)
# ═══════════════════════════════════════
with tab_gmsc:

    st.subheader("GMSC Risk Prediction")

    age = st.number_input("Age", 18, 100, 40)
    income = st.number_input("Monthly Income", 0, 100000, 5000)
    debt = st.slider("Debt Ratio", 0.0, 10.0, 0.5)

    gmsc_input = pd.DataFrame([{
        "age": age,
        "MonthlyIncome": income,
        "DebtRatio": debt,
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfDependents": 0,
        "debt_ratio": debt,
        "income_to_debt": income/(debt+1)
    }])

    if st.button("Predict GMSC"):
        model = models["gmsc"]
        if model:
            prob = model.predict_proba(gmsc_input)[0][1]
            st.success(f"Risk: {risk_label(prob)} ({prob:.2%})")
        else:
            st.error("Model not found")

# ═══════════════════════════════════════
# AMEX TAB (FULLY FIXED)
# ═══════════════════════════════════════
with tab_amex:

    st.subheader("AMEX Default Prediction")

    model = models["amex"]

    if model is None:
        st.error("AMEX model not found!")
    else:

        # ✅ Get EXACT features from model
        try:
            FEATURES = list(model.feature_names_in_)
        except:
            st.error("Model missing feature names. Retrain with DataFrame!")
            st.stop()

        st.info("Fill customer details below")

        # Auto Fill
        if st.button("⚡ Auto Fill Sample"):
            for f in FEATURES:
                st.session_state[f] = np.random.uniform(0, 1000)

        # Input UI
        amex_vals = {}
        for f in FEATURES:
            amex_vals[f] = st.number_input(
                f,
                value=st.session_state.get(f, 0.0),
                key=f
            )

        amex_input = pd.DataFrame([amex_vals])

        # ── Prediction ─────────────────
        if st.button("🔍 Predict AMEX Risk"):

            try:
                # ✅ Ensure correct columns
                for col in FEATURES:
                    if col not in amex_input:
                        amex_input[col] = 0

                amex_input = amex_input[FEATURES]

                prob = model.predict_proba(amex_input)[0][1]
                label = risk_label(prob)

                # Result
                st.subheader("📊 Result")
                col1, col2 = st.columns(2)

                col1.metric("Probability", f"{prob:.2%}")
                col2.markdown(f"### {label}")

                st.progress(prob)

                # ── Explanation ─────────────────
                st.subheader("🧠 Explanation")

                if prob > 0.7:
                    st.write("• High risk due to unstable financial behavior.")
                elif prob > 0.3:
                    st.write("• Moderate risk detected.")
                else:
                    st.write("• Low risk, stable customer.")

                st.write("• Based on payment patterns, credit usage and history.")

            except Exception as e:
                st.error("Prediction failed due to feature mismatch")
                st.write(e)
