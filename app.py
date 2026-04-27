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

# ── Risk label ─────────────────────────
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
# GMSC TAB (FIXED)
# ═══════════════════════════════════════
with tab_gmsc:

    st.subheader("GMSC Risk Prediction")

    # Inputs
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

    model_choice = st.selectbox("Select Model", ["gmsc", "gmsc_xgb"])

    if st.button("🔍 Predict GMSC Risk"):
        model = models.get(model_choice)

        if model is None:
            st.error("Model not found!")
        else:
            try:
                # ✅ FIX: align features
                features = model.feature_names_in_

                for col in features:
                    if col not in gmsc_input:
                        gmsc_input[col] = 0

                gmsc_input = gmsc_input[features]

                prob = model.predict_proba(gmsc_input)[0][1]

                st.metric("Default Probability", f"{prob:.2%}")
                st.markdown(f"### {risk_label(prob)}")
                st.progress(float(prob))

            except Exception as e:
                st.error("GMSC Prediction Error")
                st.write(e)
# ── GMSC Explanation ─────────────────
st.subheader("🧠 Prediction Explanation")

explanation = []

if prob > 0.7:
    explanation.append("🔴 High risk: Customer is likely to default.")
elif prob > 0.3:
    explanation.append("🟠 Moderate risk: Some financial warning signs detected.")
else:
    explanation.append("🟢 Low risk: Customer is financially stable.")

# Feature-based reasoning
if gmsc_input["NumberOfTimes90DaysLate"][0] > 0:
    explanation.append("⚠️ History of 90+ days late payments.")

if gmsc_input["DebtRatio"][0] > 0.5:
    explanation.append("⚠️ High debt compared to income.")

if gmsc_input["RevolvingUtilizationOfUnsecuredLines"][0] > 0.7:
    explanation.append("⚠️ High credit utilization.")

if gmsc_input["MonthlyIncome"][0] < 3000:
    explanation.append("⚠️ Low monthly income may affect repayment ability.")

# Show explanation
for line in explanation:
    st.write("•", line)
# ═══════════════════════════════════════
# AMEX TAB (FULLY FIXED)
# ═══════════════════════════════════════
with tab_amex:

    st.subheader("AMEX Default Prediction")

    model_choice = st.selectbox("Select Model", ["amex", "amex_xgb"])
    model = models.get(model_choice)

    if model is None:
        st.error("Model not found!")
    else:
        try:
            FEATURES = list(model.feature_names_in_)
        except:
            st.error("Model missing feature names!")
            st.stop()

        st.info("Enter customer details")

        # Auto-fill
        if st.button("⚡ Auto Fill Sample"):
            for f in FEATURES:
                st.session_state[f] = np.random.uniform(0, 1000)

        amex_vals = {}

        # Dynamic input
        for f in FEATURES:
            amex_vals[f] = st.number_input(
                f,
                value=st.session_state.get(f, 0.0),
                key=f
            )

        amex_input = pd.DataFrame([amex_vals])

        if st.button("🔍 Predict"):

    # 1️⃣ Prediction first
    prob = model.predict_proba(input_data)[0][1]
    prob = float(prob)

    # 2️⃣ Show result
    st.metric("Probability", f"{prob:.2%}")
    st.progress(prob)

    # 3️⃣ THEN explanation
    st.subheader("🧠 Prediction Explanation")

    if prob > 0.7:
        st.write("🔴 High risk")
    elif prob > 0.3:
        st.write("🟠 Medium risk")
    else:
        st.write("🟢 Low risk")
                # Explanation
                st.subheader("🧠 Explanation")
                if prob > 0.7:
                    st.write("• High risk due to unstable financial behavior.")
                elif prob > 0.3:
                    st.write("• Moderate risk detected.")
                else:
                    st.write("• Low risk, stable customer.")

            except Exception as e:
                st.error("AMEX Prediction Error")
                st.write(e)

# Footer
st.divider()
st.caption("ML Models: Logistic Regression | XGBoost | Ensemble")
