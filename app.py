import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ─────────────────────────
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# ── Load Models ─────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models["amex"] = pickle.load(open("amex_model.pkl", "rb"))
    except:
        models["amex"] = None
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

# ── Feature Labels ─────────────────────
FEATURE_LABELS = {
    "P_2": "Recent Payment Amount",
    "D_39": "Payment Delay Score",
    "B_1": "Outstanding Balance",
    "B_2": "Credit Utilization",
    "R_1": "Risk Indicator",
    "S_3": "Spending Pattern",
    "D_41": "Recent Delay Count",
    "B_3": "Balance Change",
    "D_42": "Missed Payment",
    "D_43": "Late Payment Trend",
    "D_44": "Payment Consistency",
    "B_4": "Balance Variation",
    "D_45": "Repayment Behaviour",
    "B_5": "Credit Usage",
    "R_2": "Risk Pattern",
    "D_46": "Payment Stability",
    "D_47": "Default Signal",
    "D_48": "Credit Health",
    "D_49": "Financial Stress",
    "B_6": "Outstanding Dues",
    "B_7": "Balance Risk",
    "B_8": "Utilization Pattern",
    "D_50": "Late Payment Frequency",
    "D_51": "Debt Pressure",
    "B_9": "Credit Behaviour",
    "R_3": "Risk Trend",
    "D_52": "Missed Payment Frequency",
    "P_3": "Previous Payment",
    "B_10": "Balance Trend",
    "D_53": "Default Warning",
    "S_5": "Spending Variation",
    "B_11": "Balance Consistency",
    "S_6": "Spending Stability",
    "D_54": "Financial Stability",
    "R_4": "Risk Escalation",
    "S_7": "Spending Risk",
    "B_12": "Credit Stability",
    "S_8": "Spending Intensity",
    "D_55": "Delay Severity",
    "D_56": "Risk Signal",
    "B_13": "Balance Exposure",
    "R_5": "Risk Index",
    "D_57": "Stress Level",
    "B_14": "Credit Burden",
    "D_58": "Payment Irregularity",
    "B_15": "Debt Load",
    "D_59": "Late Payment Risk",
    "D_60": "Credit Instability",
    "D_61": "Default Indicator",
    "B_16": "Final Balance Risk",
}

AMEX_FEATURES = list(FEATURE_LABELS.keys())

# ── UI ─────────────────────────
st.title("💳 Credit Risk Predictor")
st.subheader("📊 AmEx Customer Risk Analysis")

# Auto Fill
if st.button("⚡ Auto Fill Sample Data"):
    for feat in AMEX_FEATURES:
        st.session_state[f"amex_{feat}"] = np.random.uniform(0, 5000)

# Input UI
amex_vals = {}
groups = [AMEX_FEATURES[i:i+10] for i in range(0, len(AMEX_FEATURES), 10)]

for idx, group in enumerate(groups):
    st.markdown(f"### 🔹 Section {idx+1}")
    cols = st.columns(2)

    for i, feat in enumerate(group):
        with cols[i % 2]:
            amex_vals[feat] = st.slider(
                FEATURE_LABELS[feat],
                0.0, 10000.0,
                st.session_state.get(f"amex_{feat}", 100.0),
                key=f"amex_{feat}"
            )

# Convert to DataFrame
amex_input = pd.DataFrame([amex_vals])

st.divider()

# ── Prediction ─────────────────────────
if st.button("🔍 Predict Risk", use_container_width=True):

    model = models["amex"]

    if model is None:
        st.error("❌ Model file not found!")
    else:
        try:
            # ✅ FIX: correct column order
            amex_input = amex_input[AMEX_FEATURES]

            prob = model.predict_proba(amex_input)[0][1]
            label = risk_label(prob)

            # Result
            st.subheader("📊 Prediction Result")
            col1, col2 = st.columns(2)

            col1.metric("Default Probability", f"{prob:.2%}")
            col2.markdown(f"### {label}")

            st.progress(prob)

            # ── Explanation ─────────────────
            st.subheader("🧠 AI Explanation")

            explanation = []

            if prob > 0.7:
                explanation.append("High risk due to unstable financial behaviour.")
            elif prob > 0.3:
                explanation.append("Moderate risk detected.")
            else:
                explanation.append("Low risk, stable customer.")

            if amex_input["D_39"][0] > 50:
                explanation.append("Frequent payment delays.")
            if amex_input["B_1"][0] > 5000:
                explanation.append("High outstanding balance.")
            if amex_input["P_2"][0] < 500:
                explanation.append("Low recent payments.")
            if amex_input["D_48"][0] > 70:
                explanation.append("Poor credit health.")

            for line in explanation:
                st.write("•", line)

            # ── Summary ─────────────────
            st.subheader("📌 Interpretation")

            st.markdown(f"""
            - **Probability:** {prob:.2%}  
            - **Risk Level:** {label}  

            Based on customer financial behaviour and credit activity.
            """)

        except Exception as e:
            st.error("⚠️ Input mismatch or preprocessing issue!")
            st.write(e)
