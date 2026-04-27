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

if st.button("🔍 Predict GMSC Risk", use_container_width=True):

    model_key = "gmsc_xgb" if "XGBoost" in model_choice_gmsc else "gmsc"
    model = models.get(model_key)

    if model is None:
        st.error("Model not found")
    else:
        try:
            # ✅ Get correct feature names
            gmsc_features = model.feature_names_in_

            # Add missing columns
            for col in gmsc_features:
                if col not in gmsc_input:
                    gmsc_input[col] = 0

            # Keep only required columns in correct order
            gmsc_input = gmsc_input[gmsc_features]

            prob = model.predict_proba(gmsc_input)[0][1]
            label, _ = risk_label(prob)

            st.metric("Default Probability", f"{prob:.2%}")
            st.markdown(f"### {label}")
            st.progress(prob)

        except Exception as e:
            st.error("⚠️ Feature mismatch in GMSC model")
            st.write(e)

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
