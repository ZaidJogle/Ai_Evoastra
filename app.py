import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
)

# ── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        "gmsc":     "gmsc_model.pkl",
        "gmsc_xgb": "gmsc_xgb_model.pkl",
        "amex":     "amex_model.pkl",
        "amex_xgb": "amex_xgb_model.pkl",
    }
    for key, fname in files.items():
        try:
            with open(fname, "rb") as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            models[key] = None
    return models

models = load_models()

# ── Risk badge helper ─────────────────────────────────────────────────────────
def risk_label(prob: float) -> tuple[str, str]:
    """Return (label, colour)."""
    if prob < 0.3:
        return "🟢 Low Risk", "green"
    elif prob < 0.7:
        return "🟠 Medium Risk", "orange"
    else:
        return "🔴 High Risk", "red"

# ── GMSC features (from notebook) ────────────────────────────────────────────
GMSC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "debt_ratio",
    "income_to_debt",
]

# ── AMEX features (first 50 numeric cols from notebook) ──────────────────────
AMEX_FEATURES = [
    "P_2", "D_39", "B_1", "B_2", "R_1", "S_3", "D_41", "B_3", "D_42", "D_43",
    "D_44", "B_4", "D_45", "B_5", "R_2", "D_46", "D_47", "D_48", "D_49", "B_6",
    "B_7", "B_8", "D_50", "D_51", "B_9", "R_3", "D_52", "P_3", "B_10", "D_53",
    "S_5", "B_11", "S_6", "D_54", "R_4", "S_7", "B_12", "S_8", "D_55", "D_56",
    "B_13", "R_5", "D_57", "B_14", "D_58", "B_15", "D_59", "D_60", "D_61", "B_16",
]

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("💳 Credit Risk Predictor")
st.caption("Predict default probability using trained ML models on GMSC and AmEx datasets.")

tab_gmsc, tab_amex = st.tabs(["🏦 GMSC – Give Me Some Credit", "🏢 AmEx Default Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – GMSC
# ══════════════════════════════════════════════════════════════════════════════
with tab_gmsc:
    st.subheader("Give Me Some Credit – Delinquency Predictor")
    st.markdown(
        "Enter borrower details below. The model predicts the probability of "
        "experiencing **90-day past-due delinquency** in the next two years."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Utilisation & Debt**")
        revolving_util = st.slider(
            "Revolving Utilisation of Unsecured Lines",
            0.0, 1.0, 0.3, step=0.01,
            help="Total balance on credit cards / personal lines ÷ sum of credit limits"
        )
        debt_ratio_raw = st.slider(
            "Debt Ratio",
            0.0, 10.0, 0.3, step=0.01,
            help="Monthly debt payments, alimony, living costs ÷ monthly gross income"
        )
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=0, max_value=500_000, value=5_000, step=500
        )

    with col2:
        st.markdown("**📅 Past-Due Events**")
        past_due_30_59 = st.number_input(
            "Times 30–59 Days Past Due (not worse)", 0, 20, 0
        )
        past_due_60_89 = st.number_input(
            "Times 60–89 Days Past Due (not worse)", 0, 20, 0
        )
        times_90_late = st.number_input(
            "Times 90+ Days Late", 0, 20, 0
        )

    with col3:
        st.markdown("**🏠 Profile**")
        age = st.number_input("Age", 18, 100, 40)
        open_credit_lines = st.number_input(
            "Open Credit Lines & Loans", 0, 60, 5
        )
        real_estate_loans = st.number_input(
            "Real Estate Loans or Lines", 0, 20, 1
        )
        dependents = st.number_input("Number of Dependents", 0, 20, 0)

    # Derived features (same as notebook)
    debt_ratio_feat  = debt_ratio_raw
    income_to_debt   = monthly_income / (debt_ratio_raw + 1)

    gmsc_input = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines":  revolving_util,
        "age":                                   age,
        "NumberOfTime30-59DaysPastDueNotWorse":  past_due_30_59,
        "DebtRatio":                             debt_ratio_raw,
        "MonthlyIncome":                         monthly_income,
        "NumberOfOpenCreditLinesAndLoans":        open_credit_lines,
        "NumberOfTimes90DaysLate":               times_90_late,
        "NumberRealEstateLoansOrLines":           real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse":  past_due_60_89,
        "NumberOfDependents":                    dependents,
        "debt_ratio":                            debt_ratio_feat,
        "income_to_debt":                        income_to_debt,
    }])

    st.divider()
    model_choice_gmsc = st.radio(
        "Choose model",
        ["XGBoost (gmsc_xgb_model)", "Ensemble (gmsc_model)"],
        horizontal=True,
        key="gmsc_model_radio"
    )

    if st.button("🔍 Predict GMSC Risk", use_container_width=True):
        model_key = "gmsc_xgb" if "XGBoost" in model_choice_gmsc else "gmsc"
        model = models.get(model_key)

        if model is None:
            st.error(
                f"Model file **{'gmsc_xgb_model.pkl' if model_key == 'gmsc_xgb' else 'gmsc_model.pkl'}** "
                "not found. Place it in the same directory as app.py."
            )
        else:
            prob = model.predict_proba(gmsc_input)[0][1]
            label, colour = risk_label(prob)

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Default Probability", f"{prob:.1%}")
            with r2:
                st.markdown(f"### {label}")

            st.progress(float(prob), text=f"Risk score: {prob:.2%}")

            with st.expander("📋 Input summary"):
                st.dataframe(gmsc_input.T.rename(columns={0: "Value"}))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – AMEX
# ══════════════════════════════════════════════════════════════════════════════
with tab_amex:
    st.subheader("American Express – Default Prediction")
    st.markdown(
        "Enter the **last statement's** feature values for a customer. "
        "The model predicts whether the customer will **default** on their AmEx account."
    )

    st.info(
        "ℹ️ The model uses the first 50 numeric features from the AmEx dataset. "
        "Defaults shown are column medians from training data. Adjust as needed.",
        icon="ℹ️"
    )

    # Split into groups of ~10 for layout
    amex_vals = {}
    groups = [AMEX_FEATURES[i:i+10] for i in range(0, len(AMEX_FEATURES), 10)]
    for g_idx, group in enumerate(groups):
        cols = st.columns(5)
        for i, feat in enumerate(group):
            with cols[i % 5]:
                amex_vals[feat] = st.number_input(
                    feat,
                    value=0.0,
                    format="%.4f",
                    key=f"amex_{feat}"
                )

    amex_input = pd.DataFrame([amex_vals])

    st.divider()
    model_choice_amex = st.radio(
        "Choose model",
        ["Ensemble LR+XGB (amex_model)", "XGBoost only (amex_xgb_model)"],
        horizontal=True,
        key="amex_model_radio"
    )

    if st.button("🔍 Predict AmEx Default Risk", use_container_width=True):
        model_key = "amex" if "Ensemble" in model_choice_amex else "amex_xgb"
        model = models.get(model_key)

        if model is None:
            st.error(
                f"Model file not found. Place **{'amex_model.pkl' if model_key == 'amex' else 'amex_xgb_model.pkl'}** "
                "in the same directory as app.py."
            )
        else:
            prob = model.predict_proba(amex_input)[0][1]
            label, colour = risk_label(prob)

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Default Probability", f"{prob:.1%}")
            with r2:
                st.markdown(f"### {label}")

            st.progress(float(prob), text=f"Risk score: {prob:.2%}")

            with st.expander("📋 Input summary"):
                st.dataframe(amex_input.T.rename(columns={0: "Value"}))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Models: Logistic Regression · XGBoost · VotingClassifier (soft) | "
    "Trained with SMOTE oversampling | Risk bands: <30% Low · 30–70% Medium · >70% High"
)
