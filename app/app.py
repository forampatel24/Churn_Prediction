import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Allow imports from src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from predict import predict_single, predict_batch, load_artifacts

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .risk-high   { background:#fee2e2; color:#991b1b; padding:6px 14px; border-radius:8px; font-weight:600; }
    .risk-medium { background:#fef3c7; color:#92400e; padding:6px 14px; border-radius:8px; font-weight:600; }
    .risk-low    { background:#d1fae5; color:#065f46; padding:6px 14px; border-radius:8px; font-weight:600; }
    .metric-box  { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("📡 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn using their profile and usage data.")
st.divider()

# -----------------------------
# SIDEBAR — MODE SELECT
# -----------------------------
mode = st.sidebar.radio(
    "Prediction mode",
    ["Single Customer", "Batch CSV Upload"],
    index=0
)
st.sidebar.divider()
st.sidebar.markdown("**Model info**")
try:
    _, _, _, threshold, _ = load_artifacts()
    st.sidebar.success("Model loaded successfully")
    st.sidebar.metric("Decision threshold", f"{threshold:.2f}")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")
    st.stop()

# ==============================
# MODE 1 — SINGLE CUSTOMER
# ==============================
if mode == "Single Customer":

    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account info**")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        payment_method  = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless       = st.checkbox("Paperless billing", value=True)
        senior          = st.checkbox("Senior citizen", value=False)
        partner         = st.checkbox("Has partner", value=False)
        dependents      = st.checkbox("Has dependents", value=False)

    with col2:
        st.markdown("**Services**")
        phone_service   = st.checkbox("Phone service", value=True)
        multiple_lines  = st.checkbox("Multiple lines", value=False)
        internet        = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security",  ["Yes", "No", "No internet service"])
        online_backup   = st.selectbox("Online backup",    ["Yes", "No", "No internet service"])
        device_prot     = st.selectbox("Device protection",["Yes", "No", "No internet service"])
        tech_support    = st.selectbox("Tech support",     ["Yes", "No", "No internet service"])
        streaming_tv    = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
        streaming_movies= st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**Charges**")
        monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total charges ($)",   0.0, 10000.0,
                                          float(monthly_charges * tenure), step=1.0)

    # Encode inputs to match training encoding
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map  = {
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

    customer = {
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "Contract":         contract_map[contract],
        "PaymentMethod":    payment_map[payment_method],
        "InternetService":  internet_map[internet],
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "PaperlessBilling": int(paperless),
        "SeniorCitizen":    int(senior),
        "Partner":          int(partner),
        "Dependents":       int(dependents),
        "PhoneService":     int(phone_service),
        "MultipleLines":    int(multiple_lines),
    }

    st.divider()

    if st.button("Predict churn", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            result = predict_single(customer)

        prob  = result["churn_probability"]
        pred  = result["churn_prediction"]
        risk  = result["risk_level"]

        # --- result row ---
        r1, r2, r3 = st.columns(3)

        with r1:
            if pred == 1:
                st.error("⚠️  This customer is likely to churn")
            else:
                st.success("✅  This customer is likely to stay")

        with r2:
            st.metric("Churn probability", f"{prob:.1%}")

        with r3:
            risk_class = f"risk-{risk.lower()}"
            st.markdown(f"Risk level: <span class='{risk_class}'>{risk}</span>",
                        unsafe_allow_html=True)

        # --- gauge chart ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#ef4444" if pred else "#10b981", "thickness": 0.3},
                "steps": [
                    {"range": [0,  40], "color": "#d1fae5"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#6b7280", "width": 2},
                    "thickness": 0.75,
                    "value": threshold * 100
                }
            },
            title={"text": "Churn probability", "font": {"size": 16}},
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # --- retention suggestions ---
        st.subheader("Retention recommendations")
        suggestions = []
        if contract == "Month-to-month":
            suggestions.append("📋 Offer a discounted annual or two-year contract")
        if internet == "Fiber optic" and online_security == "No":
            suggestions.append("🔒 Bundle Online Security — fiber customers churn more without it")
        if tenure < 12:
            suggestions.append("🎁 Apply a new-customer loyalty discount for the first year")
        if monthly_charges > 70 and contract == "Month-to-month":
            suggestions.append("💰 Offer a price-lock plan — high monthly charges + no contract = high risk")
        if tech_support == "No" and internet != "No":
            suggestions.append("🛠️ Offer a free Tech Support trial")
        if not suggestions:
            suggestions.append("✅ Customer profile looks stable — continue regular engagement")

        for s in suggestions:
            st.markdown(f"- {s}")

# ==============================
# MODE 2 — BATCH CSV
# ==============================
else:
    st.subheader("Upload a CSV file to score multiple customers")
    st.markdown("The CSV should have the same columns as the training data (Churn column optional).")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_input)}** customers. Preview:")
        st.dataframe(df_input.head(), use_container_width=True)

        if st.button("Run batch prediction", type="primary"):
            with st.spinner("Scoring all customers..."):
                # Save temp file and call predict_batch
                tmp_path = os.path.join(BASE_DIR, "data", "_tmp_batch.csv")
                df_input.to_csv(tmp_path, index=False)
                results = predict_batch(tmp_path)
                os.remove(tmp_path)

            st.success("Batch prediction complete!")
            st.divider()

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            total     = len(results)
            n_churn   = results["churn_prediction"].sum()
            high_risk = (results["risk_level"] == "High").sum()
            avg_prob  = results["churn_probability"].mean()

            m1.metric("Total customers",  total)
            m2.metric("Predicted churners", f"{n_churn} ({n_churn/total:.0%})")
            m3.metric("High-risk customers", high_risk)
            m4.metric("Avg churn probability", f"{avg_prob:.1%}")

            # Risk breakdown pie
            risk_counts = results["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk level", "Count"]
            colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
            fig2 = px.pie(
                risk_counts, names="Risk level", values="Count",
                color="Risk level",
                color_discrete_map=colors,
                title="Customer risk breakdown"
            )
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)

            # Full results table
            st.subheader("Full results")
            st.dataframe(
                results.sort_values("churn_probability", ascending=False),
                use_container_width=True
            )

            # Download button
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )