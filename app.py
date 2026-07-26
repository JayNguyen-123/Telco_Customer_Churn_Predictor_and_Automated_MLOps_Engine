import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.drift_detector import check_numerical_drift

# Configure Streamlit page layout properties
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# =====================================================================
# 1. CACHED LIFECYCLE ASSET LOADERS
# =====================================================================
@st.cache_resource
def load_production_pipeline():
    """Loads the pre-compiled binary model pipeline from disk."""
    model_path = "models/optimized_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model missing at '{model_path}'. Please run your training pipeline or API setup first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_monitoring_datasets():
    """Fetches baseline records and simulates shifted live traffic for drift monitoring."""
    url = "https://githubusercontent.com"
    try:
        reference = pd.read_csv(url)
        reference['TotalCharges'] = pd.to_numeric(reference['TotalCharges'], errors='coerce')
        reference.dropna(subset=['TotalCharges'], inplace=True)
        
        # Simulate production logs where MonthlyCharges have drifted upwards (simulating inflation)
        np.random.seed(42)
        production = reference.copy().sample(n=1000, random_state=42)
        production['MonthlyCharges'] = production['MonthlyCharges'] * np.random.uniform(1.15, 1.30, size=len(production))
        
        return reference, production
    except Exception as e:
        st.error(f"❌ Failed to download baseline dataset for monitoring views: {str(e)}")
        st.stop()


# Load the models and data into memory
model_pipeline = load_production_pipeline()
reference_df, production_df = load_monitoring_datasets()

# =====================================================================
# 2. APPLICATION HEADER LAYOUT
# =====================================================================
st.title("📞 Telco Customer Churn & Operations Dashboard")
st.markdown("An end-to-end operational hub facilitating individual risk assessment simulations and feature drift metrics.")
st.markdown("---")

# Establish core application display tab split
tab1, tab2 = st.tabs(["🕹️ Interactive Churn Simulator", "📊 Operations & Drift Monitor"])

# =====================================================================
# 3. TAB 1: INTERACTIVE CHURN SIMULATOR
# =====================================================================
with tab1:
    st.header("Customer Profile Input Simulator")
    st.markdown("Modify the profile parameters below via the sidebar panel to calculate instantaneous target propensity scores.")
    
    # Setup row layouts for structured sidebar widget forms
    st.sidebar.header("⚙️ Profile Metric Vectors")
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen Status", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Account Tenure (Months Active)", 1, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service Subscribed", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines Service", ["No phone service", "No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service Provider (ISP)", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup Add-on", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection Add-on", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Premium Tech Support Add-on", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV Service", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies Service", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract Type Term", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing Active", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method Type", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.slider("Monthly Bill Charges ($)", 18.0, 120.0, 65.0)
    
    # Calculate automated structural guess for lifetime total charges to keep data realistic
    inferred_total = float(tenure * monthly_charges)
    total_charges = st.sidebar.number_input("Total Lifetime Revenue Cost ($)", min_value=18.0, value=inferred_total)

    # Reconstruct dictionary framework to parse matching features
    input_data = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔮 Pipeline Real-Time Inference")
        if st.button("Calculate Churn Propensity", type="primary"):
            # Calculate raw evaluations from your pipeline
            prediction = model_pipeline.predict(input_data)[0]
            probability = model_pipeline.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            if prediction == 1:
                st.error("🚨 **High Risk of Imminent Cancellation!**")
                st.metric(label="Calculated Churn Probability Score", value=f"{probability * 100:.2f}%")
                st.warning("Action Recommended: Propose a loyalty upgrade discount or transition to a multi-year contract tier.")
            else:
                st.success("✅ **Loyal / Low-Risk Customer Profile**")
                st.metric(label="Calculated Churn Probability Score", value=f"{probability * 100:.2f}%")
                st.info("Action Recommended: Maintain baseline marketing cadence; eligible for standard cross-sell programs.")

    with col2:
        st.subheader("📋 Parsed Pipeline Request Data Frame")
        st.info("Below is the structured data frame vector currently being passed down to the pipeline's preprocessing transformations:")
        st.dataframe(input_data.T.rename(columns={0: "Input Vector Values"}))

# =====================================================================
# 4. TAB 2: OPERATIONS & DRIFT MONITOR
# =====================================================================
with tab2:
    st.header("📡 Live Feature Distribution Stability Tracking")
    st.markdown("This tab checks for data drift by running a two-sample Kolmogorov-Smirnov test, comparing production logs against training baselines.")
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Calculate drift metrics using your module functions
    drift_results = check_numerical_drift(reference_df, production_df, numeric_cols)
    
    # Render scorecard metric summary visuals
    metric_cols = st.columns(3)
    for idx, col_name in enumerate(numeric_cols):
        with metric_cols[idx]:
            status = drift_results[col_name]
            if status["drift_detected"]:
                st.error(f"🚨 **{col_name}**: DRIFT DETECTED")
            else:
                st.success(f"✅ **{col_name}**: Stable")
            st.metric(label="KS Distance Statistic", value=status["ks_statistic"])
            st.caption(f"Calculated p-value: **{status['p_value']}**")
            
    st.markdown("---")
    
    # Select box controls to inspect feature shifts interactively
    st.subheader("📈 Interactive Density Curve Splits")
    selected_feature = st.selectbox("Choose a Numerical Feature to Inspect Distribution Shift:", numeric_cols)
    
    # Render matplotlib visualization graph plots
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.kdeplot(reference_df[selected_feature], fill=True, label="Baseline (Training Data)", ax=ax, color="#1f77b4", alpha=0.4)
    sns.kdeplot(production_df[selected_feature], fill=True, label="Live Traffic (Simulated Production Logs)", ax=ax, color="#ff7f0e", alpha=0.4)
    ax.set_title(f"Probability Density Distribution Splitting Matrix for: {selected_feature}", fontsize=12)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Density Scale")
    ax.legend(loc="upper right")
    
    st.pyplot(fig)
