# ============================================================================
# MATERNAL HEALTH RISK PREDICTOR - STREAMLIT APP
# ============================================================================
# Simple, beautiful web app to make predictions using your trained model
# Deploy with: streamlit run STREAMLIT_APP.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    try:
        import os
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "trained_models", "maternal_risk_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "trained_models", "feature_scaler.pkl")

        

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        return model, scaler

    except Exception as e:
        st.error("Real Error:")
        st.write(e)
        return None, None
# Load model
model, scaler = load_model_and_scaler()

# ============================================================================
# APP HEADER
# ============================================================================

st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🏥 Maternal Health Risk Predictor</h1>
        <p style="font-size: 18px; color: gray;">
            AI-powered risk assessment for maternal health
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("---")

# ============================================================================
# INFORMATION SECTION
# ============================================================================

with st.expander("ℹ️ How This Works", expanded=False):
    st.write("""
    This app uses a machine learning model trained on maternal health data to predict risk levels.
    
    **The model evaluates:**
    - Age of the patient
    - Systolic Blood Pressure
    - Diastolic Blood Pressure
    - Blood Sugar Level
    - Body Temperature
    - Heart Rate
    
    **Risk Levels:**
    - 🟢 **Low Risk**: Healthy pregnancy, routine monitoring
    - 🟡 **Mid Risk**: Some concerns, regular checkups recommended
    - 🔴 **High Risk**: Significant concerns, immediate medical attention needed
    
    **Disclaimer:** This is an educational tool. Always consult qualified healthcare professionals
    for medical decisions.
    """)

# ============================================================================
# MAIN INPUT SECTION
# ============================================================================

st.subheader("📋 Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "👤 Age (years)",
        min_value=10,
        max_value=100,
        value=25,
        step=1,
        help="Patient age in years"
    )
    
    systolic_bp = st.number_input(
        "💓 Systolic BP (mmHg)",
        min_value=50,
        max_value=250,
        value=120,
        step=1,
        help="Systolic blood pressure (top number)"
    )
    
    diastolic_bp = st.number_input(
        "💓 Diastolic BP (mmHg)",
        min_value=30,
        max_value=150,
        value=80,
        step=1,
        help="Diastolic blood pressure (bottom number)"
    )

with col2:
    blood_sugar = st.number_input(
        "🩸 Blood Sugar (mg/dL)",
        min_value=30.0,
        max_value=500.0,
        value=100.0,
        step=1.0,
        help="Fasting blood sugar level"
    )
    
    body_temp = st.number_input(
        "🌡️ Body Temperature (°F)",
        min_value=90.0,
        max_value=108.0,
        value=98.6,
        step=0.1,
        help="Body temperature in Fahrenheit"
    )
    
    heart_rate = st.number_input(
        "❤️ Heart Rate (bpm)",
        min_value=40,
        max_value=200,
        value=75,
        step=1,
        help="Beats per minute"
    )

with col3:
    st.write("**Input Summary:**")
    summary_data = {
        "Age": f"{age} years",
        "Systolic BP": f"{systolic_bp} mmHg",
        "Diastolic BP": f"{diastolic_bp} mmHg",
        "Blood Sugar": f"{blood_sugar} mg/dL",
        "Body Temp": f"{body_temp}°F",
        "Heart Rate": f"{heart_rate} bpm"
    }
    
    for key, value in summary_data.items():
        st.text(f"• {key}: {value}")

# ============================================================================
# PREDICTION SECTION
# ============================================================================

st.write("---")

col_button, col_reset = st.columns(2)

with col_button:
    predict_button = st.button("🔍 Predict Risk Level", use_container_width=True)

with col_reset:
    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

# ============================================================================
# MAKE PREDICTION
# ============================================================================

if predict_button:
    if model is None or scaler is None:
        st.error("❌ Cannot make prediction: Model not loaded")
    else:
        with st.spinner("🔄 Analyzing patient data..."):
            try:
                # Prepare input data
                input_data = np.array([[
                    age,
                    systolic_bp,
                    diastolic_bp,
                    blood_sugar,
                    body_temp,
                    heart_rate
                ]])
                
                # Scale input
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                
                # Get prediction probability (if available)
                try:
                    probabilities = model.predict_proba(input_scaled)[0]
                except:
                    probabilities = None
                
                # Map to risk level
                risk_levels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
                risk_icons = {0: "🟢", 1: "🟡", 2: "🔴"}
                
                result = risk_levels[prediction]
                icon = risk_icons[prediction]
                
                # Display result
                st.write("---")
                st.subheader("📊 Prediction Results")
                
                # Main result box
                col_result = st.columns(1)[0]
                with col_result:
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 30px;
                        border-radius: 10px;
                        text-align: center;
                    ">
                        <h2>{icon} {result}</h2>
                        <p style="font-size: 18px; margin: 20px 0;">
                            Based on the patient's vital signs
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics
                st.subheader("📈 Detailed Metrics")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    if probabilities is not None:
                        st.metric(
                            label="Low Risk Probability",
                            value=f"{probabilities[0]*100:.1f}%"
                        )
                
                with metric_col2:
                    if probabilities is not None:
                        st.metric(
                            label="Mid Risk Probability",
                            value=f"{probabilities[1]*100:.1f}%"
                        )
                
                with metric_col3:
                    if probabilities is not None:
                        st.metric(
                            label="High Risk Probability",
                            value=f"{probabilities[2]*100:.1f}%"
                        )
                
                # Risk indicators
                if probabilities is not None:
                    st.subheader("📊 Risk Distribution")
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    risk_names = ['Low Risk', 'Mid Risk', 'High Risk']
                    colors = ['#green', '#orange', '#red']
                    
                    bars = ax.bar(risk_names, probabilities * 100, color=['#90EE90', '#FFB347', '#FF6B6B'])
                    
                    ax.set_ylabel('Probability (%)', fontsize=12)
                    ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
                    ax.set_ylim(0, 100)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=11)
                    
                    st.pyplot(fig)
                
                # Risk recommendations
                st.subheader("💡 Recommendations")
                
                if prediction == 0:  # Low Risk
                    st.success("""
                    ✅ **Low Risk Assessment**
                    
                    • Continue routine prenatal care
                    • Maintain healthy lifestyle (diet, exercise)
                    • Regular monitoring recommended
                    • Schedule regular checkups
                    """)
                
                elif prediction == 1:  # Mid Risk
                    st.warning("""
                    ⚠️ **Mid Risk Assessment**
                    
                    • Increased monitoring recommended
                    • Regular checkups required
                    • Consider specialist consultation
                    • Monitor vital signs closely
                    • Report any symptoms immediately
                    """)
                
                else:  # High Risk
                    st.error("""
                    🚨 **High Risk Assessment**
                    
                    • **IMMEDIATE MEDICAL ATTENTION RECOMMENDED**
                    • Contact healthcare provider immediately
                    • Consider hospitalization
                    • Specialized care may be needed
                    • Do not delay - seek professional help
                    """)
                
                # Disclaimer
                st.info("""
                📌 **Important Disclaimer:**
                
                This prediction is based on a machine learning model trained on historical data.
                It should NOT be used as the sole basis for medical decisions.
                
                **Always consult qualified healthcare professionals** for:
                - Medical diagnosis
                - Treatment recommendations
                - Risk assessment confirmation
                - Personal medical advice
                """)
                
                # Save prediction (optional)
                st.write("---")
                
                if st.checkbox("💾 Save this prediction", value=False):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    prediction_data = {
                        "Timestamp": timestamp,
                        "Age": age,
                        "Systolic_BP": systolic_bp,
                        "Diastolic_BP": diastolic_bp,
                        "Blood_Sugar": blood_sugar,
                        "Body_Temp": body_temp,
                        "Heart_Rate": heart_rate,
                        "Prediction": result
                    }
                    
                    # Display prediction data
                    st.json(prediction_data)
                    
                    # Download button
                    csv_data = pd.DataFrame([prediction_data]).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction as CSV",
                        data=csv_data,
                        file_name=f"prediction_{timestamp.replace(' ', '_').replace(':', '-')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

# ============================================================================
# SIDEBAR - INFORMATION
# ============================================================================

with st.sidebar:
    st.header("📋 About This App")
    
    st.write("""
    **Maternal Health Risk Predictor**
    
    This application uses an AI model trained on maternal health data to predict
    risk levels for pregnant patients.
    
    **Model Information:**
    - Algorithm: Gradient Boosting
    - Training Accuracy: 82.16%
    - Features: 6 vital signs
    - Classes: 3 risk levels
    
    **Created:** 2026
    **Version:** 1.0
    """)
    
    st.write("---")
    
    st.header("🔗 Resources")
    
    col1, col2 = st.columns(2)
    with col1:
        st.link_button(
            "GitHub Repository",
            "https://github.com"
        )
    with col2:
        st.link_button(
            "Documentation",
            "https://github.com"
        )
    
    st.write("---")
    
    st.header("⚕️ Health Resources")
    st.write("""
    - [WHO Maternal Health](https://www.who.int)
    - [Medical Advice](https://www.healthgov.org)
    - [Emergency Help](https://www.911.gov)
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.write("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>Made with ❤️ using Streamlit | Maternal Health Risk Predictor v1.0</p>
        <p><small>This is an educational tool. Always consult healthcare professionals.</small></p>
    </div>
""", unsafe_allow_html=True)

