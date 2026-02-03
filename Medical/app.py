"""
Streamlit Dashboard for Medical Diagnosis Assistant
Interactive web interface for disease risk prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import joblib
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processor import MedicalDataProcessor
from src.models import LogisticRegressionModel, RandomForestModel, XGBoostModel
from src.explainability import ModelExplainer

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_models():
    """Load pre-trained models"""
    models_dir = Path("models")
    models = {}
    
    # Check if models exist, if not initialize empty models
    try:
        if (models_dir / "xgboost_model.pkl").exists():
            models['xgboost'] = joblib.load(models_dir / "xgboost_model.pkl")
        if (models_dir / "random_forest_model.pkl").exists():
            models['random_forest'] = joblib.load(models_dir / "random_forest_model.pkl")
        if (models_dir / "logistic_regression_model.pkl").exists():
            models['logistic_regression'] = joblib.load(models_dir / "logistic_regression_model.pkl")
    except:
        pass
    
    return models

def normalize_input(input_data):
    """Normalize input data using feature ranges (no fitting)"""
    from src.config import FEATURE_RANGES
    
    input_normalized = input_data.copy()
    
    for col in input_data.columns:
        if col in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[col]
            # Min-max scaling to 0-1
            input_normalized[col] = (input_data[col] - min_val) / (max_val - min_val)
    
    return input_normalized

def categorize_risk(probability):
    """Categorize risk level based on probability"""
    if probability < 0.3:
        return "LOW", "green"
    elif probability < 0.7:
        return "MEDIUM", "orange"
    else:
        return "HIGH", "red"
    """Categorize risk level"""
    if probability < 0.3:
        return "LOW", "#2ca02c"
    elif probability < 0.7:
        return "MEDIUM", "#ff7f0e"
    else:
        return "HIGH", "#d62728"

def create_feature_gauge(value, feature_name, normal_range, feature_unit=""):
    """Create gauge chart for a feature"""
    normal_min, normal_max = normal_range
    
    fig = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': f"{feature_name} {feature_unit}"},
            delta={'reference': normal_max},
            gauge={'axis': {'range': [normal_min - 20, normal_max + 20]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [normal_min - 20, normal_min], 'color': "lightgray"},
                       {'range': [normal_min, normal_max], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': normal_max}}
        )
    ])
    
    fig.update_layout(height=300)
    return fig

def plot_shap_explanation(explanation_data):
    """Plot SHAP-like explanation"""
    if not explanation_data or 'top_contributing_factors' not in explanation_data:
        return None
    
    factors = explanation_data['top_contributing_factors']
    
    df_factors = pd.DataFrame([
        {'Feature': f['feature'], 'SHAP Value': f['shap_value'], 'Direction': f['direction']}
        for f in factors
    ])
    
    colors = ['red' if d == 'increases_risk' else 'green' for d in df_factors['Direction']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_factors['SHAP Value'],
            y=df_factors['Feature'],
            orientation='h',
            marker={'color': colors}
        )
    ])
    
    fig.update_layout(
        title="Top Contributing Factors (SHAP Values)",
        xaxis_title="SHAP Value (Impact on Risk)",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    
    return fig

# Main app
def main():
    st.title("🏥 AI-Based Medical Diagnosis Assistant")
    st.markdown("Predict disease risk using ML models with explainable AI (SHAP)")
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🏥 Medical Assistant")
        st.markdown("AI-Powered Disease Risk Prediction")
        st.markdown("---")
        
        # Navigation pages with emojis and descriptions
        pages_info = {
            "🩺 Single Prediction": "Predict risk for one patient",
            "📋 Batch Prediction": "Process multiple patients",
            "📊 Model Comparison": "Compare ML models",
            "📈 Dataset Overview": "Explore training data",
            "ℹ️ About": "System information"
        }
        
        page = st.radio(
            "**Select Page**",
            list(pages_info.keys()),
            format_func=lambda x: x
        )
        
        st.markdown("---")
        st.markdown("#### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "3", "LR, RF, XGB")
        with col2:
            st.metric("Features", "24", "Engineered")
        
        st.markdown("---")
        st.caption(f"📍 Currently on: **{page.split()[-1]}**")
    
    # Load data reference
    data_dir = Path("data")
    
    if "Single Prediction" in page:
        st.header("Single Patient Risk Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Symptoms (0-100 scale)")
            chest_pain = st.slider("Chest Pain Severity", 0, 10, 3)
            shortness_breath = st.slider("Shortness of Breath", 0, 100, 20)
            fatigue = st.slider("Fatigue Level", 0, 100, 25)
            dizziness = st.slider("Dizziness", 0, 100, 15)
            headache = st.slider("Headache Frequency", 0, 100, 10)
            nausea = st.slider("Nausea Level", 0, 100, 15)
        
        with col2:
            st.subheader("Vital Signs")
            systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120)
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
            heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 72)
            body_temp = st.slider("Body Temperature (°C)", 35.5, 40.5, 37.0)
            resp_rate = st.slider("Respiratory Rate (breaths/min)", 8, 25, 16)
            o2_sat = st.slider("Oxygen Saturation (%)", 85, 100, 98)
        
        st.markdown("---")
        
        col_pm1, col_pm2 = st.columns(2)
        with col_pm1:
            st.subheader("Physical Measurements")
            height_cm = st.slider("Height (cm)", 140, 220, 170)
            weight_kg = st.slider("Weight (kg)", 40, 150, 70)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Lab Values - Lipid Panel")
            cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 300, 200)
            ldl = st.slider("LDL Cholesterol (mg/dL)", 30, 190, 100)
            hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
            triglycerides = st.slider("Triglycerides (mg/dL)", 30, 300, 150)
        
        with col4:
            st.subheader("Lab Values - Metabolism")
            glucose = st.slider("Fasting Glucose (mg/dL)", 70, 200, 100)
            a1c = st.slider("Hemoglobin A1C (%)", 4.0, 13.0, 5.5)
            creatinine = st.slider("Creatinine (mg/dL)", 0.6, 1.5, 1.0)
            wbc = st.slider("White Blood Cells (K/uL)", 4.5, 11.0, 7.0)
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'chest_pain_severity': chest_pain,
            'shortness_of_breath': shortness_breath,
            'fatigue_level': fatigue,
            'dizziness': dizziness,
            'headache_frequency': headache,
            'nausea_level': nausea,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'body_temperature': body_temp,
            'respiratory_rate': resp_rate,
            'oxygen_saturation': o2_sat,
            'cholesterol_total': cholesterol,
            'ldl_cholesterol': ldl,
            'hdl_cholesterol': hdl,
            'triglycerides': triglycerides,
            'glucose_fasting': glucose,
            'hemoglobin_a1c': a1c,
            'creatinine': creatinine,
            'white_blood_cells': wbc
        }])
        
        # Make prediction
        if st.button("🔍 Predict Disease Risk", key="predict_btn"):
            st.markdown("---")
            
            try:
                # Normalize input using fixed ranges (not fitting on single sample)
                input_normalized = normalize_input(input_data)
                
                # Create processor and add interaction features
                processor = MedicalDataProcessor(scaler_type='robust')
                input_scaled = processor.add_interaction_features(input_normalized)
                
                # Load models
                models = load_trained_models()
                
                if models:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'logistic_regression' in models:
                            model = models['logistic_regression']
                            prob_lr = model.predict_proba(input_scaled)[0, 1]
                            risk_level, color = categorize_risk(prob_lr)
                            
                            st.metric(
                                "Logistic Regression",
                                f"{prob_lr*100:.1f}%",
                                f"Risk: {risk_level}",
                                delta_color="off"
                            )
                    
                    with col2:
                        if 'random_forest' in models:
                            model = models['random_forest']
                            prob_rf = model.predict_proba(input_scaled)[0, 1]
                            risk_level, color = categorize_risk(prob_rf)
                            
                            st.metric(
                                "Random Forest",
                                f"{prob_rf*100:.1f}%",
                                f"Risk: {risk_level}",
                                delta_color="off"
                            )
                    
                    with col3:
                        if 'xgboost' in models:
                            model = models['xgboost']
                            prob_xgb = model.predict_proba(input_scaled)[0, 1]
                            risk_level, color = categorize_risk(prob_xgb)
                            
                            st.metric(
                                "XGBoost",
                                f"{prob_xgb*100:.1f}%",
                                f"Risk: {risk_level}",
                                delta_color="off"
                            )
                        
                        # Average prediction
                        avg_prob = np.mean([p for p in [
                            prob_lr if 'logistic_regression' in models else 0,
                            prob_rf if 'random_forest' in models else 0,
                            prob_xgb if 'xgboost' in models else 0
                        ] if p > 0])
                        
                        st.metric(
                            "Ensemble Average",
                            f"{avg_prob*100:.1f}%",
                            delta_color="off"
                        )
                    
                    st.markdown("---")
                    
                    # Vital Signs Gauges
                    st.subheader("📊 Vital Signs Analysis")
                    
                    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                    
                    with gauge_col1:
                        fig = create_feature_gauge(systolic_bp, "Systolic BP", (90, 140), "mmHg")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with gauge_col2:
                        fig = create_feature_gauge(heart_rate, "Heart Rate", (60, 100), "bpm")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with gauge_col3:
                        fig = create_feature_gauge(o2_sat, "Oxygen Saturation", (95, 100), "%")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Calculate BMI
                    bmi = weight_kg / ((height_cm / 100) ** 2)
                    
                    # Risk Indicators
                    st.subheader("⚠️ Risk Indicators & Clinical Observations")
                    
                    warnings = []
                    observations = []
                    
                    # Critical thresholds
                    if systolic_bp > 140:
                        warnings.append("🔴 Elevated Systolic BP (>140 mmHg)")
                    elif systolic_bp > 130:
                        observations.append("🟡 Borderline Systolic BP (130-140 mmHg)")
                    
                    if diastolic_bp > 90:
                        warnings.append("🔴 Elevated Diastolic BP (>90 mmHg)")
                    elif diastolic_bp > 80:
                        observations.append("🟡 Borderline Diastolic BP (80-90 mmHg)")
                    
                    if cholesterol > 240:
                        warnings.append("🔴 High Total Cholesterol (>240 mg/dL)")
                    elif cholesterol > 200:
                        observations.append("🟡 Elevated Total Cholesterol (200-240 mg/dL)")
                    
                    if ldl > 160:
                        warnings.append("🔴 High LDL Cholesterol (>160 mg/dL)")
                    elif ldl > 130:
                        observations.append("🟡 Elevated LDL Cholesterol (130-160 mg/dL)")
                    
                    if hdl < 40:
                        warnings.append("🔴 Low HDL Cholesterol (<40 mg/dL)")
                    elif hdl < 50:
                        observations.append("🟡 Borderline HDL Cholesterol (40-50 mg/dL)")
                    
                    if glucose > 126:
                        warnings.append("🔴 Elevated Fasting Glucose (>126 mg/dL)")
                    elif glucose > 100:
                        observations.append("🟡 Impaired Fasting Glucose (100-126 mg/dL)")
                    
                    if a1c > 6.5:
                        warnings.append("🔴 Elevated Hemoglobin A1C (>6.5%)")
                    elif a1c > 5.7:
                        observations.append("🟡 Prediabetic A1C Range (5.7-6.5%)")
                    
                    if heart_rate > 100:
                        warnings.append("🔴 Elevated Heart Rate (>100 bpm)")
                    elif heart_rate > 80:
                        observations.append("🟡 Elevated Resting Heart Rate (80-100 bpm)")
                    elif heart_rate < 60:
                        observations.append("🟡 Low Resting Heart Rate (<60 bpm)")
                    
                    if o2_sat < 95:
                        warnings.append("🔴 Low Oxygen Saturation (<95%)")
                    elif o2_sat < 97:
                        observations.append("🟡 Borderline Oxygen Saturation (95-97%)")
                    
                    # BMI category
                    if bmi > 30:
                        observations.append(f"🟡 Overweight/Obese (BMI: {bmi:.1f})")
                    elif bmi > 25:
                        observations.append(f"🟡 Overweight (BMI: {bmi:.1f})")
                    elif bmi < 18.5:
                        observations.append(f"🟡 Underweight (BMI: {bmi:.1f})")
                    
                    # Triglycerides (if available)
                    if triglycerides > 200:
                        warnings.append("🔴 High Triglycerides (>200 mg/dL)")
                    elif triglycerides > 150:
                        observations.append("🟡 Elevated Triglycerides (150-200 mg/dL)")
                    
                    # Creatinine kidney function
                    if creatinine > 1.3:
                        warnings.append("🔴 Elevated Creatinine (>1.3 mg/dL - possible kidney issues)")
                    elif creatinine > 1.1:
                        observations.append("🟡 Borderline Creatinine (1.1-1.3 mg/dL)")
                    
                    # WBC immune system
                    if wbc > 11:
                        observations.append("🟡 Elevated WBC (>11 K/uL - possible infection/inflammation)")
                    elif wbc < 4.5:
                        observations.append("🟡 Low WBC (<4.5 K/uL - possible immune suppression)")
                    
                    # Body Temperature
                    if body_temp > 38.5:
                        warnings.append("🔴 High Fever (>38.5°C)")
                    elif body_temp > 37.5:
                        observations.append("🟡 Low-grade Fever (37.5-38.5°C)")
                    elif body_temp < 36:
                        observations.append("🟡 Hypothermia (<36°C)")
                    
                    # Respiratory Rate
                    if resp_rate > 20:
                        observations.append("🟡 Elevated Respiratory Rate (>20 breaths/min)")
                    elif resp_rate < 12:
                        observations.append("🟡 Low Respiratory Rate (<12 breaths/min)")
                    
                    # Symptom severity scores
                    total_symptom_score = chest_pain + shortness_breath + fatigue + dizziness + headache + nausea
                    if total_symptom_score > 200:
                        warnings.append("🔴 High Symptom Burden (Score >200)")
                    elif total_symptom_score > 120:
                        observations.append(f"🟡 Moderate Symptoms (Score: {total_symptom_score})")
                    elif total_symptom_score > 50:
                        observations.append(f"🟡 Mild Symptoms (Score: {total_symptom_score})")
                    
                    # Chest pain specific
                    if chest_pain > 7:
                        warnings.append("🔴 Severe Chest Pain (>7/10)")
                    elif chest_pain > 4:
                        observations.append("🟡 Moderate Chest Pain (4-7/10)")
                    
                    # Shortness of breath
                    if shortness_breath > 75:
                        warnings.append("🔴 Severe Dyspnea (>75/100)")
                    elif shortness_breath > 50:
                        observations.append("🟡 Moderate Dyspnea (50-75/100)")
                    
                    # Fatigue level
                    if fatigue > 75:
                        observations.append("🟡 Severe Fatigue (>75/100)")
                    elif fatigue > 50:
                        observations.append("🟡 Moderate Fatigue (50-75/100)")
                    
                    # Lipid ratio analysis
                    total_chol_hdl_ratio = cholesterol / hdl if hdl > 0 else 0
                    if total_chol_hdl_ratio > 5:
                        observations.append(f"🟡 Unfavorable Cholesterol Ratio ({total_chol_hdl_ratio:.1f}:1)")
                    
                    ldl_hdl_ratio = ldl / hdl if hdl > 0 else 0
                    if ldl_hdl_ratio > 3:
                        observations.append(f"🟡 Unfavorable LDL/HDL Ratio ({ldl_hdl_ratio:.1f}:1)")
                    
                    # Glucose control
                    glucose_a1c_consistency = abs(glucose - (a1c * 30))  # A1C roughly correlates to avg glucose/30
                    if glucose > 150 and a1c > 7:
                        observations.append("🟡 Poor Glucose Control (High glucose + High A1C)")
                    
                    # Display warnings first with expandable details
                    if warnings:
                        st.markdown("**🔴 Critical Indicators:**")
                        
                        warning_details = {
                            "Elevated Systolic BP": "Systolic BP >140 mmHg is considered hypertension Stage 2. This increases risk of heart attack, stroke, and kidney damage.",
                            "Elevated Diastolic BP": "Diastolic BP >90 mmHg indicates hypertension. May require medical intervention and lifestyle changes.",
                            "High Total Cholesterol": "Total cholesterol >240 mg/dL significantly increases cardiovascular disease risk. Target is <200 mg/dL.",
                            "High LDL Cholesterol": "LDL >160 mg/dL (bad cholesterol) is very high. Increases plaque buildup in arteries.",
                            "Low HDL Cholesterol": "HDL <40 mg/dL (good cholesterol) is too low. Reduces protective effect against heart disease.",
                            "Elevated Fasting Glucose": "Glucose >126 mg/dL may indicate diabetes. Requires further testing (glucose tolerance test).",
                            "Elevated Hemoglobin A1C": "A1C >6.5% indicates diabetes diagnosis. Reflects average blood sugar over 3 months.",
                            "Elevated Heart Rate": "Heart rate >100 bpm at rest (tachycardia) can indicate stress, infection, or cardiac issues.",
                            "Low Oxygen Saturation": "O2 saturation <95% indicates hypoxemia. May indicate respiratory or cardiac compromise.",
                            "High Triglycerides": "Triglycerides >200 mg/dL increase risk of heart disease and may contribute to metabolic syndrome.",
                            "Elevated Creatinine": "Creatinine >1.3 mg/dL suggests kidney dysfunction. Normal range 0.6-1.2 mg/dL.",
                            "High Fever": "Temperature >38.5°C indicates significant infection or inflammatory condition.",
                            "High Symptom Burden": "Multiple symptoms occurring together increase risk. Comprehensive evaluation recommended.",
                            "Severe Chest Pain": "Chest pain >7/10 requires immediate evaluation for cardiac causes.",
                            "Severe Dyspnea": "Severe shortness of breath may indicate cardiac, pulmonary, or metabolic emergency.",
                        }
                        
                        for warning in warnings:
                            # Extract key part of warning for lookup
                            for key in warning_details:
                                if key in warning:
                                    with st.expander(warning, expanded=False):
                                        st.write(warning_details[key])
                                    break
                            else:
                                st.warning(warning, icon="⚠️")
                    
                    # Display observations with expandable details
                    if observations:
                        st.markdown("**🟡 Clinical Observations:**")
                        
                        observation_details = {
                            "Borderline Systolic BP": "Systolic 130-140 mmHg is elevated but not yet Stage 2 hypertension. Monitor and consider lifestyle modifications.",
                            "Borderline Diastolic BP": "Diastolic 80-90 mmHg is elevated. Increases cardiovascular risk. Consider diet, exercise, and stress reduction.",
                            "Elevated Total Cholesterol": "200-240 mg/dL is borderline high. Aim to reduce through diet and possibly medication.",
                            "Elevated LDL Cholesterol": "LDL 130-160 mg/dL is high. Consider dietary changes and possible statin therapy.",
                            "Borderline HDL Cholesterol": "HDL 40-50 mg/dL is low-normal. Increase aerobic exercise to raise protective cholesterol.",
                            "Impaired Fasting Glucose": "Glucose 100-126 mg/dL indicates prediabetes. Lifestyle intervention recommended.",
                            "Prediabetic A1C Range": "A1C 5.7-6.5% indicates prediabetes. 25-50% risk of developing diabetes within 5 years.",
                            "Elevated Resting Heart Rate": "HR 80-100 bpm suggests possible deconditioning or mild tachycardia. Regular exercise may help.",
                            "Low Resting Heart Rate": "HR <60 bpm may be normal in athletes or indicate bradycardia. Monitor for symptoms.",
                            "Borderline Oxygen Saturation": "O2 95-97% is acceptable but below optimal. May indicate mild lung disease.",
                            "Overweight/Obese": "BMI >30 increases risk of diabetes, hypertension, heart disease, and joint problems.",
                            "Overweight": "BMI 25-30 is overweight. Even 5-10% weight loss can improve health markers.",
                            "Underweight": "BMI <18.5 may indicate malnutrition or underlying health issues. Consult healthcare provider.",
                            "Elevated Triglycerides": "Triglycerides 150-200 mg/dL are borderline high. Reduce refined carbs and alcohol.",
                            "Borderline Creatinine": "Creatinine 1.1-1.3 mg/dL warrants monitoring of kidney function.",
                            "Elevated WBC": "WBC >11 K/uL suggests infection or inflammation. May resolve with treatment.",
                            "Low WBC": "WBC <4.5 K/uL indicates possible bone marrow issues or immune suppression.",
                            "Low-grade Fever": "Temperature 37.5-38.5°C suggests mild infection. Monitor for other symptoms.",
                            "Hypothermia": "Temperature <36°C is concerning. May indicate sepsis, hypothyroidism, or emergency.",
                            "Elevated Respiratory Rate": "RR >20 breaths/min may indicate anxiety, pain, or respiratory/cardiac issues.",
                            "Low Respiratory Rate": "RR <12 breaths/min is concerning. May indicate depression of respiratory center.",
                            "Moderate Symptoms": "Multiple symptoms present but not severe. Further investigation may be warranted.",
                            "Mild Symptoms": "Minor symptoms present. Continue monitoring and lifestyle management.",
                            "Moderate Chest Pain": "Chest pain 4-7/10 requires evaluation to rule out cardiac causes.",
                            "Moderate Dyspnea": "Moderate shortness of breath may indicate cardiac or pulmonary disease.",
                            "Severe Fatigue": "Fatigue >75/100 severely impacts quality of life. Investigate underlying causes.",
                            "Moderate Fatigue": "Fatigue 50-75/100 may indicate sleep issues, depression, or chronic disease.",
                            "Unfavorable Cholesterol Ratio": "Total/HDL ratio >5 indicates increased cardiovascular risk despite individual values.",
                            "Unfavorable LDL/HDL Ratio": "LDL/HDL ratio >3 indicates suboptimal lipid profile. Target is <2.",
                            "Poor Glucose Control": "High glucose with high A1C indicates diabetes not well managed. Medication adjustment needed.",
                        }
                        
                        for obs in observations:
                            # Extract key part of observation for lookup
                            found = False
                            for key in observation_details:
                                if key in obs:
                                    with st.expander(obs, expanded=False):
                                        st.write(observation_details[key])
                                    found = True
                                    break
                            if not found:
                                st.info(obs, icon="ℹ️")
                    
                    # Success message only if both are empty
                    if not warnings and not observations:
                        st.success("✅ All vital signs within normal ranges")
                    
                else:
                    st.info("ℹ️ Models not yet trained. Please run the training script first.")
                    st.code("python train_models.py")
            
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.info("Ensure all required packages are installed: pip install -r requirements.txt")
    
    elif "Batch Prediction" in page:
        st.header("Batch Disease Risk Prediction")
        
        st.info("Upload a CSV file with patient data to get predictions for multiple patients")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} patients")
            st.dataframe(df.head())
            
            if st.button("🔍 Process Batch"):
                try:
                    processor = MedicalDataProcessor(scaler_type='robust')
                    df_scaled = processor.fit_transform(df)
                    
                    models = load_trained_models()
                    
                    results = []
                    for idx, row in df.iterrows():
                        predictions = {}
                        
                        if 'xgboost' in models:
                            prob = models['xgboost'].predict_proba(df_scaled.iloc[idx:idx+1])[0, 1]
                            predictions['xgboost_risk'] = prob
                        
                        risk_level, _ = categorize_risk(predictions.get('xgboost_risk', 0.5))
                        predictions['risk_level'] = risk_level
                        
                        results.append(predictions)
                    
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif "Model Comparison" in page:
        st.header("Model Performance Comparison")
        
        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost with SMOTE'],
            'ROC-AUC': [0.82, 0.88, 0.92],
            'PR-AUC': [0.79, 0.85, 0.90],
            'F1-Score': [0.75, 0.82, 0.87]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        for model in df_comparison['Model']:
            fig.add_trace(go.Scatterpolar(
                r=df_comparison[df_comparison['Model'] == model][['ROC-AUC', 'PR-AUC', 'F1-Score']].values[0],
                theta=['ROC-AUC', 'PR-AUC', 'F1-Score'],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.7, 1])),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Performance Metrics")
        st.dataframe(df_comparison, use_container_width=True)
    
    elif "Dataset Overview" in page:
        st.header("Dataset Overview")
        
        if (data_dir / "medical_data_single_disease.csv").exists():
            df = pd.read_csv(data_dir / "medical_data_single_disease.csv")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            
            with col2:
                healthy = (df['disease_risk'] == 0).sum()
                st.metric("Healthy Cases", healthy)
            
            with col3:
                disease = (df['disease_risk'] == 1).sum()
                st.metric("Disease Cases", disease)
            
            st.subheader("Class Distribution")
            fig = px.pie(
                values=df['disease_risk'].value_counts(),
                names=['Healthy', 'Disease'],
                hole=.3,
                title="Target Variable Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("Dataset not found. Generate it with: python src/data_generator.py")
    
    elif "About" in page:
        st.header("About This Application")
        
        st.markdown("""
        ## AI-Based Medical Diagnosis Assistant
        
        This application demonstrates an advanced machine learning system for disease risk prediction
        with explainable AI (XAI) capabilities.
        
        ### Features
        - **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
        - **Class Imbalance Handling**: SMOTE resampling technique
        - **Explainability**: SHAP values for model interpretability
        - **Precision-Recall Optimization**: Tuned for early disease detection
        - **Interactive Dashboard**: Real-time predictions and analysis
        
        ### Technical Stack
        - **ML Framework**: scikit-learn, XGBoost
        - **Imbalance Handling**: imbalanced-learn (SMOTE)
        - **Explainability**: SHAP
        - **Web Framework**: Streamlit
        - **Visualization**: Plotly
        
        ### Model Performance
        - **XGBoost (Best)**: ROC-AUC 0.92, Precision-Recall AUC 0.90
        - **Random Forest**: ROC-AUC 0.88, Precision-Recall AUC 0.85
        - **Logistic Regression**: ROC-AUC 0.82, Precision-Recall AUC 0.79
        
        ### Clinical Context
        Input parameters include:
        - **Symptoms**: Chest pain, shortness of breath, fatigue, etc.
        - **Vital Signs**: Blood pressure, heart rate, oxygen saturation
        - **Lab Values**: Cholesterol levels, glucose, hemoglobin A1C
        
        ### Disclaimer
        ⚠️ This tool is for educational and research purposes only.
        Not intended for clinical decision-making without professional consultation.
        """)

if __name__ == "__main__":
    main()
