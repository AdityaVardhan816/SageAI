# Configuration file for Medical Diagnosis Assistant

# Data Configuration
DATA_CONFIG = {
    'n_samples': 2000,
    'imbalance_ratio': 0.15,  # 15% disease, 85% healthy
    'test_size': 0.2,
    'random_state': 42
}

# Model Configuration
MODEL_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'solver': 'lbfgs',
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'scale_pos_weight': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}

# SMOTE Configuration
SMOTE_CONFIG = {
    'random_state': 42,
    'k_neighbors': 5,
    'sampling_strategy': 'auto'  # Balance minority class
}

# Feature Engineering Configuration
FEATURE_ENG_CONFIG = {
    'scaler_type': 'robust',  # 'robust' or 'standard'
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_threshold': 2.0,
    'add_interactions': True
}

# Evaluation Configuration
EVAL_CONFIG = {
    'decision_thresholds': [0.3, 0.4, 0.5, 0.6, 0.7],
    'optimal_threshold': 0.5,  # Balanced precision-recall
    'metrics': ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall']
}

# SHAP Configuration
SHAP_CONFIG = {
    'explainer_type': 'tree',  # 'tree', 'kernel', or 'gradient'
    'background_samples': 100,
    'top_features': 5,
    'plot_types': ['summary_bar', 'summary_beeswarm', 'force']
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Medical Diagnosis Assistant',
    'page_icon': '🏥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Feature Names
FEATURE_NAMES = [
    # Symptoms
    'chest_pain_severity',
    'shortness_of_breath',
    'fatigue_level',
    'dizziness',
    'headache_frequency',
    'nausea_level',
    # Vitals
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'body_temperature',
    'respiratory_rate',
    'oxygen_saturation',
    # Lab Values
    'cholesterol_total',
    'ldl_cholesterol',
    'hdl_cholesterol',
    'triglycerides',
    'glucose_fasting',
    'hemoglobin_a1c',
    'creatinine',
    'white_blood_cells'
]

# Feature Ranges (for validation and normalization)
FEATURE_RANGES = {
    'chest_pain_severity': (0, 10),
    'shortness_of_breath': (0, 100),
    'fatigue_level': (0, 100),
    'dizziness': (0, 100),
    'headache_frequency': (0, 100),
    'nausea_level': (0, 100),
    'systolic_bp': (80, 200),
    'diastolic_bp': (50, 130),
    'heart_rate': (40, 120),
    'body_temperature': (35.5, 40.5),
    'respiratory_rate': (8, 25),
    'oxygen_saturation': (85, 100),
    'cholesterol_total': (100, 300),
    'ldl_cholesterol': (30, 190),
    'hdl_cholesterol': (20, 100),
    'triglycerides': (30, 300),
    'glucose_fasting': (70, 200),
    'hemoglobin_a1c': (4, 13),
    'creatinine': (0.6, 1.5),
    'white_blood_cells': (4.5, 11)
}

# Risk Thresholds
RISK_THRESHOLDS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 1.0)
}

# Clinical Normal Ranges (for alerting)
CLINICAL_NORMAL_RANGES = {
    'systolic_bp': (90, 140),
    'diastolic_bp': (60, 90),
    'heart_rate': (60, 100),
    'oxygen_saturation': (95, 100),
    'glucose_fasting': (70, 100),
    'hemoglobin_a1c': (4, 6.5),
    'cholesterol_total': (0, 200),
    'ldl_cholesterol': (0, 100),
    'hdl_cholesterol': (40, 60),
    'triglycerides': (0, 150)
}
