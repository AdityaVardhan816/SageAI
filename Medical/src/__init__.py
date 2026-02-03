"""
Medical Diagnosis Assistant Package
Complete ML pipeline for disease risk prediction with explainable AI
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "AI-Based Medical Diagnosis Assistant with SHAP Explainability"

# Import main modules
from .data_generator import generate_medical_dataset, create_multi_disease_dataset
from .data_processor import MedicalDataProcessor, prepare_data
from .models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    get_feature_importance,
    save_model,
    load_model
)
from .explainability import ModelExplainer, create_model_explanation_report
from .config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    FEATURE_NAMES,
    FEATURE_RANGES,
    RISK_THRESHOLDS,
    CLINICAL_NORMAL_RANGES
)

__all__ = [
    'generate_medical_dataset',
    'create_multi_disease_dataset',
    'MedicalDataProcessor',
    'prepare_data',
    'LogisticRegressionModel',
    'RandomForestModel',
    'XGBoostModel',
    'get_feature_importance',
    'save_model',
    'load_model',
    'ModelExplainer',
    'create_model_explanation_report',
    'DATA_CONFIG',
    'MODEL_CONFIG',
    'FEATURE_NAMES',
    'FEATURE_RANGES',
    'RISK_THRESHOLDS',
    'CLINICAL_NORMAL_RANGES'
]
