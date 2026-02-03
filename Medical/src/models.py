"""
Model Training Module
Implements progressive ML stack: Logistic Regression → Random Forest → XGBoost with SMOTE
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, roc_curve, auc
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MedicalDiagnosisModel:
    """Base class for medical diagnosis models"""
    
    def __init__(self, model_type='logistic_regression', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.model_metrics = {}
        self.feature_importances = None
        self.feature_names = None
        
    def get_model_info(self):
        """Return model information"""
        return {
            'type': self.model_type,
            'parameters': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }

class LogisticRegressionModel(MedicalDiagnosisModel):
    """Baseline Logistic Regression model"""
    
    def __init__(self, random_state=42, max_iter=1000):
        super().__init__(model_type='logistic_regression', random_state=random_state)
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced'  # Handle imbalance
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"Training Logistic Regression Model")
        print(f"{'='*60}")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            X_train_values = X_train
        
        self.model.fit(X_train_values, y_train)
        print("Model trained successfully!")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict(X_values)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict_proba(X_values)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        self.model_metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        
        return y_pred, y_pred_proba

class RandomForestModel(MedicalDiagnosisModel):
    """Random Forest model"""
    
    def __init__(self, random_state=42, n_estimators=100, max_depth=15):
        super().__init__(model_type='random_forest', random_state=random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"Training Random Forest Model")
        print(f"{'='*60}")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            X_train_values = X_train
        
        self.model.fit(X_train_values, y_train)
        self.feature_importances = self.model.feature_importances_
        print("Model trained successfully!")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict(X_values)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict_proba(X_values)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        self.model_metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        
        return y_pred, y_pred_proba

class XGBoostModel(MedicalDiagnosisModel):
    """XGBoost model with SMOTE for handling imbalance"""
    
    def __init__(self, random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1):
        super().__init__(model_type='xgboost', random_state=random_state)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            scale_pos_weight=4,  # Handle imbalance (tune based on class ratio)
            eval_metric='logloss',
            tree_method='hist'
        )
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)
    
    def train(self, X_train, y_train, apply_smote=True):
        """
        Train the model
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - apply_smote: Whether to apply SMOTE for handling imbalance
        """
        print(f"\n{'='*60}")
        print(f"Training XGBoost Model (with {'SMOTE' if apply_smote else 'class weights'})")
        print(f"{'='*60}")
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            X_train_values = X_train
        
        # Apply SMOTE to handle class imbalance
        if apply_smote:
            print(f"Original class distribution: {np.bincount(y_train)}")
            X_train_values, y_train = self.smote.fit_resample(X_train_values, y_train)
            print(f"After SMOTE: {np.bincount(y_train)}")
        
        self.model.fit(X_train_values, y_train, verbose=0)
        self.feature_importances = self.model.feature_importances_
        print("Model trained successfully!")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict(X_values)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict_proba(X_values)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        self.model_metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        
        return y_pred, y_pred_proba

def get_feature_importance(model, feature_names=None, top_n=10):
    """
    Get feature importance from tree-based models
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    if feature_names is None:
        feature_names = model.feature_names or [f"feature_{i}" for i in range(len(importances))]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Important Features:")
    print(importance_df.head(top_n).to_string())
    
    return importance_df

def save_model(model, save_path):
    """Save model to disk"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path):
    """Load model from disk"""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
