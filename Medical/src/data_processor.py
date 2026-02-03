"""
Feature Engineering Module
Handles data preprocessing, scaling, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

class MedicalDataProcessor:
    """
    Preprocesses medical data with appropriate scaling and feature engineering
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize processor
        
        Parameters:
        - scaler_type: 'standard' or 'robust' (robust handles outliers better)
        """
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Fit scaler on training data"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        self.scaler.fit(X_values)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(X_values.shape[1])]
        
        X_scaled = self.scaler.transform(X_values)
        return pd.DataFrame(X_scaled, columns=feature_names)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def add_interaction_features(self, X):
        """Add interaction features for medical data"""
        X_copy = X.copy()
        
        # Convert to numeric if DataFrame
        if isinstance(X_copy, pd.DataFrame):
            X_values = X_copy.values
            cols = X_copy.columns.tolist()
        else:
            X_values = X_copy
            cols = [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Add useful medical interactions
        interactions = []
        interaction_names = []
        
        # BP ratio (systolic/diastolic)
        if len(cols) >= 7:  # Has BP columns
            interactions.append(X_values[:, 6] / (X_values[:, 7] + 1e-6))  # systolic/diastolic
            interaction_names.append('bp_ratio')
        
        # Cholesterol ratio (total/HDL)
        if len(cols) >= 14:
            interactions.append(X_values[:, 12] / (X_values[:, 14] + 1e-6))  # total/HDL
            interaction_names.append('cholesterol_ratio')
        
        # Cholesterol risk score
        if len(cols) >= 15:
            interactions.append(X_values[:, 13] - X_values[:, 14])  # LDL - HDL
            interaction_names.append('ldl_hdl_diff')
        
        # Glucose-A1C interaction
        if len(cols) >= 18:
            interactions.append(X_values[:, 16] * X_values[:, 17])  # glucose * A1C
            interaction_names.append('glucose_a1c_product')
        
        if interactions:
            interactions_array = np.column_stack(interactions)
            if isinstance(X, pd.DataFrame):
                X_new = X.copy()
                for name, values in zip(interaction_names, interactions):
                    X_new[name] = values
                return X_new
            else:
                return np.hstack([X_values, interactions_array])
        
        return X
    
    def handle_outliers(self, X, method='iqr', threshold=1.5):
        """
        Handle outliers
        
        Parameters:
        - method: 'iqr' (Interquartile Range) or 'zscore'
        - threshold: IQR multiplier or Z-score threshold
        """
        X_copy = X.copy()
        
        if isinstance(X_copy, pd.DataFrame):
            cols = X_copy.columns
            for col in cols:
                if method == 'iqr':
                    Q1 = X_copy[col].quantile(0.25)
                    Q3 = X_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    X_copy[col] = X_copy[col].clip(lower_bound, upper_bound)
                elif method == 'zscore':
                    mean = X_copy[col].mean()
                    std = X_copy[col].std()
                    X_copy[col] = X_copy[col].clip(mean - threshold * std, mean + threshold * std)
        else:
            X_copy = np.clip(X_copy, np.percentile(X_copy, 0.1), np.percentile(X_copy, 99.9))
        
        return X_copy

def prepare_data(df, target_column='disease_risk', test_size=0.2, random_state=42):
    """
    Prepare data for modeling
    
    Parameters:
    - df: DataFrame with features and target
    - target_column: Name of target column
    - test_size: Test set size
    - random_state: Reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: Processed data splits
    """
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Process features
    processor = MedicalDataProcessor(scaler_type='robust')
    X_train_processed = processor.fit_transform(X_train)
    X_test_processed = processor.transform(X_test)
    
    # Add interaction features
    X_train_processed = processor.add_interaction_features(X_train_processed)
    X_test_processed = processor.add_interaction_features(X_test_processed)
    
    # Handle outliers
    X_train_processed = processor.handle_outliers(X_train_processed, method='iqr', threshold=2)
    X_test_processed = processor.handle_outliers(X_test_processed, method='iqr', threshold=2)
    
    print(f"Training set: {X_train_processed.shape}")
    print(f"Test set: {X_test_processed.shape}")
    print(f"Target distribution in train: {y_train.value_counts().to_dict()}")
    print(f"Target distribution in test: {y_test.value_counts().to_dict()}")
    
    return X_train_processed, X_test_processed, y_train, y_test, processor

def calculate_precision_recall_metrics(y_true, y_pred_proba, thresholds=[0.5]):
    """
    Calculate precision, recall, and F1 at different thresholds
    Useful for finding optimal threshold for precision-recall tradeoff
    """
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    metrics = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics.append({
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'f1': f1_val,
            'specificity': specificity,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        })
    
    return metrics, pr_auc, (precision, recall, pr_thresholds)
