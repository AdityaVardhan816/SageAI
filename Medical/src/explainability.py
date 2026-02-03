"""
Explainability Module
Implements SHAP values and other explainability techniques
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    Provides explainability for model predictions using SHAP values
    """
    
    def __init__(self, model, X_background, feature_names=None):
        """
        Initialize explainer
        
        Parameters:
        - model: Trained model with predict function
        - X_background: Background data for SHAP (typically training data sample)
        - feature_names: Names of features
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, explainer_type='tree'):
        """
        Create SHAP explainer
        
        Parameters:
        - explainer_type: 'tree', 'kernel', or 'gradient'
        """
        print(f"Creating {explainer_type} SHAP explainer...")
        
        # Convert data if needed
        if isinstance(self.X_background, pd.DataFrame):
            X_bg = self.X_background.values
        else:
            X_bg = self.X_background
        
        # Create explainer based on model type
        model_obj = self.model.model if hasattr(self.model, 'model') else self.model
        
        if explainer_type == 'tree':
            # Tree explainer works with tree-based models
            try:
                self.explainer = shap.TreeExplainer(model_obj)
            except Exception as e:
                print(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
                self.explainer = shap.KernelExplainer(
                    lambda x: model_obj.predict_proba(x)[:, 1],
                    X_bg[:min(100, len(X_bg))]
                )
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                lambda x: model_obj.predict_proba(x)[:, 1],
                X_bg[:min(100, len(X_bg))]
            )
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(
                model_obj,
                X_bg[:min(100, len(X_bg))]
            )
        
        print("Explainer created successfully!")
        return self
    
    def explain_prediction(self, X_instance, top_features=5):
        """
        Explain prediction for a single instance
        
        Parameters:
        - X_instance: Single instance to explain
        - top_features: Number of top contributing features to return
        
        Returns:
        - Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Get model prediction
        model_obj = self.model.model if hasattr(self.model, 'model') else self.model
        
        if isinstance(X_instance, pd.DataFrame):
            X_inst = X_instance.values
        else:
            X_inst = X_instance
        
        # Ensure 2D array
        if X_inst.ndim == 1:
            X_inst = X_inst.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_inst)
        
        # Handle output format
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class SHAP values
        
        # Get prediction probability
        pred_proba = model_obj.predict_proba(X_inst)[0, 1]
        
        # Get feature values
        if isinstance(X_instance, pd.DataFrame):
            features = X_instance.columns.tolist()
            values = X_instance.values[0]
        else:
            features = self.feature_names or [f"feature_{i}" for i in range(X_inst.shape[1])]
            values = X_inst[0]
        
        # Create explanation
        shap_vals = shap_values[0]
        feature_importance = list(zip(features, shap_vals, values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation = {
            'prediction_probability': float(pred_proba),
            'top_contributing_factors': [
                {
                    'feature': f[0],
                    'shap_value': float(f[1]),
                    'feature_value': float(f[2]),
                    'direction': 'increases_risk' if f[1] > 0 else 'decreases_risk'
                }
                for f in feature_importance[:top_features]
            ],
            'all_features': [
                {
                    'feature': f[0],
                    'shap_value': float(f[1]),
                    'feature_value': float(f[2])
                }
                for f in feature_importance
            ]
        }
        
        return explanation
    
    def explain_dataset(self, X_data, max_samples=None):
        """
        Calculate SHAP values for entire dataset
        
        Parameters:
        - X_data: Data to explain
        - max_samples: Maximum samples to explain (for performance)
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        if isinstance(X_data, pd.DataFrame):
            X = X_data.values
        else:
            X = X_data
        
        if max_samples and len(X) > max_samples:
            X = X[:max_samples]
            print(f"Using {max_samples} samples for explanation")
        
        print(f"Calculating SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)
        
        # Handle output format
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class SHAP values
        
        self.shap_values = shap_values
        return shap_values
    
    def plot_summary(self, X_data, plot_type='bar', save_path=None):
        """
        Create SHAP summary plots
        
        Parameters:
        - X_data: Data to visualize
        - plot_type: 'bar', 'beeswarm', or 'violin'
        - save_path: Path to save plot
        """
        if self.shap_values is None:
            print("Calculating SHAP values for dataset...")
            self.explain_dataset(X_data)
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'bar':
            shap.summary_plot(self.shap_values, X_data, plot_type="bar", show=False)
        elif plot_type == 'beeswarm':
            shap.summary_plot(self.shap_values, X_data, show=False)
        elif plot_type == 'violin':
            shap.summary_plot(self.shap_values, X_data, plot_type="violin", show=False)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_force(self, X_instance, save_path=None):
        """
        Create SHAP force plot for single instance
        
        Parameters:
        - X_instance: Single instance to visualize
        - save_path: Path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created.")
        
        if isinstance(X_instance, pd.DataFrame):
            X_inst = X_instance.values
        else:
            X_inst = X_instance
        
        if X_inst.ndim == 1:
            X_inst = X_inst.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X_inst)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create force plot
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        shap.force_plot(base_value, shap_values[0], X_inst[0], matplotlib=True, show=False)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to {save_path}")
        
        plt.close()
    
    def get_feature_importance_from_shap(self, top_n=10):
        """
        Get global feature importance from SHAP values
        """
        if self.shap_values is None:
            raise ValueError("Must call explain_dataset() first")
        
        # Mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)

def create_model_explanation_report(model, X_test, y_test, feature_names=None, 
                                     sample_indices=None, output_dir=None):
    """
    Create comprehensive explanation report
    
    Parameters:
    - model: Trained model
    - X_test: Test data
    - y_test: Test labels
    - feature_names: Feature names
    - sample_indices: Specific indices to explain (default: first 3 misclassified + first correct)
    - output_dir: Directory to save reports
    """
    output_dir = Path(output_dir) if output_dir else Path('explanations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create explainer
    X_background = X_test.iloc[:min(100, len(X_test))] if isinstance(X_test, pd.DataFrame) else X_test[:min(100, len(X_test))]
    explainer = ModelExplainer(model, X_background, feature_names=feature_names)
    explainer.create_explainer(explainer_type='tree')
    
    # Explain dataset
    print("\nGenerating SHAP explanations for test set...")
    explainer.explain_dataset(X_test)
    
    # Create summary plots
    print("\nCreating SHAP summary plots...")
    explainer.plot_summary(X_test, plot_type='bar', save_path=output_dir / 'shap_summary_bar.png')
    
    # Get predictions for sample selection
    y_pred = model.predict(X_test)
    
    # Select samples to explain
    if sample_indices is None:
        # Get misclassified samples
        misclassified = np.where(y_pred != y_test)[0]
        correct = np.where(y_pred == y_test)[0]
        
        sample_indices = []
        if len(misclassified) > 0:
            sample_indices.extend(misclassified[:3])
        if len(correct) > 0:
            sample_indices.extend(correct[:2])
        
        sample_indices = sample_indices[:5]
    
    # Create explanations for selected samples
    explanations = []
    for idx in sample_indices:
        if isinstance(X_test, pd.DataFrame):
            X_inst = X_test.iloc[idx]
        else:
            X_inst = X_test[idx]
        
        explanation = explainer.explain_prediction(X_inst, top_features=5)
        explanations.append({
            'sample_index': int(idx),
            'true_label': int(y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]),
            'predicted_label': int(y_pred[idx]),
            'explanation': explanation
        })
    
    # Save explanations
    report = {
        'model_type': model.model_type,
        'total_samples': len(X_test),
        'sample_explanations': explanations,
        'feature_importance': explainer.get_feature_importance_from_shap().to_dict('records')
    }
    
    report_path = output_dir / 'explanation_report.json'
    with open(report_path, 'w') as f:
        import json
        json.dump(report, f, indent=2)
    
    print(f"Explanation report saved to {report_path}")
    
    return report
