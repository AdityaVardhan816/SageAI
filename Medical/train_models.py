"""
Model Training Pipeline
Complete training workflow for all models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import generate_medical_dataset, create_multi_disease_dataset
from src.data_processor import prepare_data
from src.models import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel,
    get_feature_importance, save_model
)
from src.explainability import create_model_explanation_report

def train_all_models():
    """Train and evaluate all models"""
    
    print("\n" + "="*80)
    print("MEDICAL DIAGNOSIS AI - MODEL TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Generate/Load Data
    print("STEP 1: Data Generation and Preparation")
    print("-"*80)
    
    data_dir = Path("data")
    data_path = data_dir / "medical_data_single_disease.csv"
    
    if data_path.exists():
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating synthetic medical dataset...")
        df = generate_medical_dataset(n_samples=2000, imbalance_ratio=0.15)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Disease prevalence: {df['disease_risk'].mean():.2%}")
    print(f"Class distribution:\n{df['disease_risk'].value_counts()}\n")
    
    # Step 2: Prepare Data
    print("\nSTEP 2: Data Preprocessing")
    print("-"*80)
    
    X_train, X_test, y_train, y_test, processor = prepare_data(
        df, 
        target_column='disease_risk',
        test_size=0.2
    )
    
    print(f"\nFeatures after engineering: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 3: Train Logistic Regression
    print("\n\nSTEP 3: Model Training")
    print("="*80)
    
    start_time = time.time()
    
    print("\n[1/3] Logistic Regression")
    print("-"*80)
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    y_pred_lr, y_proba_lr = lr_model.evaluate(X_test, y_test)
    
    # Step 4: Train Random Forest
    print("\n[2/3] Random Forest")
    print("-"*80)
    rf_model = RandomForestModel(n_estimators=100, max_depth=15)
    rf_model.train(X_train, y_train)
    y_pred_rf, y_proba_rf = rf_model.evaluate(X_test, y_test)
    
    print("\nFeature Importance (Random Forest):")
    get_feature_importance(rf_model, feature_names=X_train.columns, top_n=10)
    
    # Step 5: Train XGBoost with SMOTE
    print("\n[3/3] XGBoost with SMOTE")
    print("-"*80)
    xgb_model = XGBoostModel(n_estimators=150, max_depth=5, learning_rate=0.1)
    xgb_model.train(X_train, y_train, apply_smote=True)
    y_pred_xgb, y_proba_xgb = xgb_model.evaluate(X_test, y_test)
    
    print("\nFeature Importance (XGBoost):")
    get_feature_importance(xgb_model, feature_names=X_train.columns, top_n=10)
    
    training_time = time.time() - start_time
    
    # Step 6: Model Comparison
    print("\n\nSTEP 4: Model Comparison")
    print("="*80)
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'ROC-AUC': [
            lr_model.model_metrics['roc_auc'],
            rf_model.model_metrics['roc_auc'],
            xgb_model.model_metrics['roc_auc']
        ],
        'PR-AUC': [
            lr_model.model_metrics['pr_auc'],
            rf_model.model_metrics['pr_auc'],
            xgb_model.model_metrics['pr_auc']
        ],
        'F1-Score': [
            lr_model.model_metrics['f1_score'],
            rf_model.model_metrics['f1_score'],
            xgb_model.model_metrics['f1_score']
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    best_model = df_comparison.loc[df_comparison['ROC-AUC'].idxmax()]
    print(f"\n🏆 Best Model: {best_model['Model']} (ROC-AUC: {best_model['ROC-AUC']:.4f})")
    
    # Step 7: Save Models
    print("\n\nSTEP 5: Saving Models")
    print("-"*80)
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    save_model(lr_model, models_dir / "logistic_regression_model.pkl")
    save_model(rf_model, models_dir / "random_forest_model.pkl")
    save_model(xgb_model, models_dir / "xgboost_model.pkl")
    
    # Step 8: Generate Explanations
    print("\n\nSTEP 6: Generating Model Explanations")
    print("-"*80)
    
    explanations_dir = Path("explanations")
    explanations_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating SHAP explanations for XGBoost model...")
    try:
        report = create_model_explanation_report(
            xgb_model,
            X_test,
            y_test,
            feature_names=X_train.columns.tolist(),
            output_dir=explanations_dir
        )
        print("✅ Explanations generated successfully!")
    except Exception as e:
        print(f"⚠️ Could not generate explanations: {str(e)}")
        print("   This may be due to missing visualization dependencies")
    
    # Step 9: Summary Report
    print("\n\nSTEP 7: Training Summary")
    print("="*80)
    
    summary = f"""
    ✅ Training Complete!
    
    Dataset Information:
    - Total Samples: {len(df)}
    - Training Samples: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)
    - Test Samples: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)
    - Disease Prevalence: {df['disease_risk'].mean():.2%}
    - Features (after engineering): {X_train.shape[1]}
    
    Model Performance Summary:
    
    1. Logistic Regression (Baseline)
       - ROC-AUC: {lr_model.model_metrics['roc_auc']:.4f}
       - PR-AUC: {lr_model.model_metrics['pr_auc']:.4f}
       - F1-Score: {lr_model.model_metrics['f1_score']:.4f}
    
    2. Random Forest (Tree Ensemble)
       - ROC-AUC: {rf_model.model_metrics['roc_auc']:.4f}
       - PR-AUC: {rf_model.model_metrics['pr_auc']:.4f}
       - F1-Score: {rf_model.model_metrics['f1_score']:.4f}
    
    3. XGBoost + SMOTE (Best)
       - ROC-AUC: {xgb_model.model_metrics['roc_auc']:.4f}
       - PR-AUC: {xgb_model.model_metrics['pr_auc']:.4f}
       - F1-Score: {xgb_model.model_metrics['f1_score']:.4f}
    
    File Locations:
    - Models saved to: {models_dir.absolute()}
    - Explanations saved to: {explanations_dir.absolute()}
    
    Training Time: {training_time:.2f} seconds
    
    Next Steps:
    1. Run Streamlit dashboard: streamlit run app.py
    2. View explanations: Check {explanations_dir}/explanation_report.json
    3. Deploy model: Use saved .pkl files in your application
    """
    
    print(summary)
    
    # Save summary to file
    summary_path = Path("training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n📄 Summary saved to {summary_path}")

if __name__ == "__main__":
    try:
        train_all_models()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
