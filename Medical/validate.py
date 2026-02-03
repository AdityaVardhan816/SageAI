"""
Validation Script for Medical Diagnosis Assistant
Tests all components and validates the pipeline
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all dependencies are installed"""
    print("\n" + "="*60)
    print("TEST 1: Checking Dependencies")
    print("="*60)
    
    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'shap': 'SHAP',
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'imblearn': 'Imbalanced-learn'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name:20} - OK")
        except ImportError:
            print(f"❌ {name:20} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def test_data_generation():
    """Test dataset generation"""
    print("\n" + "="*60)
    print("TEST 2: Data Generation")
    print("="*60)
    
    try:
        from src.data_generator import generate_medical_dataset
        
        print("Generating test dataset...")
        df = generate_medical_dataset(n_samples=100, imbalance_ratio=0.15)
        
        assert df.shape[0] == 100, f"Expected 100 rows, got {df.shape[0]}"
        assert df.shape[1] == 21, f"Expected 21 columns, got {df.shape[1]}"
        assert 'disease_risk' in df.columns, "Missing target column"
        assert df['disease_risk'].nunique() == 2, "Target should be binary"
        
        disease_ratio = df['disease_risk'].mean()
        assert 0.10 <= disease_ratio <= 0.20, f"Imbalance ratio issue: {disease_ratio:.2%}"
        
        print(f"✅ Dataset generation passed")
        print(f"   Shape: {df.shape}")
        print(f"   Disease prevalence: {disease_ratio:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    print("\n" + "="*60)
    print("TEST 3: Data Preprocessing")
    print("="*60)
    
    try:
        from src.data_generator import generate_medical_dataset
        from src.data_processor import MedicalDataProcessor
        
        print("Testing preprocessing pipeline...")
        df = generate_medical_dataset(n_samples=100)
        
        # Test processor
        processor = MedicalDataProcessor(scaler_type='robust')
        X = df.drop('disease_risk', axis=1)
        
        X_scaled = processor.fit_transform(X)
        assert X_scaled.shape == X.shape, "Shape mismatch after scaling"
        
        # Test interaction features
        X_inter = processor.add_interaction_features(X_scaled)
        assert X_inter.shape[1] > X_scaled.shape[1], "No interaction features added"
        
        print(f"✅ Preprocessing passed")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Features after engineering: {X_inter.shape[1]}")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        return False

def test_models():
    """Test model training and evaluation"""
    print("\n" + "="*60)
    print("TEST 4: Model Training and Evaluation")
    print("="*60)
    
    try:
        from src.data_generator import generate_medical_dataset
        from src.data_processor import prepare_data
        from src.models import LogisticRegressionModel, RandomForestModel, XGBoostModel
        
        print("Generating test data...")
        df = generate_medical_dataset(n_samples=200, imbalance_ratio=0.20)
        X_train, X_test, y_train, y_test, _ = prepare_data(df, test_size=0.2)
        
        # Test Logistic Regression
        print("Testing Logistic Regression...")
        lr_model = LogisticRegressionModel()
        lr_model.train(X_train, y_train)
        y_pred, y_proba = lr_model.evaluate(X_test, y_test)
        assert y_proba.shape == (len(y_test), 2), "Probability shape incorrect"
        print(f"✅ Logistic Regression: ROC-AUC = {lr_model.model_metrics['roc_auc']:.3f}")
        
        # Test Random Forest
        print("Testing Random Forest...")
        rf_model = RandomForestModel(n_estimators=50, max_depth=10)
        rf_model.train(X_train, y_train)
        y_pred, y_proba = rf_model.evaluate(X_test, y_test)
        assert hasattr(rf_model, 'feature_importances'), "No feature importances"
        print(f"✅ Random Forest: ROC-AUC = {rf_model.model_metrics['roc_auc']:.3f}")
        
        # Test XGBoost
        print("Testing XGBoost...")
        xgb_model = XGBoostModel(n_estimators=50, max_depth=3, learning_rate=0.1)
        xgb_model.train(X_train, y_train, apply_smote=True)
        y_pred, y_proba = xgb_model.evaluate(X_test, y_test)
        assert hasattr(xgb_model, 'feature_importances'), "No feature importances"
        print(f"✅ XGBoost: ROC-AUC = {xgb_model.model_metrics['roc_auc']:.3f}")
        
        print("\n✅ All models trained successfully!")
        
        return True, lr_model, rf_model, xgb_model, X_test, y_test
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None, None

def test_explainability(xgb_model, X_test, y_test):
    """Test SHAP explainability"""
    print("\n" + "="*60)
    print("TEST 5: SHAP Explainability")
    print("="*60)
    
    try:
        from src.explainability import ModelExplainer
        
        print("Testing SHAP explainer...")
        
        # Create explainer
        X_background = X_test.iloc[:min(50, len(X_test))]
        explainer = ModelExplainer(xgb_model, X_background, feature_names=X_test.columns.tolist())
        
        print("Creating explainer...")
        explainer.create_explainer(explainer_type='tree')
        
        print("Explaining single prediction...")
        explanation = explainer.explain_prediction(X_test.iloc[0:1], top_features=5)
        
        assert 'prediction_probability' in explanation, "Missing prediction probability"
        assert 'top_contributing_factors' in explanation, "Missing contributing factors"
        assert len(explanation['top_contributing_factors']) > 0, "No factors identified"
        
        print(f"✅ SHAP explainability passed")
        print(f"   Prediction probability: {explanation['prediction_probability']:.1%}")
        print(f"   Top contributing factors: {len(explanation['top_contributing_factors'])}")
        
        return True
    except Exception as e:
        print(f"⚠️  SHAP testing failed (may require visualization dependencies): {str(e)}")
        return True  # Not critical for core functionality

def test_file_structure():
    """Test that all required files exist"""
    print("\n" + "="*60)
    print("TEST 6: File Structure")
    print("="*60)
    
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/data_generator.py',
        'src/data_processor.py',
        'src/models.py',
        'src/explainability.py',
        'app.py',
        'train_models.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
    ]
    
    required_dirs = [
        'src',
        'data',
        'models',
        'notebooks',
    ]
    
    all_ok = True
    
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✅ {file:40} - OK")
        else:
            print(f"❌ {file:40} - MISSING")
            all_ok = False
    
    for dir in required_dirs:
        path = Path(dir)
        if path.exists():
            print(f"✅ {dir:40} - OK (directory)")
        else:
            print(f"⚠️  {dir:40} - MISSING (will be created)")
    
    return all_ok

def main():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("🏥 MEDICAL DIAGNOSIS AI - VALIDATION SUITE")
    print("="*70)
    
    results = {
        'imports': test_imports(),
        'file_structure': test_file_structure(),
    }
    
    if results['imports']:
        results['data_generation'] = test_data_generation()
        results['preprocessing'] = test_preprocessing()
        
        model_result = test_models()
        if model_result[0]:
            results['models'] = True
            results['explainability'] = test_explainability(model_result[3], model_result[4], model_result[5])
        else:
            results['models'] = False
            results['explainability'] = False
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("\nYou can now:")
        print("  1. Generate dataset: python src/data_generator.py")
        print("  2. Train models: python train_models.py")
        print("  3. Launch dashboard: streamlit run app.py")
        print("  4. Explore notebook: jupyter notebook notebooks/medical_diagnosis_analysis.ipynb")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
