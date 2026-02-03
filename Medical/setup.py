#!/usr/bin/env python
"""
Setup Script for Medical Diagnosis AI
Run this to initialize the project
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'notebooks', 'explanations', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Created project directories")

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required (found {sys.version_info.major}.{sys.version_info.minor})")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def print_welcome():
    """Print welcome message"""
    welcome = """
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          🏥 MEDICAL DIAGNOSIS AI ASSISTANT - SETUP 🏥            ║
║                                                                    ║
║        AI-Based Disease Risk Prediction with Explainable AI       ║
║                    (SHAP Integration)                             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """
    print(welcome)

def print_next_steps():
    """Print next steps"""
    steps = """
╔════════════════════════════════════════════════════════════════════╗
║                        NEXT STEPS                                  ║
╚════════════════════════════════════════════════════════════════════╝

1. Validate Installation
   └─ python validate.py

2. Generate Dataset
   └─ python src/data_generator.py

3. Train Models
   └─ python train_models.py

4. Launch Dashboard
   └─ streamlit run app.py
   └─ Opens at: http://localhost:8501

5. Explore Notebook
   └─ jupyter notebook notebooks/medical_diagnosis_analysis.ipynb

╔════════════════════════════════════════════════════════════════════╗
║                    PROJECT FEATURES                                ║
╚════════════════════════════════════════════════════════════════════╝

✨ ML Models (Progressive Stack)
   ├─ Logistic Regression (Baseline)
   ├─ Random Forest (Ensemble)
   └─ XGBoost + SMOTE (Best Performance)

🔬 Advanced Techniques
   ├─ SMOTE for Class Imbalance
   ├─ Feature Engineering
   ├─ Precision-Recall Optimization
   └─ SHAP for Explainability

📊 Interactive Dashboard
   ├─ Single Patient Predictions
   ├─ Batch Processing
   ├─ Model Comparison
   ├─ Dataset Exploration
   └─ SHAP Visualizations

📚 Documentation
   ├─ README.md (Comprehensive)
   ├─ QUICKSTART.md (5-min guide)
   ├─ PROJECT_SUMMARY.md (Overview)
   └─ Jupyter Notebook (Interactive)

╔════════════════════════════════════════════════════════════════════╗
║                      IMPORTANT NOTES                               ║
╚════════════════════════════════════════════════════════════════════╝

⚠️  DISCLAIMER
    This tool is for EDUCATIONAL & RESEARCH use only.
    NOT intended for clinical decision-making without
    professional medical consultation.

📋 REQUIREMENTS
    ✓ Python 3.8+
    ✓ pip/conda package manager
    ✓ 2GB disk space
    ✓ 4GB RAM recommended

🎓 LEARNING RESOURCES
    • SHAP: https://github.com/slundberg/shap
    • XGBoost: https://xgboost.readthedocs.io/
    • Streamlit: https://docs.streamlit.io/

💡 TIPS
    1. Start with QUICKSTART.md for quick setup
    2. Read README.md for detailed documentation
    3. Run validate.py to check installation
    4. Explore Jupyter notebook for in-depth analysis
    5. Customize config.py for your needs

📞 SUPPORT
    • Check README.md for troubleshooting
    • Review training_summary.txt after training
    • Inspect explanation_report.json for SHAP outputs

═════════════════════════════════════════════════════════════════════

            🚀 Ready to predict? Run: python validate.py

═════════════════════════════════════════════════════════════════════
    """
    print(steps)

def main():
    """Main setup function"""
    print_welcome()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install requirements
    print("\n" + "="*70)
    print("Installing Python dependencies...")
    print("="*70)
    if not install_requirements():
        return 1
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)
