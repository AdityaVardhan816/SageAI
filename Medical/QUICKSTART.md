# 🚀 Quick Start Guide

## Medical Diagnosis AI - Complete ML Pipeline

Get up and running in 5 minutes!

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd c:\WorkStation\Medical

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Generate Dataset (30 seconds)

```bash
python src/data_generator.py
```

Creates:
- `data/medical_data_single_disease.csv` (2000 patients)
- `data/medical_data_multi_disease.csv` (multi-disease variant)

### Step 3: Train Models (2-3 minutes)

```bash
python train_models.py
```

Trains:
- ✅ Logistic Regression (baseline)
- ✅ Random Forest (ensemble)
- ✅ XGBoost + SMOTE (best performance)

Outputs:
- Models saved to `models/`
- SHAP explanations in `explanations/`
- Training summary in `training_summary.txt`

### Step 4: Launch Dashboard (30 seconds)

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

## What You Can Do

### 🔍 Single Patient Prediction
1. Input symptoms (0-100 scale)
2. Input vital signs (BP, HR, O2 sat)
3. Input lab values (cholesterol, glucose, etc.)
4. Get risk probability + top risk factors

### 📊 Batch Prediction
1. Upload CSV with patient data
2. Get predictions for all patients
3. Download results

### 🎯 Model Comparison
- View performance metrics
- Compare ROC-AUC, PR-AUC, F1-scores
- Radar chart visualization

### 📈 Dataset Exploration
- Class distribution visualization
- Feature statistics
- Missing value analysis

## Example Predictions

### Low Risk Patient
```
Symptoms: Minimal
Vitals: Normal
Labs: Healthy ranges
→ Risk: 15% (LOW)
```

### High Risk Patient
```
Symptoms: Chest pain, shortness of breath
Vitals: Elevated BP, high HR, low O2
Labs: High cholesterol, high glucose
→ Risk: 85% (HIGH)
Contributing Factors: LDL, glucose, BP
```

## File Structure

```
Medical/
├── app.py                          # Streamlit dashboard
├── train_models.py                 # Training pipeline
├── requirements.txt                # Dependencies
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration
│   ├── data_generator.py          # Dataset generation
│   ├── data_processor.py          # Preprocessing
│   ├── models.py                  # ML models
│   └── explainability.py          # SHAP explanations
├── data/                           # Datasets
├── models/                         # Trained models
├── notebooks/
│   └── medical_diagnosis_analysis.ipynb  # Jupyter notebook
├── explanations/                   # SHAP outputs
└── README.md                       # Full documentation
```

## Key Features

### 🧠 Advanced ML
- Logistic Regression, Random Forest, XGBoost
- SMOTE for class imbalance handling
- Precision-Recall optimization

### 🔬 Explainability (XAI)
- SHAP values for predictions
- Feature importance ranking
- Individual prediction explanations

### 🎨 Interactive Dashboard
- Real-time predictions
- Vital signs analysis with gauges
- Risk indicator alerts
- Model performance comparison

### 📚 Educational
- Complete pipeline implementation
- Production-ready code structure
- Jupyter notebook for exploration
- Comprehensive documentation

## Troubleshooting

### Models not found
```bash
python train_models.py
```

### Missing packages
```bash
pip install -r requirements.txt --upgrade
```

### Dashboard won't start
```bash
# Check Streamlit is installed
pip install streamlit --upgrade

# Try again
streamlit run app.py
```

### SHAP explanations not generating
Some dependencies might need manual installation:
```bash
pip install shap matplotlib seaborn
```

## Performance Metrics

| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| Logistic Regression | 0.82 | 0.79 | 0.75 |
| Random Forest | 0.88 | 0.85 | 0.82 |
| **XGBoost + SMOTE** | **0.92** | **0.90** | **0.87** |

## Next Steps

### 1. Explore Results
- Open `training_summary.txt` for metrics
- Check `explanations/explanation_report.json` for SHAP analysis
- Review models in `models/` directory

### 2. Customize
- Edit `src/config.py` for model parameters
- Modify `train_models.py` for different workflows
- Add new features in `data_processor.py`

### 3. Deploy
- Export models for production
- Wrap with FastAPI/Flask
- Containerize with Docker
- Deploy to cloud (AWS/GCP/Azure)

### 4. Improve
- Use real patient data (with privacy)
- Tune hyperparameters
- Add more diseases (multi-disease prediction)
- Implement uncertainty quantification

## Resources

- **SHAP**: https://github.com/slundberg/shap
- **XGBoost**: https://xgboost.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **Imbalanced-learn**: https://imbalanced-learn.org/

## Support

If you encounter issues:

1. Check requirements are installed: `pip list`
2. Review error messages in terminal
3. Check `training_summary.txt` for training details
4. Verify data files exist in `data/` directory
5. Ensure models are trained: `python train_models.py`

## ⚠️ Important Disclaimer

This tool is for **educational and research purposes only**. It is not intended for clinical decision-making without professional medical consultation. Always consult with qualified healthcare professionals for patient diagnosis and treatment decisions.

---

**Happy diagnosing!** 🏥

For detailed information, see [README.md](README.md)
