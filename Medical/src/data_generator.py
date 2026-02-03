"""
Medical Dataset Generator
Generates synthetic medical data with class imbalance (realistic for disease datasets)
Includes symptoms, vitals, and lab values
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from pathlib import Path

def generate_medical_dataset(n_samples=2000, imbalance_ratio=0.15, random_state=42):
    """
    Generate synthetic medical dataset with realistic features
    
    Parameters:
    - n_samples: Total number of samples
    - imbalance_ratio: Proportion of positive cases (disease)
    - random_state: For reproducibility
    
    Returns:
    - DataFrame with medical features
    """
    np.random.seed(random_state)
    
    # Generate synthetic data with built-in imbalance
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        random_state=random_state,
        flip_y=0.02
    )
    
    # Define feature names with medical context
    feature_names = [
        # Symptoms (binary/categorical converted to continuous 0-1)
        'chest_pain_severity',
        'shortness_of_breath',
        'fatigue_level',
        'dizziness',
        'headache_frequency',
        'nausea_level',
        
        # Vital Signs
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
        'white_blood_cells',
    ]
    
    # Scale features to realistic medical ranges
    X[:, 0] = (X[:, 0] + 3) * 3.33  # Chest pain: 0-10 scale
    X[:, 1] = np.clip((X[:, 1] + 2) * 20, 0, 100)  # SOB: 0-100
    X[:, 2] = np.clip((X[:, 2] + 2) * 15, 0, 100)  # Fatigue: 0-100
    X[:, 3] = np.clip((X[:, 3] + 2) * 15, 0, 100)  # Dizziness: 0-100
    X[:, 4] = np.clip((X[:, 4] + 2) * 10, 0, 100)  # Headache freq: 0-100
    X[:, 5] = np.clip((X[:, 5] + 2) * 15, 0, 100)  # Nausea: 0-100
    
    X[:, 6] = np.clip(100 + X[:, 6] * 25, 80, 200)  # Systolic BP: 80-200
    X[:, 7] = np.clip(70 + X[:, 7] * 20, 50, 130)   # Diastolic BP: 50-130
    X[:, 8] = np.clip(70 + X[:, 8] * 15, 40, 120)   # Heart rate: 40-120
    X[:, 9] = np.clip(37 + X[:, 9] * 1.5, 35.5, 40.5)  # Temperature: 35.5-40.5
    X[:, 10] = np.clip(16 + X[:, 10] * 6, 8, 25)    # Respiratory rate: 8-25
    X[:, 11] = np.clip(95 + X[:, 11] * 3, 85, 100)  # O2 Sat: 85-100
    
    X[:, 12] = np.clip(150 + X[:, 12] * 50, 100, 300)  # Total cholesterol: 100-300
    X[:, 13] = np.clip(100 + X[:, 13] * 40, 30, 190)   # LDL: 30-190
    X[:, 14] = np.clip(50 + X[:, 14] * 15, 20, 100)    # HDL: 20-100
    X[:, 15] = np.clip(100 + X[:, 15] * 60, 30, 300)   # Triglycerides: 30-300
    X[:, 16] = np.clip(100 + X[:, 16] * 40, 70, 200)   # Glucose: 70-200
    X[:, 17] = np.clip(5.5 + X[:, 17] * 2, 4, 13)      # A1C: 4-13%
    X[:, 18] = np.clip(1 + X[:, 18] * 0.4, 0.6, 1.5)   # Creatinine: 0.6-1.5
    X[:, 19] = np.clip(7 + X[:, 19] * 2, 4.5, 11)      # WBC: 4.5-11 K/uL
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['disease_risk'] = y  # 1 = disease present, 0 = healthy
    
    return df

def create_multi_disease_dataset(n_samples=2000, random_state=42):
    """
    Generate dataset for multi-disease prediction
    """
    np.random.seed(random_state)
    
    df = generate_medical_dataset(n_samples, random_state=random_state)
    
    # Create multiple disease predictions based on feature combinations
    X = df.drop('disease_risk', axis=1).values
    
    # Disease 1: Cardiovascular (correlates with BP, cholesterol, heart rate)
    cardiovascular_risk = (
        (X[:, 6] > 140).astype(int) * 0.3 +  # High systolic
        (X[:, 12] > 200).astype(int) * 0.25 +  # High cholesterol
        (X[:, 8] > 100).astype(int) * 0.2 +  # High heart rate
        np.random.binomial(1, 0.1, n_samples) * 0.25
    ) > 0.5
    
    # Disease 2: Diabetes (correlates with glucose, A1C)
    diabetes_risk = (
        (X[:, 16] > 125).astype(int) * 0.4 +  # High glucose
        (X[:, 17] > 6.5).astype(int) * 0.4 +  # High A1C
        np.random.binomial(1, 0.15, n_samples) * 0.2
    ) > 0.5
    
    # Disease 3: Respiratory (correlates with SOB, O2 sat, respiration rate)
    respiratory_risk = (
        (X[:, 1] > 50).astype(int) * 0.3 +  # High SOB
        (X[:, 11] < 92).astype(int) * 0.35 +  # Low O2
        (X[:, 10] > 20).astype(int) * 0.25 +  # High resp rate
        np.random.binomial(1, 0.12, n_samples) * 0.1
    ) > 0.5
    
    df['cardiovascular_risk'] = cardiovascular_risk.astype(int)
    df['diabetes_risk'] = diabetes_risk.astype(int)
    df['respiratory_risk'] = respiratory_risk.astype(int)
    
    return df

if __name__ == "__main__":
    # Generate datasets
    print("Generating medical dataset...")
    df_single = generate_medical_dataset(n_samples=2000)
    
    print("Generating multi-disease dataset...")
    df_multi = create_multi_disease_dataset(n_samples=2000)
    
    # Save datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    df_single.to_csv(data_dir / 'medical_data_single_disease.csv', index=False)
    df_multi.to_csv(data_dir / 'medical_data_multi_disease.csv', index=False)
    
    print(f"\nDataset Statistics (Single Disease):")
    print(f"Shape: {df_single.shape}")
    print(f"Class distribution:\n{df_single['disease_risk'].value_counts()}")
    print(f"Imbalance ratio: {df_single['disease_risk'].mean():.2%}")
    
    print(f"\nDataset Statistics (Multi Disease):")
    print(f"Cardiovascular risk: {df_multi['cardiovascular_risk'].mean():.2%}")
    print(f"Diabetes risk: {df_multi['diabetes_risk'].mean():.2%}")
    print(f"Respiratory risk: {df_multi['respiratory_risk'].mean():.2%}")
    
    print("\nDatasets saved to /data directory")
