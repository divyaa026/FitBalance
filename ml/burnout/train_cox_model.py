"""
Train Cox Proportional Hazards model for burnout prediction
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import json

def preprocess_data(df):
    """Preprocess data for Cox model"""
    
    # Encode categorical variables
    df['gender_M'] = (df['gender'] == 'M').astype(int)
    df['gender_F'] = (df['gender'] == 'F').astype(int)
    
    df['intensity_moderate'] = (df['training_intensity'] == 'moderate').astype(int)
    df['intensity_high'] = (df['training_intensity'] == 'high').astype(int)
    df['intensity_extreme'] = (df['training_intensity'] == 'extreme').astype(int)
    
    # Select features for model
    feature_cols = [
        'age', 'experience_years', 'workout_frequency', 'avg_sleep_hours',
        'stress_level', 'recovery_days', 'hrv_avg', 'resting_hr',
        'injury_history', 'nutrition_quality',
        'gender_M', 'gender_F',
        'intensity_moderate', 'intensity_high', 'intensity_extreme'
    ]
    
    # Create modeling dataframe
    model_df = df[feature_cols + ['observation_days', 'event_burnout']].copy()
    model_df.rename(columns={
        'observation_days': 'duration',
        'event_burnout': 'event'
    }, inplace=True)
    
    return model_df, feature_cols

def train_cox_model():
    """Train Cox PH model"""
    
    # Load data
    df = pd.read_csv('ml_models/burnout/burnout_dataset.csv')
    print(f"[INFO] Loaded {len(df)} athlete records")
    
    # Preprocess
    model_df, feature_cols = preprocess_data(df)
    
    # Split data
    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)
    print(f"   Training: {len(train_df)}, Testing: {len(test_df)}")
    
    # Initialize Cox model
    cph = CoxPHFitter(penalizer=0.01)
    
    # Train model
    print("\n[INFO] Training Cox Proportional Hazards model...")
    cph.fit(train_df, duration_col='duration', event_col='event')
    
    # Print model summary
    print("\n[INFO] Model Summary:")
    print(cph.summary)
    
    # Evaluate on test set
    print("\n[INFO] Model Performance:")
    
    # Concordance index (C-index)
    c_index_train = concordance_index(
        train_df['duration'], 
        -cph.predict_partial_hazard(train_df),
        train_df['event']
    )
    c_index_test = concordance_index(
        test_df['duration'],
        -cph.predict_partial_hazard(test_df),
        test_df['event']
    )
    
    print(f"   C-index (Train): {c_index_train:.3f}")
    print(f"   C-index (Test): {c_index_test:.3f}")
    print(f"   Target: >0.70 (Good), >0.80 (Excellent)")
    
    # Save model
    model_path = 'ml_models/burnout/cox_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(cph, f)
    print(f"\n[SUCCESS] Model saved to {model_path}")
    
    # Save feature names
    with open('ml_models/burnout/feature_names.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    # Save performance metrics
    metrics = {
        'c_index_train': float(c_index_train),
        'c_index_test': float(c_index_test),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'features': feature_cols
    }
    with open('ml_models/burnout/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return cph, test_df

def plot_survival_curves(cph, test_df):
    """Plot survival curves for different risk groups"""
    
    # Predict risk scores
    test_df['risk_score'] = cph.predict_partial_hazard(test_df)
    
    # Define risk groups
    low_risk = test_df[test_df['risk_score'] < test_df['risk_score'].quantile(0.33)]
    medium_risk = test_df[
        (test_df['risk_score'] >= test_df['risk_score'].quantile(0.33)) &
        (test_df['risk_score'] < test_df['risk_score'].quantile(0.67))
    ]
    high_risk = test_df[test_df['risk_score'] >= test_df['risk_score'].quantile(0.67)]
    
    # Fit Kaplan-Meier curves
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(12, 6))
    
    kmf.fit(low_risk['duration'], low_risk['event'], label='Low Risk')
    kmf.plot_survival_function()
    
    kmf.fit(medium_risk['duration'], medium_risk['event'], label='Medium Risk')
    kmf.plot_survival_function()
    
    kmf.fit(high_risk['duration'], high_risk['event'], label='High Risk')
    kmf.plot_survival_function()
    
    plt.xlabel('Days')
    plt.ylabel('Survival Probability (No Burnout)')
    plt.title('Burnout Survival Curves by Risk Group')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ml_models/burnout/survival_curves.png', dpi=300)
    print("[SUCCESS] Survival curves saved to survival_curves.png")

def create_inference_function():
    """Create inference function for deployment"""
    
    code = '''"""
Burnout Model Inference
"""
import pickle
import pandas as pd
import numpy as np

def load_model():
    with open('ml_models/burnout/cox_model.pkl', 'rb') as f:
        return pickle.load(f)

def predict_burnout_risk(
    age, experience_years, workout_frequency, avg_sleep_hours,
    stress_level, recovery_days, hrv_avg, resting_hr,
    injury_history, nutrition_quality, gender, training_intensity
):
    """
    Predict burnout risk for an athlete
    
    Returns:
        - risk_score: Relative hazard (1.0 = average risk)
        - time_to_burnout: Expected days until burnout
        - survival_probability_1yr: Probability of no burnout in 1 year
    """
    model = load_model()
    
    # Prepare features
    features = pd.DataFrame([{
        'age': age,
        'experience_years': experience_years,
        'workout_frequency': workout_frequency,
        'avg_sleep_hours': avg_sleep_hours,
        'stress_level': stress_level,
        'recovery_days': recovery_days,
        'hrv_avg': hrv_avg,
        'resting_hr': resting_hr,
        'injury_history': injury_history,
        'nutrition_quality': nutrition_quality,
        'gender_M': 1 if gender == 'M' else 0,
        'gender_F': 1 if gender == 'F' else 0,
        'intensity_moderate': 1 if training_intensity == 'moderate' else 0,
        'intensity_high': 1 if training_intensity == 'high' else 0,
        'intensity_extreme': 1 if training_intensity == 'extreme' else 0,
    }])
    
    # Predict
    risk_score = float(model.predict_partial_hazard(features).iloc[0])
    expected_time = float(model.predict_expectation(features).iloc[0])
    
    # Survival at 365 days
    survival_prob = float(model.predict_survival_function(features, times=[365]).iloc[0, 0])
    
    return {
        'risk_score': risk_score,
        'time_to_burnout_days': expected_time,
        'survival_probability_1yr': survival_prob
    }
'''
    
    with open('ml_models/burnout/inference.py', 'w') as f:
        f.write(code)
    print("[SUCCESS] Inference function created: ml_models/burnout/inference.py")

if __name__ == "__main__":
    cph, test_df = train_cox_model()
    plot_survival_curves(cph, test_df)
    create_inference_function()
    print("\n[SUCCESS] Training complete!")

