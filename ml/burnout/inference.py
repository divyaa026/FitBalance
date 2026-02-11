"""
Burnout Model Inference
"""
import pickle
import pandas as pd
import numpy as np

def load_model():
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'cox_model.pkl')
    with open(model_path, 'rb') as f:
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
