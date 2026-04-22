"""
Burnout Model Inference
"""
import pickle
import pandas as pd
import numpy as np

def load_model():
    import os
    # Try multiple possible paths for the model
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'models', 'cox_ph_model.pkl'),
        os.path.join(os.path.dirname(__file__), 'cox_model.pkl'),
        os.path.join(os.path.dirname(__file__), 'models', 'cox_model.pkl'),
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    
    raise FileNotFoundError(f"Cox model not found. Tried: {possible_paths}")

def predict_burnout_risk(
    age, experience_years, workout_frequency, avg_sleep_hours,
    stress_level, recovery_days, hrv_avg, resting_hr,
    injury_history, nutrition_quality, gender, training_intensity
):
    """
    Predict burnout risk for an athlete
    
    Maps user inputs to the model's expected features:
    - sleep_quality: derived from avg_sleep_hours (1-10 scale)
    - stress_level: direct input (1-10)
    - workload: derived from workout_frequency and training_intensity
    - social_support: default moderate (5)
    - exercise_frequency: workout_frequency
    - hrv_score: derived from hrv_avg
    - recovery_time: recovery_days
    - work_life_balance: inverse of stress_level
    - nutrition_quality: direct input (1-10)
    - mental_fatigue: derived from stress and sleep
    
    Returns:
        - risk_score: Relative hazard (1.0 = average risk)
        - time_to_burnout: Expected days until burnout
        - survival_probability_1yr: Probability of no burnout in 1 year
    """
    model = load_model()
    
    # Map inputs to model features
    # sleep_quality: 1-10 scale based on hours (8h = 10, 5h = 1)
    sleep_quality = max(1, min(10, (avg_sleep_hours - 5) * 3))
    
    # workload: 1-10 based on frequency and intensity
    intensity_multiplier = {'low': 0.7, 'moderate': 1.0, 'high': 1.3, 'extreme': 1.6}.get(training_intensity, 1.0)
    workload = min(10, workout_frequency * intensity_multiplier)
    
    # social_support: default moderate value
    social_support = 5
    
    # exercise_frequency: map to 1-10 scale (7 workouts/week = 10)
    exercise_frequency = min(10, workout_frequency * 10 / 7)
    
    # hrv_score: normalize HRV (typical range 20-100 to 1-10)
    hrv_score = max(1, min(10, (hrv_avg - 20) / 8))
    
    # work_life_balance: inverse of stress (high stress = low balance)
    work_life_balance = max(1, 11 - stress_level)
    
    # mental_fatigue: derived from stress and sleep quality
    mental_fatigue = max(1, min(10, stress_level * 0.6 + (10 - sleep_quality) * 0.4))
    
    # Prepare features in the order the model expects
    features = pd.DataFrame([{
        'sleep_quality': sleep_quality,
        'stress_level': stress_level,
        'workload': workload,
        'social_support': social_support,
        'exercise_frequency': exercise_frequency,
        'hrv_score': hrv_score,
        'recovery_time': recovery_days,
        'work_life_balance': work_life_balance,
        'nutrition_quality': nutrition_quality,
        'mental_fatigue': mental_fatigue,
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
