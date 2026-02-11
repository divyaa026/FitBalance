"""
Generate synthetic burnout prediction dataset
Real data would come from athlete tracking over time
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_burnout_dataset(n_athletes=2000):
    """Generate realistic athlete burnout dataset"""
    
    np.random.seed(42)
    
    data = {
        'athlete_id': range(n_athletes),
        'age': np.random.randint(18, 65, n_athletes),
        'gender': np.random.choice(['M', 'F', 'Other'], n_athletes),
        'experience_years': np.random.randint(0, 20, n_athletes),
        'workout_frequency': np.random.randint(1, 8, n_athletes),  # per week
        'avg_sleep_hours': np.random.uniform(4, 10, n_athletes),
        'stress_level': np.random.randint(1, 11, n_athletes),  # 1-10
        'recovery_days': np.random.randint(0, 7, n_athletes),
        'training_intensity': np.random.choice(['low', 'moderate', 'high', 'extreme'], n_athletes),
        'hrv_avg': np.random.uniform(20, 100, n_athletes),  # Heart Rate Variability
        'resting_hr': np.random.randint(45, 90, n_athletes),
        'injury_history': np.random.randint(0, 5, n_athletes),
        'nutrition_quality': np.random.uniform(1, 10, n_athletes),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate burnout risk factors
    risk_score = (
        (8 - df['workout_frequency']) * -5 +  # More workouts = higher risk
        (7 - df['avg_sleep_hours']) * 15 +    # Less sleep = higher risk
        df['stress_level'] * 10 +             # Higher stress = higher risk
        (7 - df['recovery_days']) * 12 +      # Less recovery = higher risk
        df['injury_history'] * 8 +
        (10 - df['nutrition_quality']) * 5 +
        (100 - df['hrv_avg']) * 0.5
    )
    
    # Calculate time to burnout (survival time in days)
    base_survival = 730  # 2 years baseline
    df['time_to_burnout'] = np.maximum(
        30, 
        base_survival - risk_score * 2 + np.random.normal(0, 50, n_athletes)
    )
    
    # Event indicator: 1 = burnout occurred, 0 = censored
    # Higher risk athletes more likely to experience burnout
    burnout_probability = 1 / (1 + np.exp(-0.01 * (risk_score - 50)))
    df['event_burnout'] = np.random.binomial(1, burnout_probability)
    
    # Add observation time (may be less than time_to_burnout if censored)
    df['observation_days'] = np.where(
        df['event_burnout'] == 1,
        df['time_to_burnout'],
        np.random.uniform(df['time_to_burnout'] * 0.5, df['time_to_burnout'] * 1.5, n_athletes)
    )
    
    return df

# Generate and save dataset
if __name__ == "__main__":
    df = generate_burnout_dataset(n_athletes=2000)
    df.to_csv('ml_models/burnout/burnout_dataset.csv', index=False)
    print(f"[SUCCESS] Generated dataset with {len(df)} athletes")
    print(f"   Burnout rate: {df['event_burnout'].mean()*100:.1f}%")
    print(f"   Avg time to burnout: {df['time_to_burnout'].mean():.0f} days")
