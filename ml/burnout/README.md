# Burnout Prediction Model

## Overview
Cox Proportional Hazards survival model for predicting athletic burnout risk.

## Model Performance
- **C-index (Test)**: 0.842 (from model_metrics.json)
- **Dataset**: 2,000 synthetic athlete records
- **Features**: 15 features including sleep, stress, workout frequency, HRV

## Features
1. **Demographics**: Age, gender, experience years
2. **Training**: Workout frequency, intensity, recovery days
3. **Health**: Sleep hours, stress level, HRV, resting HR
4. **History**: Injury history, nutrition quality

## Usage

```python
from ml_models.burnout.inference import predict_burnout_risk

result = predict_burnout_risk(
    age=28,
    experience_years=5,
    workout_frequency=4,
    avg_sleep_hours=8.0,
    stress_level=4,
    recovery_days=2,
    hrv_avg=70,
    resting_hr=60,
    injury_history=0,
    nutrition_quality=8.0,
    gender='M',
    training_intensity='moderate'
)

print(f"Risk score: {result['risk_score']}")
print(f"Time to burnout: {result['time_to_burnout_days']} days")
print(f"1-year survival: {result['survival_probability_1yr']}")
```

## Interpretation
- **Risk Score**: Relative hazard (1.0 = average, 2.0 = 2x average risk)
- **Time to Burnout**: Expected days until burnout event
- **Survival Probability**: Probability of not experiencing burnout in timeframe

## Risk Levels
- **Low Risk**: Survival prob > 80%, Risk score < 0.7
- **Medium Risk**: Survival prob 50-80%, Risk score 0.7-1.5
- **High Risk**: Survival prob 30-50%, Risk score 1.5-2.5
- **Critical Risk**: Survival prob < 30%, Risk score > 2.5

## Files
- `cox_model.pkl` - Trained Cox PH model
- `inference.py` - Inference function
- `train_cox_model.py` - Training script
- `burnout_dataset.csv` - Training dataset