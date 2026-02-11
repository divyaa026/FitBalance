"""
Test burnout model with sample cases
"""
from inference import predict_burnout_risk

# Test case 1: Low risk athlete
print("=" * 60)
print("TEST CASE 1: Low Risk Athlete")
print("=" * 60)
result = predict_burnout_risk(
    age=28,
    experience_years=5,
    workout_frequency=4,  # Moderate
    avg_sleep_hours=8.5,  # Good sleep
    stress_level=3,       # Low stress
    recovery_days=2,      # Good recovery
    hrv_avg=75,
    resting_hr=55,
    injury_history=0,
    nutrition_quality=8.5,
    gender='M',
    training_intensity='moderate'
)
print(f"Risk Score: {result['risk_score']:.2f}x average")
print(f"Expected time to burnout: {result['time_to_burnout_days']:.0f} days ({result['time_to_burnout_days']/365:.1f} years)")
print(f"1-year survival probability: {result['survival_probability_1yr']*100:.1f}%")

# Test case 2: High risk athlete
print("\n" + "=" * 60)
print("TEST CASE 2: High Risk Athlete")
print("=" * 60)
result = predict_burnout_risk(
    age=24,
    experience_years=1,    # Inexperienced
    workout_frequency=7,   # Excessive
    avg_sleep_hours=5.5,   # Poor sleep
    stress_level=9,        # High stress
    recovery_days=0,       # No recovery
    hrv_avg=35,            # Low HRV
    resting_hr=80,
    injury_history=3,
    nutrition_quality=4.0,
    gender='F',
    training_intensity='extreme'
)
print(f"Risk Score: {result['risk_score']:.2f}x average")
print(f"Expected time to burnout: {result['time_to_burnout_days']:.0f} days ({result['time_to_burnout_days']/365:.1f} years)")
print(f"1-year survival probability: {result['survival_probability_1yr']*100:.1f}%")

# Test case 3: Medium risk athlete
print("\n" + "=" * 60)
print("TEST CASE 3: Medium Risk Athlete")
print("=" * 60)
result = predict_burnout_risk(
    age=32,
    experience_years=8,
    workout_frequency=5,   # Above average
    avg_sleep_hours=7.0,   # Borderline sleep
    stress_level=6,        # Moderate stress
    recovery_days=1,       # Minimal recovery
    hrv_avg=55,
    resting_hr=65,
    injury_history=1,
    nutrition_quality=6.5,
    gender='M',
    training_intensity='high'
)
print(f"Risk Score: {result['risk_score']:.2f}x average")
print(f"Expected time to burnout: {result['time_to_burnout_days']:.0f} days ({result['time_to_burnout_days']/365:.1f} years)")
print(f"1-year survival probability: {result['survival_probability_1yr']*100:.1f}%")

