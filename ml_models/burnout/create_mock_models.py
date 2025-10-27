"""
Create mock burnout prediction models for initial deployment
These can be replaced with real trained models later
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

def create_burnout_models():
    """Create mock burnout prediction models"""
    
    output_dir = "ml_models/burnout/models"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating mock burnout models...")
    
    # 1. Cox Proportional Hazards Model
    print("  - Creating Cox PH model...")
    
    try:
        from lifelines import CoxPHFitter
        
        # Generate small synthetic dataset for Cox model
        n_samples = 1000
        np.random.seed(42)
        
        data = pd.DataFrame({
            'sleep_quality': np.random.normal(70, 15, n_samples),
            'stress_level': np.random.normal(50, 20, n_samples),
            'workload': np.random.normal(60, 20, n_samples),
            'social_support': np.random.normal(65, 18, n_samples),
            'exercise_frequency': np.random.poisson(3, n_samples),
            'hrv_score': np.random.normal(60, 15, n_samples),
            'recovery_time': np.random.normal(7, 2, n_samples),
            'work_life_balance': np.random.normal(65, 18, n_samples),
            'nutrition_quality': np.random.normal(70, 15, n_samples),
            'mental_fatigue': np.random.normal(50, 20, n_samples)
        })
        
        # Generate outcome variables
        # Higher stress/workload/fatigue = higher burnout risk
        risk_score = (
            data['stress_level'] * 0.3 +
            data['workload'] * 0.3 +
            data['mental_fatigue'] * 0.2 -
            data['sleep_quality'] * 0.1 -
            data['social_support'] * 0.1
        )
        
        # Time to burnout (days)
        base_time = 365
        data['training_days'] = np.maximum(30, base_time - risk_score * 2 + np.random.normal(0, 50, n_samples))
        
        # Burnout event (1 if burnout occurred, 0 if censored)
        data['burnout_event'] = (risk_score > 50).astype(int)
        
        # Train Cox PH model
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(data, duration_col='training_days', event_col='burnout_event')
        
        # Save Cox model
        with open(f"{output_dir}/cox_ph_model.pkl", 'wb') as f:
            pickle.dump(cph, f)
        
        print(f"    ✓ Saved Cox PH model to {output_dir}/cox_ph_model.pkl")
        print(f"      Concordance index: {cph.concordance_index_:.3f}")
        
    except ImportError:
        print("    ⚠ lifelines not installed, creating placeholder model")
        # Create placeholder
        with open(f"{output_dir}/cox_ph_model.pkl", 'wb') as f:
            pickle.dump({'type': 'placeholder', 'note': 'Install lifelines to train real model'}, f)
    
    # 2. Risk Assessment Model (simple logistic model)
    print("  - Creating risk assessment model...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features and target
        X = data[['sleep_quality', 'stress_level', 'workload', 'social_support',
                  'exercise_frequency', 'hrv_score', 'recovery_time', 
                  'work_life_balance', 'nutrition_quality', 'mental_fatigue']]
        y = data['burnout_event']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_scaled, y)
        
        # Save model and scaler
        with open(f"{output_dir}/risk_assessment_model.pkl", 'wb') as f:
            pickle.dump({'model': rf_model, 'scaler': scaler}, f)
        
        print(f"    ✓ Saved risk assessment model to {output_dir}/risk_assessment_model.pkl")
        print(f"      Training accuracy: {rf_model.score(X_scaled, y):.3f}")
        
        # 3. Feature importance for SHAP-like explanations
        print("  - Creating feature importance data...")
        
        feature_importance = {
            'features': list(X.columns),
            'importance': rf_model.feature_importances_.tolist()
        }
        
        with open(f"{output_dir}/feature_importance.pkl", 'wb') as f:
            pickle.dump(feature_importance, f)
        
        print(f"    ✓ Saved feature importance to {output_dir}/feature_importance.pkl")
        
    except Exception as e:
        print(f"    ⚠ sklearn error: {e}, creating placeholder")
        with open(f"{output_dir}/risk_assessment_model.pkl", 'wb') as f:
            pickle.dump({'type': 'placeholder'}, f)
    
    # 4. Recommendation templates
    print("  - Creating recommendation templates...")
    
    recommendations = {
        'low_sleep': [
            "Aim for 7-9 hours of quality sleep per night",
            "Establish a consistent sleep schedule",
            "Avoid screens 1 hour before bedtime"
        ],
        'high_stress': [
            "Practice stress management techniques (meditation, deep breathing)",
            "Consider talking to a mental health professional",
            "Take regular breaks throughout the day"
        ],
        'high_workload': [
            "Set boundaries between work and personal time",
            "Prioritize tasks and delegate when possible",
            "Schedule regular time off and vacations"
        ],
        'low_social_support': [
            "Connect with friends and family regularly",
            "Join social or hobby groups",
            "Consider professional support groups"
        ],
        'low_exercise': [
            "Aim for 150 minutes of moderate activity per week",
            "Start with short, manageable workout sessions",
            "Find physical activities you enjoy"
        ],
        'low_hrv': [
            "Focus on recovery and rest days",
            "Practice relaxation techniques",
            "Consult with a healthcare provider if HRV remains low"
        ],
        'poor_work_life_balance': [
            "Set clear work hours and stick to them",
            "Make time for hobbies and personal interests",
            "Learn to say no to non-essential commitments"
        ],
        'high_mental_fatigue': [
            "Take regular mental breaks throughout the day",
            "Practice mindfulness or meditation",
            "Ensure adequate rest and recovery time"
        ]
    }
    
    with open(f"{output_dir}/recommendation_templates.pkl", 'wb') as f:
        pickle.dump(recommendations, f)
    
    print(f"    ✓ Saved recommendation templates to {output_dir}/recommendation_templates.pkl")
    
    # 5. Create metadata
    metadata = {
        'models': {
            'cox_ph': {
                'file': 'cox_ph_model.pkl',
                'type': 'cox_proportional_hazards',
                'description': 'Cox PH model for survival analysis and time-to-burnout prediction'
            },
            'risk_assessment': {
                'file': 'risk_assessment_model.pkl',
                'type': 'random_forest',
                'description': 'Random forest model for burnout risk classification'
            }
        },
        'created_date': datetime.now().isoformat(),
        'version': '1.0.0',
        'training_samples': n_samples,
        'note': 'These are models trained on synthetic data. Replace with real data for production.'
    }
    
    with open(f"{output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"    ✓ Saved metadata to {output_dir}/metadata.pkl")
    
    print("\n✅ Burnout mock models created successfully!")
    print(f"📁 Location: {output_dir}")
    print("📝 Note: These models use synthetic data. Team should retrain with real user data.")

if __name__ == "__main__":
    create_burnout_models()
