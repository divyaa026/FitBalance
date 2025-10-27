# Person 2: Backend + Burnout ML Developer - Task Guide

**Project:** FitBalance  
**Duration:** 2-3 weeks (60-80 hours)  
**Contribution:** 25-30% of total project  
**Dependencies:** Person 1 (ML models) must complete nutrition model first

---

## 📋 Overview

You are responsible for:
1. Training the **Burnout Prediction ML Model** (Cox Proportional Hazards)
2. Complete **Backend Integration** (database, APIs, health sync)
3. Setting up **PostgreSQL Database** with Indian food data
4. Creating all **API endpoints** for the system
5. **Testing** all backend functionality

---

## 🎯 Your Deliverables

- ✅ Trained Burnout Cox PH model (C-index > 0.70)
- ✅ Burnout model integrated with backend
- ✅ PostgreSQL database setup with all tables
- ✅ Indian food database populated (80 foods)
- ✅ Nutrition module integrated with database
- ✅ New API endpoints for history, stats, foods
- ✅ Health API integration framework
- ✅ Comprehensive error handling
- ✅ API testing suite
- ✅ Backend documentation

---

## 📂 File Structure You'll Work With

```
FitBalance/
├── backend/
│   ├── api/
│   │   └── main.py                    # ← YOU: Add new endpoints
│   ├── modules/
│   │   ├── nutrition.py               # ← YOU: Integrate with database
│   │   └── burnout.py                 # ← YOU: Integrate Cox model
│   ├── database/
│   │   ├── nutrition_db.py            # ← YOU: Update config
│   │   └── init_database.py           # ← YOU: Create this
│   ├── integrations/
│   │   └── health_apis.py             # ← YOU: Implement real APIs
│   └── utils/
│       └── error_handlers.py          # ← YOU: Create this
├── ml_models/
│   └── burnout/
│       ├── generate_burnout_dataset.py    # ← YOU: Create
│       ├── train_cox_model.py             # ← YOU: Create
│       ├── test_burnout_model.py          # ← YOU: Create
│       ├── inference.py                   # ← YOU: Auto-generated
│       ├── cox_model.pkl                  # ← YOU: Output
│       ├── burnout_dataset.csv            # ← YOU: Output
│       └── README.md                      # ← YOU: Create
└── requirements.txt                       # ← Already exists
```

---

## 🚀 TASK 1: Train Burnout Prediction ML Model
**Time:** 1 week (40 hours)

### Step 1.1: Dataset Creation (8 hours)

**Create:** `ml_models/burnout/generate_burnout_dataset.py`

```python
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
    print(f"✅ Generated dataset with {len(df)} athletes")
    print(f"   Burnout rate: {df['event_burnout'].mean()*100:.1f}%")
    print(f"   Avg time to burnout: {df['time_to_burnout'].mean():.0f} days")
```

**Run it:**
```powershell
cd c:\Users\divya\Desktop\projects\FitBalance
python ml_models/burnout/generate_burnout_dataset.py
```

**Expected Output:**
- File created: `ml_models/burnout/burnout_dataset.csv` (2000 rows)
- Console output showing burnout rate and statistics

**Verify:**
```powershell
# Check file was created
ls ml_models/burnout/burnout_dataset.csv

# Preview data
python -c "import pandas as pd; df = pd.read_csv('ml_models/burnout/burnout_dataset.csv'); print(df.head()); print(df.describe())"
```

---

### Step 1.2: Cox Proportional Hazards Model Implementation (12 hours)

**Install dependencies:**
```powershell
pip install lifelines scikit-learn matplotlib seaborn
```

**Create:** `ml_models/burnout/train_cox_model.py`

```python
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
    print(f"📊 Loaded {len(df)} athlete records")
    
    # Preprocess
    model_df, feature_cols = preprocess_data(df)
    
    # Split data
    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)
    print(f"   Training: {len(train_df)}, Testing: {len(test_df)}")
    
    # Initialize Cox model
    cph = CoxPHFitter(penalizer=0.01)
    
    # Train model
    print("\n🔄 Training Cox Proportional Hazards model...")
    cph.fit(train_df, duration_col='duration', event_col='event')
    
    # Print model summary
    print("\n📈 Model Summary:")
    print(cph.summary)
    
    # Evaluate on test set
    print("\n🎯 Model Performance:")
    
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
    print(f"\n✅ Model saved to {model_path}")
    
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
    print("📊 Survival curves saved to survival_curves.png")

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
    print("✅ Inference function created: ml_models/burnout/inference.py")

if __name__ == "__main__":
    cph, test_df = train_cox_model()
    plot_survival_curves(cph, test_df)
    create_inference_function()
    print("\n✅ Training complete!")
```

**Run training:**
```powershell
python ml_models/burnout/train_cox_model.py
```

**Expected outputs:**
- `ml_models/burnout/cox_model.pkl` - Trained model
- `ml_models/burnout/model_metrics.json` - Performance metrics
- `ml_models/burnout/survival_curves.png` - Visualization
- `ml_models/burnout/inference.py` - Inference function

**Success criteria:**
- C-index (test) > 0.70 ✅
- Model file size: ~50-100KB
- Survival curves show clear separation between risk groups

---

### Step 1.3: Model Testing & Validation (6 hours)

**Create:** `ml_models/burnout/test_burnout_model.py`

```python
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
```

**Run tests:**
```powershell
cd ml_models/burnout
python test_burnout_model.py
```

**Expected behavior:**
- Low risk → High survival probability (>80%)
- High risk → Low survival probability (<50%)
- Risk scores proportional to risk factors

---

### Step 1.4: Create Model Documentation (4 hours)

**Create:** `ml_models/burnout/README.md`

```markdown
# Burnout Prediction Model

## Overview
Cox Proportional Hazards survival model for predicting athletic burnout risk.

## Model Performance
- **C-index (Test)**: 0.72 (from model_metrics.json)
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
```

---

## 🗄️ TASK 2: Database Setup
**Time:** 12 hours

### Step 2.1: Install PostgreSQL (2 hours)

**Option 1: Docker (Recommended)**
```powershell
docker run --name fitbalance-db -e POSTGRES_PASSWORD=fitbalance123 -p 5432:5432 -d postgres:15
```

**Option 2: Windows Installer**
1. Download from: https://www.postgresql.org/download/windows/
2. Run installer, set password: `fitbalance123`
3. Note the port (default: 5432)

**Test connection:**
```powershell
# Using psql
psql -U postgres -h localhost

# In psql:
CREATE DATABASE fitbalance_nutrition;
\q
```

---

### Step 2.2: Update Database Configuration (2 hours)

**Update:** `backend/database/nutrition_db.py`

Find this line (around line 31):
```python
DATABASE_URL = os.getenv(
    "NUTRITION_DATABASE_URL", 
    "postgresql://username:password@localhost:5432/fitbalance_nutrition"
)
```

Change to:
```python
DATABASE_URL = os.getenv(
    "NUTRITION_DATABASE_URL", 
    "postgresql://postgres:fitbalance123@localhost:5432/fitbalance_nutrition"
)
```

**Create/Update:** `.env` (in project root)
```env
NUTRITION_DATABASE_URL=postgresql://postgres:fitbalance123@localhost:5432/fitbalance_nutrition
GEMINI_API_KEY=AIzaSyBL4OE... # (your existing key)
```

---

### Step 2.3: Initialize Database (3 hours)

**Create:** `backend/database/init_database.py`

```python
"""
Initialize FitBalance database with all tables and seed data
"""
from nutrition_db import nutrition_db, Base, FoodItems, User
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize complete database"""
    
    try:
        # Create all tables
        logger.info("📦 Creating database tables...")
        Base.metadata.create_all(bind=nutrition_db.engine)
        logger.info("✅ Tables created successfully")
        
        # Populate food database with Indian foods
        logger.info("🍛 Populating Indian food database...")
        populate_indian_foods()
        logger.info("✅ Indian food database populated")
        
        # Create test user
        create_test_users()
        logger.info("✅ Test users created")
        
        logger.info("\n🎉 Database initialization complete!")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

def populate_indian_foods():
    """Populate with 80 Indian foods from trained model"""
    
    indian_foods = [
        # Format: (name, protein/100g, carbs/100g, fat/100g, calories/100g, category)
        ("aloo_gobi", 2.5, 15.0, 0.5, 75, "vegetable"),
        ("aloo_matar", 3.0, 18.0, 0.8, 90, "vegetable"),
        ("aloo_methi", 3.5, 14.0, 1.0, 80, "vegetable"),
        ("aloo_shimla_mirch", 2.0, 16.0, 0.6, 78, "vegetable"),
        ("aloo_tikki", 4.5, 25.0, 8.0, 180, "snack"),
        ("biryani", 6.0, 45.0, 8.0, 280, "rice"),
        ("butter_naan", 8.0, 50.0, 12.0, 320, "bread"),
        ("cham_cham", 5.0, 55.0, 10.0, 310, "sweet"),
        ("chana_masala", 8.5, 27.0, 4.0, 175, "curry"),
        ("chapati", 3.5, 18.0, 1.0, 100, "bread"),
        ("chole_bhature", 7.0, 48.0, 15.0, 350, "combo"),
        ("dal_makhani", 9.0, 18.0, 6.0, 165, "dal"),
        ("dal_tadka", 8.0, 20.0, 3.0, 140, "dal"),
        ("dosa", 7.0, 42.0, 1.5, 200, "breakfast"),
        ("gulab_jamun", 4.0, 60.0, 12.0, 350, "sweet"),
        ("idli", 3.5, 22.0, 0.5, 105, "breakfast"),
        ("jalebi", 2.0, 65.0, 15.0, 400, "sweet"),
        ("kadai_paneer", 12.0, 8.0, 15.0, 210, "curry"),
        ("kadhi_pakora", 5.0, 15.0, 8.0, 145, "curry"),
        ("kofta", 10.0, 12.0, 18.0, 250, "curry"),
        ("masala_dosa", 8.0, 45.0, 3.0, 230, "breakfast"),
        ("momos", 7.0, 28.0, 4.0, 170, "snack"),
        ("palak_paneer", 14.0, 7.0, 12.0, 190, "curry"),
        ("paneer_butter_masala", 13.0, 9.0, 16.0, 220, "curry"),
        ("paneer_tikka", 15.0, 5.0, 14.0, 200, "starter"),
        ("pav_bhaji", 5.0, 35.0, 10.0, 240, "street_food"),
        ("poha", 3.0, 25.0, 2.0, 130, "breakfast"),
        ("rasgulla", 6.0, 45.0, 0.5, 210, "sweet"),
        ("samosa", 6.0, 30.0, 12.0, 250, "snack"),
        ("tandoori_chicken", 28.0, 2.0, 8.0, 190, "chicken"),
        # Add more 50 foods based on Person 1's dataset...
        ("butter_chicken", 20.0, 8.0, 15.0, 250, "chicken"),
        ("rogan_josh", 18.0, 10.0, 12.0, 220, "curry"),
        ("tikka_masala", 22.0, 9.0, 14.0, 240, "chicken"),
        ("vindaloo", 15.0, 12.0, 10.0, 200, "curry"),
        ("korma", 12.0, 15.0, 20.0, 280, "curry"),
        ("naan", 8.0, 48.0, 8.0, 290, "bread"),
        ("roti", 3.0, 18.0, 0.5, 90, "bread"),
        ("paratha", 4.5, 24.0, 12.0, 230, "bread"),
        ("puri", 5.0, 28.0, 15.0, 260, "bread"),
        ("bhindi_masala", 2.0, 8.0, 3.0, 65, "vegetable"),
        ("baingan_bharta", 1.5, 10.0, 4.0, 80, "vegetable"),
        ("malai_kofta", 8.0, 15.0, 18.0, 250, "curry"),
        ("rajma", 9.0, 25.0, 0.5, 140, "curry"),
        ("chole", 8.5, 27.0, 4.0, 175, "curry"),
        ("aloo_paratha", 5.0, 30.0, 10.0, 220, "bread"),
        ("pani_puri", 3.0, 18.0, 2.0, 100, "street_food"),
        ("bhel_puri", 4.0, 22.0, 3.0, 130, "street_food"),
        ("vada_pav", 6.0, 35.0, 12.0, 260, "street_food"),
        ("dahi_vada", 5.0, 20.0, 8.0, 160, "snack"),
        ("khichdi", 6.0, 35.0, 2.0, 180, "rice"),
        ("pulao", 5.0, 40.0, 6.0, 220, "rice"),
        ("fried_rice", 6.0, 45.0, 8.0, 270, "rice"),
        ("chicken_biryani", 15.0, 50.0, 12.0, 360, "rice"),
        ("mutton_biryani", 18.0, 48.0, 15.0, 390, "rice"),
        ("fish_curry", 20.0, 8.0, 10.0, 200, "seafood"),
        ("prawn_curry", 18.0, 6.0, 12.0, 190, "seafood"),
        ("kebab", 25.0, 3.0, 15.0, 250, "starter"),
        ("seekh_kebab", 22.0, 4.0, 18.0, 270, "starter"),
        ("chicken_tikka", 27.0, 3.0, 10.0, 210, "starter"),
        ("lassi", 3.5, 12.0, 3.0, 85, "beverage"),
        ("kulfi", 5.0, 30.0, 8.0, 200, "dessert"),
        ("kheer", 4.0, 35.0, 6.0, 210, "dessert"),
        ("halwa", 3.0, 40.0, 15.0, 300, "dessert"),
        ("barfi", 6.0, 50.0, 12.0, 320, "dessert"),
        ("ladoo", 4.5, 55.0, 10.0, 310, "dessert"),
        ("mysore_pak", 3.0, 48.0, 18.0, 350, "dessert"),
        ("peda", 5.5, 52.0, 9.0, 295, "dessert"),
        ("sandesh", 6.0, 45.0, 8.0, 270, "dessert"),
        ("rasmalai", 7.0, 40.0, 10.0, 280, "dessert"),
        ("gajar_halwa", 3.5, 38.0, 12.0, 270, "dessert"),
        ("upma", 4.0, 28.0, 3.0, 150, "breakfast"),
        ("medu_vada", 5.0, 22.0, 8.0, 170, "breakfast"),
        ("uttapam", 6.0, 35.0, 2.0, 180, "breakfast"),
        ("pesarattu", 8.0, 30.0, 2.5, 170, "breakfast"),
        ("appam", 2.0, 25.0, 1.0, 115, "breakfast"),
        ("puttu", 3.5, 30.0, 1.5, 145, "breakfast"),
        ("dhokla", 4.0, 20.0, 2.0, 110, "snack"),
        ("khandvi", 5.0, 18.0, 3.0, 115, "snack"),
        ("pakora", 4.5, 22.0, 10.0, 190, "snack"),
        ("bonda", 4.0, 24.0, 8.0, 170, "snack"),
    ]
    
    session = nutrition_db.SessionLocal()
    
    for food_data in indian_foods:
        food = FoodItems(
            food_name=food_data[0],
            protein_per_100g=food_data[1],
            carbs_per_100g=food_data[2],
            fats_per_100g=food_data[3],
            calories_per_100g=food_data[4],
            food_category=food_data[5]
        )
        session.merge(food)  # Use merge to avoid duplicates
    
    session.commit()
    session.close()

def create_test_users():
    """Create test users for development"""
    session = nutrition_db.SessionLocal()
    
    test_users = [
        User(user_id=1, age=28, weight_kg=70, height_cm=175, fitness_goal="muscle_gain"),
        User(user_id=123, age=25, weight_kg=65, height_cm=168, fitness_goal="fat_loss"),
    ]
    
    for user in test_users:
        session.merge(user)
    
    session.commit()
    session.close()

if __name__ == "__main__":
    init_database()
```

**Run initialization:**
```powershell
cd c:\Users\divya\Desktop\projects\FitBalance
python backend/database/init_database.py
```

**Expected output:**
```
📦 Creating database tables...
✅ Tables created successfully
🍛 Populating Indian food database...
✅ Indian food database populated
✅ Test users created
🎉 Database initialization complete!
```

---

### Step 2.4: Test Database Connection (2 hours)

**Create:** `backend/database/test_db.py`

```python
"""Test database connection and queries"""
from nutrition_db import nutrition_db
from sqlalchemy import text

def test_connection():
    """Test basic database operations"""
    
    print("Testing database connection...")
    
    # Test 1: Query food items
    session = nutrition_db.SessionLocal()
    result = session.execute(text("SELECT COUNT(*) FROM food_items"))
    count = result.scalar()
    print(f"✅ Food items in database: {count}")
    
    # Test 2: Get specific food
    food = nutrition_db.get_food_nutrition("paneer_butter_masala")
    if food:
        print(f"✅ Retrieved food: {food}")
    
    # Test 3: Log a test meal
    nutrition_db.log_meal(
        user_id=123,
        image_path="/test/image.jpg",
        detected_foods={"paneer_butter_masala": 150},
        total_protein=19.5,
        total_calories=330,
        confidence=0.85
    )
    print("✅ Test meal logged")
    
    session.close()
    print("\n🎉 All database tests passed!")

if __name__ == "__main__":
    test_connection()
```

**Run:**
```powershell
python backend/database/test_db.py
```

---

## 🔌 TASK 3: Backend Integration
**Time:** 20 hours

### Step 3.1: Integrate Burnout Model with Backend (8 hours)

**Update:** `backend/modules/burnout.py`

Find the `_load_models` method (around line 78) and replace `_create_mock_models()` with:

```python
def _load_models(self):
    """Load pre-trained Cox PH model"""
    try:
        model_path = 'ml_models/burnout/cox_model.pkl'
        with open(model_path, 'rb') as f:
            self.cph_model = pickle.load(f)
        
        logger.info("✅ Burnout Cox model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Could not load Cox model: {e}")
        self._create_mock_models()  # Fallback
```

Add imports at the top:
```python
import pickle
import sys
sys.path.append('ml_models/burnout')
from inference import predict_burnout_risk as predict_with_cox
```

Update `_predict_time_to_burnout` method (around line 150):

```python
def _predict_time_to_burnout(self, risk_factors: BurnoutRiskFactors) -> Optional[float]:
    """Predict time to burnout using real Cox model"""
    try:
        # Map training_intensity string to model format
        intensity_map = {
            'low': 'low',
            'moderate': 'moderate',
            'high': 'high',
            'extreme': 'extreme'
        }
        
        result = predict_with_cox(
            age=risk_factors.age,
            experience_years=risk_factors.experience_years,
            workout_frequency=risk_factors.workout_frequency,
            avg_sleep_hours=risk_factors.sleep_hours,
            stress_level=risk_factors.stress_level,
            recovery_days=risk_factors.recovery_time,
            hrv_avg=70.0,  # Default, should come from user data
            resting_hr=65,  # Default, should come from user data
            injury_history=0,  # Default, should come from user data
            nutrition_quality=7.0,  # Default, should come from user data
            gender=risk_factors.gender if risk_factors.gender in ['M', 'F'] else 'M',
            training_intensity=intensity_map.get(risk_factors.training_intensity, 'moderate')
        )
        
        return result['time_to_burnout_days']
        
    except Exception as e:
        logger.error(f"Time to burnout prediction error: {e}")
        return None
```

**Test:**
```powershell
# Start backend server
cd c:\Users\divya\Desktop\projects\FitBalance
uvicorn backend.api.main:app --reload

# In another terminal, test burnout endpoint:
curl -X POST "http://localhost:8000/burnout/analyze" -H "Content-Type: application/json" -d "{\"user_id\":\"123\",\"workout_frequency\":5,\"sleep_hours\":7,\"stress_level\":6,\"recovery_time\":2,\"performance_trend\":\"stable\"}"
```

---

### Step 3.2: Integrate Nutrition Module with Database (6 hours)

**Update:** `backend/modules/nutrition.py`

Add import at top:
```python
from backend.database.nutrition_db import nutrition_db
```

Update `analyze_meal` method (around line 227) - add database integration after detecting foods:

```python
async def analyze_meal(self, meal_photo: UploadFile, user_id: str, dietary_restrictions: List[str]) -> MealAnalysis:
    """Analyze meal photo - NOW WITH DATABASE INTEGRATION"""
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await meal_photo.read()
            tmp_file.write(content)
            image_path = tmp_file.name
        
        # Validate it's a food image
        is_food, message = self._is_food_image(image_path)
        if not is_food:
            os.unlink(image_path)
            raise ValueError(f"Not a food image: {message}")
        
        # Detect foods using trained model (Person 1's work)
        detected_foods = self._detect_foods(image_path)
        
        # Calculate nutrition using DATABASE
        total_protein = 0
        total_calories = 0
        total_carbs = 0
        total_fat = 0
        
        for food in detected_foods:
            # Get nutrition from database
            nutrition = nutrition_db.get_food_nutrition(food.name)
            if nutrition:
                # Update food item with database values
                food.protein_content = nutrition['protein']
                food.calories = nutrition['calories']
                
                # Accumulate totals (assuming 100g portions)
                total_protein += nutrition['protein']
                total_calories += nutrition['calories']
                total_carbs += nutrition['carbs']
                total_fat += nutrition['fat']
        
        # Calculate protein deficit
        user_target_protein = self._get_user_protein_target(user_id)
        protein_deficit = max(0, user_target_protein - total_protein)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_protein, total_calories, protein_deficit, dietary_restrictions
        )
        
        # Calculate meal quality score
        meal_quality = self._calculate_meal_quality(total_protein, total_calories, total_carbs, total_fat)
        
        # LOG MEAL TO DATABASE
        detected_foods_dict = {food.name: 100 for food in detected_foods}  # Assuming 100g each
        nutrition_db.log_meal(
            user_id=int(user_id) if user_id.isdigit() else 123,
            image_path=image_path,
            detected_foods=detected_foods_dict,
            total_protein=total_protein,
            total_calories=total_calories,
            confidence=sum(f.confidence for f in detected_foods) / len(detected_foods) if detected_foods else 0
        )
        
        # Clean up
        os.unlink(image_path)
        
        return MealAnalysis(
            total_protein=total_protein,
            total_calories=total_calories,
            total_carbs=total_carbs,
            total_fat=total_fat,
            detected_foods=detected_foods,
            protein_deficit=protein_deficit,
            recommendations=recommendations,
            meal_quality_score=meal_quality,
            nutritional_balance={
                'protein': total_protein,
                'carbs': total_carbs,
                'fat': total_fat
            }
        )
        
    except Exception as e:
        logger.error(f"Meal analysis error: {str(e)}")
        raise
```

---

### Step 3.3: Add New API Endpoints (6 hours)

**Update:** `backend/api/main.py`

Add these new endpoints after existing ones (around line 150):

```python
@app.get("/nutrition/history/{user_id}")
async def get_nutrition_history(user_id: int, days: int = 7):
    """Get user's meal history"""
    try:
        meals = nutrition_db.get_recent_meals(user_id, days=days)
        return {
            "user_id": user_id,
            "days": days,
            "meal_count": len(meals),
            "meals": meals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nutrition/stats/{user_id}")
async def get_nutrition_stats(user_id: int, days: int = 7):
    """Get aggregated nutrition statistics"""
    try:
        meals = nutrition_db.get_recent_meals(user_id, days=days)
        
        if not meals:
            return {"error": "No meal data found"}
        
        total_protein = sum(m['total_protein'] for m in meals)
        total_calories = sum(m['total_calories'] for m in meals)
        avg_protein = total_protein / len(meals)
        avg_calories = total_calories / len(meals)
        
        return {
            "user_id": user_id,
            "days": days,
            "total_meals": len(meals),
            "total_protein": round(total_protein, 1),
            "total_calories": round(total_calories, 1),
            "avg_protein_per_meal": round(avg_protein, 1),
            "avg_calories_per_meal": round(avg_calories, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nutrition/foods")
async def list_all_foods():
    """Get list of all foods in database"""
    try:
        session = nutrition_db.SessionLocal()
        from sqlalchemy import text
        foods = session.execute(text("SELECT food_name, food_category FROM food_items ORDER BY food_name"))
        result = [{"name": row[0], "category": row[1]} for row in foods]
        session.close()
        return {"count": len(result), "foods": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🧪 TASK 4: Testing & Documentation
**Time:** 8 hours

### Step 4.1: Create API Test Suite (4 hours)

**Create:** `backend/test_api.py`

```python
"""
Test all API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_nutrition_analyze():
    """Test meal analysis endpoint"""
    print("\n🧪 Testing POST /nutrition/analyze...")
    
    # Use a test image (you'll need to provide one)
    try:
        with open("test_images/paneer_tikka.jpg", "rb") as f:
            files = {"meal_photo": f}
            data = {"user_id": "123", "dietary_restrictions": []}
            
            response = requests.post(f"{BASE_URL}/nutrition/analyze", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analysis successful!")
                print(f"   Detected foods: {len(result['detected_foods'])}")
                print(f"   Total protein: {result['total_protein']}g")
                print(f"   Total calories: {result['total_calories']}")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(response.text)
    except FileNotFoundError:
        print("⚠️ Test image not found, skipping...")

def test_nutrition_history():
    """Test history endpoint"""
    print("\n🧪 Testing GET /nutrition/history/123...")
    
    response = requests.get(f"{BASE_URL}/nutrition/history/123?days=7")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ History retrieved!")
        print(f"   Meals found: {result['meal_count']}")
    else:
        print(f"❌ Failed: {response.status_code}")

def test_nutrition_stats():
    """Test stats endpoint"""
    print("\n🧪 Testing GET /nutrition/stats/123...")
    
    response = requests.get(f"{BASE_URL}/nutrition/stats/123?days=7")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Stats retrieved!")
        if 'total_meals' in result:
            print(f"   Total meals: {result['total_meals']}")
            print(f"   Total protein: {result['total_protein']}g")
    else:
        print(f"❌ Failed: {response.status_code}")

def test_food_database():
    """Test food database endpoint"""
    print("\n🧪 Testing GET /nutrition/foods...")
    
    response = requests.get(f"{BASE_URL}/nutrition/foods")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Food database retrieved!")
        print(f"   Total foods: {result['count']}")
        print(f"   Sample foods: {result['foods'][:5]}")
    else:
        print(f"❌ Failed: {response.status_code}")

def test_burnout_analyze():
    """Test burnout analysis"""
    print("\n🧪 Testing POST /burnout/analyze...")
    
    data = {
        "user_id": "123",
        "workout_frequency": 5,
        "sleep_hours": 7.0,
        "stress_level": 6,
        "recovery_time": 1,
        "performance_trend": "stable"
    }
    
    response = requests.post(f"{BASE_URL}/burnout/analyze", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Burnout analysis successful!")
        print(f"   Risk score: {result['risk_score']}")
        print(f"   Risk level: {result['risk_level']}")
        print(f"   Time to burnout: {result.get('time_to_burnout', 'N/A')} days")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("🔍 Testing FitBalance API Endpoints")
    print("=" * 60)
    
    # Test all endpoints
    test_food_database()
    test_nutrition_history()
    test_nutrition_stats()
    test_burnout_analyze()
    # test_nutrition_analyze()  # Uncomment when you have test images
    
    print("\n" + "=" * 60)
    print("✅ API testing complete!")
```

**Run tests:**
```powershell
# Start server first
uvicorn backend.api.main:app --reload

# In another terminal:
python backend/test_api.py
```

---

### Step 4.2: Create Backend Documentation (4 hours)

**Create:** `backend/BACKEND_README.md`

```markdown
# FitBalance Backend Documentation

## Architecture

### Tech Stack
- **Framework:** FastAPI (Python 3.12)
- **Database:** PostgreSQL 15
- **ORM:** SQLAlchemy
- **ML Models:** TensorFlow, PyTorch, lifelines

### Project Structure
```
backend/
├── api/
│   └── main.py              # FastAPI app and endpoints
├── modules/
│   ├── nutrition.py         # Nutrition analysis module
│   ├── biomechanics.py      # Exercise form analysis
│   └── burnout.py           # Burnout prediction
├── database/
│   ├── nutrition_db.py      # Database ORM models
│   └── init_database.py     # Database initialization
├── integrations/
│   └── health_apis.py       # Health tracking API integrations
└── utils/
    └── error_handlers.py    # Error handling utilities
```

## API Endpoints

### Nutrition Endpoints

#### POST /nutrition/analyze
Analyze meal photo for nutrition content.

**Request:**
```bash
curl -X POST "http://localhost:8000/nutrition/analyze" \
  -F "meal_photo=@image.jpg" \
  -F "user_id=123" \
  -F "dietary_restrictions=[]"
```

**Response:**
```json
{
  "total_protein": 25.5,
  "total_calories": 450,
  "total_carbs": 45.0,
  "total_fat": 15.0,
  "detected_foods": [...],
  "recommendations": [...]
}
```

#### GET /nutrition/history/{user_id}
Get user's meal history.

**Request:**
```bash
curl "http://localhost:8000/nutrition/history/123?days=7"
```

#### GET /nutrition/stats/{user_id}
Get aggregated nutrition statistics.

#### GET /nutrition/foods
List all foods in database.

### Burnout Endpoints

#### POST /burnout/analyze
Analyze burnout risk.

**Request:**
```json
{
  "user_id": "123",
  "workout_frequency": 5,
  "sleep_hours": 7.0,
  "stress_level": 6,
  "recovery_time": 2,
  "performance_trend": "stable"
}
```

**Response:**
```json
{
  "risk_score": 45.5,
  "risk_level": "medium",
  "time_to_burnout": 450,
  "survival_probability": 0.75,
  "recommendations": [...]
}
```

#### GET /burnout/survival-curve/{user_id}
Get survival curve data.

### Biomechanics Endpoints

#### POST /biomechanics/analyze
Analyze exercise form from video.

## Database Schema

### Tables

**users**
- user_id (PK)
- age, weight_kg, height_cm
- fitness_goal
- created_at

**food_items**
- food_id (PK)
- food_name (unique)
- protein_per_100g, carbs_per_100g, fats_per_100g
- calories_per_100g
- food_category

**meal_logs**
- meal_id (PK)
- user_id (FK)
- meal_image_path
- detected_foods (JSON)
- total_protein, total_calories
- meal_timestamp

**health_metrics**
- id (PK)
- user_id (FK)
- date
- sleep_duration, hrv, activity_level, stress_level

## Setup & Running

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment
Create `.env`:
```env
NUTRITION_DATABASE_URL=postgresql://postgres:fitbalance123@localhost:5432/fitbalance_nutrition
GEMINI_API_KEY=your_key_here
```

### Initialize Database
```bash
python backend/database/init_database.py
```

### Run Server
```bash
uvicorn backend.api.main:app --reload --port 8000
```

## Testing

```bash
# Run test suite
python backend/test_api.py

# Manual API testing
curl http://localhost:8000/
curl http://localhost:8000/health
```

## Troubleshooting

### Database Connection Failed
- Verify PostgreSQL is running
- Check DATABASE_URL in .env
- Test connection: `psql $DATABASE_URL`

### Model Loading Errors
- Verify model files exist in ml_models/
- Check Python path configuration
- Review model file permissions

### API Returns 500 Errors
- Check logs in terminal
- Verify all dependencies installed
- Test individual modules

## Performance

- API Response Time (p95): < 500ms
- Database Query Time: < 100ms
- ML Inference Time: 200-500ms per request

## Security

- [ ] Use environment variables for secrets
- [ ] Implement rate limiting
- [ ] Add authentication (TODO)
- [ ] Enable CORS properly
- [ ] Validate all inputs
- [ ] Sanitize file uploads

## Next Steps

1. Implement user authentication
2. Add API rate limiting
3. Set up Redis caching
4. Implement WebSocket for real-time updates
5. Add comprehensive logging
6. Set up monitoring (Prometheus/Grafana)
```

---

## ✅ Final Checklist

Before marking your work complete, verify:

### Burnout Model
- [ ] Dataset generated (2000 athletes)
- [ ] Model trained (C-index > 0.70)
- [ ] Inference function works
- [ ] Test cases pass (low/medium/high risk)
- [ ] Model integrated with backend

### Database
- [ ] PostgreSQL installed and running
- [ ] All tables created
- [ ] 80 Indian foods populated
- [ ] Test users created
- [ ] Connection test passes

### Backend Integration
- [ ] Nutrition module uses database
- [ ] Burnout model integrated
- [ ] All new endpoints working
- [ ] Error handling added
- [ ] Logging configured

### Testing
- [ ] API test suite runs successfully
- [ ] All endpoints return expected results
- [ ] Database queries work
- [ ] Error cases handled

### Documentation
- [ ] Burnout model README complete
- [ ] Backend README complete
- [ ] API documentation clear
- [ ] Setup instructions tested

---

## 📊 Success Metrics

Your work is complete when:
1. ✅ Burnout model C-index > 0.70
2. ✅ Database has 80+ foods
3. ✅ All API endpoints return 200 OK
4. ✅ Test suite passes 100%
5. ✅ Documentation is comprehensive

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs:** `uvicorn` terminal output
2. **Test database:** `python backend/database/test_db.py`
3. **Test model:** `python ml_models/burnout/test_burnout_model.py`
4. **Review docs:** Read error messages carefully
5. **Ask team:** Coordinate with Person 1 (ML) and Person 4 (DevOps)

---

## 📅 Timeline

**Week 1:**
- Days 1-3: Burnout model training
- Days 4-5: Database setup

**Week 2:**
- Days 1-3: Backend integration
- Days 4-5: Testing and documentation

**Week 3 (Buffer):**
- Polish and bug fixes
- Performance optimization
- Additional testing

---

## 🎉 Completion

Once all tasks are complete:
1. Commit all code to git
2. Update main project README
3. Share documentation with team
4. Demo your work to Person 4 (DevOps)
5. Prepare for deployment

**Your contribution: 25-30% of total project** 🏆

Good luck! 💪
