from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional, Dict
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FitBalance AI Fitness Platform",
    description="AI-powered fitness platform with biomechanics coaching, protein optimization, and burnout prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock implementations for development
class MockBiomechanicsCoach:
    async def analyze_movement(self, video_file, exercise_type, user_id):
        # Simulate processing time
        time.sleep(1)
        return {
            "form_score": 85.5,
            "joint_angles": [
                {"joint_name": "knee", "angle": 95.2, "is_abnormal": True},
                {"joint_name": "hip", "angle": 85.1, "is_abnormal": False}
            ],
            "torques": [
                {"joint_name": "knee", "torque_magnitude": 45.2, "risk_level": "medium"}
            ],
            "recommendations": ["Keep knees aligned with toes", "Maintain proper hip hinge"],
            "exercise_type": exercise_type,
            "user_id": user_id
        }
    
    async def generate_heatmap(self, user_id, exercise_type):
        return {
            "heatmap_data": "mock_heatmap_data",
            "user_id": user_id,
            "exercise_type": exercise_type
        }

class MockProteinOptimizer:
    async def analyze_meal(self, meal_photo, user_id, dietary_restrictions):
        # Import the real nutrition logic from our demo
        import random
        from datetime import datetime
        
        # Simulate processing time
        time.sleep(1)
        
        # Mock food database
        food_combinations = [
            [("Chicken Breast", 25.0, 165), ("Broccoli", 2.8, 34), ("Rice", 2.7, 130)],
            [("Salmon", 25.0, 208), ("Quinoa", 4.4, 120), ("Spinach", 2.9, 23)],
            [("Eggs", 13.0, 155), ("Toast", 3.0, 70), ("Avocado", 2.0, 160)],
            [("Greek Yogurt", 10.0, 59), ("Almonds", 21.0, 579), ("Berries", 1.0, 50)]
        ]
        
        selected = random.choice(food_combinations)
        detected_foods = []
        total_protein = 0
        total_calories = 0
        
        for food_name, protein, calories in selected:
            portion = random.uniform(80, 150)  # grams
            protein_content = protein * (portion / 100)
            calorie_content = calories * (portion / 100)
            
            detected_foods.append({
                "name": food_name,
                "protein_content": round(protein_content, 1),
                "calories": round(calorie_content, 0),
                "confidence": random.uniform(0.85, 0.95),
                "portion_grams": round(portion, 0)
            })
            
            total_protein += protein_content
            total_calories += calorie_content
        
        # Calculate meal quality score
        variety_score = len(detected_foods) * 15
        protein_score = min((total_protein / 30) * 40, 40)  # Target 30g per meal
        balance_score = 30 if total_calories > 200 else 20
        meal_quality_score = min(variety_score + protein_score + balance_score, 100)
        
        return {
            "total_protein": round(total_protein, 1),
            "total_calories": round(total_calories, 0),
            "detected_foods": detected_foods,
            "meal_quality_score": round(meal_quality_score, 1),
            "recommendations": [
                "Great protein variety!" if len(detected_foods) > 2 else "Consider adding more protein sources",
                "Well-balanced meal" if meal_quality_score > 70 else "Try adding more vegetables"
            ],
            "user_id": user_id,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_recommendations(self, user_id, target_protein, activity_level):
        # Generate health-based recommendations
        import random
        
        base_protein = target_protein or 120
        
        # Activity-based adjustments
        activity_multipliers = {"low": 0.9, "moderate": 1.0, "high": 1.3}
        adjusted_protein = base_protein * activity_multipliers.get(activity_level, 1.0)
        
        recommendations = [
            f"Target {adjusted_protein:.0f}g protein daily based on {activity_level} activity",
            "Include protein in every meal (20-30g per meal)",
            "Time protein intake within 2 hours post-workout" if activity_level == "high" else "Distribute protein evenly throughout the day"
        ]
        
        return {
            "recommended_protein": round(adjusted_protein, 1),
            "baseline_protein": base_protein,
            "recommendations": recommendations,
            "activity_level": activity_level,
            "user_id": user_id
        }

class MockBurnoutPredictor:
    async def analyze_risk(self, user_id, workout_frequency, sleep_hours, stress_level, recovery_time, performance_trend):
        # Calculate mock risk score
        risk_score = (stress_level * 10) + (10 - sleep_hours) * 5 + (7 - workout_frequency) * 3
        risk_score = min(max(risk_score, 0), 100)
        
        return {
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 70 else "medium" if risk_score > 40 else "low",
            "time_to_burnout": 45.0 if risk_score > 70 else 90.0,
            "survival_probability": 0.35 if risk_score > 70 else 0.65,
            "risk_factors": ["High stress levels", "Poor sleep quality", "Excessive workload"],
            "recommendations": [
                "Implement stress management techniques",
                "Improve sleep hygiene",
                "Consider workload reduction"
            ]
        }
    
    async def generate_survival_curve(self, user_id):
        return {
            "time_points": list(range(0, 365, 7)),
            "survival_probabilities": [0.95 - i * 0.001 for i in range(0, 365, 7)],
            "user_id": user_id
        }
    
    async def get_recommendations(self, user_id):
        return {
            "recommendations": [
                "Take regular breaks during work",
                "Practice mindfulness meditation",
                "Maintain work-life balance",
                "Get adequate sleep (7-9 hours)"
            ]
        }

# Initialize mock modules
biomechanics_coach = MockBiomechanicsCoach()
protein_optimizer = MockProteinOptimizer()
burnout_predictor = MockBurnoutPredictor()

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to FitBalance AI-powered fitness platform!",
        "modules": [
            "biomechanics-coaching",
            "protein-optimization", 
            "burnout-prediction"
        ],
        "status": "running_with_mock_data"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "modules": {
        "biomechanics": "active (mock)",
        "nutrition": "active (mock)", 
        "burnout": "active (mock)"
    }}

# Biomechanics Coaching Endpoints
@app.post("/biomechanics/analyze")
async def analyze_biomechanics(
    video_file: UploadFile = File(...),
    exercise_type: str = "squat",
    user_id: str = "default"
):
    """Analyze biomechanics from uploaded video"""
    try:
        if not video_file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save video temporarily and analyze
        result = await biomechanics_coach.analyze_movement(
            video_file, exercise_type, user_id
        )
        return result
    except Exception as e:
        logger.error(f"Biomechanics analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/biomechanics/heatmap/{user_id}")
async def get_torque_heatmap(user_id: str, exercise_type: str = "squat"):
    """Get torque heatmap for user's exercise"""
    try:
        heatmap = await biomechanics_coach.generate_heatmap(user_id, exercise_type)
        return {"heatmap": heatmap, "user_id": user_id, "exercise": exercise_type}
    except Exception as e:
        logger.error(f"Heatmap generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Protein Optimization Endpoints
@app.post("/nutrition/analyze-meal")
async def analyze_meal_photo(
    meal_photo: UploadFile = File(...),
    user_id: str = "default",
    dietary_restrictions: Optional[List[str]] = None
):
    """Analyze meal photo for protein content and optimization"""
    try:
        if not meal_photo.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Use the mock meal analysis
        result = await protein_optimizer.analyze_meal(
            meal_photo, user_id, dietary_restrictions or []
        )
        return result
    except Exception as e:
        logger.error(f"Meal analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nutrition/recommendations/{user_id}")
async def get_nutrition_recommendations(
    user_id: str,
    target_protein: Optional[float] = None,
    activity_level: str = "moderate"
):
    """Get personalized nutrition recommendations"""
    try:
        result = await protein_optimizer.get_recommendations(
            user_id, target_protein, activity_level
        )
        return result
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nutrition/health-data/{user_id}")
async def update_health_data(
    user_id: str,
    health_data: Dict
):
    """Update user health data for protein optimization"""
    try:
        # Simple validation
        required_fields = ["sleep_quality", "activity_level"]
        for field in required_fields:
            if field not in health_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Log the data (in a real system, this would save to database)
        logger.info(f"Health data updated for user {user_id}: {health_data}")
        
        from datetime import datetime
        return {
            "status": "success", 
            "message": "Health data updated successfully",
            "user_id": user_id,
            "updated_fields": list(health_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health data update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nutrition/shap-explanation/{user_id}")
async def get_shap_explanation(
    user_id: str,
    date_str: Optional[str] = None
):
    """Get SHAP explanation for protein recommendation"""
    try:
        from backend.integrations.health_apis import create_demo_health_data
        from datetime import date, datetime
        
        # Get health data for the specified date
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else date.today()
        health_data_point = create_demo_health_data(target_date)
        
        # Convert to HealthMetrics format
        from ml_models.nutrition.gru_protein_optimizer import HealthMetrics
        health_metric = HealthMetrics(
            date=datetime.combine(health_data_point.date, datetime.min.time()),
            sleep_duration=health_data_point.sleep_duration or 7.5,
            sleep_quality=health_data_point.sleep_quality or 7.0,
            hrv=health_data_point.hrv or 45.0,
            activity_level=health_data_point.activity_score or 5.0,
            stress_level=int(health_data_point.stress_level or 4),
            protein_intake=0.0,
            recovery_score=health_data_point.recovery_score or 7.0
        )
        
        # Try to get SHAP explanation
        try:
            from ml_models.nutrition.shap_explainer import shap_explainer
            if shap_explainer:
                explanation = shap_explainer.explain_recommendation(health_metric, user_id)
                
                return {
                    'feature_importance': explanation.feature_importance,
                    'explanation_text': explanation.explanation_text,
                    'confidence_score': explanation.confidence_score,
                    'predicted_value': explanation.predicted_value,
                    'base_value': explanation.base_value,
                    'health_metrics': {
                        'sleep_quality': health_metric.sleep_quality,
                        'sleep_duration': health_metric.sleep_duration,
                        'hrv': health_metric.hrv,
                        'activity_level': health_metric.activity_level,
                        'stress_level': health_metric.stress_level,
                        'recovery_score': health_metric.recovery_score
                    }
                }
            else:
                raise ValueError("SHAP explainer not available")
        except:
            # Fallback explanation
            return {
                'feature_importance': {
                    'sleep_quality': {'value': health_metric.sleep_quality, 'impact': 'moderate'},
                    'activity_level': {'value': health_metric.activity_level, 'impact': 'high'},
                    'hrv': {'value': health_metric.hrv, 'impact': 'moderate'}
                },
                'explanation_text': "Basic protein recommendation based on sleep, activity, and recovery patterns.",
                'confidence_score': 0.7,
                'predicted_value': 120.0,
                'base_value': 120.0,
                'health_metrics': {
                    'sleep_quality': health_metric.sleep_quality,
                    'sleep_duration': health_metric.sleep_duration,
                    'hrv': health_metric.hrv,
                    'activity_level': health_metric.activity_level,
                    'stress_level': health_metric.stress_level,
                    'recovery_score': health_metric.recovery_score
                }
            }
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nutrition/food-database")
async def get_food_database():
    """Get comprehensive food database with nutritional information"""
    try:
        # Basic food database for demo
        food_database = {
            'chicken_breast': {'name': 'Chicken Breast', 'protein': 31.0, 'carbs': 0.0, 'fat': 3.6, 'calories': 165},
            'salmon': {'name': 'Salmon', 'protein': 25.0, 'carbs': 0.0, 'fat': 12.0, 'calories': 208},
            'eggs': {'name': 'Eggs', 'protein': 13.0, 'carbs': 1.1, 'fat': 11.0, 'calories': 155},
            'greek_yogurt': {'name': 'Greek Yogurt', 'protein': 10.0, 'carbs': 3.6, 'fat': 0.4, 'calories': 59},
            'quinoa': {'name': 'Quinoa', 'protein': 4.4, 'carbs': 22.0, 'fat': 1.9, 'calories': 120},
            'broccoli': {'name': 'Broccoli', 'protein': 2.8, 'carbs': 7.0, 'fat': 0.4, 'calories': 34},
            'tofu': {'name': 'Tofu', 'protein': 15.0, 'carbs': 4.0, 'fat': 8.0, 'calories': 144},
            'almonds': {'name': 'Almonds', 'protein': 21.0, 'carbs': 22.0, 'fat': 49.0, 'calories': 579}
        }
        
        # Format for API response
        formatted_foods = {}
        for food_id, nutrition in food_database.items():
            formatted_foods[food_id] = {
                **nutrition,
                'protein_per_100g': nutrition['protein'],
                'carbs_per_100g': nutrition['carbs'],
                'fat_per_100g': nutrition['fat'],
                'calories_per_100g': nutrition['calories'],
                'category': 'protein' if nutrition['protein'] > 15 else 'carbs' if nutrition['carbs'] > 20 else 'vegetable'
            }
        
        return {
            'foods': formatted_foods,
            'total_foods': len(formatted_foods),
            'categories': ['protein', 'carbs', 'vegetable', 'dairy', 'nuts']
        }
        
    except Exception as e:
        logger.error(f"Food database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Burnout Prediction Endpoints
@app.post("/burnout/analyze")
async def analyze_burnout_risk(
    user_id: str,
    workout_frequency: int,
    sleep_hours: float,
    stress_level: int,  # 1-10 scale
    recovery_time: int,  # days
    performance_trend: str = "stable"  # improving, stable, declining
):
    """Analyze burnout risk based on user metrics"""
    try:
        risk_analysis = await burnout_predictor.analyze_risk(
            user_id, workout_frequency, sleep_hours, 
            stress_level, recovery_time, performance_trend
        )
        return risk_analysis
    except Exception as e:
        logger.error(f"Burnout analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/burnout/survival-curve/{user_id}")
async def get_survival_curve(user_id: str):
    """Get survival curve for burnout prediction"""
    try:
        survival_data = await burnout_predictor.generate_survival_curve(user_id)
        return survival_data
    except Exception as e:
        logger.error(f"Survival curve generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/burnout/recommendations/{user_id}")
async def get_burnout_recommendations(user_id: str):
    """Get personalized recommendations to prevent burnout"""
    try:
        recommendations = await burnout_predictor.get_recommendations(user_id)
        return recommendations
    except Exception as e:
        logger.error(f"Burnout recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
