from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import logging
import json
import time
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import the actual modules
from modules.nutrition import ProteinOptimizer

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
        # Simulate processing time
        time.sleep(1)
        return {
            "total_protein": 32.3,
            "total_calories": 436,
            "detected_foods": [
                {"name": "grilled_chicken", "protein_content": 25.0, "confidence": 0.92},
                {"name": "brown_rice", "protein_content": 4.5, "confidence": 0.88}
            ],
            "meal_quality_score": 85.0,
            "recommendations": ["Great protein content!", "Consider adding healthy fats"],
            "user_id": user_id
        }
    
    async def get_recommendations(self, user_id, target_protein, activity_level):
        return {
            "recommendations": [
                "Aim for 1.6g protein per kg body weight",
                "Include protein in every meal",
                "Consider protein timing around workouts"
            ],
            "target_protein": target_protein or 120,
            "activity_level": activity_level
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

# Keep mock modules for non-nutrition features
biomechanics_coach = MockBiomechanicsCoach()
burnout_predictor = MockBurnoutPredictor()

# Initialize actual modules
try:
    protein_optimizer = ProteinOptimizer()
    logger.info("Successfully initialized ProteinOptimizer")
except Exception as e:
    logger.error(f"Failed to initialize ProteinOptimizer: {e}")
    protein_optimizer = None

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
    user_id: str = Form("default"),
    dietary_restrictions: Optional[str] = Form(None)
):
    """Analyze meal photo for protein content and optimization"""
    try:
        print(f"DEBUG: Received meal analysis request")
        print(f"DEBUG: File name: {meal_photo.filename}")
        print(f"DEBUG: File content type: {meal_photo.content_type}")
        print(f"DEBUG: User ID: {user_id}")
        print(f"DEBUG: Dietary restrictions: {dietary_restrictions}")
        
        # Validate file type
        if not meal_photo.content_type or not meal_photo.content_type.startswith("image/"):
            print(f"DEBUG: Invalid content type: {meal_photo.content_type}")
            raise HTTPException(status_code=422, detail="File must be an image (JPEG, PNG, etc.)")
        
        # Parse dietary restrictions
        dietary_restrictions_list = []
        if dietary_restrictions:
            try:
                if dietary_restrictions.startswith('['):
                    dietary_restrictions_list = json.loads(dietary_restrictions)
                else:
                    dietary_restrictions_list = [dietary_restrictions]
            except json.JSONDecodeError:
                dietary_restrictions_list = [dietary_restrictions]
        
        # Use real nutrition analyzer if available, fallback to mock
        if protein_optimizer and hasattr(protein_optimizer, 'analyze_meal'):
            try:
                meal_analysis = await protein_optimizer.analyze_meal(
                    meal_photo, user_id, dietary_restrictions_list
                )
                
                # Convert MealAnalysis to dict for JSON response
                return {
                    "total_protein": meal_analysis.total_protein,
                    "total_calories": meal_analysis.total_calories,
                    "detected_foods": [
                        {
                            "name": food.name,
                            "confidence": food.confidence,
                            "protein_content": food.protein_content,
                            "calories": food.calories,
                            "serving_size": food.serving_size
                        }
                        for food in meal_analysis.detected_foods
                    ],
                    "protein_deficit": meal_analysis.protein_deficit,
                    "recommendations": meal_analysis.recommendations,
                    "meal_quality_score": meal_analysis.meal_quality_score,
                    "nutritional_balance": meal_analysis.nutritional_balance,
                    "user_id": user_id,
                    "analysis_method": "real_nutrition_model"
                }
            except Exception as real_error:
                logger.error(f"Real nutrition analysis failed: {real_error}")
                # Fallback to mock analysis
                mock_optimizer = MockProteinOptimizer()
                result = await mock_optimizer.analyze_meal(meal_photo, user_id, dietary_restrictions_list)
                result["analysis_method"] = "mock_fallback"
                result["fallback_reason"] = str(real_error)
                return result
        else:
            # Use mock implementation
            mock_optimizer = MockProteinOptimizer()
            result = await mock_optimizer.analyze_meal(meal_photo, user_id, dietary_restrictions_list)
            result["analysis_method"] = "mock_only"
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meal analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/nutrition/recommendations/{user_id}")
async def get_nutrition_recommendations(
    user_id: str,
    target_protein: Optional[float] = None,
    activity_level: str = "moderate"
):
    """Get personalized nutrition recommendations"""
    try:
        # Use real nutrition analyzer if available, fallback to mock
        if protein_optimizer and hasattr(protein_optimizer, 'get_recommendations'):
            try:
                recommendations = await protein_optimizer.get_recommendations(
                    user_id, target_protein, activity_level
                )
                recommendations["analysis_method"] = "real_nutrition_model"
                return recommendations
            except Exception as real_error:
                logger.error(f"Real nutrition recommendations failed: {real_error}")
                # Fallback to mock
                mock_optimizer = MockProteinOptimizer()
                result = await mock_optimizer.get_recommendations(user_id, target_protein, activity_level)
                result["analysis_method"] = "mock_fallback"
                result["fallback_reason"] = str(real_error)
                return result
        else:
            # Use mock implementation
            mock_optimizer = MockProteinOptimizer()
            result = await mock_optimizer.get_recommendations(user_id, target_protein, activity_level)
            result["analysis_method"] = "mock_only"
            return result
            
    except Exception as e:
        logger.error(f"Nutrition recommendations error: {str(e)}")
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
