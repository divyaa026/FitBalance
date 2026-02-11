from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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

# Import database module for new endpoints
from database.nutrition_db import nutrition_db

# Import burnout module for real integration
from modules.burnout import BurnoutPredictor

# Import the actual modules
from modules.nutrition import ProteinOptimizer
from modules.biomechanics import BiomechanicsCoach

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
        time.sleep(0.5)
        
        import random
        
        # Mock food combinations for more realistic results
        food_combinations = [
            [
                {"name": "grilled_chicken", "protein_content": 25.0, "calories": 165, "confidence": 0.92},
                {"name": "brown_rice", "protein_content": 4.5, "calories": 130, "confidence": 0.88},
                {"name": "broccoli", "protein_content": 2.8, "calories": 34, "confidence": 0.85}
            ],
            [
                {"name": "salmon", "protein_content": 22.0, "calories": 206, "confidence": 0.94},
                {"name": "quinoa", "protein_content": 4.4, "calories": 120, "confidence": 0.89},
                {"name": "spinach", "protein_content": 2.9, "calories": 23, "confidence": 0.82}
            ],
            [
                {"name": "greek_yogurt", "protein_content": 10.0, "calories": 59, "confidence": 0.91},
                {"name": "almonds", "protein_content": 21.0, "calories": 579, "confidence": 0.87},
                {"name": "berries", "protein_content": 1.0, "calories": 50, "confidence": 0.90}
            ]
        ]
        
        selected_foods = random.choice(food_combinations)
        total_protein = sum(food["protein_content"] for food in selected_foods)
        total_calories = sum(food["calories"] for food in selected_foods)
        
        return {
            "total_protein": round(total_protein, 1),
            "total_calories": round(total_calories, 0),
            "detected_foods": selected_foods,
            "meal_quality_score": random.randint(75, 95),
            "recommendations": [
                "Great protein variety!" if len(selected_foods) > 2 else "Consider adding protein sources",
                "Well balanced meal!" if total_protein > 20 else "Try adding more protein"
            ],
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

# Initialize actual modules
try:
    biomechanics_coach = BiomechanicsCoach()
    logger.info("Successfully initialized BiomechanicsCoach")
except Exception as e:
    logger.error(f"Failed to initialize BiomechanicsCoach: {e}")
    biomechanics_coach = MockBiomechanicsCoach()
    logger.info("Using MockBiomechanicsCoach as fallback")

try:
    protein_optimizer = ProteinOptimizer()
    logger.info("Successfully initialized ProteinOptimizer")
except Exception as e:
    logger.error(f"Failed to initialize ProteinOptimizer: {e}")
    protein_optimizer = None

try:
    burnout_predictor = BurnoutPredictor()
    logger.info("Successfully initialized BurnoutPredictor")
except Exception as e:
    logger.error(f"Failed to initialize BurnoutPredictor: {e}")
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
    """Analyze biomechanics from uploaded video or photo"""
    try:
        logger.info(f"Received biomechanics analysis request: exercise={exercise_type}, user={user_id}, file={video_file.filename}")
        
        # Accept both video and image files
        if not (video_file.content_type.startswith("video/") or video_file.content_type.startswith("image/")):
            raise HTTPException(status_code=400, detail="File must be a video or image")
        
        # Analyze movement
        result = await biomechanics_coach.analyze_movement(
            video_file, exercise_type, user_id
        )
        
        # Check if valid exercise was detected
        if not result.is_valid_exercise:
            return {
                "is_valid_exercise": False,
                "error_message": result.error_message,
                "exercise_type": result.exercise_type,
                "form_score": 0,
                "risk_factors": [],
                "recommendations": ["Please upload a video showing a person performing an exercise."],
                "joint_angles": [],
                "torques": [],
                "heatmap_data": {},
                "form_errors": [],
                "user_id": user_id,
                "analysis_method": "GNN-LSTM with MediaPipe pose detection"
            }
        
        # Convert BiomechanicsAnalysis to dict for JSON response
        # Calculate overall risk level based on form score
        def get_risk_level(form_score):
            if form_score < 50:
                return "high"
            elif form_score < 70:
                return "medium"
            else:
                return "low"
        
        overall_risk_level = get_risk_level(result.form_score)
        
        return {
            "is_valid_exercise": True,
            "error_message": "",
            "exercise_type": result.exercise_type,
            "form_score": result.form_score,
            "risk_factors": result.risk_factors,
            "recommendations": result.recommendations,
            "joint_angles": [
                {"joint_name": joint, "angle": angle, "is_abnormal": False}
                for joint, angle in result.joint_angles.items()
            ],
            "torques": [
                {"joint_name": joint, "torque_magnitude": sum(torques) / len(torques) if torques else 0, "risk_level": overall_risk_level}
                for joint, torques in result.torque_data.items()
            ],
            "heatmap_data": result.heatmap_data,
            "form_errors": [
                {
                    "body_part": err.body_part,
                    "issue": err.issue,
                    "current_value": err.current_value,
                    "expected_range": list(err.expected_range),
                    "severity": err.severity,
                    "correction_tip": err.correction_tip
                }
                for err in (result.form_errors or [])
            ],
            "user_id": user_id,
            "analysis_method": "GNN-LSTM with MediaPipe pose detection"
        }
    except Exception as e:
        logger.error(f"Biomechanics analysis error: {str(e)}", exc_info=True)
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
            except ValueError as validation_error:
                # Image validation failed (not a food image)
                logger.warning(f"Image validation failed: {validation_error}")
                raise HTTPException(
                    status_code=400, 
                    detail=str(validation_error)
                )
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
        session = None
        try:
            session = nutrition_db.SessionLocal()
            from sqlalchemy import text
            foods = session.execute(text("SELECT food_name, food_category FROM food_items ORDER BY food_name"))
            result = [{"name": row[0], "category": row[1]} for row in foods]
            return {"count": len(result), "foods": result}
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
    except Exception as e:
        # Graceful fallback when DB is not running
        return {"count": 0, "foods": [], "note": "Database unavailable; start PostgreSQL to list foods."}

# Burnout Prediction Endpoints
class BurnoutAnalysisRequest(BaseModel):
    user_id: str
    workout_frequency: int
    sleep_hours: float
    stress_level: int  # 1-10 scale
    recovery_time: int  # days
    performance_trend: str = "stable"  # improving, stable, declining

@app.post("/burnout/analyze")
async def analyze_burnout_risk(request: BurnoutAnalysisRequest):
    """Analyze burnout risk based on user metrics"""
    try:
        risk_analysis = await burnout_predictor.analyze_risk(
            request.user_id, request.workout_frequency, request.sleep_hours, 
            request.stress_level, request.recovery_time, request.performance_trend
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
