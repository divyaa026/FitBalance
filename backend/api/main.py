from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import logging

# Import core modules
from modules.biomechanics import BiomechanicsCoach
from modules.nutrition import ProteinOptimizer
from modules.burnout import BurnoutPredictor

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

# Initialize core modules
biomechanics_coach = BiomechanicsCoach()
protein_optimizer = ProteinOptimizer()
burnout_predictor = BurnoutPredictor()

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to FitBalance AI-powered fitness platform!",
        "modules": [
            "biomechanics-coaching",
            "protein-optimization", 
            "burnout-prediction"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "modules": {
        "biomechanics": "active",
        "nutrition": "active", 
        "burnout": "active"
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
        recommendations = await protein_optimizer.get_recommendations(
            user_id, target_protein, activity_level
        )
        return recommendations
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
