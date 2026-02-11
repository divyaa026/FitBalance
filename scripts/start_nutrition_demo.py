"""
Simplified Nutrition System Demo
Works with basic dependencies - no complex ML models required for initial testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional, Dict
import logging
import json
import tempfile
from PIL import Image
import numpy as np
from datetime import datetime, date, timedelta
import random
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FitBalance Nutrition System - Demo",
    description="Simplified nutrition optimization system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    logger.info(f"Request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = datetime.now() - start_time
        logger.info(f"Response: {response.status_code} - Time: {process_time.total_seconds():.3f}s")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Simple food database
FOOD_DATABASE = {
    "chicken_breast": {"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165, "category": "protein"},
    "salmon": {"protein": 25.0, "carbs": 0.0, "fat": 12.0, "calories": 208, "category": "protein"},
    "eggs": {"protein": 13.0, "carbs": 1.1, "fat": 11.0, "calories": 155, "category": "protein"},
    "greek_yogurt": {"protein": 10.0, "carbs": 3.6, "fat": 0.4, "calories": 59, "category": "dairy"},
    "quinoa": {"protein": 4.4, "carbs": 22.0, "fat": 1.9, "calories": 120, "category": "grain"},
    "broccoli": {"protein": 2.8, "carbs": 7.0, "fat": 0.4, "calories": 34, "category": "vegetable"},
    "rice": {"protein": 2.7, "carbs": 28.0, "fat": 0.3, "calories": 130, "category": "grain"},
    "tofu": {"protein": 15.0, "carbs": 4.0, "fat": 8.0, "calories": 144, "category": "protein"},
    "almonds": {"protein": 21.0, "carbs": 22.0, "fat": 49.0, "calories": 579, "category": "nuts"},
    "spinach": {"protein": 2.9, "carbs": 3.6, "fat": 0.4, "calories": 23, "category": "vegetable"},
}

def generate_demo_health_data():
    """Generate realistic demo health data"""
    return {
        "sleep_duration": random.uniform(6.0, 9.0),
        "sleep_quality": random.uniform(5.0, 9.5),
        "hrv": random.uniform(25, 75),
        "activity_level": random.uniform(3.0, 9.0),
        "stress_level": random.uniform(2.0, 8.0),
        "recovery_score": random.uniform(5.0, 9.5)
    }

def mock_food_detection(image_size):
    """Mock food detection based on image analysis"""
    # Simulate different food combinations based on image characteristics
    combinations = [
        [("chicken_breast", 150), ("broccoli", 100), ("rice", 80)],
        [("salmon", 120), ("quinoa", 100), ("spinach", 60)],
        [("eggs", 2), ("toast", 50), ("avocado", 40)],
        [("greek_yogurt", 200), ("almonds", 30), ("berries", 80)],
        [("tofu", 100), ("vegetables", 150), ("rice", 100)]
    ]
    
    selected = random.choice(combinations)
    detected_foods = []
    
    for food_name, portion in selected:
        if food_name in FOOD_DATABASE:
            nutrition = FOOD_DATABASE[food_name]
            detected_foods.append({
                "name": food_name.replace("_", " ").title(),
                "confidence": random.uniform(0.75, 0.95),
                "protein_content": nutrition["protein"] * (portion / 100),
                "calories": nutrition["calories"] * (portion / 100),
                "portion_grams": portion,
                "category": nutrition["category"]
            })
    
    return detected_foods

def calculate_protein_recommendation(health_data, user_profile=None):
    """Calculate personalized protein recommendation"""
    # Base protein calculation
    base_protein = 120.0  # Default base
    
    # Adjust based on health factors
    sleep_factor = 1.0
    if health_data["sleep_quality"] < 6:
        sleep_factor = 1.15  # Poor sleep = more protein
    elif health_data["sleep_quality"] > 8:
        sleep_factor = 0.95  # Great sleep = slightly less needed
    
    # Activity factor
    activity_factor = 1.0
    if health_data["activity_level"] > 7:
        activity_factor = 1.25  # High activity = more protein
    elif health_data["activity_level"] < 4:
        activity_factor = 0.9   # Low activity = less protein
    
    # HRV factor (stress indicator)
    hrv_factor = 1.0
    if health_data["hrv"] < 35:
        hrv_factor = 1.1   # Low HRV = more stress = more protein
    elif health_data["hrv"] > 60:
        hrv_factor = 0.98  # High HRV = good recovery
    
    # Calculate adjusted protein
    adjusted_protein = base_protein * sleep_factor * activity_factor * hrv_factor
    adjusted_protein = max(80, min(adjusted_protein, 200))  # Reasonable bounds
    
    return {
        "baseline_protein": base_protein,
        "adjusted_protein": adjusted_protein,
        "sleep_factor": sleep_factor,
        "activity_factor": activity_factor,
        "hrv_factor": hrv_factor,
        "confidence": 0.85
    }

def generate_recommendations(detected_foods, protein_deficit, health_data):
    """Generate nutrition recommendations"""
    recommendations = []
    
    if protein_deficit > 10:
        recommendations.append(f"Add {protein_deficit:.1f}g more protein to reach your target")
        recommendations.append("Consider protein-rich snacks like Greek yogurt or nuts")
    elif protein_deficit < -10:
        recommendations.append("Your protein intake is above target - well done!")
    else:
        recommendations.append("Your protein intake is well-balanced")
    
    # Activity-based recommendations
    if health_data["activity_level"] > 7:
        recommendations.append("High activity detected - ensure protein within 2 hours post-workout")
    
    # Sleep-based recommendations
    if health_data["sleep_quality"] < 6:
        recommendations.append("Poor sleep may affect protein synthesis - consider timing")
    
    # Food variety
    categories = set(food["category"] for food in detected_foods)
    if len(categories) < 3:
        recommendations.append("Try to include more food variety for complete nutrition")
    
    return recommendations

def create_simple_explanation(health_data, protein_optimization):
    """Create simple explanation for protein recommendation"""
    factors = []
    
    if protein_optimization["sleep_factor"] > 1.05:
        factors.append(f"Poor sleep quality ({health_data['sleep_quality']:.1f}/10) increases protein needs")
    elif protein_optimization["sleep_factor"] < 0.95:
        factors.append(f"Excellent sleep quality ({health_data['sleep_quality']:.1f}/10) optimizes protein use")
    
    if protein_optimization["activity_factor"] > 1.1:
        factors.append(f"High activity level ({health_data['activity_level']:.1f}/10) increases protein requirements")
    elif protein_optimization["activity_factor"] < 0.95:
        factors.append(f"Lower activity level reduces protein needs")
    
    if protein_optimization["hrv_factor"] > 1.05:
        factors.append(f"Low HRV ({health_data['hrv']:.0f}ms) indicates stress, requiring more protein")
    elif protein_optimization["hrv_factor"] < 0.98:
        factors.append(f"Good HRV ({health_data['hrv']:.0f}ms) shows optimal recovery")
    
    if not factors:
        factors.append("Your health metrics are well-balanced, maintaining baseline protein recommendation")
    
    return ". ".join(factors)

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "FitBalance Nutrition System - Demo Version", 
        "status": "running",
        "features": [
            "Meal photo analysis",
            "Personalized protein recommendations", 
            "Health data integration",
            "Nutrition insights"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Add OPTIONS handler for CORS preflight requests
@app.options("/nutrition/{path:path}")
async def options_handler():
    return {"message": "OK"}

@app.post("/nutrition/analyze-meal")
async def analyze_meal_photo(
    meal_photo: UploadFile = File(...),
    user_id: str = "default",
    dietary_restrictions: Optional[List[str]] = None
):
    """Analyze meal photo for protein content and optimization - Flutter compatible"""
    try:
        # Validate file type
        if not meal_photo.content_type or not meal_photo.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await meal_photo.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Save and process image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load image to get basic info
            image = Image.open(tmp_file_path)
            image_size = image.size
            
            # Mock food detection
            detected_foods = mock_food_detection(image_size)
            
            # Calculate totals
            total_protein = sum(food["protein_content"] for food in detected_foods)
            total_calories = sum(food["calories"] for food in detected_foods)
            
            # Generate health data for user
            health_data = generate_demo_health_data()
            
            # Calculate protein recommendation
            protein_optimization = calculate_protein_recommendation(health_data)
            target_protein = protein_optimization["adjusted_protein"]
            protein_deficit = target_protein - total_protein
            
            # Generate recommendations
            recommendations = generate_recommendations(detected_foods, protein_deficit, health_data)
            
            # Create explanation
            explanation = create_simple_explanation(health_data, protein_optimization)
            
            # Calculate meal quality score
            variety_score = len(set(food["category"] for food in detected_foods)) * 15
            protein_score = min((total_protein / target_protein) * 40, 40) if target_protein > 0 else 20
            balance_score = 30 if total_calories > 200 else 20
            meal_quality_score = min(variety_score + protein_score + balance_score, 100)
            
            # Format response to match Flutter frontend expectations
            response = {
                "success": True,
                "message": "Meal analyzed successfully",
                "analysis": {
                    "total_protein": round(total_protein, 1),
                    "total_calories": round(total_calories, 0),
                    "detected_foods": detected_foods,
                    "protein_deficit": round(max(0, protein_deficit), 1),
                    "target_protein": round(target_protein, 1),
                    "recommendations": recommendations,
                    "meal_quality_score": round(meal_quality_score, 1),
                    "confidence": 0.85,
                    "explanation": explanation,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "protein_optimization": {
                        **protein_optimization,
                        "explanation": explanation
                    },
                    "health_metrics": health_data
                }
            }
            
            logger.info(f"Successfully analyzed meal for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meal analysis error: {str(e)}")
        # Return error response in expected format
        return {
            "success": False,
            "message": f"Analysis failed: {str(e)}",
            "analysis": None
        }

@app.get("/nutrition/recommendations/{user_id}")
async def get_nutrition_recommendations(
    user_id: str,
    target_protein: Optional[float] = None,
    activity_level: str = "moderate"
):
    """Get personalized nutrition recommendations - Flutter compatible"""
    try:
        # Generate health data
        health_data = generate_demo_health_data()
        
        # Override activity level if provided
        activity_mapping = {"low": 3.0, "moderate": 5.5, "high": 8.0}
        health_data["activity_level"] = activity_mapping.get(activity_level, 5.5)
        
        # Calculate protein recommendation
        protein_optimization = calculate_protein_recommendation(health_data)
        
        if target_protein:
            protein_optimization["baseline_protein"] = target_protein
            protein_optimization["adjusted_protein"] = target_protein * (
                protein_optimization["sleep_factor"] * 
                protein_optimization["activity_factor"] * 
                protein_optimization["hrv_factor"]
            )
        
        # Generate recommendations
        recommendations = []
        current_protein = 100  # Simulated current intake
        protein_gap = protein_optimization["adjusted_protein"] - current_protein
        
        if protein_gap > 10:
            recommendations.append(f"Increase daily protein by {protein_gap:.1f}g")
            recommendations.append("Consider protein-rich snacks like Greek yogurt or nuts")
        elif protein_gap < -10:
            recommendations.append("Consider reducing protein intake slightly")
        else:
            recommendations.append("Current protein intake is well-optimized")
        
        # Activity-specific recommendations
        if health_data["activity_level"] > 7:
            recommendations.append("High activity - ensure post-workout protein timing")
        
        # Sleep-specific recommendations
        if health_data["sleep_quality"] < 6:
            recommendations.append("Poor sleep - protein supports recovery")
        
        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "recommended_protein": round(protein_optimization["adjusted_protein"], 1),
                "baseline_protein": round(protein_optimization["baseline_protein"], 1),
                "current_average_protein": current_protein,
                "optimization_factors": {
                    "sleep_factor": round(protein_optimization["sleep_factor"], 2),
                    "activity_factor": round(protein_optimization["activity_factor"], 2),
                    "hrv_factor": round(protein_optimization["hrv_factor"], 2),
                    "confidence": protein_optimization["confidence"]
                },
                "recommendations": recommendations,
                "explanation": create_simple_explanation(health_data, protein_optimization),
                "health_metrics": health_data,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to get recommendations: {str(e)}",
            "data": None
        }

@app.post("/nutrition/health-data/{user_id}")
async def update_health_data(user_id: str, health_data: Dict):
    """Update user health data"""
    try:
        # Validate health data
        required_fields = ["sleep_quality", "activity_level"]
        for field in required_fields:
            if field not in health_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # In a real system, this would save to database
        logger.info(f"Health data updated for user {user_id}: {health_data}")
        
        return {
            "success": True,
            "status": "success", 
            "message": "Health data updated successfully",
            "user_id": user_id,
            "updated_fields": list(health_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health data update error: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to update health data: {str(e)}"
        }

@app.get("/nutrition/food-database")
async def get_food_database():
    """Get food nutrition database"""
    try:
        formatted_foods = {}
        for food_id, nutrition in FOOD_DATABASE.items():
            formatted_foods[food_id] = {
                "name": food_id.replace("_", " ").title(),
                "protein_per_100g": nutrition["protein"],
                "carbs_per_100g": nutrition["carbs"],
                "fat_per_100g": nutrition["fat"],
                "calories_per_100g": nutrition["calories"],
                "category": nutrition["category"]
            }
        
        categories = list(set(food["category"] for food in FOOD_DATABASE.values()))
        
        return {
            "success": True,
            "data": {
                "foods": formatted_foods,
                "total_foods": len(formatted_foods),
                "categories": categories,
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Food database error: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to get food database: {str(e)}",
            "data": None
        }

@app.get("/nutrition/demo-explanation/{user_id}")
async def get_demo_explanation(user_id: str):
    """Get demo explanation for nutrition recommendations"""
    try:
        health_data = generate_demo_health_data()
        protein_optimization = calculate_protein_recommendation(health_data)
        
        # Create feature importance (simplified)
        feature_importance = {
            "sleep_quality": {
                "value": health_data["sleep_quality"],
                "impact": "positive" if protein_optimization["sleep_factor"] > 1 else "neutral",
                "explanation": f"Sleep quality of {health_data['sleep_quality']:.1f}/10"
            },
            "activity_level": {
                "value": health_data["activity_level"],
                "impact": "positive" if protein_optimization["activity_factor"] > 1 else "neutral",
                "explanation": f"Activity level of {health_data['activity_level']:.1f}/10"
            },
            "hrv": {
                "value": health_data["hrv"],
                "impact": "positive" if protein_optimization["hrv_factor"] > 1 else "neutral",
                "explanation": f"HRV of {health_data['hrv']:.0f}ms"
            }
        }
        
        return {
            "success": True,
            "data": {
                "feature_importance": feature_importance,
                "explanation_text": create_simple_explanation(health_data, protein_optimization),
                "confidence_score": protein_optimization["confidence"],
                "predicted_value": protein_optimization["adjusted_protein"],
                "base_value": protein_optimization["baseline_protein"],
                "health_metrics": health_data,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Demo explanation error: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to get explanation: {str(e)}",
            "data": None
        }

if __name__ == "__main__":
    print("🚀 Starting FitBalance Nutrition System (Demo)")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("🧪 Health check: http://localhost:8000/health")
    
    uvicorn.run("start_nutrition_demo:app", host="0.0.0.0", port=8000, reload=True)