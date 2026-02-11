"""
Nutrition Database Schema and Connection
PostgreSQL database for storing user nutrition data, health metrics, and meal logs
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, JSON, ForeignKey, Text
from datetime import datetime, date
import logging
import os
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

# Database URL - In production, use environment variables
DATABASE_URL = os.getenv(
    "NUTRITION_DATABASE_URL", 
    "postgresql://postgres:fitbalance123@localhost:5432/fitbalance_nutrition"
)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    age = Column(Integer)
    weight_kg = Column(Float)
    height_cm = Column(Float)
    fitness_goal = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)

class HealthMetrics(Base):
    __tablename__ = "health_metrics"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    date = Column(Date, nullable=False)
    sleep_duration = Column(Float)
    sleep_quality = Column(Float)  # 0-10 scale
    hrv = Column(Float)  # Heart Rate Variability
    activity_level = Column(Float)  # Activity score
    stress_level = Column(Integer)  # 1-10 scale
    created_at = Column(DateTime, default=datetime.now)

class FoodItems(Base):
    __tablename__ = "food_items"
    
    food_id = Column(Integer, primary_key=True)
    food_name = Column(String(100), unique=True)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fats_per_100g = Column(Float)
    calories_per_100g = Column(Float)
    food_category = Column(String(50))

class MealLogs(Base):
    __tablename__ = "meal_logs"
    
    meal_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    meal_image_path = Column(String(255))
    meal_timestamp = Column(DateTime, default=datetime.now)
    detected_foods = Column(JSON)  # {food_id: portion_grams}
    total_protein = Column(Float)
    total_calories = Column(Float)
    confidence_score = Column(Float)

class ProteinRecommendations(Base):
    __tablename__ = "protein_recommendations"
    
    rec_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    date = Column(Date, nullable=False)
    baseline_protein = Column(Float)
    adjusted_protein = Column(Float)
    sleep_factor = Column(Float)
    hrv_factor = Column(Float)
    activity_factor = Column(Float)
    shap_explanation = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)

class NutritionDatabase:
    """Database connection and operations for nutrition data"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._init_connection()
    
    def _init_connection(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(self.database_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Fallback to in-memory storage for demo
            self._init_fallback_storage()
    
    def _init_fallback_storage(self):
        """Initialize in-memory storage as fallback"""
        self.users = {}
        self.health_metrics = {}
        self.meal_logs = {}
        self.protein_recommendations = {}
        logger.info("Using in-memory storage as fallback")
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self._populate_food_database()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
    
    def _populate_food_database(self):
        """Populate food items table with nutritional data"""
        foods_data = [
            ("chicken_breast", 31.0, 0.0, 3.6, 165, "protein"),
            ("salmon", 25.0, 0.0, 12.0, 208, "protein"),
            ("eggs", 13.0, 1.1, 11.0, 155, "protein"),
            ("greek_yogurt", 10.0, 3.6, 0.4, 59, "dairy"),
            ("quinoa", 4.4, 22.0, 1.9, 120, "grain"),
            ("broccoli", 2.8, 7.0, 0.4, 34, "vegetable"),
            ("rice", 2.7, 28.0, 0.3, 130, "grain"),
            ("beans", 21.0, 23.0, 0.5, 127, "legume"),
            ("tofu", 15.0, 4.0, 8.0, 144, "protein"),
            ("almonds", 21.0, 22.0, 49.0, 579, "nuts"),
            ("spinach", 2.9, 3.6, 0.4, 23, "vegetable"),
            ("sweet_potato", 2.0, 20.0, 0.1, 86, "vegetable"),
            ("avocado", 2.0, 9.0, 15.0, 160, "fruit"),
            ("tuna", 30.0, 0.0, 1.0, 132, "protein"),
            ("oats", 11.0, 66.0, 6.0, 389, "grain"),
        ]
        
        try:
            session = self.SessionLocal()
            for food_data in foods_data:
                food_item = FoodItems(
                    food_name=food_data[0],
                    protein_per_100g=food_data[1],
                    carbs_per_100g=food_data[2],
                    fats_per_100g=food_data[3],
                    calories_per_100g=food_data[4],
                    food_category=food_data[5]
                )
                session.merge(food_item)  # Use merge to avoid duplicates
            session.commit()
            session.close()
            logger.info("Food database populated successfully")
        except Exception as e:
            logger.error(f"Food database population failed: {e}")
    
    def log_meal(self, user_id: int, image_path: str, detected_foods: Dict, 
                 total_protein: float, total_calories: float, confidence: float):
        """Log a meal with detected foods and nutrition"""
        try:
            if self.engine:
                session = self.SessionLocal()
                meal_log = MealLogs(
                    user_id=user_id,
                    meal_image_path=image_path,
                    detected_foods=detected_foods,
                    total_protein=total_protein,
                    total_calories=total_calories,
                    confidence_score=confidence
                )
                session.add(meal_log)
                session.commit()
                session.close()
                logger.info(f"Meal logged for user {user_id}")
            else:
                # Fallback storage
                if user_id not in self.meal_logs:
                    self.meal_logs[user_id] = []
                self.meal_logs[user_id].append({
                    'timestamp': datetime.now(),
                    'detected_foods': detected_foods,
                    'total_protein': total_protein,
                    'total_calories': total_calories,
                    'confidence': confidence
                })
        except Exception as e:
            logger.error(f"Meal logging failed: {e}")
    
    def get_user_health_metrics(self, user_id: int, days: int = 7) -> pd.DataFrame:
        """Get recent health metrics for a user"""
        try:
            if self.engine:
                query = text(f"""
                SELECT * FROM health_metrics 
                WHERE user_id = :user_id AND date >= CURRENT_DATE - INTERVAL '{days} days'
                ORDER BY date DESC
                """)
                return pd.read_sql(query, self.engine, params={"user_id": user_id})
            else:
                # Fallback: return mock data
                return pd.DataFrame({
                    'date': [date.today()],
                    'sleep_quality': [7.5],
                    'hrv': [45.0],
                    'activity_level': [6.5],
                    'stress_level': [3]
                })
        except Exception as e:
            logger.error(f"Health metrics retrieval failed: {e}")
            return pd.DataFrame()
    
    def save_protein_recommendation(self, user_id: int, rec_date: date,
                                  baseline_protein: float, adjusted_protein: float,
                                  sleep_factor: float, hrv_factor: float,
                                  activity_factor: float, shap_explanation: Dict):
        """Save protein recommendation with SHAP explanation"""
        try:
            if self.engine:
                session = self.SessionLocal()
                recommendation = ProteinRecommendations(
                    user_id=user_id,
                    date=rec_date,
                    baseline_protein=baseline_protein,
                    adjusted_protein=adjusted_protein,
                    sleep_factor=sleep_factor,
                    hrv_factor=hrv_factor,
                    activity_factor=activity_factor,
                    shap_explanation=shap_explanation
                )
                session.add(recommendation)
                session.commit()
                session.close()
            else:
                # Fallback storage
                if user_id not in self.protein_recommendations:
                    self.protein_recommendations[user_id] = []
                self.protein_recommendations[user_id].append({
                    'date': rec_date,
                    'baseline_protein': baseline_protein,
                    'adjusted_protein': adjusted_protein,
                    'shap_explanation': shap_explanation
                })
        except Exception as e:
            logger.error(f"Protein recommendation save failed: {e}")
    
    def get_food_nutrition(self, food_name: str) -> Optional[Dict]:
        """Get nutritional information for a food item"""
        try:
            # Normalize the food name
            normalized_name = self._normalize_food_name(food_name)
            
            if self.engine:
                session = self.SessionLocal()
                food_item = session.query(FoodItems).filter(
                    FoodItems.food_name == normalized_name
                ).first()
                session.close()
                
                if food_item:
                    return {
                        'protein': food_item.protein_per_100g,
                        'carbs': food_item.carbs_per_100g,
                        'fat': food_item.fats_per_100g,
                        'calories': food_item.calories_per_100g,
                        'category': food_item.food_category
                    }
            
            # Fallback: comprehensive food database
            food_db = self._get_comprehensive_food_database()
            result = food_db.get(normalized_name)
            
            if result:
                return result
            
            # Try fuzzy matching for common variations
            result = self._fuzzy_match_food(normalized_name, food_db)
            if result:
                return result
                
            # Last resort: return estimated values based on food type
            logger.warning(f"No nutrition data found for: {food_name}, using estimates")
            return self._estimate_nutrition(normalized_name)
            
        except Exception as e:
            logger.error(f"Food nutrition retrieval failed: {e}")
            return None
    
    def _normalize_food_name(self, food_name: str) -> str:
        """Normalize food name for better matching"""
        normalized = food_name.lower().strip()
        # Remove common descriptors
        descriptors = ['grilled', 'roasted', 'steamed', 'boiled', 'baked', 'fried', 
                      'cooked', 'raw', 'fresh', 'sliced', 'diced', 'chopped', 'sauteed']
        for desc in descriptors:
            normalized = normalized.replace(f"{desc}_", "").replace(f"_{desc}", "")
        return normalized
    
    def _get_comprehensive_food_database(self) -> Dict:
        """Comprehensive food nutrition database (per 100g)"""
        return {
            # Proteins
            "chicken_breast": {"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165, "category": "protein"},
            "chicken": {"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165, "category": "protein"},
            "turkey_breast": {"protein": 29.0, "carbs": 0.0, "fat": 1.0, "calories": 135, "category": "protein"},
            "turkey": {"protein": 29.0, "carbs": 0.0, "fat": 1.0, "calories": 135, "category": "protein"},
            "salmon": {"protein": 25.4, "carbs": 0.0, "fat": 12.4, "calories": 208, "category": "protein"},
            "salmon_fillet": {"protein": 25.4, "carbs": 0.0, "fat": 12.4, "calories": 208, "category": "protein"},
            "tuna": {"protein": 30.0, "carbs": 0.0, "fat": 1.0, "calories": 132, "category": "protein"},
            "tuna_steak": {"protein": 30.0, "carbs": 0.0, "fat": 1.0, "calories": 132, "category": "protein"},
            "beef": {"protein": 26.0, "carbs": 0.0, "fat": 15.0, "calories": 250, "category": "protein"},
            "lean_beef": {"protein": 26.0, "carbs": 0.0, "fat": 15.0, "calories": 250, "category": "protein"},
            "egg": {"protein": 13.0, "carbs": 1.1, "fat": 11.0, "calories": 155, "category": "protein"},
            "hard_boiled_egg": {"protein": 13.0, "carbs": 1.1, "fat": 11.0, "calories": 155, "category": "protein"},
            "tofu": {"protein": 15.7, "carbs": 4.0, "fat": 8.0, "calories": 144, "category": "protein"},
            "greek_yogurt": {"protein": 10.0, "carbs": 3.6, "fat": 0.4, "calories": 59, "category": "dairy"},
            
            # Vegetables
            "broccoli": {"protein": 2.8, "carbs": 7.0, "fat": 0.4, "calories": 34, "category": "vegetable"},
            "spinach": {"protein": 2.9, "carbs": 3.6, "fat": 0.4, "calories": 23, "category": "vegetable"},
            "asparagus": {"protein": 2.2, "carbs": 3.9, "fat": 0.2, "calories": 27, "category": "vegetable"},
            "bell_peppers": {"protein": 1.0, "carbs": 6.0, "fat": 0.3, "calories": 31, "category": "vegetable"},
            "bell_pepper": {"protein": 1.0, "carbs": 6.0, "fat": 0.3, "calories": 31, "category": "vegetable"},
            "green_beans": {"protein": 1.8, "carbs": 7.0, "fat": 0.1, "calories": 35, "category": "vegetable"},
            "carrots": {"protein": 0.9, "carbs": 10.0, "fat": 0.2, "calories": 41, "category": "vegetable"},
            "lettuce": {"protein": 1.4, "carbs": 2.9, "fat": 0.2, "calories": 15, "category": "vegetable"},
            "tomato": {"protein": 0.9, "carbs": 3.9, "fat": 0.2, "calories": 18, "category": "vegetable"},
            "cucumber": {"protein": 0.7, "carbs": 3.6, "fat": 0.1, "calories": 16, "category": "vegetable"},
            
            # Carbs
            "rice": {"protein": 2.7, "carbs": 28.0, "fat": 0.3, "calories": 130, "category": "grain"},
            "brown_rice": {"protein": 2.6, "carbs": 23.0, "fat": 0.9, "calories": 112, "category": "grain"},
            "quinoa": {"protein": 4.4, "carbs": 21.3, "fat": 1.9, "calories": 120, "category": "grain"},
            "quinoa_cooked": {"protein": 4.4, "carbs": 21.3, "fat": 1.9, "calories": 120, "category": "grain"},
            "sweet_potato": {"protein": 2.0, "carbs": 20.1, "fat": 0.2, "calories": 86, "category": "vegetable"},
            "potato": {"protein": 2.0, "carbs": 17.0, "fat": 0.1, "calories": 77, "category": "vegetable"},
            "oats": {"protein": 13.2, "carbs": 67.7, "fat": 6.5, "calories": 389, "category": "grain"},
            "pasta": {"protein": 5.0, "carbs": 25.0, "fat": 0.9, "calories": 131, "category": "grain"},
            "bread": {"protein": 9.0, "carbs": 49.0, "fat": 3.2, "calories": 265, "category": "grain"},
            "white_bread": {"protein": 9.0, "carbs": 49.0, "fat": 3.2, "calories": 265, "category": "grain"},
            
            # Fats & Oils
            "avocado": {"protein": 2.0, "carbs": 8.5, "fat": 14.7, "calories": 160, "category": "fruit"},
            "almonds": {"protein": 21.2, "carbs": 21.6, "fat": 49.9, "calories": 579, "category": "nuts"},
            "olive_oil": {"protein": 0.0, "carbs": 0.0, "fat": 100.0, "calories": 884, "category": "oil"},
            "olive_oil_drizzle": {"protein": 0.0, "carbs": 0.0, "fat": 100.0, "calories": 884, "category": "oil"},
            
            # Legumes
            "beans": {"protein": 21.0, "carbs": 62.4, "fat": 0.9, "calories": 347, "category": "legume"},
            "lentils": {"protein": 9.0, "carbs": 20.0, "fat": 0.4, "calories": 116, "category": "legume"},
            "chickpeas": {"protein": 8.9, "carbs": 27.4, "fat": 2.6, "calories": 164, "category": "legume"},
            
            # JUNK FOOD / FAST FOOD (realistic values)
            "pizza": {"protein": 11.0, "carbs": 33.0, "fat": 10.0, "calories": 266, "category": "junk_food"},
            "cheese_pizza": {"protein": 12.0, "carbs": 33.0, "fat": 11.0, "calories": 276, "category": "junk_food"},
            "pepperoni_pizza": {"protein": 12.0, "carbs": 30.0, "fat": 13.0, "calories": 298, "category": "junk_food"},
            "burger": {"protein": 17.0, "carbs": 28.0, "fat": 15.0, "calories": 295, "category": "junk_food"},
            "hamburger": {"protein": 17.0, "carbs": 28.0, "fat": 15.0, "calories": 295, "category": "junk_food"},
            "cheeseburger": {"protein": 18.0, "carbs": 28.0, "fat": 18.0, "calories": 327, "category": "junk_food"},
            "fries": {"protein": 3.4, "carbs": 41.4, "fat": 15.0, "calories": 312, "category": "junk_food"},
            "french_fries": {"protein": 3.4, "carbs": 41.4, "fat": 15.0, "calories": 312, "category": "junk_food"},
            "chips": {"protein": 6.6, "carbs": 53.0, "fat": 35.0, "calories": 536, "category": "junk_food"},
            "potato_chips": {"protein": 6.6, "carbs": 53.0, "fat": 35.0, "calories": 536, "category": "junk_food"},
            "hot_dog": {"protein": 10.4, "carbs": 2.0, "fat": 26.0, "calories": 290, "category": "junk_food"},
            "soda": {"protein": 0.0, "carbs": 10.6, "fat": 0.0, "calories": 41, "category": "junk_food"},
            "cola": {"protein": 0.0, "carbs": 10.6, "fat": 0.0, "calories": 41, "category": "junk_food"},
            "candy": {"protein": 0.0, "carbs": 75.0, "fat": 5.0, "calories": 400, "category": "junk_food"},
            "chocolate": {"protein": 4.9, "carbs": 61.0, "fat": 30.0, "calories": 546, "category": "junk_food"},
            "chocolate_bar": {"protein": 4.9, "carbs": 61.0, "fat": 30.0, "calories": 546, "category": "junk_food"},
            "ice_cream": {"protein": 3.5, "carbs": 23.0, "fat": 11.0, "calories": 207, "category": "junk_food"},
            "donut": {"protein": 4.9, "carbs": 51.0, "fat": 25.0, "calories": 452, "category": "junk_food"},
            "doughnut": {"protein": 4.9, "carbs": 51.0, "fat": 25.0, "calories": 452, "category": "junk_food"},
            "cake": {"protein": 4.0, "carbs": 58.0, "fat": 15.0, "calories": 387, "category": "junk_food"},
            "cookie": {"protein": 5.0, "carbs": 68.0, "fat": 20.0, "calories": 480, "category": "junk_food"},
            "cookies": {"protein": 5.0, "carbs": 68.0, "fat": 20.0, "calories": 480, "category": "junk_food"},
            "pastry": {"protein": 5.6, "carbs": 45.0, "fat": 20.0, "calories": 384, "category": "junk_food"},
            "fried_chicken": {"protein": 25.0, "carbs": 10.0, "fat": 18.0, "calories": 290, "category": "junk_food"},
            "chicken_nuggets": {"protein": 15.0, "carbs": 16.0, "fat": 18.0, "calories": 296, "category": "junk_food"},
            "nachos": {"protein": 9.0, "carbs": 56.0, "fat": 24.0, "calories": 467, "category": "junk_food"},
            "taco": {"protein": 12.0, "carbs": 28.0, "fat": 14.0, "calories": 226, "category": "junk_food"},
            "burrito": {"protein": 12.0, "carbs": 38.0, "fat": 12.0, "calories": 314, "category": "junk_food"},
            "milkshake": {"protein": 6.0, "carbs": 48.0, "fat": 9.0, "calories": 300, "category": "junk_food"},
            "energy_drink": {"protein": 0.0, "carbs": 11.0, "fat": 0.0, "calories": 45, "category": "junk_food"},
            "popcorn": {"protein": 12.6, "carbs": 77.8, "fat": 4.5, "calories": 387, "category": "snack"},
            "onion_rings": {"protein": 3.8, "carbs": 38.0, "fat": 16.0, "calories": 310, "category": "junk_food"},
        }
    
    def _fuzzy_match_food(self, food_name: str, food_db: Dict) -> Optional[Dict]:
        """Try to find a match using fuzzy string matching"""
        # Check for partial matches
        for db_name, nutrition in food_db.items():
            # Check if either name contains the other
            if food_name in db_name or db_name in food_name:
                logger.info(f"Fuzzy matched '{food_name}' to '{db_name}'")
                return nutrition
        return None
    
    def _estimate_nutrition(self, food_name: str) -> Dict:
        """Estimate nutrition based on food name keywords"""
        # Default estimates based on common food types
        if any(word in food_name for word in ['chicken', 'turkey', 'fish', 'meat', 'beef', 'pork']):
            return {"protein": 25.0, "carbs": 0.0, "fat": 8.0, "calories": 180, "category": "protein"}
        elif any(word in food_name for word in ['vegetable', 'salad', 'greens', 'veggie']):
            return {"protein": 2.0, "carbs": 5.0, "fat": 0.3, "calories": 30, "category": "vegetable"}
        elif any(word in food_name for word in ['rice', 'pasta', 'bread', 'grain', 'potato']):
            return {"protein": 3.0, "carbs": 25.0, "fat": 0.5, "calories": 120, "category": "carb"}
        elif any(word in food_name for word in ['oil', 'butter', 'fat']):
            return {"protein": 0.0, "carbs": 0.0, "fat": 100.0, "calories": 884, "category": "fat"}
        else:
            # Generic estimate
            return {"protein": 5.0, "carbs": 15.0, "fat": 3.0, "calories": 100, "category": "unknown"}
    
    def get_recent_meals(self, user_id: int, days: int = 7) -> List[Dict]:
        """Get recent meals for a user"""
        try:
            if self.engine:
                session = self.SessionLocal()
                recent_meals = session.query(MealLogs).filter(
                    MealLogs.user_id == user_id,
                    MealLogs.meal_timestamp >= datetime.now() - pd.Timedelta(days=days)
                ).order_by(MealLogs.meal_timestamp.desc()).all()
                session.close()
                
                return [{
                    'timestamp': meal.meal_timestamp,
                    'detected_foods': meal.detected_foods,
                    'total_protein': meal.total_protein,
                    'total_calories': meal.total_calories
                } for meal in recent_meals]
            else:
                # Fallback
                return self.meal_logs.get(user_id, [])[-days:]
        except Exception as e:
            logger.error(f"Recent meals retrieval failed: {e}")
            return []

# Global database instance
nutrition_db = NutritionDatabase()