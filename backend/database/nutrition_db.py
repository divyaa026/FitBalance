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
            if self.engine:
                session = self.SessionLocal()
                food_item = session.query(FoodItems).filter(
                    FoodItems.food_name == food_name
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
            else:
                # Fallback: basic food database
                food_db = {
                    "chicken_breast": {"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165},
                    "salmon": {"protein": 25.0, "carbs": 0.0, "fat": 12.0, "calories": 208},
                    "broccoli": {"protein": 2.8, "carbs": 7.0, "fat": 0.4, "calories": 34},
                    "quinoa": {"protein": 4.4, "carbs": 22.0, "fat": 1.9, "calories": 120},
                }
                return food_db.get(food_name, {"protein": 0, "carbs": 0, "fat": 0, "calories": 0})
        except Exception as e:
            logger.error(f"Food nutrition retrieval failed: {e}")
            return None
    
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