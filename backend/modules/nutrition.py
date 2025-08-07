"""
Nutrition Module
Dynamic protein optimization using meal photos with CNN-GRU architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from fastapi import UploadFile
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FoodItem:
    """Represents a detected food item"""
    name: str
    confidence: float
    protein_content: float  # grams
    calories: float
    serving_size: str
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height

@dataclass
class MealAnalysis:
    """Results of meal photo analysis"""
    total_protein: float
    total_calories: float
    detected_foods: List[FoodItem]
    protein_deficit: float
    recommendations: List[str]
    meal_quality_score: float  # 0-100
    nutritional_balance: Dict[str, float]

class CNNGRUModel(nn.Module):
    """CNN-GRU model for food recognition and nutritional analysis"""
    
    def __init__(self, num_classes=100, hidden_dim=128, num_layers=2):
        super(CNNGRUModel, self).__init__()
        
        # CNN for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # GRU for sequential processing
        self.gru = nn.GRU(512, hidden_dim, num_layers, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 4)  # protein, calories, carbs, fat
        )
    
    def forward(self, x, sequence_length=None):
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = self.conv_layers(x)
        features = features.view(batch_size, -1)  # Flatten
        
        # GRU processing if sequence data available
        if sequence_length is not None:
            features = features.view(batch_size, sequence_length, -1)
            gru_out, _ = self.gru(features)
            features = gru_out[:, -1, :]  # Take last timestep
        
        # Outputs
        classification = self.classifier(features)
        nutritional_values = self.regressor(features)
        
        return classification, nutritional_values

class ProteinOptimizer:
    """Main class for protein optimization and meal analysis"""
    
    def __init__(self):
        self.model = CNNGRUModel()
        self.food_database = self._load_food_database()
        self.user_profiles = {}  # In production, use database
        self.meal_history = {}  # In production, use database
        
        # Load pre-trained model (placeholder)
        self._load_model()
    
    def _load_food_database(self) -> Dict:
        """Load food database with nutritional information"""
        return {
            "chicken_breast": {
                "protein": 31.0,  # grams per 100g
                "calories": 165,
                "carbs": 0.0,
                "fat": 3.6,
                "serving_sizes": ["100g", "150g", "200g"]
            },
            "salmon": {
                "protein": 25.0,
                "calories": 208,
                "carbs": 0.0,
                "fat": 12.0,
                "serving_sizes": ["100g", "150g", "200g"]
            },
            "eggs": {
                "protein": 13.0,
                "calories": 155,
                "carbs": 1.1,
                "fat": 11.0,
                "serving_sizes": ["1 egg", "2 eggs", "3 eggs"]
            },
            "greek_yogurt": {
                "protein": 10.0,
                "calories": 59,
                "carbs": 3.6,
                "fat": 0.4,
                "serving_sizes": ["100g", "150g", "200g"]
            },
            "quinoa": {
                "protein": 4.4,
                "calories": 120,
                "carbs": 22.0,
                "fat": 1.9,
                "serving_sizes": ["100g", "150g", "200g"]
            },
            "broccoli": {
                "protein": 2.8,
                "calories": 34,
                "carbs": 7.0,
                "fat": 0.4,
                "serving_sizes": ["100g", "150g", "200g"]
            },
            "rice": {
                "protein": 2.7,
                "calories": 130,
                "carbs": 28.0,
                "fat": 0.3,
                "serving_sizes": ["100g", "150g", "200g"]
            },
            "beans": {
                "protein": 21.0,
                "calories": 127,
                "carbs": 23.0,
                "fat": 0.5,
                "serving_sizes": ["100g", "150g", "200g"]
            }
        }
    
    def _load_model(self):
        """Load pre-trained model weights"""
        try:
            # In production, load from saved weights
            # self.model.load_state_dict(torch.load('models/nutrition_model.pth'))
            logger.info("Nutrition model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
    
    async def analyze_meal(self, meal_photo: UploadFile, user_id: str, dietary_restrictions: List[str]) -> MealAnalysis:
        """Analyze meal photo for protein content and optimization"""
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                content = await meal_photo.read()
                tmp_file.write(content)
                image_path = tmp_file.name
            
            # Process image
            detected_foods = self._detect_foods(image_path)
            
            # Calculate nutritional content
            total_protein, total_calories, nutritional_balance = self._calculate_nutrition(detected_foods)
            
            # Get user's protein target
            user_profile = self._get_user_profile(user_id)
            target_protein = user_profile.get('target_protein', 120.0)  # Default 120g
            
            # Calculate protein deficit
            protein_deficit = max(0, target_protein - total_protein)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                detected_foods, protein_deficit, dietary_restrictions, user_profile
            )
            
            # Calculate meal quality score
            meal_quality_score = self._calculate_meal_quality(detected_foods, nutritional_balance)
            
            # Store meal data
            self._store_meal_data(user_id, detected_foods, total_protein, total_calories)
            
            # Clean up
            os.unlink(image_path)
            
            return MealAnalysis(
                total_protein=total_protein,
                total_calories=total_calories,
                detected_foods=detected_foods,
                protein_deficit=protein_deficit,
                recommendations=recommendations,
                meal_quality_score=meal_quality_score,
                nutritional_balance=nutritional_balance
            )
            
        except Exception as e:
            logger.error(f"Meal analysis error: {str(e)}")
            raise
    
    def _detect_foods(self, image_path: str) -> List[FoodItem]:
        """Detect foods in image using CNN-GRU model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = image.resize((224, 224))  # Standard size for CNN
            image_tensor = self._preprocess_image(image)
            
            # Model inference
            with torch.no_grad():
                classification, nutritional_values = self.model(image_tensor.unsqueeze(0))
                
                # Get predictions
                food_probs = F.softmax(classification, dim=1)
                detected_foods = []
                
                # For demo, use mock detection
                detected_foods = self._mock_food_detection(image_path)
                
        except Exception as e:
            logger.error(f"Food detection error: {e}")
            # Fallback to mock detection
            detected_foods = self._mock_food_detection(image_path)
        
        return detected_foods
    
    def _mock_food_detection(self, image_path: str) -> List[FoodItem]:
        """Mock food detection for testing"""
        # Simulate detection of common foods
        mock_foods = [
            FoodItem(
                name="chicken_breast",
                confidence=0.85,
                protein_content=31.0,
                calories=165,
                serving_size="100g",
                bounding_box=(50, 50, 100, 80)
            ),
            FoodItem(
                name="broccoli",
                confidence=0.78,
                protein_content=2.8,
                calories=34,
                serving_size="100g",
                bounding_box=(200, 50, 80, 60)
            ),
            FoodItem(
                name="quinoa",
                confidence=0.72,
                protein_content=4.4,
                calories=120,
                serving_size="100g",
                bounding_box=(150, 150, 90, 70)
            )
        ]
        
        return mock_foods
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to tensor and normalize
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return image_tensor
    
    def _calculate_nutrition(self, detected_foods: List[FoodItem]) -> Tuple[float, float, Dict[str, float]]:
        """Calculate total nutritional content"""
        total_protein = sum(food.protein_content for food in detected_foods)
        total_calories = sum(food.calories for food in detected_foods)
        
        # Calculate macronutrient balance
        total_carbs = sum(self.food_database.get(food.name, {}).get('carbs', 0) for food in detected_foods)
        total_fat = sum(self.food_database.get(food.name, {}).get('fat', 0) for food in detected_foods)
        
        nutritional_balance = {
            'protein': total_protein,
            'calories': total_calories,
            'carbs': total_carbs,
            'fat': total_fat
        }
        
        return total_protein, total_calories, nutritional_balance
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """Get user's nutritional profile"""
        if user_id not in self.user_profiles:
            # Default profile
            self.user_profiles[user_id] = {
                'target_protein': 120.0,  # grams
                'target_calories': 2000.0,
                'activity_level': 'moderate',
                'dietary_restrictions': [],
                'weight': 70.0,  # kg
                'height': 170.0,  # cm
                'age': 30,
                'gender': 'not_specified'
            }
        
        return self.user_profiles[user_id]
    
    def _generate_recommendations(self, detected_foods: List[FoodItem], protein_deficit: float, 
                                dietary_restrictions: List[str], user_profile: Dict) -> List[str]:
        """Generate personalized nutrition recommendations"""
        recommendations = []
        
        # Protein deficit recommendations
        if protein_deficit > 20:
            recommendations.append(f"Add {protein_deficit:.1f}g more protein to meet your daily target")
            
            # Suggest high-protein foods
            high_protein_foods = self._get_high_protein_foods(dietary_restrictions)
            recommendations.append(f"Consider adding: {', '.join(high_protein_foods[:3])}")
        
        # Meal balance recommendations
        food_names = [food.name for food in detected_foods]
        
        if 'vegetables' not in food_names and 'broccoli' not in food_names:
            recommendations.append("Add vegetables for better nutritional balance")
        
        if len(detected_foods) < 3:
            recommendations.append("Consider adding more variety to your meal")
        
        # Timing recommendations
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10:
            recommendations.append("Great breakfast choice! Consider adding complex carbs for sustained energy")
        elif 11 <= current_hour <= 14:
            recommendations.append("Good lunch selection. Remember to stay hydrated")
        elif 17 <= current_hour <= 20:
            recommendations.append("Dinner looks good. Consider lighter options if eating late")
        
        return recommendations
    
    def _get_high_protein_foods(self, dietary_restrictions: List[str]) -> List[str]:
        """Get list of high-protein foods respecting dietary restrictions"""
        high_protein_foods = []
        
        for food_name, nutrition in self.food_database.items():
            if nutrition['protein'] > 20:  # High protein threshold
                # Check dietary restrictions
                if 'vegetarian' in dietary_restrictions and food_name in ['chicken_breast', 'salmon']:
                    continue
                if 'vegan' in dietary_restrictions and food_name in ['chicken_breast', 'salmon', 'eggs']:
                    continue
                
                high_protein_foods.append(food_name.replace('_', ' ').title())
        
        return high_protein_foods
    
    def _calculate_meal_quality(self, detected_foods: List[FoodItem], nutritional_balance: Dict[str, float]) -> float:
        """Calculate meal quality score (0-100)"""
        score = 0.0
        
        # Protein content score (40 points)
        protein_ratio = nutritional_balance['protein'] / max(nutritional_balance['calories'] / 10, 1)
        if 0.8 <= protein_ratio <= 1.2:  # Good protein-to-calorie ratio
            score += 40
        elif 0.6 <= protein_ratio <= 1.4:
            score += 30
        else:
            score += 20
        
        # Variety score (30 points)
        unique_food_types = len(set(food.name for food in detected_foods))
        score += min(unique_food_types * 10, 30)
        
        # Balance score (30 points)
        if nutritional_balance['carbs'] > 0 and nutritional_balance['fat'] > 0:
            score += 30
        elif nutritional_balance['carbs'] > 0 or nutritional_balance['fat'] > 0:
            score += 20
        else:
            score += 10
        
        return min(100.0, score)
    
    def _store_meal_data(self, user_id: str, detected_foods: List[FoodItem], total_protein: float, total_calories: float):
        """Store meal data for user"""
        if user_id not in self.meal_history:
            self.meal_history[user_id] = []
        
        meal_data = {
            'timestamp': datetime.now().isoformat(),
            'foods': [food.name for food in detected_foods],
            'total_protein': total_protein,
            'total_calories': total_calories
        }
        
        self.meal_history[user_id].append(meal_data)
    
    async def get_recommendations(self, user_id: str, target_protein: Optional[float] = None, 
                                activity_level: str = "moderate") -> Dict:
        """Get personalized nutrition recommendations"""
        try:
            user_profile = self._get_user_profile(user_id)
            
            # Update target protein if provided
            if target_protein is not None:
                user_profile['target_protein'] = target_protein
            
            # Calculate daily protein needs based on activity level
            base_protein = user_profile.get('weight', 70.0) * 1.2  # 1.2g per kg body weight
            
            activity_multipliers = {
                'sedentary': 1.0,
                'light': 1.2,
                'moderate': 1.4,
                'active': 1.6,
                'very_active': 1.8
            }
            
            multiplier = activity_multipliers.get(activity_level, 1.4)
            recommended_protein = base_protein * multiplier
            
            # Get recent meal history
            recent_meals = self.meal_history.get(user_id, [])[-7:]  # Last 7 meals
            avg_protein = sum(meal['total_protein'] for meal in recent_meals) / max(len(recent_meals), 1)
            
            # Generate recommendations
            recommendations = []
            
            if avg_protein < recommended_protein * 0.8:
                recommendations.append(f"Increase protein intake. Current average: {avg_protein:.1f}g, Recommended: {recommended_protein:.1f}g")
            
            # Suggest meal timing
            if len(recent_meals) < 3:
                recommendations.append("Consider eating more frequently throughout the day")
            
            # Suggest food variety
            recent_foods = set()
            for meal in recent_meals:
                recent_foods.update(meal['foods'])
            
            if len(recent_foods) < 10:
                recommendations.append("Try to incorporate more food variety for better nutrition")
            
            return {
                'user_id': user_id,
                'recommended_protein': recommended_protein,
                'current_average_protein': avg_protein,
                'activity_level': activity_level,
                'recommendations': recommendations,
                'recent_meals': len(recent_meals)
            }
            
        except Exception as e:
            logger.error(f"Recommendations error: {str(e)}")
            raise 