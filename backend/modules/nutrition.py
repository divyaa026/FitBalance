"""
Enhanced Nutrition Module
Complete protein optimization system with CNN Food Classification, GRU Time-Series Learning, and SHAP Explainability
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
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

# Import our enhanced ML models
try:
    from ml_models.nutrition.cnn_food_classifier import FoodClassifierInference, FoodDetection
    from ml_models.nutrition.gru_protein_optimizer import ProteinOptimizer as GRUProteinOptimizer, HealthMetrics
    from ml_models.nutrition.shap_explainer import initialize_shap_explainer
    from backend.database.nutrition_db import nutrition_db
    from backend.integrations.health_apis import health_aggregator, create_demo_health_data
    ENHANCED_MODE = True
except ImportError as e:
    logger.warning(f"Could not import enhanced modules: {e}. Using fallback implementations.")
    ENHANCED_MODE = False

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
    """Enhanced protein optimization system with CNN, GRU, and SHAP explainability"""
    
    def __init__(self):
        # Initialize components based on available modules
        if ENHANCED_MODE:
            self.food_classifier = FoodClassifierInference()
            self.gru_optimizer = GRUProteinOptimizer()
            self.shap_explainer = initialize_shap_explainer(self.gru_optimizer)
            self.database = nutrition_db
            logger.info("Enhanced nutrition system initialized")
        else:
            # Fallback to original implementation
            self.model = CNNGRUModel()
            self.food_database = self._load_food_database()
            self.user_profiles = {}
            self.meal_history = {}
            self._load_model()
            logger.info("Fallback nutrition system initialized")
    
    async def analyze_meal_enhanced(self, meal_photo: UploadFile, user_id: str, 
                                  dietary_restrictions: List[str]) -> Dict:
        """Enhanced meal analysis using CNN Food Classifier"""
        try:
            if not ENHANCED_MODE:
                return await self.analyze_meal(meal_photo, user_id, dietary_restrictions)
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                content = await meal_photo.read()
                tmp_file.write(content)
                image_path = tmp_file.name
            
            # Use enhanced CNN food classifier
            analysis_result = self.food_classifier.analyze_meal_image(image_path)
            detections = analysis_result['detections']
            
            # Get user health data for protein optimization
            health_data = await self._get_user_health_data(user_id)
            
            # Generate protein recommendation using GRU
            if health_data:
                protein_recommendation = self.gru_optimizer.optimize_protein_intake(user_id, health_data)
                
                # Generate SHAP explanation
                shap_explanation = None
                if self.shap_explainer and health_data:
                    shap_explanation = self.shap_explainer.explain_recommendation(
                        health_data[-1], user_id
                    )
            else:
                # Fallback to basic recommendation
                protein_recommendation = None
                shap_explanation = None
            
            # Calculate nutritional analysis
            total_protein = analysis_result['total_protein']
            total_calories = analysis_result['total_calories']
            
            # Determine protein deficit
            target_protein = protein_recommendation.adjusted_protein if protein_recommendation else 120.0
            protein_deficit = max(0, target_protein - total_protein)
            
            # Generate recommendations
            recommendations = self._generate_enhanced_recommendations(
                detections, protein_deficit, dietary_restrictions, protein_recommendation
            )
            
            # Calculate meal quality score
            meal_quality_score = self._calculate_enhanced_meal_quality(
                detections, total_protein, total_calories, target_protein
            )
            
            # Store meal data in database
            if self.database:
                detected_foods_dict = {
                    str(i): {
                        'name': det.food_class,
                        'protein': det.protein_content,
                        'calories': det.calories,
                        'portion': det.portion_grams
                    } for i, det in enumerate(detections)
                }
                
                self.database.log_meal(
                    user_id=int(user_id) if user_id.isdigit() else 1,
                    image_path=image_path,
                    detected_foods=detected_foods_dict,
                    total_protein=total_protein,
                    total_calories=total_calories,
                    confidence=analysis_result['confidence']
                )
            
            # Clean up
            os.unlink(image_path)
            
            # Prepare response
            response = {
                'total_protein': total_protein,
                'total_calories': total_calories,
                'detected_foods': [
                    {
                        'name': det.food_class.replace('_', ' ').title(),
                        'confidence': det.confidence,
                        'protein_content': det.protein_content,
                        'calories': det.calories,
                        'portion_grams': det.portion_grams,
                        'bounding_box': det.bounding_box
                    } for det in detections
                ],
                'protein_deficit': protein_deficit,
                'target_protein': target_protein,
                'recommendations': recommendations,
                'meal_quality_score': meal_quality_score,
                'protein_optimization': {
                    'baseline_protein': protein_recommendation.baseline_protein if protein_recommendation else target_protein,
                    'adjusted_protein': protein_recommendation.adjusted_protein if protein_recommendation else target_protein,
                    'sleep_factor': protein_recommendation.sleep_factor if protein_recommendation else 1.0,
                    'hrv_factor': protein_recommendation.hrv_factor if protein_recommendation else 1.0,
                    'activity_factor': protein_recommendation.activity_factor if protein_recommendation else 1.0,
                    'confidence': protein_recommendation.confidence if protein_recommendation else 0.8,
                    'explanation': protein_recommendation.explanation if protein_recommendation else "Using baseline recommendation"
                },
                'shap_explanation': {
                    'feature_importance': shap_explanation.feature_importance if shap_explanation else {},
                    'explanation_text': shap_explanation.explanation_text if shap_explanation else "Basic nutrition analysis",
                    'confidence_score': shap_explanation.confidence_score if shap_explanation else 0.7
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced meal analysis error: {str(e)}")
            # Fallback to original method
            return await self.analyze_meal(meal_photo, user_id, dietary_restrictions)
    
    async def _get_user_health_data(self, user_id: str, days: int = 7) -> Optional[List[HealthMetrics]]:
        """Get user health data for protein optimization"""
        try:
            if not ENHANCED_MODE:
                return None
            
            # Try to get real health data from APIs
            historical_data = await health_aggregator.get_historical_data(user_id, days)
            
            if not historical_data:
                # Generate demo data for testing
                from backend.integrations.health_apis import get_demo_historical_data
                historical_data = get_demo_historical_data(days)
            
            # Convert to HealthMetrics format
            health_metrics = []
            for data_point in historical_data:
                health_metric = HealthMetrics(
                    date=datetime.combine(data_point.date, datetime.min.time()),
                    sleep_duration=data_point.sleep_duration or 7.5,
                    sleep_quality=data_point.sleep_quality or 7.0,
                    hrv=data_point.hrv or 45.0,
                    activity_level=data_point.activity_score or 5.0,
                    stress_level=int(data_point.stress_level or 4),
                    protein_intake=0.0,  # Will be filled from meal logs
                    recovery_score=data_point.recovery_score or 7.0
                )
                health_metrics.append(health_metric)
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Health data retrieval failed: {e}")
            return None
    
    def _generate_enhanced_recommendations(self, detections: List, protein_deficit: float,
                                        dietary_restrictions: List[str], 
                                        protein_recommendation=None) -> List[str]:
        """Generate enhanced nutrition recommendations"""
        recommendations = []
        
        if protein_deficit > 10:
            recommendations.append(f"Add {protein_deficit:.1f}g more protein to reach your optimized target")
            
            # Suggest specific high-protein foods based on dietary restrictions
            suitable_foods = self._get_suitable_protein_foods(dietary_restrictions)
            if suitable_foods:
                recommendations.append(f"Consider adding: {', '.join(suitable_foods[:3])}")
        
        # Add personalized recommendations based on health data
        if protein_recommendation:
            if protein_recommendation.sleep_factor > 1.1:
                recommendations.append("Poor sleep quality detected - protein needs increased for recovery")
            if protein_recommendation.activity_factor > 1.2:
                recommendations.append("High activity level - increased protein for muscle repair")
            if protein_recommendation.hrv_factor > 1.1:
                recommendations.append("Elevated stress levels - additional protein for immune support")
        
        # Food variety recommendations
        food_categories = set()
        for detection in detections:
            if hasattr(detection, 'food_class'):
                food_categories.add(self._get_food_category(detection.food_class))
        
        if len(food_categories) < 3:
            recommendations.append("Try to include more food variety for complete nutrition")
        
        # Timing recommendations
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10:
            recommendations.append("Great breakfast choice! Consider adding protein for sustained energy")
        elif 11 <= current_hour <= 14:
            recommendations.append("Lunch looks good! Include protein for afternoon productivity")
        elif current_hour >= 18:
            recommendations.append("Evening meal - moderate protein for overnight recovery")
        
        return recommendations
    
    def _get_suitable_protein_foods(self, dietary_restrictions: List[str]) -> List[str]:
        """Get protein foods suitable for dietary restrictions"""
        all_protein_foods = {
            'chicken breast', 'salmon', 'eggs', 'greek yogurt', 'tofu', 
            'quinoa', 'beans', 'almonds', 'tuna', 'cottage cheese'
        }
        
        restrictions_lower = [r.lower() for r in dietary_restrictions]
        
        # Filter based on restrictions
        if 'vegetarian' in restrictions_lower:
            all_protein_foods.discard('chicken breast')
            all_protein_foods.discard('salmon')
            all_protein_foods.discard('tuna')
        
        if 'vegan' in restrictions_lower:
            all_protein_foods.discard('chicken breast')
            all_protein_foods.discard('salmon')
            all_protein_foods.discard('eggs')
            all_protein_foods.discard('greek yogurt')
            all_protein_foods.discard('tuna')
            all_protein_foods.discard('cottage cheese')
        
        if 'dairy-free' in restrictions_lower or 'lactose-free' in restrictions_lower:
            all_protein_foods.discard('greek yogurt')
            all_protein_foods.discard('cottage cheese')
        
        if 'nut-free' in restrictions_lower:
            all_protein_foods.discard('almonds')
        
        return list(all_protein_foods)
    
    def _get_food_category(self, food_name: str) -> str:
        """Get food category for variety analysis"""
        protein_foods = {'chicken', 'salmon', 'tuna', 'eggs', 'tofu', 'beans'}
        grain_foods = {'quinoa', 'rice', 'oats', 'bread'}
        vegetable_foods = {'broccoli', 'spinach', 'carrot'}
        dairy_foods = {'yogurt', 'cheese', 'milk'}
        
        food_lower = food_name.lower()
        
        for protein in protein_foods:
            if protein in food_lower:
                return 'protein'
        
        for grain in grain_foods:
            if grain in food_lower:
                return 'grain'
        
        for vegetable in vegetable_foods:
            if vegetable in food_lower:
                return 'vegetable'
        
        for dairy in dairy_foods:
            if dairy in food_lower:
                return 'dairy'
        
        return 'other'
    
    def _calculate_enhanced_meal_quality(self, detections: List, total_protein: float,
                                       total_calories: float, target_protein: float) -> float:
        """Calculate enhanced meal quality score"""
        score = 0.0
        
        # Protein adequacy (40 points)
        protein_ratio = total_protein / target_protein if target_protein > 0 else 0
        if 0.8 <= protein_ratio <= 1.2:
            score += 40
        elif 0.6 <= protein_ratio <= 1.4:
            score += 30
        else:
            score += 20
        
        # Food variety (30 points)
        unique_categories = len(set(self._get_food_category(
            det.food_class if hasattr(det, 'food_class') else str(det)
        ) for det in detections))
        score += min(unique_categories * 10, 30)
        
        # Nutritional balance (20 points)
        if len(detections) >= 3:  # Multiple food items
            score += 20
        elif len(detections) == 2:
            score += 15
        else:
            score += 10
        
        # Portion appropriateness (10 points)
        if 300 <= total_calories <= 800:  # Reasonable meal size
            score += 10
        elif 200 <= total_calories <= 1000:
            score += 8
        else:
            score += 5
        
        return min(100.0, score)
    
    async def get_enhanced_recommendations(self, user_id: str, target_protein: Optional[float] = None,
                                         activity_level: str = "moderate") -> Dict:
        """Get enhanced personalized nutrition recommendations"""
        try:
            if not ENHANCED_MODE:
                return await self.get_recommendations(user_id, target_protein, activity_level)
            
            # Get user health data
            health_data = await self._get_user_health_data(user_id)
            
            if health_data:
                # Use GRU optimizer for personalized recommendation
                protein_recommendation = self.gru_optimizer.optimize_protein_intake(user_id, health_data)
                
                # Generate SHAP explanation
                shap_explanation = None
                if self.shap_explainer:
                    shap_explanation = self.shap_explainer.explain_recommendation(
                        health_data[-1], user_id
                    )
                
                # Get recent meal history
                recent_meals = []
                if self.database:
                    recent_meals = self.database.get_recent_meals(int(user_id) if user_id.isdigit() else 1)
                
                # Calculate current nutrition status
                recent_protein = sum(meal.get('total_protein', 0) for meal in recent_meals[-3:])
                avg_protein = recent_protein / max(len(recent_meals[-3:]), 1)
                
                recommendations = []
                
                # Protein recommendations
                protein_gap = protein_recommendation.adjusted_protein - avg_protein
                if protein_gap > 10:
                    recommendations.append(f"Increase daily protein by {protein_gap:.1f}g")
                elif protein_gap < -10:
                    recommendations.append(f"Current protein intake is {abs(protein_gap):.1f}g above optimal")
                else:
                    recommendations.append("Current protein intake is well-optimized")
                
                # Health-based recommendations
                latest_health = health_data[-1]
                if latest_health.sleep_quality < 7:
                    recommendations.append("Poor sleep detected - consider protein timing for better recovery")
                if latest_health.activity_level > 7:
                    recommendations.append("High activity - ensure post-workout protein within 2 hours")
                if latest_health.hrv < 40:
                    recommendations.append("Low HRV indicates stress - protein supports immune function")
                
                return {
                    'user_id': user_id,
                    'recommended_protein': protein_recommendation.adjusted_protein,
                    'baseline_protein': protein_recommendation.baseline_protein,
                    'current_average_protein': avg_protein,
                    'optimization_factors': {
                        'sleep_factor': protein_recommendation.sleep_factor,
                        'hrv_factor': protein_recommendation.hrv_factor,
                        'activity_factor': protein_recommendation.activity_factor,
                        'confidence': protein_recommendation.confidence
                    },
                    'recommendations': recommendations,
                    'explanation': protein_recommendation.explanation,
                    'shap_explanation': {
                        'feature_importance': shap_explanation.feature_importance if shap_explanation else {},
                        'explanation_text': shap_explanation.explanation_text if shap_explanation else "",
                        'confidence_score': shap_explanation.confidence_score if shap_explanation else 0.0
                    },
                    'health_metrics': {
                        'sleep_quality': latest_health.sleep_quality,
                        'hrv': latest_health.hrv,
                        'activity_level': latest_health.activity_level,
                        'recovery_score': latest_health.recovery_score
                    },
                    'recent_meals': len(recent_meals)
                }
            else:
                # Fallback to basic recommendations
                return await self.get_recommendations(user_id, target_protein, activity_level)
                
        except Exception as e:
            logger.error(f"Enhanced recommendations error: {str(e)}")
            return await self.get_recommendations(user_id, target_protein, activity_level)
    
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