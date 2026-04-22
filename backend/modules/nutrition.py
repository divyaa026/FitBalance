"""
Nutrition Module
Dynamic protein optimization using meal photos with CNN-GRU architecture and Gemini Vision API
"""

from __future__ import annotations

# Optional PyTorch imports - fallback gracefully if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
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

# Optional Gemini import - fallback gracefully if not available
try:
    # Try the new package first, fallback to legacy package
    try:
        import google.genai as genai
        from google.genai import types as genai_types
        GEMINI_NEW_API = True
    except ImportError:
        import google.generativeai as genai
        genai_types = None
        GEMINI_NEW_API = False
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GEMINI_AVAILABLE = False
    GEMINI_NEW_API = False

import base64
import io
# Import database module
from database.nutrition_db import nutrition_db

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
    carbs: float = 0.0  # grams
    fat: float = 0.0  # grams
    fiber: float = 0.0  # grams

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

if TORCH_AVAILABLE:
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
            features = features.view(batch_size, -1)  # Flatten to [batch_size, 512]
            
            # For single image input, create a sequence of length 1
            if sequence_length is None:
                # Reshape for GRU: [batch_size, seq_len=1, features]
                features = features.unsqueeze(1)  # [batch_size, 1, 512]
                gru_out, _ = self.gru(features)
                features = gru_out.squeeze(1)  # [batch_size, hidden_dim]
            else:
                # Handle sequence input if needed
                features = features.view(batch_size, sequence_length, -1)
                gru_out, _ = self.gru(features)
                features = gru_out[:, -1, :]  # Take last timestep
            
            # Outputs
            classification = self.classifier(features)
            nutritional_values = self.regressor(features)
            
            return classification, nutritional_values
else:
    # Fallback when PyTorch is not available
    class CNNGRUModel:
        """Placeholder CNN-GRU model when PyTorch is not available"""
        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch not available - CNNGRUModel is a placeholder")

class ProteinOptimizer:
    """Main class for protein optimization and meal analysis"""
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.model = CNNGRUModel()
        else:
            self.model = None
            logger.warning("PyTorch not available - model-based features will be disabled")
        
        self.food_database = self._load_food_database()
        self.user_profiles = {}  # In production, use database
        self.meal_history = {}  # In production, use database
        
        # Initialize Gemini AI
        self._setup_gemini()
        
        # Load pre-trained model (placeholder)
        self._load_model()
    
    # Model to use for all Gemini calls
    GEMINI_MODEL = 'gemini-2.5-flash'

    def _setup_gemini(self):
        """Setup Google Gemini AI for enhanced food detection."""
        try:
            if not GEMINI_AVAILABLE:
                self.use_gemini = False
                logger.warning("google-generativeai package not installed, food detection disabled")
                return

            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                self.use_gemini = False
                logger.warning("GEMINI_API_KEY not found, food detection disabled")
                return

            if GEMINI_NEW_API:
                self.gemini_client = genai.Client(api_key=api_key)
                self.gemini_model = self.GEMINI_MODEL
            else:
                # Legacy google.generativeai package
                genai.configure(api_key=api_key)
                self.gemini_client = None
                self.gemini_model = genai.GenerativeModel(self.GEMINI_MODEL)

            self.use_gemini = True
            logger.info(f"Gemini AI initialised with model: {self.GEMINI_MODEL}")
        except Exception as e:
            self.use_gemini = False
            logger.warning(f"Could not initialise Gemini AI: {e}, food detection disabled")
    
    # Transient error substrings from the Gemini / gRPC layer
    _GEMINI_TRANSIENT_ERRORS = (
        'unavailable',
        'resource_exhausted',
        'quota',
        'high demand',
        'try again later',
        'rate limit',
        'deadline exceeded',
        'internal error',
        '429',
        '503',
        '500',
    )

    def _is_gemini_transient_error(self, exc: Exception) -> bool:
        """Return True when the exception looks like a temporary Gemini service error."""
        msg = str(exc).lower()
        return any(marker in msg for marker in self._GEMINI_TRANSIENT_ERRORS)

    def _build_gemini_contents(self, contents: list):
        """Convert a mixed list of strings and PIL Images into the correct
        SDK-specific content structure.

        New google.genai SDK requires an explicit Content(role='user', parts=[...])
        wrapper — passing a raw list of strings/Parts is silently malformed.
        """
        if GEMINI_NEW_API and genai_types is not None:
            parts = []
            for item in contents:
                if isinstance(item, Image.Image):
                    buf = io.BytesIO()
                    item.convert('RGB').save(buf, format='JPEG', quality=85)
                    parts.append(genai_types.Part.from_bytes(
                        data=buf.getvalue(), mime_type='image/jpeg'
                    ))
                elif isinstance(item, str):
                    parts.append(genai_types.Part.from_text(text=item))
                else:
                    # Already a Part object — pass through
                    parts.append(item)
            return [genai_types.Content(role='user', parts=parts)]
        else:
            # Legacy SDK accepts a flat list of strings and PIL images
            return contents

    def _call_gemini(self, contents: list):
        """Call Gemini API with support for both new and legacy packages.

        Raises the original exception on transient service errors so callers
        can distinguish them from malformed-response errors.
        """
        structured = self._build_gemini_contents(contents)
        try:
            if GEMINI_NEW_API:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=structured
                )
                return response
            else:
                return self.gemini_model.generate_content(structured)
        except Exception as e:
            if self._is_gemini_transient_error(e):
                logger.warning(f"Gemini API transient error: {e}")
            raise
    
    def _load_food_database(self) -> Dict:
        """Load food database with nutritional information - includes international cuisines"""
        return {
            # === WESTERN FOODS ===
            "chicken_breast": {"protein": 31.0, "calories": 165, "carbs": 0.0, "fat": 3.6, "serving_sizes": ["100g", "150g", "200g"]},
            "grilled_chicken_breast": {"protein": 31.0, "calories": 165, "carbs": 0.0, "fat": 3.6, "serving_sizes": ["100g", "150g", "200g"]},
            "salmon": {"protein": 25.0, "calories": 208, "carbs": 0.0, "fat": 12.0, "serving_sizes": ["100g", "150g", "200g"]},
            "eggs": {"protein": 13.0, "calories": 155, "carbs": 1.1, "fat": 11.0, "serving_sizes": ["1 egg", "2 eggs", "3 eggs"]},
            "greek_yogurt": {"protein": 10.0, "calories": 59, "carbs": 3.6, "fat": 0.4, "serving_sizes": ["100g", "150g", "200g"]},
            "quinoa": {"protein": 4.4, "calories": 120, "carbs": 22.0, "fat": 1.9, "serving_sizes": ["100g", "150g", "200g"]},
            "broccoli": {"protein": 2.8, "calories": 34, "carbs": 7.0, "fat": 0.4, "serving_sizes": ["100g", "150g", "200g"]},
            "rice": {"protein": 2.7, "calories": 130, "carbs": 28.0, "fat": 0.3, "serving_sizes": ["100g", "150g", "200g"]},
            "beans": {"protein": 21.0, "calories": 127, "carbs": 23.0, "fat": 0.5, "serving_sizes": ["100g", "150g", "200g"]},
            
            # === INDIAN FOODS - South Indian ===
            "dosa": {"protein": 2.6, "calories": 133, "carbs": 28.0, "fat": 1.2, "serving_sizes": ["1 dosa", "2 dosas"]},
            "masala_dosa": {"protein": 4.5, "calories": 206, "carbs": 32.0, "fat": 7.0, "serving_sizes": ["1 dosa"]},
            "plain_dosa": {"protein": 2.6, "calories": 133, "carbs": 28.0, "fat": 1.2, "serving_sizes": ["1 dosa"]},
            "idli": {"protein": 2.0, "calories": 39, "carbs": 8.0, "fat": 0.1, "serving_sizes": ["1 idli", "2 idlis", "3 idlis"]},
            "sambar": {"protein": 2.3, "calories": 65, "carbs": 9.0, "fat": 2.0, "serving_sizes": ["100ml", "150ml"]},
            "coconut_chutney": {"protein": 2.0, "calories": 108, "carbs": 4.5, "fat": 9.5, "serving_sizes": ["30g", "50g"]},
            "uttapam": {"protein": 3.5, "calories": 150, "carbs": 25.0, "fat": 4.0, "serving_sizes": ["1 uttapam"]},
            "vada": {"protein": 5.0, "calories": 180, "carbs": 18.0, "fat": 10.0, "serving_sizes": ["1 vada", "2 vadas"]},
            "medu_vada": {"protein": 5.0, "calories": 180, "carbs": 18.0, "fat": 10.0, "serving_sizes": ["1 vada"]},
            "upma": {"protein": 3.2, "calories": 150, "carbs": 22.0, "fat": 5.5, "serving_sizes": ["100g", "150g"]},
            "pongal": {"protein": 4.0, "calories": 145, "carbs": 24.0, "fat": 4.0, "serving_sizes": ["100g", "150g"]},
            "rava_dosa": {"protein": 3.0, "calories": 165, "carbs": 26.0, "fat": 5.5, "serving_sizes": ["1 dosa"]},
            "appam": {"protein": 2.5, "calories": 120, "carbs": 24.0, "fat": 1.5, "serving_sizes": ["1 appam"]},
            
            # === INDIAN FOODS - North Indian ===
            "roti": {"protein": 3.0, "calories": 71, "carbs": 15.0, "fat": 0.4, "serving_sizes": ["1 roti", "2 rotis"]},
            "chapati": {"protein": 3.0, "calories": 71, "carbs": 15.0, "fat": 0.4, "serving_sizes": ["1 chapati"]},
            "naan": {"protein": 3.5, "calories": 130, "carbs": 22.0, "fat": 3.5, "serving_sizes": ["1 naan"]},
            "paratha": {"protein": 4.0, "calories": 180, "carbs": 25.0, "fat": 7.0, "serving_sizes": ["1 paratha"]},
            "aloo_paratha": {"protein": 5.0, "calories": 210, "carbs": 30.0, "fat": 8.0, "serving_sizes": ["1 paratha"]},
            "paneer": {"protein": 18.0, "calories": 265, "carbs": 1.2, "fat": 21.0, "serving_sizes": ["100g", "150g"]},
            "paneer_tikka": {"protein": 16.0, "calories": 250, "carbs": 5.0, "fat": 18.0, "serving_sizes": ["100g", "150g"]},
            "palak_paneer": {"protein": 12.0, "calories": 220, "carbs": 8.0, "fat": 16.0, "serving_sizes": ["150g", "200g"]},
            "butter_chicken": {"protein": 18.0, "calories": 250, "carbs": 10.0, "fat": 16.0, "serving_sizes": ["150g", "200g"]},
            "chicken_tikka_masala": {"protein": 20.0, "calories": 240, "carbs": 8.0, "fat": 14.0, "serving_sizes": ["150g", "200g"]},
            "dal": {"protein": 9.0, "calories": 120, "carbs": 20.0, "fat": 1.5, "serving_sizes": ["100g", "150g"]},
            "dal_tadka": {"protein": 9.0, "calories": 130, "carbs": 18.0, "fat": 4.0, "serving_sizes": ["150g"]},
            "dal_makhani": {"protein": 8.0, "calories": 180, "carbs": 22.0, "fat": 7.0, "serving_sizes": ["150g"]},
            "rajma": {"protein": 8.5, "calories": 140, "carbs": 22.0, "fat": 2.5, "serving_sizes": ["150g"]},
            "chole": {"protein": 9.0, "calories": 165, "carbs": 27.0, "fat": 3.0, "serving_sizes": ["150g"]},
            "chana_masala": {"protein": 9.0, "calories": 165, "carbs": 27.0, "fat": 3.0, "serving_sizes": ["150g"]},
            "biryani": {"protein": 12.0, "calories": 250, "carbs": 35.0, "fat": 8.0, "serving_sizes": ["200g", "250g"]},
            "chicken_biryani": {"protein": 15.0, "calories": 270, "carbs": 32.0, "fat": 10.0, "serving_sizes": ["250g"]},
            "vegetable_biryani": {"protein": 6.0, "calories": 200, "carbs": 38.0, "fat": 5.0, "serving_sizes": ["250g"]},
            "pulao": {"protein": 5.0, "calories": 180, "carbs": 32.0, "fat": 4.0, "serving_sizes": ["150g", "200g"]},
            "tandoori_chicken": {"protein": 28.0, "calories": 195, "carbs": 4.0, "fat": 8.0, "serving_sizes": ["150g", "200g"]},
            "seekh_kebab": {"protein": 22.0, "calories": 230, "carbs": 5.0, "fat": 14.0, "serving_sizes": ["100g"]},
            "korma": {"protein": 14.0, "calories": 280, "carbs": 12.0, "fat": 20.0, "serving_sizes": ["150g"]},
            "aloo_gobi": {"protein": 3.0, "calories": 110, "carbs": 18.0, "fat": 4.0, "serving_sizes": ["150g"]},
            "bhindi_masala": {"protein": 2.5, "calories": 95, "carbs": 12.0, "fat": 5.0, "serving_sizes": ["100g"]},
            "baingan_bharta": {"protein": 2.0, "calories": 100, "carbs": 10.0, "fat": 6.0, "serving_sizes": ["100g"]},
            "raita": {"protein": 3.0, "calories": 60, "carbs": 5.0, "fat": 3.0, "serving_sizes": ["100g"]},
            "lassi": {"protein": 3.5, "calories": 110, "carbs": 18.0, "fat": 2.5, "serving_sizes": ["200ml"]},
            "mango_lassi": {"protein": 3.0, "calories": 150, "carbs": 28.0, "fat": 2.5, "serving_sizes": ["200ml"]},
            "kheer": {"protein": 4.0, "calories": 160, "carbs": 28.0, "fat": 4.0, "serving_sizes": ["100g"]},
            "gulab_jamun": {"protein": 3.0, "calories": 150, "carbs": 25.0, "fat": 5.0, "serving_sizes": ["2 pieces"]},
            "jalebi": {"protein": 1.5, "calories": 150, "carbs": 30.0, "fat": 4.0, "serving_sizes": ["50g"]},
            "samosa": {"protein": 4.0, "calories": 260, "carbs": 30.0, "fat": 14.0, "serving_sizes": ["1 samosa"]},
            "pakora": {"protein": 3.0, "calories": 175, "carbs": 18.0, "fat": 10.0, "serving_sizes": ["50g"]},
            "puri": {"protein": 2.5, "calories": 125, "carbs": 18.0, "fat": 5.0, "serving_sizes": ["1 puri"]},
            "bhatura": {"protein": 4.0, "calories": 200, "carbs": 28.0, "fat": 8.0, "serving_sizes": ["1 bhatura"]},
            "poha": {"protein": 3.0, "calories": 130, "carbs": 25.0, "fat": 3.0, "serving_sizes": ["100g"]},
            "dhokla": {"protein": 4.0, "calories": 120, "carbs": 20.0, "fat": 2.5, "serving_sizes": ["100g"]},
            "khandvi": {"protein": 3.5, "calories": 100, "carbs": 15.0, "fat": 3.0, "serving_sizes": ["100g"]},
            "thepla": {"protein": 4.0, "calories": 140, "carbs": 22.0, "fat": 4.5, "serving_sizes": ["1 thepla"]},
            
            # === CHINESE/ASIAN FOODS ===
            "fried_rice": {"protein": 5.0, "calories": 180, "carbs": 30.0, "fat": 5.0, "serving_sizes": ["150g", "200g"]},
            "noodles": {"protein": 5.0, "calories": 190, "carbs": 35.0, "fat": 4.0, "serving_sizes": ["150g", "200g"]},
            "manchurian": {"protein": 6.0, "calories": 220, "carbs": 25.0, "fat": 12.0, "serving_sizes": ["150g"]},
            "spring_roll": {"protein": 3.0, "calories": 150, "carbs": 20.0, "fat": 7.0, "serving_sizes": ["1 roll"]},
            "momos": {"protein": 6.0, "calories": 140, "carbs": 18.0, "fat": 5.0, "serving_sizes": ["5 pieces"]},
            "dim_sum": {"protein": 5.0, "calories": 130, "carbs": 15.0, "fat": 5.5, "serving_sizes": ["4 pieces"]},
            "tofu": {"protein": 8.0, "calories": 76, "carbs": 2.0, "fat": 4.5, "serving_sizes": ["100g"]},
            "sushi": {"protein": 7.0, "calories": 145, "carbs": 25.0, "fat": 2.0, "serving_sizes": ["6 pieces"]},
            "ramen": {"protein": 12.0, "calories": 350, "carbs": 45.0, "fat": 12.0, "serving_sizes": ["400ml"]},
            
            # === MIDDLE EASTERN/MEDITERRANEAN ===
            "hummus": {"protein": 8.0, "calories": 166, "carbs": 14.0, "fat": 10.0, "serving_sizes": ["100g"]},
            "falafel": {"protein": 13.0, "calories": 333, "carbs": 32.0, "fat": 18.0, "serving_sizes": ["100g"]},
            "shawarma": {"protein": 18.0, "calories": 275, "carbs": 15.0, "fat": 18.0, "serving_sizes": ["200g"]},
            "kebab": {"protein": 20.0, "calories": 220, "carbs": 5.0, "fat": 12.0, "serving_sizes": ["100g"]},
            "pita_bread": {"protein": 5.5, "calories": 165, "carbs": 33.0, "fat": 0.7, "serving_sizes": ["1 pita"]},
            "tabbouleh": {"protein": 2.5, "calories": 85, "carbs": 15.0, "fat": 2.5, "serving_sizes": ["100g"]},
            
            # === MEXICAN ===
            "tacos": {"protein": 10.0, "calories": 210, "carbs": 20.0, "fat": 10.0, "serving_sizes": ["2 tacos"]},
            "burrito": {"protein": 15.0, "calories": 300, "carbs": 40.0, "fat": 10.0, "serving_sizes": ["1 burrito"]},
            "quesadilla": {"protein": 12.0, "calories": 280, "carbs": 25.0, "fat": 15.0, "serving_sizes": ["1 quesadilla"]},
            "guacamole": {"protein": 2.0, "calories": 160, "carbs": 8.5, "fat": 15.0, "serving_sizes": ["100g"]},
            
            # === THAI ===
            "pad_thai": {"protein": 12.0, "calories": 270, "carbs": 35.0, "fat": 10.0, "serving_sizes": ["200g"]},
            "green_curry": {"protein": 15.0, "calories": 220, "carbs": 10.0, "fat": 16.0, "serving_sizes": ["200g"]},
            "tom_yum_soup": {"protein": 8.0, "calories": 100, "carbs": 8.0, "fat": 4.0, "serving_sizes": ["250ml"]},
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
            # Save and preprocess image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                content = await meal_photo.read()
                
                # Preprocess image for optimal Gemini recognition
                image_path = self._preprocess_uploaded_image(content, tmp_file.name)
            
            logger.info(f"Image preprocessed and saved to: {image_path}")

            # Detect foods (Gemini handles food/non-food validation in the same call)
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
        """Detect foods in image using Gemini Vision API.

        Never falls back to random mock detection — if Gemini is unavailable or
        returns no food, a ValueError is raised so the caller can inform the user.
        """
        if not self.use_gemini:
            raise ValueError(
                "Food analysis is temporarily unavailable (AI service not configured). "
                "Please try again later."
            )

        logger.info("Using Gemini Vision API for food detection")
        detected_foods = self._detect_foods_with_gemini(image_path)

        if not detected_foods:
            raise ValueError(
                "No food items could be identified in this image. "
                "Please upload a clear photo of your meal."
            )

        logger.info(f"Gemini detected {len(detected_foods)} food item(s)")
        return detected_foods

    def _basic_food_check(self, image_path: str) -> Tuple[bool, str]:
        """Basic OpenCV-based check to detect obvious non-food images.
        
        Uses face detection and color analysis to identify:
        - Selfies/portraits (face detection)
        - Pure scenery (blue sky, green landscapes - limited food colors)
        - Obviously non-food images
        
        This is a PRE-FILTER before the main Gemini analysis.
        Returns (is_likely_food, reason)
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return True, "Could not read image for pre-filtering"
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. FACE DETECTION - Selfies/Portraits
            # Use OpenCV's built-in cascade classifier for face detection
            face_cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            ]
            
            face_cascade = None
            for cascade_path in face_cascade_paths:
                if os.path.exists(cascade_path):
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    break
            
            if face_cascade is not None and not face_cascade.empty():
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60)  # Minimum face size
                )
                
                if len(faces) > 0:
                    # Calculate total face area
                    total_face_area = sum(w * h for (x, y, w, h) in faces)
                    image_area = image.shape[0] * image.shape[1]
                    face_percentage = (total_face_area / image_area) * 100
                    
                    # If faces occupy significant portion of image, it's likely a selfie
                    if face_percentage > 8 or len(faces) >= 2:
                        logger.info(f"Face detection: {len(faces)} faces detected, {face_percentage:.1f}% of image")
                        return False, f"This looks like a selfie or portrait (detected {len(faces)} face(s)). Please upload a photo of your meal instead."
                    else:
                        logger.info(f"Face detected but small ({face_percentage:.1f}%), allowing image")
            
            # 2. COLOR ANALYSIS - Scenery Detection
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for scenery (sky blue, grass green, etc.)
            # Blue sky: H=100-130, S>50, V>100
            sky_blue_lower = np.array([100, 50, 100])
            sky_blue_upper = np.array([130, 255, 255])
            sky_mask = cv2.inRange(hsv, sky_blue_lower, sky_blue_upper)
            sky_percentage = (np.sum(sky_mask > 0) / (image.shape[0] * image.shape[1])) * 100
            
            # Green grass/trees: H=40-80, S>40, V>40
            green_lower = np.array([40, 40, 40])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_percentage = (np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])) * 100
            
            # If image is predominantly sky blue and/or grass green, it's likely scenery
            if sky_percentage > 35:
                logger.info(f"Scenery detection: {sky_percentage:.1f}% sky blue")
                return False, "This looks like a scenery/landscape photo. Please upload a photo of your meal instead."
            
            if green_percentage > 50:
                # Check if it might be vegetables (smaller green portions are OK)
                logger.info(f"Green detection: {green_percentage:.1f}% green - checking if vegetables...")
                # If very high green with no other food colors, it's likely scenery
                if green_percentage > 70 and sky_percentage > 15:
                    return False, "This looks like an outdoor/nature photo. Please upload a photo of your meal instead."
            
            # 3. EDGE DENSITY CHECK - Food typically has more texture/edges than scenery
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            
            # Very smooth images with low edge density might be scenery or abstract images
            if edge_density < 0.02 and (sky_percentage > 20 or green_percentage > 30):
                logger.info(f"Low edge density ({edge_density:.3f}) with scenery colors")
                return False, "This doesn't appear to be a food photo. Please upload a clear image of your meal."
            
            # 4. FOOD COLOR PRESENCE CHECK
            # Common food colors: browns, reds, oranges, yellows, whites
            # Brown (bread, meat, rice): H=10-30, S>30, V>30
            brown_lower = np.array([10, 30, 30])
            brown_upper = np.array([30, 255, 200])
            brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
            brown_percentage = (np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])) * 100
            
            # Red (tomatoes, meat, peppers): H=0-10 or 160-180, S>50, V>50
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
            red_percentage = (np.sum(red_mask > 0) / (image.shape[0] * image.shape[1])) * 100
            
            # Orange/Yellow (citrus, eggs, cheese): H=15-35, S>50, V>100
            orange_lower = np.array([15, 50, 100])
            orange_upper = np.array([35, 255, 255])
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            orange_percentage = (np.sum(orange_mask > 0) / (image.shape[0] * image.shape[1])) * 100
            
            # White (rice, bread, dairy): H=0-180, S<30, V>200
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            white_percentage = (np.sum(white_mask > 0) / (image.shape[0] * image.shape[1])) * 100
            
            food_color_total = brown_percentage + red_percentage + orange_percentage + (green_percentage * 0.3)  # Green vegetables
            
            logger.info(f"Color analysis: brown={brown_percentage:.1f}%, red={red_percentage:.1f}%, orange={orange_percentage:.1f}%, green={green_percentage:.1f}%, white={white_percentage:.1f}%")
            
            # If no food-typical colors are present, might not be food
            if food_color_total < 5 and white_percentage < 10 and green_percentage < 20:
                logger.info(f"Low food color presence ({food_color_total:.1f}%)")
                # Don't reject outright, but note it
                return True, "Image analyzed - proceeding with food detection (low food color signals)"
            
            # Image passed all checks - likely food
            return True, "Basic analysis: Image appears to contain food"
            
        except Exception as e:
            logger.error(f"Basic food check error: {e}")
            # On error, allow the image to proceed (conservative approach)
            return True, f"Pre-filter check failed: {str(e)}, proceeding with analysis"
    
    def _is_food_image(self, image_path: str) -> Tuple[bool, str]:
        """Check if the image contains food using Gemini Vision API with OpenCV fallback.
        
        This method detects non-food images like selfies, portraits, scenery, etc.
        When Gemini is available, it uses AI-powered detection.
        When Gemini is not available, it uses basic OpenCV-based heuristics.
        """
        # First, try basic OpenCV-based detection as a pre-filter
        # This catches obvious non-food images even before calling Gemini
        is_likely_food, cv_reason = self._basic_food_check(image_path)
        
        if not is_likely_food:
            # OpenCV detected this is definitely NOT food (like a face/portrait)
            logger.warning(f"OpenCV pre-filter rejected image: {cv_reason}")
            return False, cv_reason
        
        if not GEMINI_AVAILABLE or not self.use_gemini:
            # Fallback: use OpenCV-based analysis
            return is_likely_food, cv_reason
            
        try:
            image = Image.open(image_path)
            
            # Simple validation prompt
            validation_prompt = """
            Analyze this image and answer: Does this image contain FOOD or a MEAL?
            
            Respond with a JSON object in this exact format:
            {
                "is_food": true or false,
                "reason": "brief explanation",
                "detected_subjects": "what you see in the image"
            }
            
            Return is_food: true ONLY if the image shows actual food, meals, dishes, or edible items.
            Return is_food: false if the image shows people, faces, portraits, animals, objects, scenery, or anything that is NOT food.
            """
            
            response = self._call_gemini([validation_prompt, image])
            
            if response and response.text:
                response_text = response.text.strip()
                
                # Extract JSON from response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.rfind("```")
                    response_text = response_text[json_start:json_end]
                
                result = json.loads(response_text.strip())
                is_food = result.get("is_food", False)
                reason = result.get("reason", "Unknown")
                detected = result.get("detected_subjects", "Unknown")
                
                logger.info(f"Food validation: is_food={is_food}, reason={reason}, detected={detected}")
                return is_food, f"{reason} - Detected: {detected}"
                
        except Exception as e:
            # If this is a transient Gemini service error, fall back to the
            # OpenCV result rather than wrongly rejecting the image.
            if self._is_gemini_transient_error(e):
                logger.warning(f"Gemini unavailable during food validation, using OpenCV result: {e}")
                self.use_gemini = False  # disable for this request cycle
                return is_likely_food, cv_reason
            logger.error(f"Food validation error: {e}")
            # For non-transient errors fall back to OpenCV as well; we
            # should never reject a valid food image due to an API outage.
            return is_likely_food, f"Validation fallback (API error): {cv_reason}"

        return False, "Unable to validate image"
    
    def _detect_foods_with_gemini(self, image_path: str) -> List[FoodItem]:
        """Use Gemini Vision API to detect and analyse foods in the image.

        Performs both food/non-food validation AND item detection in a single API
        call to conserve quota and eliminate double-call inconsistencies.

        Raises ValueError for non-food images or unrecognisable content.
        Raises RuntimeError for API/service failures.
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini package not installed")

        image = Image.open(image_path)
        logger.info(f"Sending image to Gemini ({self.gemini_model}): size={image.size}, mode={image.mode}")

        # IMPORTANT: No food name examples in this prompt.
        # Any food name written here (even as an example) biases the model
        # toward returning that food regardless of what is in the photo.
        prompt = """Look at this image carefully.

STEP 1 — Is this a food image?
If the image does NOT show food or a meal (e.g. it shows a person, animal, scenery,
object, text, or anything non-edible), respond with ONLY this JSON and nothing else:
{"not_food": true, "detected_subjects": "<describe what you actually see>"}

STEP 2 — If it IS food, identify every visible food item accurately.
Rules:
- Name only what you can actually see. Do not invent or guess foods.
- Use precise descriptive names that include the food type and how it is prepared.
- Estimate realistic portion weights based on what is visible on the plate.
- Calculate nutritional totals: total = (per_100g_value × estimated_grams) / 100
- Confidence: 0.9+ for clearly visible items, 0.7-0.89 for partially visible, 0.5-0.69 for uncertain.

Respond with ONLY this JSON and nothing else (no markdown, no explanation):
{
    "detected_foods": [
        {
            "name": "<descriptive food name based solely on what you see>",
            "confidence": 0.95,
            "estimated_grams": 200,
            "protein_per_100g": 17.0,
            "calories_per_100g": 295,
            "carbs_per_100g": 24.0,
            "fat_per_100g": 14.0,
            "fiber_per_100g": 1.0,
            "total_protein": 34.0,
            "total_calories": 590,
            "total_carbs": 48.0,
            "total_fat": 28.0,
            "preparation_method": "<how it is cooked>",
            "food_category": "<protein/vegetable/grain/dairy/fruit/junk_food/beverage>"
        }
    ],
    "meal_summary": {
        "meal_type": "<breakfast/lunch/dinner/snack>",
        "overall_quality": "<high/medium/low>",
        "health_assessment": "<one sentence assessment>"
    }
}
"""

        try:
            response = self._call_gemini([prompt, image])
            logger.info(f"Gemini API response received")
        except Exception as api_error:
            logger.error(f"Gemini API call failed ({type(api_error).__name__}): {api_error}")
            if self._is_gemini_transient_error(api_error):
                self.use_gemini = False
                raise RuntimeError(
                    "Food analysis service is temporarily unavailable due to high demand. "
                    "Please try again in a moment."
                ) from api_error
            raise RuntimeError(f"Food analysis failed: {api_error}") from api_error

        if not response or not response.text:
            raise RuntimeError("Gemini returned an empty response. Please try again.")

        response_text = response.text.strip()
        logger.info(f"Gemini raw response text: {response_text[:500]}")

        # Strip markdown code fences if present
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.rfind("```")
            response_text = response_text[json_start:json_end]

        try:
            gemini_result = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON: {e} | raw: {response_text[:300]}")
            raise RuntimeError("Unexpected response from food analysis service. Please try again.") from e

        # Gemini explicitly flagged this as not food
        if gemini_result.get("not_food"):
            detected = gemini_result.get("detected_subjects", "non-food content")
            raise ValueError(
                f"This image does not appear to contain food ({detected}). "
                "Please upload a photo of your meal."
            )

        detected_foods = []
        for food_data in gemini_result.get("detected_foods", []):
            estimated_grams = float(food_data.get("estimated_grams", 100))

            total_protein = float(food_data.get("total_protein", 0.0))
            total_calories = int(food_data.get("total_calories", 0))
            total_carbs = float(food_data.get("total_carbs", 0.0))
            total_fat = float(food_data.get("total_fat", 0.0))
            total_fiber = float(food_data.get("total_fiber", 0.0))

            # Calculate totals from per-100g values if not already provided
            if total_protein == 0 and "protein_per_100g" in food_data:
                total_protein = float(food_data["protein_per_100g"]) * estimated_grams / 100.0
            if total_calories == 0 and "calories_per_100g" in food_data:
                total_calories = int(float(food_data["calories_per_100g"]) * estimated_grams / 100.0)
            if total_carbs == 0 and "carbs_per_100g" in food_data:
                total_carbs = float(food_data["carbs_per_100g"]) * estimated_grams / 100.0
            if total_fat == 0 and "fat_per_100g" in food_data:
                total_fat = float(food_data["fat_per_100g"]) * estimated_grams / 100.0
            if total_fiber == 0 and "fiber_per_100g" in food_data:
                total_fiber = float(food_data["fiber_per_100g"]) * estimated_grams / 100.0

            food_item = FoodItem(
                name=food_data.get("name", "unknown_food"),
                confidence=float(food_data.get("confidence", 0.5)),
                protein_content=round(total_protein, 1),
                calories=total_calories,
                serving_size=food_data.get("serving_size", f"{int(estimated_grams)}g"),
                bounding_box=(0, 0, 100, 100),
                carbs=round(total_carbs, 1),
                fat=round(total_fat, 1),
                fiber=round(total_fiber, 1)
            )
            detected_foods.append(food_item)
            logger.info(
                f"  {food_item.name}: {food_item.protein_content}g protein, "
                f"{food_item.carbs}g carbs, {food_item.fat}g fat, {food_item.calories} cal "
                f"(conf {food_item.confidence:.2f})"
            )

        if detected_foods:
            logger.info(f"Gemini detected {len(detected_foods)} food item(s)")
        else:
            logger.warning("Gemini response contained no food items")

        return detected_foods

    
    def _mock_food_detection(self, image_path: str) -> List[FoodItem]:
        """Enhanced mock food detection for testing (simulates Gemini-quality results)"""
        import random
        
        # Comprehensive food pool with multi-cuisine support
        food_pool = [
            # === WESTERN - High-protein mains ===
            {"name": "grilled_chicken_breast", "protein": 31.0, "calories": 165, "confidence": 0.92},
            {"name": "salmon_fillet", "protein": 25.4, "calories": 208, "confidence": 0.89},
            {"name": "lean_beef", "protein": 26.1, "calories": 250, "confidence": 0.85},
            {"name": "turkey_breast", "protein": 29.0, "calories": 189, "confidence": 0.88},
            {"name": "tuna_steak", "protein": 30.0, "calories": 184, "confidence": 0.91},
            {"name": "tofu_grilled", "protein": 15.7, "calories": 144, "confidence": 0.78},
            
            # === WESTERN - Protein sides ===
            {"name": "hard_boiled_egg", "protein": 13.0, "calories": 155, "confidence": 0.95},
            {"name": "greek_yogurt", "protein": 17.0, "calories": 100, "confidence": 0.87},
            {"name": "cottage_cheese", "protein": 11.1, "calories": 98, "confidence": 0.83},
            {"name": "lentils_cooked", "protein": 9.0, "calories": 116, "confidence": 0.80},
            
            # === INDIAN - South Indian ===
            {"name": "dosa", "protein": 2.6, "calories": 133, "confidence": 0.92},
            {"name": "masala_dosa", "protein": 4.5, "calories": 206, "confidence": 0.90},
            {"name": "idli", "protein": 2.0, "calories": 39, "confidence": 0.93},
            {"name": "sambar", "protein": 2.3, "calories": 65, "confidence": 0.88},
            {"name": "coconut_chutney", "protein": 2.0, "calories": 108, "confidence": 0.85},
            {"name": "uttapam", "protein": 3.5, "calories": 150, "confidence": 0.87},
            {"name": "medu_vada", "protein": 5.0, "calories": 180, "confidence": 0.89},
            {"name": "upma", "protein": 3.2, "calories": 150, "confidence": 0.86},
            {"name": "pongal", "protein": 4.0, "calories": 145, "confidence": 0.84},
            
            # === INDIAN - North Indian ===
            {"name": "roti", "protein": 3.0, "calories": 71, "confidence": 0.94},
            {"name": "naan", "protein": 3.5, "calories": 130, "confidence": 0.91},
            {"name": "paratha", "protein": 4.0, "calories": 180, "confidence": 0.88},
            {"name": "aloo_paratha", "protein": 5.0, "calories": 210, "confidence": 0.87},
            {"name": "paneer_tikka", "protein": 16.0, "calories": 250, "confidence": 0.89},
            {"name": "palak_paneer", "protein": 12.0, "calories": 220, "confidence": 0.88},
            {"name": "butter_chicken", "protein": 18.0, "calories": 250, "confidence": 0.92},
            {"name": "chicken_tikka_masala", "protein": 20.0, "calories": 240, "confidence": 0.90},
            {"name": "dal_tadka", "protein": 9.0, "calories": 130, "confidence": 0.89},
            {"name": "dal_makhani", "protein": 8.0, "calories": 180, "confidence": 0.87},
            {"name": "rajma", "protein": 8.5, "calories": 140, "confidence": 0.86},
            {"name": "chole", "protein": 9.0, "calories": 165, "confidence": 0.88},
            {"name": "chicken_biryani", "protein": 15.0, "calories": 270, "confidence": 0.93},
            {"name": "vegetable_biryani", "protein": 6.0, "calories": 200, "confidence": 0.87},
            {"name": "tandoori_chicken", "protein": 28.0, "calories": 195, "confidence": 0.91},
            {"name": "samosa", "protein": 4.0, "calories": 260, "confidence": 0.92},
            {"name": "pakora", "protein": 3.0, "calories": 175, "confidence": 0.85},
            {"name": "poha", "protein": 3.0, "calories": 130, "confidence": 0.88},
            {"name": "dhokla", "protein": 4.0, "calories": 120, "confidence": 0.84},
            
            # === CHINESE/ASIAN ===
            {"name": "fried_rice", "protein": 5.0, "calories": 180, "confidence": 0.90},
            {"name": "noodles", "protein": 5.0, "calories": 190, "confidence": 0.88},
            {"name": "momos", "protein": 6.0, "calories": 140, "confidence": 0.87},
            {"name": "spring_roll", "protein": 3.0, "calories": 150, "confidence": 0.85},
            
            # === VEGETABLES ===
            {"name": "steamed_broccoli", "protein": 2.8, "calories": 34, "confidence": 0.93},
            {"name": "roasted_asparagus", "protein": 2.2, "calories": 27, "confidence": 0.85},
            {"name": "spinach_sauteed", "protein": 2.9, "calories": 23, "confidence": 0.82},
            {"name": "bell_peppers", "protein": 1.0, "calories": 31, "confidence": 0.88},
            {"name": "aloo_gobi", "protein": 3.0, "calories": 110, "confidence": 0.86},
            {"name": "bhindi_masala", "protein": 2.5, "calories": 95, "confidence": 0.83},
            
            # === HEALTHY CARBS ===
            {"name": "quinoa_cooked", "protein": 4.4, "calories": 120, "confidence": 0.76},
            {"name": "brown_rice", "protein": 2.6, "calories": 112, "confidence": 0.81},
            {"name": "basmati_rice", "protein": 2.7, "calories": 130, "confidence": 0.89},
            {"name": "sweet_potato", "protein": 2.0, "calories": 103, "confidence": 0.86},
            
            # === HEALTHY FATS ===
            {"name": "avocado_sliced", "protein": 2.0, "calories": 160, "confidence": 0.91},
            {"name": "almonds", "protein": 21.2, "calories": 579, "confidence": 0.73},
            {"name": "raita", "protein": 3.0, "calories": 60, "confidence": 0.87},
        ]
        
        # Randomly select 2-5 foods to simulate realistic meal detection
        num_foods = random.randint(2, 5)
        selected_foods = random.sample(food_pool, min(num_foods, len(food_pool)))
        
        # Ensure at least one high-protein food if possible
        has_protein = any(food["protein"] > 15 for food in selected_foods)
        if not has_protein and len(food_pool) > num_foods:
            high_protein_foods = [f for f in food_pool if f["protein"] > 15]
            if high_protein_foods:
                selected_foods[0] = random.choice(high_protein_foods)
        
        mock_foods = []
        for i, food in enumerate(selected_foods):
            # Add realistic portion variation
            portion_multiplier = random.uniform(0.6, 1.8)  # More realistic range
            
            # Adjust confidence based on food type (some are easier to detect)
            confidence_adjustment = random.uniform(-0.1, 0.05)
            final_confidence = max(0.5, min(0.98, food["confidence"] + confidence_adjustment))
            
            mock_foods.append(FoodItem(
                name=food["name"],
                confidence=round(final_confidence, 2),
                protein_content=round(food["protein"] * portion_multiplier, 1),
                calories=round(food["calories"] * portion_multiplier),
                serving_size=f"{round(100 * portion_multiplier)}g",
                bounding_box=(
                    random.randint(20, 150), 
                    random.randint(20, 120), 
                    random.randint(80, 140), 
                    random.randint(60, 120)
                )
            ))
        
        logger.info(f"Enhanced mock detected {len(mock_foods)} foods: {[f.name for f in mock_foods]}")
        logger.info(f"Total protein: {sum(f.protein_content for f in mock_foods):.1f}g, Total calories: {sum(f.calories for f in mock_foods)}")
        return mock_foods
    
    def _preprocess_uploaded_image(self, image_bytes: bytes, output_path: str) -> str:
        """Preprocess uploaded image for optimal Gemini recognition"""
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.info(f"Original image: size={image.size}, mode={image.mode}, format={image.format}")
            
            # Convert RGBA/P to RGB
            if image.mode in ('RGBA', 'P', 'LA'):
                # Create white background
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA' or image.mode == 'LA':
                    rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:
                    rgb_image.paste(image)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if image is too large (Gemini works best with images under 4MB and reasonable dimensions)
            max_dimension = 2048  # Gemini's recommended max dimension
            if image.size[0] > max_dimension or image.size[1] > max_dimension:
                # Calculate new size maintaining aspect ratio
                ratio = min(max_dimension / image.size[0], max_dimension / image.size[1])
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                logger.info(f"Resized image to: {new_size}")
            
            # Ensure minimum size (Gemini needs images to be at least a certain size for good recognition)
            min_dimension = 512
            if image.size[0] < min_dimension or image.size[1] < min_dimension:
                # Calculate new size maintaining aspect ratio
                ratio = max(min_dimension / image.size[0], min_dimension / image.size[1])
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                logger.info(f"Upscaled image to: {new_size}")
            
            # Save with optimal quality for Gemini
            image.save(output_path, 'JPEG', quality=85, optimize=True)
            
            # Check file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Preprocessed image saved: size={image.size}, file_size={file_size:.2f}MB")
            
            # If still too large, reduce quality
            if file_size > 4.0:
                image.save(output_path, 'JPEG', quality=70, optimize=True)
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Reduced quality to fit size limit: {file_size:.2f}MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            # Fallback: save original bytes
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            return output_path
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input (PyTorch CNN-GRU)"""
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor and normalize
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return image_tensor
    
    def _calculate_nutrition(self, detected_foods: List[FoodItem]) -> Tuple[float, float, Dict[str, float]]:
        """Calculate total nutritional content - uses values from Gemini or database"""
        total_protein = 0
        total_calories = 0
        total_carbs = 0
        total_fat = 0
        
        for food in detected_foods:
            # If food already has nutrition values from Gemini, use them directly
            if food.protein_content > 0 or food.calories > 0:
                logger.info(f"Using Gemini nutrition for {food.name}: {food.protein_content}g protein, {food.carbs}g carbs, {food.fat}g fat, {food.calories} cal")
                total_protein += food.protein_content
                total_calories += food.calories
                total_carbs += food.carbs
                total_fat += food.fat
            else:
                # Only look up in database if food has no nutrition data
                logger.info(f"Looking up nutrition for {food.name} in database...")
                nutrition = nutrition_db.get_food_nutrition(food.name)
                if nutrition:
                    # Extract serving size from food item
                    serving_g = 100  # default
                    if food.serving_size:
                        try:
                            # Extract number from strings like "150g", "100g", etc.
                            import re
                            match = re.search(r'(\d+)', food.serving_size)
                            if match:
                                serving_g = int(match.group(1))
                        except:
                            pass
                    
                    # Calculate nutrition based on actual serving size
                    food.protein_content = round((nutrition['protein'] * serving_g) / 100, 1)
                    food.calories = int((nutrition['calories'] * serving_g) / 100)
                    food.carbs = round((nutrition['carbs'] * serving_g) / 100, 1)
                    food.fat = round((nutrition['fat'] * serving_g) / 100, 1)
                    
                    logger.info(f"Database lookup for {food.name} ({serving_g}g): {food.protein_content}g protein, {food.carbs}g carbs, {food.fat}g fat, {food.calories} cal")
                    
                    # Accumulate totals
                    total_protein += food.protein_content
                    total_calories += food.calories
                    total_carbs += food.carbs
                    total_fat += food.fat
                else:
                    logger.warning(f"No nutrition data found for {food.name}, using zeros")
                    # Keep existing values (likely 0)
                    total_protein += food.protein_content
                    total_calories += food.calories
                    total_carbs += food.carbs
                    total_fat += food.fat
        
        nutritional_balance = {
            'protein': round(total_protein, 1),
            'calories': int(total_calories),
            'carbs': round(total_carbs, 1),
            'fat': round(total_fat, 1)
        }
        
        logger.info(f"=== TOTAL NUTRITION ===")
        logger.info(f"Protein: {total_protein:.1f}g | Carbs: {total_carbs:.1f}g | Fat: {total_fat:.1f}g | Calories: {total_calories}")
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
        
        # Calculate totals for analysis
        total_protein = sum(food.protein_content for food in detected_foods)
        total_calories = sum(food.calories for food in detected_foods)
        food_names = [food.name.lower() for food in detected_foods]
        
        # Protein deficit recommendations
        if protein_deficit > 20:
            recommendations.append(f"Add {protein_deficit:.1f}g more protein to meet your daily target")
            high_protein_foods = self._get_high_protein_foods(dietary_restrictions)
            recommendations.append(f"Try adding: {', '.join(high_protein_foods[:3])}")
        elif protein_deficit > 0:
            recommendations.append(f"Good protein intake! You need {protein_deficit:.1f}g more to reach your goal")
        else:
            recommendations.append("Excellent! You've met your protein target for this meal")
        
        # Meal composition analysis
        has_vegetables = any(veg in food_names for veg in ['broccoli', 'vegetables', 'spinach', 'kale', 'carrots'])
        has_protein = any(protein in food_names for protein in ['chicken', 'salmon', 'egg', 'tofu', 'fish'])
        has_carbs = any(carb in food_names for carb in ['rice', 'quinoa', 'bread', 'pasta', 'potato'])
        
        if not has_vegetables:
            recommendations.append("Add colorful vegetables for vitamins and fiber")
        if not has_protein and total_protein < 15:
            recommendations.append("Include a lean protein source for muscle maintenance")
        if not has_carbs and total_calories < 400:
            recommendations.append("Consider adding complex carbohydrates for sustained energy")
        
        # Portion size recommendations
        if total_calories > 800:
            recommendations.append("This appears to be a large portion - consider splitting it if not post-workout")
        elif total_calories < 200:
            recommendations.append("This seems like a light meal - perfect for a snack!")
        
        # Timing recommendations
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10:
            recommendations.append("Great breakfast choice! Starting the day with protein supports metabolism")
        elif 11 <= current_hour <= 14:
            recommendations.append("Perfect lunch timing! This meal will fuel your afternoon")
        elif 17 <= current_hour <= 20:
            recommendations.append("Good dinner timing! Try to finish eating 2-3 hours before bed")
        elif current_hour >= 21:
            recommendations.append("Late evening meal - consider lighter options for better sleep")
        
        # Dietary restriction compliance
        if dietary_restrictions:
            compliant_foods = [food.name for food in detected_foods 
                             if self._check_dietary_compliance(food.name, dietary_restrictions)]
            if len(compliant_foods) == len(detected_foods):
                recommendations.append(f"Great job following your {', '.join(dietary_restrictions)} diet!")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
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
    
    def _check_dietary_compliance(self, food_name: str, dietary_restrictions: List[str]) -> bool:
        """Check if a food item complies with dietary restrictions"""
        food_name = food_name.lower()
        
        for restriction in dietary_restrictions:
            restriction = restriction.lower()
            
            if restriction == 'vegetarian':
                meat_items = ['chicken', 'beef', 'pork', 'salmon', 'fish', 'turkey', 'lamb']
                if any(meat in food_name for meat in meat_items):
                    return False
                    
            elif restriction == 'vegan':
                animal_items = ['chicken', 'beef', 'pork', 'salmon', 'fish', 'egg', 'milk', 
                               'cheese', 'yogurt', 'butter', 'turkey', 'lamb']
                if any(animal in food_name for animal in animal_items):
                    return False
                    
            elif restriction == 'gluten-free':
                gluten_items = ['bread', 'pasta', 'wheat', 'barley', 'rye', 'oats']
                if any(gluten in food_name for gluten in gluten_items):
                    return False
                    
            elif restriction == 'dairy-free':
                dairy_items = ['milk', 'cheese', 'yogurt', 'butter', 'cream']
                if any(dairy in food_name for dairy in dairy_items):
                    return False
        
        return True
    
    def _calculate_meal_quality(self, detected_foods: List[FoodItem], nutritional_balance: Dict[str, float]) -> float:
        """Calculate meal quality score (0-100) based on food types and nutritional balance"""
        score = 50.0  # Start at neutral score
        
        food_names = [food.name.lower() for food in detected_foods]
        
        # Define food categories with more comprehensive lists
        junk_foods = [
            'pizza', 'burger', 'fries', 'chips', 'soda', 'candy', 'chocolate', 'ice_cream',
            'donut', 'cake', 'cookie', 'pastry', 'fried_chicken', 'hot_dog', 'nachos',
            'popcorn', 'milkshake', 'energy_drink', 'french_fries', 'onion_rings',
            'taco', 'burrito', 'quesadilla', 'deep_fried', 'processed', 'fast_food',
            'nugget', 'mcdonalds', 'kfc', 'dominos', 'pepperoni', 'cheese_pizza'
        ]
        
        unhealthy_keywords = [
            'fried', 'deep_fried', 'battered', 'breaded', 'crispy', 'loaded',
            'double', 'triple', 'extra_cheese', 'bacon', 'creamy', 'buttery',
            'mayo', 'sauce'
        ]
        
        healthy_foods = [
            'salad', 'vegetables', 'fruit', 'broccoli', 'spinach', 'kale', 'quinoa',
            'chicken_breast', 'salmon', 'tuna', 'tofu', 'lentils', 'beans', 'egg',
            'greek_yogurt', 'oats', 'brown_rice', 'sweet_potato', 'avocado', 'nuts',
            'grilled', 'steamed', 'baked', 'roasted', 'boiled', 'lettuce', 'tomato',
            'cucumber', 'carrot', 'bell_pepper', 'asparagus', 'green_beans'
        ]
        
        # Count food types
        junk_food_count = sum(1 for food in food_names if any(junk in food for junk in junk_foods))
        unhealthy_prep_count = sum(1 for food in food_names if any(keyword in food for keyword in unhealthy_keywords))
        healthy_food_count = sum(1 for food in food_names if any(healthy in food for healthy in healthy_foods))
        
        logger.info(f"Quality analysis: {junk_food_count} junk, {healthy_food_count} healthy, {unhealthy_prep_count} unhealthy prep")
        
        # Apply heavy penalties for junk food
        if junk_food_count > 0:
            penalty = junk_food_count * 30
            score -= penalty
            logger.info(f"Junk food penalty: -{penalty} points ({junk_food_count} junk items)")
        
        # Penalty for unhealthy preparation
        if unhealthy_prep_count > 0:
            penalty = unhealthy_prep_count * 15
            score -= penalty
            logger.info(f"Unhealthy preparation penalty: -{penalty} points")
        
        # Big bonus for healthy foods
        if healthy_food_count > 0:
            bonus = min(healthy_food_count * 10, 30)
            score += bonus
            logger.info(f"Healthy food bonus: +{bonus} points ({healthy_food_count} healthy items)")
        
        # Nutritional balance score (max 25 points)
        total_calories = nutritional_balance.get('calories', 1)
        total_protein = nutritional_balance.get('protein', 0)
        total_carbs = nutritional_balance.get('carbs', 0)
        total_fat = nutritional_balance.get('fat', 0)
        
        # Protein quality scoring (max 25 points)
        if total_calories > 0 and total_protein > 0:
            protein_percentage = (total_protein * 4 / total_calories) * 100
            protein_score = 0
            
            if protein_percentage >= 30:  # Excellent protein (30%+)
                protein_score = 25
            elif protein_percentage >= 20:  # Very good (20-30%)
                protein_score = 20
            elif protein_percentage >= 15:  # Good (15-20%)
                protein_score = 15
            elif protein_percentage >= 10:  # Moderate (10-15%)
                protein_score = 10
            else:  # Low protein (<10%)
                protein_score = 5
            
            score += protein_score
            logger.info(f"Protein quality: {protein_percentage:.1f}% of calories = +{protein_score} points")
        
        # Macronutrient balance (max 15 points)
        if total_carbs > 0 and total_fat > 0 and total_protein > 0:
            # All three macros present = balanced meal
            score += 15
            logger.info(f"Balanced macros: +15 points")
        elif (total_carbs > 0 and total_protein > 0) or (total_fat > 0 and total_protein > 0):
            # Two macros present = moderately balanced
            score += 10
            logger.info(f"Moderately balanced: +10 points")
        
        # Calorie appropriateness (penalty for extreme calories)
        if total_calories > 1200:
            score -= 15
            logger.info(f"Very high calorie meal ({total_calories} cal): -15 points")
        elif total_calories > 900:
            score -= 10
            logger.info(f"High calorie meal ({total_calories} cal): -10 points")
        elif total_calories < 150:
            score -= 5
            logger.info(f"Very low calorie snack: -5 points")
        
        # Variety bonus (max 10 points)
        unique_food_count = len(detected_foods)
        variety_score = 0
        if unique_food_count >= 4:
            variety_score = 10
        elif unique_food_count >= 3:
            variety_score = 7
        elif unique_food_count >= 2:
            variety_score = 4
        
        if variety_score > 0:
            score += variety_score
            logger.info(f"Variety bonus ({unique_food_count} items): +{variety_score} points")
        
        # Ensure score is within 0-100
        final_score = max(0, min(100, score))
        
        # Log quality assessment
        quality_level = "EXCELLENT" if final_score >= 80 else "GOOD" if final_score >= 60 else "FAIR" if final_score >= 40 else "POOR"
        logger.info(f"=== FINAL MEAL QUALITY: {final_score:.1f}/100 ({quality_level}) ===")
        logger.info(f"  Summary: {total_protein:.1f}g protein, {total_calories} cal, {total_carbs:.1f}g carbs, {total_fat:.1f}g fat")
        
        return final_score
    
    def _store_meal_data(self, user_id: str, detected_foods: List[FoodItem], total_protein: float, total_calories: float):
        """Store meal data for user using database"""
        try:
            # Prepare detected foods dictionary
            detected_foods_dict = {food.name: 100 for food in detected_foods}  # Assuming 100g each
            
            # Calculate confidence score
            confidence = sum(food.confidence for food in detected_foods) / len(detected_foods) if detected_foods else 0
            
            # Log meal to database
            nutrition_db.log_meal(
                user_id=int(user_id) if user_id.isdigit() else 123,
                image_path="/temp/meal_image.jpg",  # Placeholder path
                detected_foods=detected_foods_dict,
                total_protein=total_protein,
                total_calories=total_calories,
                confidence=confidence
            )
            
            logger.info(f"Meal data stored for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store meal data: {e}")
            # Fallback to in-memory storage
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