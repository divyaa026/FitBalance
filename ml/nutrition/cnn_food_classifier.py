"""
CNN Food Classifier - EfficientNet-based Food Recognition
Train on Food-101 dataset for food classification and protein content estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class FoodDetection:
    """Represents a detected food item with bounding box"""
    food_class: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    protein_content: float
    calories: float
    portion_grams: float = 100.0

class FoodClassifierCNN(nn.Module):
    """EfficientNet-based CNN for food classification and nutrition estimation"""
    
    def __init__(self, num_food_classes: int = 101, nutrition_outputs: int = 4):
        super(FoodClassifierCNN, self).__init__()
        
        # Use EfficientNet-B3 as backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        
        # Get the number of features from the backbone
        backbone_features = self.backbone._fc.in_features
        
        # Remove the original classifier
        self.backbone._fc = nn.Identity()
        
        # Food classification head
        self.food_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_food_classes)
        )
        
        # Nutrition regression head (protein, carbs, fat, calories per 100g)
        self.nutrition_regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nutrition_outputs)
        )
        
        # Portion estimation head
        self.portion_estimator = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Estimated portion size in grams
        )
    
    def forward(self, x):
        # Extract features using EfficientNet backbone
        features = self.backbone(x)
        
        # Apply different heads
        food_classification = self.food_classifier(features)
        nutrition_values = self.nutrition_regressor(features)
        portion_size = self.portion_estimator(features)
        
        return food_classification, nutrition_values, portion_size

class Food101Dataset:
    """Food-101 dataset loader with protein content mapping"""
    
    def __init__(self, data_dir: str = "data/food-101"):
        self.data_dir = data_dir
        self.food_to_protein = self._load_protein_mapping()
        self.class_names = self._load_class_names()
        self.transform = self._get_transforms()
    
    def _load_protein_mapping(self) -> Dict[str, Dict]:
        """Load protein content mapping for Food-101 classes"""
        # Comprehensive protein mapping for Food-101 dataset
        protein_mapping = {
            "apple_pie": {"protein": 2.4, "carbs": 32.0, "fat": 11.0, "calories": 237},
            "baby_back_ribs": {"protein": 26.0, "carbs": 0.0, "fat": 20.0, "calories": 277},
            "baklava": {"protein": 5.5, "carbs": 29.0, "fat": 18.0, "calories": 307},
            "beef_carpaccio": {"protein": 22.0, "carbs": 1.0, "fat": 8.0, "calories": 158},
            "beef_tartare": {"protein": 20.0, "carbs": 2.0, "fat": 15.0, "calories": 220},
            "beet_salad": {"protein": 3.0, "carbs": 12.0, "fat": 7.0, "calories": 112},
            "beignets": {"protein": 6.0, "carbs": 35.0, "fat": 12.0, "calories": 260},
            "bibimbap": {"protein": 12.0, "carbs": 45.0, "fat": 8.0, "calories": 295},
            "bread_pudding": {"protein": 7.0, "carbs": 40.0, "fat": 8.0, "calories": 250},
            "breakfast_burrito": {"protein": 15.0, "carbs": 30.0, "fat": 12.0, "calories": 280},
            "bruschetta": {"protein": 4.0, "carbs": 20.0, "fat": 6.0, "calories": 142},
            "caesar_salad": {"protein": 8.0, "carbs": 10.0, "fat": 15.0, "calories": 200},
            "cannoli": {"protein": 6.0, "carbs": 25.0, "fat": 16.0, "calories": 260},
            "caprese_salad": {"protein": 11.0, "carbs": 8.0, "fat": 12.0, "calories": 180},
            "carrot_cake": {"protein": 4.0, "carbs": 45.0, "fat": 16.0, "calories": 320},
            "ceviche": {"protein": 20.0, "carbs": 8.0, "fat": 2.0, "calories": 130},
            "cheese_plate": {"protein": 20.0, "carbs": 4.0, "fat": 25.0, "calories": 320},
            "cheesecake": {"protein": 6.0, "carbs": 35.0, "fat": 22.0, "calories": 350},
            "chicken_curry": {"protein": 25.0, "carbs": 12.0, "fat": 15.0, "calories": 280},
            "chicken_quesadilla": {"protein": 22.0, "carbs": 25.0, "fat": 18.0, "calories": 330},
            "chicken_wings": {"protein": 26.0, "carbs": 0.0, "fat": 18.0, "calories": 250},
            "chocolate_cake": {"protein": 5.0, "carbs": 50.0, "fat": 18.0, "calories": 360},
            "chocolate_mousse": {"protein": 4.0, "carbs": 25.0, "fat": 15.0, "calories": 240},
            "churros": {"protein": 4.0, "carbs": 40.0, "fat": 12.0, "calories": 270},
            "clam_chowder": {"protein": 8.0, "carbs": 15.0, "fat": 10.0, "calories": 180},
            "club_sandwich": {"protein": 25.0, "carbs": 30.0, "fat": 15.0, "calories": 340},
            "crab_cakes": {"protein": 20.0, "carbs": 8.0, "fat": 12.0, "calories": 210},
            "creme_brulee": {"protein": 4.0, "carbs": 22.0, "fat": 20.0, "calories": 280},
            "croque_madame": {"protein": 18.0, "carbs": 25.0, "fat": 20.0, "calories": 340},
            "cup_cakes": {"protein": 3.0, "carbs": 45.0, "fat": 12.0, "calories": 280},
            "deviled_eggs": {"protein": 12.0, "carbs": 1.0, "fat": 15.0, "calories": 180},
            "donuts": {"protein": 4.0, "carbs": 35.0, "fat": 18.0, "calories": 300},
            "dumplings": {"protein": 8.0, "carbs": 25.0, "fat": 6.0, "calories": 180},
            "edamame": {"protein": 11.0, "carbs": 8.0, "fat": 5.0, "calories": 120},
            "eggs_benedict": {"protein": 20.0, "carbs": 15.0, "fat": 25.0, "calories": 350},
            "escargots": {"protein": 16.0, "carbs": 2.0, "fat": 8.0, "calories": 140},
            "falafel": {"protein": 13.0, "carbs": 32.0, "fat": 18.0, "calories": 333},
            "filet_mignon": {"protein": 30.0, "carbs": 0.0, "fat": 15.0, "calories": 250},
            "fish_and_chips": {"protein": 20.0, "carbs": 40.0, "fat": 22.0, "calories": 420},
            "foie_gras": {"protein": 11.0, "carbs": 4.0, "fat": 44.0, "calories": 462},
            "french_fries": {"protein": 4.0, "carbs": 43.0, "fat": 17.0, "calories": 330},
            "french_onion_soup": {"protein": 8.0, "carbs": 12.0, "fat": 15.0, "calories": 200},
            "french_toast": {"protein": 10.0, "carbs": 35.0, "fat": 12.0, "calories": 280},
            "fried_calamari": {"protein": 15.0, "carbs": 20.0, "fat": 15.0, "calories": 260},
            "fried_rice": {"protein": 8.0, "carbs": 35.0, "fat": 8.0, "calories": 240},
            "frozen_yogurt": {"protein": 4.0, "carbs": 20.0, "fat": 2.0, "calories": 110},
            "garlic_bread": {"protein": 8.0, "carbs": 30.0, "fat": 12.0, "calories": 250},
            "gnocchi": {"protein": 6.0, "carbs": 35.0, "fat": 2.0, "calories": 180},
            "greek_salad": {"protein": 5.0, "carbs": 10.0, "fat": 15.0, "calories": 190},
            "grilled_cheese_sandwich": {"protein": 12.0, "carbs": 25.0, "fat": 18.0, "calories": 290},
            "grilled_salmon": {"protein": 25.0, "carbs": 0.0, "fat": 12.0, "calories": 206},
            "guacamole": {"protein": 2.0, "carbs": 8.0, "fat": 15.0, "calories": 160},
            "gyoza": {"protein": 8.0, "carbs": 20.0, "fat": 6.0, "calories": 160},
            "hamburger": {"protein": 25.0, "carbs": 30.0, "fat": 20.0, "calories": 380},
            "hot_and_sour_soup": {"protein": 6.0, "carbs": 8.0, "fat": 4.0, "calories": 85},
            "hot_dog": {"protein": 12.0, "carbs": 20.0, "fat": 15.0, "calories": 250},
            "huevos_rancheros": {"protein": 15.0, "carbs": 25.0, "fat": 12.0, "calories": 260},
            "hummus": {"protein": 8.0, "carbs": 14.0, "fat": 10.0, "calories": 166},
            "ice_cream": {"protein": 4.0, "carbs": 22.0, "fat": 11.0, "calories": 200},
            "lasagna": {"protein": 18.0, "carbs": 25.0, "fat": 15.0, "calories": 300},
            "lobster_bisque": {"protein": 10.0, "carbs": 8.0, "fat": 15.0, "calories": 200},
            "lobster_roll_sandwich": {"protein": 25.0, "carbs": 20.0, "fat": 8.0, "calories": 240},
            "macaroni_and_cheese": {"protein": 12.0, "carbs": 35.0, "fat": 18.0, "calories": 320},
            "macarons": {"protein": 6.0, "carbs": 30.0, "fat": 8.0, "calories": 200},
            "miso_soup": {"protein": 3.0, "carbs": 4.0, "fat": 1.0, "calories": 35},
            "mussels": {"protein": 20.0, "carbs": 4.0, "fat": 2.0, "calories": 120},
            "nachos": {"protein": 12.0, "carbs": 30.0, "fat": 20.0, "calories": 340},
            "omelette": {"protein": 20.0, "carbs": 2.0, "fat": 18.0, "calories": 250},
            "onion_rings": {"protein": 4.0, "carbs": 35.0, "fat": 15.0, "calories": 280},
            "oysters": {"protein": 9.0, "carbs": 5.0, "fat": 2.0, "calories": 68},
            "pad_thai": {"protein": 15.0, "carbs": 40.0, "fat": 12.0, "calories": 310},
            "paella": {"protein": 20.0, "carbs": 35.0, "fat": 10.0, "calories": 300},
            "pancakes": {"protein": 8.0, "carbs": 35.0, "fat": 8.0, "calories": 230},
            "panna_cotta": {"protein": 4.0, "carbs": 20.0, "fat": 15.0, "calories": 220},
            "peking_duck": {"protein": 25.0, "carbs": 5.0, "fat": 20.0, "calories": 290},
            "pho": {"protein": 15.0, "carbs": 30.0, "fat": 5.0, "calories": 220},
            "pizza": {"protein": 12.0, "carbs": 35.0, "fat": 12.0, "calories": 290},
            "pork_chop": {"protein": 25.0, "carbs": 0.0, "fat": 12.0, "calories": 210},
            "poutine": {"protein": 8.0, "carbs": 40.0, "fat": 20.0, "calories": 350},
            "prime_rib": {"protein": 28.0, "carbs": 0.0, "fat": 18.0, "calories": 280},
            "pulled_pork_sandwich": {"protein": 22.0, "carbs": 30.0, "fat": 12.0, "calories": 300},
            "ramen": {"protein": 12.0, "carbs": 45.0, "fat": 15.0, "calories": 340},
            "ravioli": {"protein": 14.0, "carbs": 30.0, "fat": 8.0, "calories": 240},
            "red_velvet_cake": {"protein": 4.0, "carbs": 45.0, "fat": 16.0, "calories": 320},
            "risotto": {"protein": 8.0, "carbs": 35.0, "fat": 10.0, "calories": 250},
            "samosa": {"protein": 6.0, "carbs": 25.0, "fat": 12.0, "calories": 220},
            "sashimi": {"protein": 25.0, "carbs": 0.0, "fat": 5.0, "calories": 140},
            "scallops": {"protein": 20.0, "carbs": 2.0, "fat": 1.0, "calories": 95},
            "seaweed_salad": {"protein": 3.0, "carbs": 8.0, "fat": 2.0, "calories": 60},
            "shrimp_and_grits": {"protein": 18.0, "carbs": 25.0, "fat": 12.0, "calories": 270},
            "spaghetti_bolognese": {"protein": 18.0, "carbs": 35.0, "fat": 12.0, "calories": 310},
            "spaghetti_carbonara": {"protein": 20.0, "carbs": 35.0, "fat": 18.0, "calories": 360},
            "spring_rolls": {"protein": 5.0, "carbs": 20.0, "fat": 8.0, "calories": 160},
            "steak": {"protein": 30.0, "carbs": 0.0, "fat": 15.0, "calories": 250},
            "strawberry_shortcake": {"protein": 4.0, "carbs": 35.0, "fat": 12.0, "calories": 250},
            "sushi": {"protein": 15.0, "carbs": 20.0, "fat": 3.0, "calories": 160},
            "tacos": {"protein": 15.0, "carbs": 20.0, "fat": 10.0, "calories": 220},
            "takoyaki": {"protein": 8.0, "carbs": 15.0, "fat": 8.0, "calories": 160},
            "tiramisu": {"protein": 6.0, "carbs": 30.0, "fat": 18.0, "calories": 290},
            "tuna_tartare": {"protein": 25.0, "carbs": 2.0, "fat": 8.0, "calories": 170},
            "waffles": {"protein": 8.0, "carbs": 35.0, "fat": 10.0, "calories": 250}
        }
        return protein_mapping
    
    def _load_class_names(self) -> List[str]:
        """Load Food-101 class names"""
        return list(self.food_to_protein.keys())
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

class FoodClassifierTrainer:
    """Trainer for CNN Food Classifier"""
    
    def __init__(self, model: FoodClassifierCNN, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.dataset = Food101Dataset()
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.nutrition_criterion = nn.MSELoss()
        self.portion_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels, nutrition_targets, portion_targets) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            nutrition_targets = nutrition_targets.to(self.device)
            portion_targets = portion_targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            food_logits, nutrition_pred, portion_pred = self.model(images)
            
            # Calculate losses
            classification_loss = self.classification_criterion(food_logits, labels)
            nutrition_loss = self.nutrition_criterion(nutrition_pred, nutrition_targets)
            portion_loss = self.portion_criterion(portion_pred.squeeze(), portion_targets)
            
            # Combined loss
            total_batch_loss = classification_loss + 0.5 * nutrition_loss + 0.3 * portion_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, nutrition_targets, portion_targets in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                nutrition_targets = nutrition_targets.to(self.device)
                portion_targets = portion_targets.to(self.device)
                
                # Forward pass
                food_logits, nutrition_pred, portion_pred = self.model(images)
                
                # Calculate losses
                classification_loss = self.classification_criterion(food_logits, labels)
                nutrition_loss = self.nutrition_criterion(nutrition_pred, nutrition_targets)
                portion_loss = self.portion_criterion(portion_pred.squeeze(), portion_targets)
                
                total_batch_loss = classification_loss + 0.5 * nutrition_loss + 0.3 * portion_loss
                total_loss += total_batch_loss.item()
                
                # Accuracy
                _, predicted = torch.max(food_logits.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_names': self.dataset.class_names,
            'protein_mapping': self.dataset.food_to_protein
        }, filepath)
        logger.info(f"Model saved to {filepath}")

class FoodClassifierInference:
    """Inference engine for food classification"""
    
    def __init__(self, model_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.dataset = Food101Dataset()
        self.model = FoodClassifierCNN(num_food_classes=len(self.dataset.class_names))
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Using randomly initialized weights.")
        
        self.model.to(device)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
    
    def classify_image(self, image: Image.Image, top_k: int = 3) -> List[FoodDetection]:
        """Classify food in image and estimate nutrition"""
        # Preprocess image
        image_tensor = self.dataset.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Model prediction
            food_logits, nutrition_pred, portion_pred = self.model(image_tensor)
            
            # Get top-k predictions
            food_probs = F.softmax(food_logits, dim=1)
            top_probs, top_indices = torch.topk(food_probs, top_k, dim=1)
            
            detections = []
            
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                food_class = self.dataset.class_names[class_idx]
                
                # Get nutrition from prediction or database
                if food_class in self.dataset.food_to_protein:
                    nutrition_info = self.dataset.food_to_protein[food_class]
                    protein_content = nutrition_info['protein']
                    calories = nutrition_info['calories']
                else:
                    # Use model prediction
                    protein_content = max(0, nutrition_pred[0][0].item())
                    calories = max(0, nutrition_pred[0][3].item())
                
                # Estimate portion size
                portion_grams = max(50, min(500, portion_pred[0].item()))
                
                detection = FoodDetection(
                    food_class=food_class,
                    confidence=confidence,
                    bounding_box=(0, 0, image.width, image.height),  # Full image for now
                    protein_content=protein_content * (portion_grams / 100),
                    calories=calories * (portion_grams / 100),
                    portion_grams=portion_grams
                )
                
                detections.append(detection)
            
            return detections
    
    def analyze_meal_image(self, image_path: str) -> Dict:
        """Complete meal analysis from image path"""
        try:
            image = Image.open(image_path).convert('RGB')
            detections = self.classify_image(image)
            
            # Calculate total nutrition
            total_protein = sum(det.protein_content for det in detections)
            total_calories = sum(det.calories for det in detections)
            
            return {
                'detections': detections,
                'total_protein': total_protein,
                'total_calories': total_calories,
                'confidence': detections[0].confidence if detections else 0.0
            }
        
        except Exception as e:
            logger.error(f"Meal analysis failed: {e}")
            return {
                'detections': [],
                'total_protein': 0.0,
                'total_calories': 0.0,
                'confidence': 0.0
            }

# Global inference engine
food_classifier = FoodClassifierInference()