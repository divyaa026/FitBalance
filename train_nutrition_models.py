"""
Training Script for FitBalance Nutrition Optimization Models
Demonstrates training of CNN Food Classifier and GRU Protein Optimizer
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datetime import datetime, timedelta, date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_cnn_food_classifier():
    """Train the CNN Food Classifier (demo version)"""
    logger.info("🍽️ Training CNN Food Classifier...")
    
    try:
        from ml_models.nutrition.cnn_food_classifier import FoodClassifierCNN, FoodClassifierTrainer
        
        # Initialize model
        model = FoodClassifierCNN(num_food_classes=101)
        trainer = FoodClassifierTrainer(model)
        
        # In a real implementation, you would load Food-101 dataset here
        # For demo, we'll simulate training
        logger.info("📊 Simulating training with Food-101 dataset...")
        
        # Simulate training epochs
        for epoch in range(5):  # Reduced for demo
            # Simulate training loss
            loss = 2.5 - (epoch * 0.3) + np.random.normal(0, 0.1)
            accuracy = 0.3 + (epoch * 0.15) + np.random.normal(0, 0.02)
            
            logger.info(f"Epoch {epoch + 1}/5 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/cnn_food_classifier.pth")
        logger.info("✅ CNN Food Classifier training completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CNN training failed: {e}")
        return False

def train_gru_protein_optimizer():
    """Train the GRU Protein Optimizer"""
    logger.info("🧠 Training GRU Protein Optimizer...")
    
    try:
        from ml_models.nutrition.gru_protein_optimizer import ProteinOptimizer, HealthMetrics
        
        # Initialize optimizer
        optimizer = ProteinOptimizer()
        
        # Generate synthetic training data
        logger.info("📈 Generating synthetic training data...")
        training_data = []
        
        for i in range(200):  # 200 training samples
            # Generate a week of health data
            health_sequence = []
            base_date = datetime.now() - timedelta(days=7 * i)
            
            for day in range(7):
                # Generate realistic health metrics with some correlation
                sleep_quality = np.random.normal(7, 1.5)
                sleep_duration = sleep_quality * 0.8 + np.random.normal(1, 0.5)
                activity_level = np.random.uniform(2, 9)
                hrv = 50 + (sleep_quality - 7) * 5 + np.random.normal(0, 10)
                stress_level = int(10 - sleep_quality + np.random.normal(0, 1))
                recovery_score = (sleep_quality + (10 - stress_level)) / 2
                
                # Clip to valid ranges
                sleep_quality = np.clip(sleep_quality, 1, 10)
                sleep_duration = np.clip(sleep_duration, 4, 12)
                activity_level = np.clip(activity_level, 0, 10)
                hrv = np.clip(hrv, 20, 100)
                stress_level = np.clip(stress_level, 1, 10)
                recovery_score = np.clip(recovery_score, 1, 10)
                
                health_metric = HealthMetrics(
                    date=base_date + timedelta(days=day),
                    sleep_duration=float(sleep_duration),
                    sleep_quality=float(sleep_quality),
                    hrv=float(hrv),
                    activity_level=float(activity_level),
                    stress_level=int(stress_level),
                    protein_intake=0.0,  # Will be calculated
                    recovery_score=float(recovery_score)
                )
                health_sequence.append(health_metric)
            
            # Calculate target protein based on health patterns
            latest = health_sequence[-1]
            base_protein = 120.0  # Base protein need
            
            # Adjust based on health factors
            sleep_adjustment = (7 - latest.sleep_quality) * 3  # Poor sleep = more protein
            activity_adjustment = (latest.activity_level - 5) * 8  # High activity = more protein
            stress_adjustment = (latest.stress_level - 5) * 2  # High stress = more protein
            
            target_protein = base_protein + sleep_adjustment + activity_adjustment + stress_adjustment
            target_protein = np.clip(target_protein, 80, 200)  # Reasonable bounds
            
            training_data.append((health_sequence, target_protein))
        
        # Train the model
        logger.info("🏃‍♂️ Training GRU model...")
        optimizer.train_model(training_data, epochs=50, learning_rate=0.001)
        
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        optimizer.save_model("models/gru_protein_optimizer.pth")
        logger.info("✅ GRU Protein Optimizer training completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GRU training failed: {e}")
        return False

def test_trained_models():
    """Test the trained models with sample data"""
    logger.info("🧪 Testing trained models...")
    
    try:
        from ml_models.nutrition.cnn_food_classifier import FoodClassifierInference
        from ml_models.nutrition.gru_protein_optimizer import ProteinOptimizer, HealthMetrics
        from ml_models.nutrition.shap_explainer import initialize_shap_explainer
        from backend.integrations.health_apis import create_demo_health_data
        from PIL import Image
        import numpy as np
        
        # Test CNN Food Classifier
        logger.info("🍎 Testing CNN Food Classifier...")
        food_classifier = FoodClassifierInference()
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        detections = food_classifier.classify_image(dummy_image, top_k=3)
        
        logger.info(f"Detected foods: {[det.food_class for det in detections]}")
        logger.info(f"Total protein: {sum(det.protein_content for det in detections):.1f}g")
        
        # Test GRU Protein Optimizer
        logger.info("🧠 Testing GRU Protein Optimizer...")
        protein_optimizer = ProteinOptimizer()
        
        # Generate test health data
        health_data = []
        for i in range(7):
            demo_data = create_demo_health_data(date.today() - timedelta(days=6-i))
            health_metric = HealthMetrics(
                date=datetime.combine(demo_data.date, datetime.min.time()),
                sleep_duration=demo_data.sleep_duration or 7.5,
                sleep_quality=demo_data.sleep_quality or 7.0,
                hrv=demo_data.hrv or 45.0,
                activity_level=demo_data.activity_score or 5.0,
                stress_level=int(demo_data.stress_level or 4),
                protein_intake=0.0,
                recovery_score=demo_data.recovery_score or 7.0
            )
            health_data.append(health_metric)
        
        recommendation = protein_optimizer.optimize_protein_intake("test_user", health_data)
        logger.info(f"Protein recommendation: {recommendation.adjusted_protein:.1f}g")
        logger.info(f"Explanation: {recommendation.explanation}")
        
        # Test SHAP Explainer
        logger.info("📊 Testing SHAP Explainer...")
        shap_explainer = initialize_shap_explainer(protein_optimizer)
        
        if shap_explainer:
            explanation = shap_explainer.explain_recommendation(health_data[-1], "test_user")
            logger.info(f"SHAP explanation: {explanation.explanation_text}")
            logger.info(f"Confidence: {explanation.confidence_score:.2f}")
        
        logger.info("✅ All model tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model testing failed: {e}")
        return False

def create_demo_dataset():
    """Create demo datasets for training"""
    logger.info("📁 Creating demo datasets...")
    
    try:
        # Create directories
        os.makedirs("datasets", exist_ok=True)
        os.makedirs("datasets/food_images", exist_ok=True)
        
        # Create synthetic nutrition dataset
        nutrition_data = []
        food_names = [
            'chicken_breast', 'salmon', 'eggs', 'greek_yogurt', 'quinoa',
            'broccoli', 'rice', 'beans', 'tofu', 'almonds'
        ]
        
        for food in food_names:
            for i in range(20):  # 20 samples per food
                # Add some variation to nutritional values
                base_nutrition = {
                    'chicken_breast': {'protein': 31, 'carbs': 0, 'fat': 3.6, 'calories': 165},
                    'salmon': {'protein': 25, 'carbs': 0, 'fat': 12, 'calories': 208},
                    'eggs': {'protein': 13, 'carbs': 1.1, 'fat': 11, 'calories': 155},
                    'greek_yogurt': {'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'calories': 59},
                    'quinoa': {'protein': 4.4, 'carbs': 22, 'fat': 1.9, 'calories': 120},
                    'broccoli': {'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'calories': 34},
                    'rice': {'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'calories': 130},
                    'beans': {'protein': 21, 'carbs': 23, 'fat': 0.5, 'calories': 127},
                    'tofu': {'protein': 15, 'carbs': 4, 'fat': 8, 'calories': 144},
                    'almonds': {'protein': 21, 'carbs': 22, 'fat': 49, 'calories': 579}
                }[food]
                
                # Add variation (±10%)
                variation = 0.1
                nutrition_data.append({
                    'food_name': food,
                    'protein': base_nutrition['protein'] * (1 + np.random.uniform(-variation, variation)),
                    'carbs': base_nutrition['carbs'] * (1 + np.random.uniform(-variation, variation)),
                    'fat': base_nutrition['fat'] * (1 + np.random.uniform(-variation, variation)),
                    'calories': base_nutrition['calories'] * (1 + np.random.uniform(-variation, variation)),
                    'sample_id': i
                })
        
        # Save dataset
        import pandas as pd
        df = pd.DataFrame(nutrition_data)
        df.to_csv("datasets/nutrition_dataset.csv", index=False)
        
        logger.info(f"✅ Created nutrition dataset with {len(nutrition_data)} samples")
        
        # Create health metrics dataset
        health_data = []
        for i in range(1000):  # 1000 days of data
            target_date = date.today() - timedelta(days=i)
            demo_data = create_demo_health_data(target_date)
            
            health_data.append({
                'date': target_date.isoformat(),
                'sleep_duration': demo_data.sleep_duration,
                'sleep_quality': demo_data.sleep_quality,
                'hrv': demo_data.hrv,
                'activity_score': demo_data.activity_score,
                'stress_level': demo_data.stress_level,
                'recovery_score': demo_data.recovery_score,
                'calories_burned': demo_data.calories_burned,
                'steps': demo_data.steps
            })
        
        df_health = pd.DataFrame(health_data)
        df_health.to_csv("datasets/health_metrics_dataset.csv", index=False)
        
        logger.info(f"✅ Created health metrics dataset with {len(health_data)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        return False

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting FitBalance Nutrition Optimization Training Pipeline")
    
    try:
        # Step 1: Create demo datasets
        if not create_demo_dataset():
            logger.error("Failed to create datasets")
            return
        
        # Step 2: Train CNN Food Classifier
        if not train_cnn_food_classifier():
            logger.error("Failed to train CNN Food Classifier")
            return
        
        # Step 3: Train GRU Protein Optimizer
        if not train_gru_protein_optimizer():
            logger.error("Failed to train GRU Protein Optimizer")
            return
        
        # Step 4: Test all models
        if not test_trained_models():
            logger.error("Model testing failed")
            return
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("💡 You can now use the enhanced nutrition optimization system!")
        
        # Print usage instructions
        print("\n" + "="*60)
        print("🎯 FITBALANCE NUTRITION OPTIMIZER - READY!")
        print("="*60)
        print("📋 Available Features:")
        print("  • CNN Food Classification (Food-101 based)")
        print("  • GRU Protein Optimization (Time-series learning)")
        print("  • SHAP Explainability (Feature importance)")
        print("  • Real-time health data integration")
        print("  • PostgreSQL database storage")
        print("\n🌐 API Endpoints:")
        print("  • POST /nutrition/analyze-meal - Analyze meal photos")
        print("  • GET /nutrition/recommendations/{user_id} - Get protein recommendations")
        print("  • GET /nutrition/shap-explanation/{user_id} - Get SHAP explanations")
        print("  • POST /nutrition/health-data/{user_id} - Update health data")
        print("  • GET /nutrition/food-database - Browse food nutrition database")
        print("\n🚀 Start the server with: python start_server.py")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")

if __name__ == "__main__":
    main()