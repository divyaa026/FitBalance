"""
Step 3: Model Evaluation Script
Comprehensive evaluation with confusion matrix, per-class accuracy, and visualizations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_dir, dataset_dir):
        """
        Args:
            model_dir: Directory containing trained model
            dataset_dir: Directory containing dataset
        """
        self.model_dir = Path(model_dir)
        self.dataset_dir = Path(dataset_dir)
        self.model = None
        self.class_names = None
        
    def load_model(self):
        """Load trained model and class names"""
        logger.info("Loading model...")
        
        # Load model
        model_path = self.model_dir / 'best_model.keras'
        if not model_path.exists():
            model_path = self.model_dir / 'final_model.keras'
        
        self.model = keras.models.load_model(str(model_path))
        logger.info(f"✓ Loaded model from {model_path}")
        
        # Load class names
        class_names_path = self.model_dir / 'class_names.json'
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        logger.info(f"✓ Loaded {len(self.class_names)} class names")
        
        return self.model
    
    def create_test_generator(self, batch_size=32):
        """Create test data generator"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            str(self.dataset_dir),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return test_generator
    
    def evaluate_comprehensive(self):
        """Comprehensive model evaluation"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*60)
        
        # Create test generator
        test_gen = self.create_test_generator()
        
        # Get predictions
        logger.info("Generating predictions...")
        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes
        
        # Calculate metrics
        logger.info("\n📊 Overall Metrics:")
        accuracy = np.mean(y_pred == y_true)
        logger.info(f"  Accuracy: {accuracy*100:.2f}%")
        
        # Top-5 accuracy
        top5_pred = np.argsort(y_pred_probs, axis=1)[:, -5:]
        top5_accuracy = np.mean([y_true[i] in top5_pred[i] for i in range(len(y_true))])
        logger.info(f"  Top-5 Accuracy: {top5_accuracy*100:.2f}%")
        
        # Per-class report
        logger.info("\n📋 Per-Class Performance:")
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better display
        df_report = pd.DataFrame(report).transpose()
        
        # Save detailed report
        report_path = self.model_dir / 'evaluation_report.csv'
        df_report.to_csv(report_path)
        logger.info(f"✓ Saved detailed report to {report_path}")
        
        # Show top 10 best and worst performing classes
        df_classes = df_report[:-3]  # Exclude avg rows
        df_classes = df_classes.sort_values('f1-score', ascending=False)
        
        logger.info("\n🏆 Top 10 Best Performing Classes:")
        for idx, (class_name, row) in enumerate(df_classes.head(10).iterrows(), 1):
            logger.info(f"  {idx}. {class_name}: {row['f1-score']*100:.1f}% F1-score")
        
        logger.info("\n⚠️  Bottom 10 Classes (Need Improvement):")
        for idx, (class_name, row) in enumerate(df_classes.tail(10).iterrows(), 1):
            logger.info(f"  {idx}. {class_name}: {row['f1-score']*100:.1f}% F1-score")
        
        # Generate confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Save results summary
        results = {
            'accuracy': float(accuracy),
            'top5_accuracy': float(top5_accuracy),
            'num_samples': len(y_true),
            'num_classes': len(self.class_names),
            'best_classes': df_classes.head(10).to_dict('index'),
            'worst_classes': df_classes.tail(10).to_dict('index')
        }
        
        results_path = self.model_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, top_n=20):
        """Plot confusion matrix for top N classes"""
        logger.info(f"\n📊 Generating confusion matrix (top {top_n} classes)...")
        
        # Get most common classes
        unique, counts = np.unique(y_true, return_counts=True)
        top_classes_idx = unique[np.argsort(counts)[-top_n:]]
        
        # Filter predictions for top classes
        mask = np.isin(y_true, top_classes_idx)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(15, 12))
        class_labels = [self.class_names[i] for i in top_classes_idx]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Accuracy'}
        )
        
        plt.title(f'Confusion Matrix - Top {top_n} Classes', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved confusion matrix to {plot_path}")
        plt.close()
    
    def test_random_images(self, num_images=10):
        """Test model on random images and show predictions"""
        logger.info(f"\n🎲 Testing on {num_images} random images...")
        
        test_gen = self.create_test_generator(batch_size=1)
        
        # Get random indices
        np.random.seed(42)
        indices = np.random.choice(len(test_gen.filenames), num_images, replace=False)
        
        correct = 0
        for idx in indices:
            # Get image and true label
            img_path = Path(self.dataset_dir) / test_gen.filenames[idx]
            true_label_idx = test_gen.classes[idx]
            true_label = self.class_names[true_label_idx]
            
            # Predict
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            predictions = self.model.predict(img_array, verbose=0)[0]
            top5_idx = np.argsort(predictions)[-5:][::-1]
            
            pred_label = self.class_names[top5_idx[0]]
            confidence = predictions[top5_idx[0]]
            
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            logger.info(f"\n{status} True: {true_label}, Pred: {pred_label} ({confidence*100:.1f}%)")
            logger.info(f"   Top 5: {', '.join([self.class_names[i] for i in top5_idx])}")
        
        logger.info(f"\n✓ Random test accuracy: {correct}/{num_images} = {correct/num_images*100:.1f}%")


def main():
    """Main evaluation pipeline"""
    import sys
    
    # Get model directory (either from command line or find latest)
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        # Find latest model
        models_dir = Path("ml_models/nutrition/models")
        model_dirs = sorted(models_dir.glob("indian_food_*"))
        if not model_dirs:
            logger.error("No trained models found!")
            logger.error("Please train a model first using step2_train_model.py")
            return
        model_dir = model_dirs[-1]
    
    dataset_dir = "nutrition/augmented_indian_food_dataset"
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Model: {model_dir}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info("="*60 + "\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(model_dir, dataset_dir)
    
    # Load model
    evaluator.load_model()
    
    # Comprehensive evaluation
    results = evaluator.evaluate_comprehensive()
    
    # Test on random images
    evaluator.test_random_images(num_images=20)
    
    logger.info("\n" + "="*60)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"📊 Accuracy: {results['accuracy']*100:.2f}%")
    logger.info(f"🎯 Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
    logger.info(f"📁 Results saved to: {model_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
