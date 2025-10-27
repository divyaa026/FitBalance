"""
Master Script: Complete Training Pipeline
Runs all steps: augmentation → training → evaluation
"""

import subprocess
import sys
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_header(text):
    """Print formatted header"""
    logger.info("\n" + "="*70)
    logger.info(f"  {text}")
    logger.info("="*70 + "\n")

def run_step(script_name, description):
    """Run a training step"""
    print_header(description)
    
    script_path = Path("ml_models/nutrition") / script_name
    
    logger.info(f"▶ Running: {script_path}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ {description} complete! (took {elapsed/60:.1f} minutes)")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ {description} failed!")
        logger.error(f"Error: {e}")
        return False

def main():
    """Run complete training pipeline"""
    
    print_header("🚀 INDIAN FOOD CLASSIFIER - COMPLETE TRAINING PIPELINE")
    
    logger.info("This script will:")
    logger.info("  1. ✨ Augment your dataset (4,000 → 24,000 images)")
    logger.info("  2. 🏋️  Train EfficientNetB3 model (~1-2 hours)")
    logger.info("  3. 📊 Evaluate model performance")
    logger.info("")
    
    # Check if dataset exists
    dataset_dir = Path("nutrition/Indian Food Images/Indian Food Images")
    if not dataset_dir.exists():
        logger.error(f"❌ Dataset not found at: {dataset_dir}")
        logger.error("Please ensure your Indian Food Images dataset is in the correct location")
        return
    
    # Count classes
    classes = [d for d in dataset_dir.iterdir() if d.is_dir()]
    logger.info(f"✓ Found {len(classes)} food classes")
    logger.info("")
    
    response = input("Continue with training? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Training cancelled.")
        return
    
    total_start = time.time()
    
    # Step 1: Data Augmentation
    if not run_step("step1_augment_data.py", "STEP 1: DATA AUGMENTATION"):
        return
    
    # Step 2: Model Training
    if not run_step("step2_train_model_simple.py", "STEP 2: MODEL TRAINING"):
        return
    
    # Step 3: Model Evaluation
    if not run_step("step3_evaluate_model.py", "STEP 3: MODEL EVALUATION"):
        return
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print_header("🎉 TRAINING PIPELINE COMPLETE!")
    
    logger.info(f"⏱️  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    logger.info("")
    logger.info("📁 Your trained model is ready!")
    logger.info("📊 Check the evaluation report for detailed metrics")
    logger.info("")
    logger.info("🚀 Next steps:")
    logger.info("  1. Review evaluation results in ml_models/nutrition/models/")
    logger.info("  2. Run step4_integrate_backend.py to integrate with your app")
    logger.info("  3. Test with real food images!")
    logger.info("")

if __name__ == "__main__":
    main()
