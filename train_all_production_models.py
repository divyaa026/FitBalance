"""
Master Training Script for All Production ML Models
Generates datasets, trains models, evaluates performance
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd: list, description: str):
    """Run command and handle errors"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        print(f"✅ Completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s")
        print("Error:", e.stderr)
        return False

def main():
    """Train all production ML models"""
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        🏋️ FitBalance Production ML Training Pipeline 🏋️           ║
║                                                                    ║
║  This script will:                                                 ║
║  1. Generate comprehensive synthetic datasets                      ║
║  2. Train biomechanics GNN-LSTM models                            ║
║  3. Train burnout Cox PH and ML models                            ║
║  4. Evaluate all models                                           ║
║                                                                    ║
║  Estimated time: 30-60 minutes (depending on hardware)            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to start training...")
    
    results = {}
    
    # Step 1: Generate Biomechanics Dataset
    results['biomechanics_data'] = run_command(
        [sys.executable, "datasets/generate_biomechanics_dataset.py"],
        "Step 1/4: Generating Biomechanics Dataset (1,000 sequences)"
    )
    
    # Step 2: Generate Burnout Dataset
    results['burnout_data'] = run_command(
        [sys.executable, "datasets/generate_burnout_dataset.py"],
        "Step 2/4: Generating Burnout Dataset (1,000 users × 365 days)"
    )
    
    # Step 3: Train Biomechanics Models
    if results['biomechanics_data']:
        results['biomechanics_model'] = run_command(
            [sys.executable, "ml_models/biomechanics/train_production_model.py"],
            "Step 3/4: Training Biomechanics GNN-LSTM Model"
        )
    else:
        print("⚠️  Skipping biomechanics training due to data generation failure")
        results['biomechanics_model'] = False
    
    # Step 4: Train Burnout Models
    if results['burnout_data']:
        results['burnout_model'] = run_command(
            [sys.executable, "ml_models/burnout/train_production_model.py"],
            "Step 4/4: Training Burnout Cox PH and ML Models"
        )
    else:
        print("⚠️  Skipping burnout training due to data generation failure")
        results['burnout_model'] = False
    
    # Final Summary
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    
    total = len(results)
    successful = sum(1 for r in results.values() if r)
    
    print(f"\nCompleted: {successful}/{total} steps")
    print("\nDetailed Results:")
    
    status_emoji = {True: "✅", False: "❌"}
    for step, success in results.items():
        print(f"  {status_emoji[success]} {step.replace('_', ' ').title()}")
    
    if successful == total:
        print("\n🎉 All models trained successfully!")
        print("\n📁 Model Locations:")
        print("  - Biomechanics: ml_models/biomechanics/gnn_lstm_best.pth")
        print("  - Burnout Cox: ml_models/burnout/cox_ph_model.pkl")
        print("  - Burnout RF: ml_models/burnout/random_forest_model.pkl")
        print("  - Burnout GB: ml_models/burnout/gradient_boosting_model.pkl")
        print("\n📊 Dataset Locations:")
        print("  - Biomechanics: datasets/biomechanics/")
        print("  - Burnout: datasets/burnout/")
        print("\n✅ Ready for production deployment!")
    else:
        print(f"\n⚠️  {total - successful} step(s) failed. Check logs above.")
        print("   Fix errors and re-run this script.")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
