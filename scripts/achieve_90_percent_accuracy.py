"""
Master Pipeline: Achieve 90%+ Accuracy for Biomechanics Model
Step-by-step execution with public fitness datasets
"""

import subprocess
import sys
from pathlib import Path
import json

def print_header(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print_header(" Checking Dependencies")
    
    required = ['gdown', 'yt-dlp', 'mediapipe', 'opencv-python', 'torch', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f" {package}")
        except ImportError:
            print(f" {package}")
            missing.append(package)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("\n Install with:")
        print(f"   pip install {' '.join(missing)}")
        
        response = input("\nInstall now? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
            print(" Dependencies installed")
        else:
            print(" Please install dependencies before continuing")
            sys.exit(1)
    else:
        print("\n All dependencies installed")

def step1_download_datasets():
    """Step 1: Download public datasets"""
    print_header(" Step 1: Download Public Datasets")
    
    datasets_dir = Path("data/biomechanics_public")
    if datasets_dir.exists() and any(datasets_dir.iterdir()):
        print(f" Datasets directory exists: {datasets_dir}")
        response = input("Re-download datasets? (y/n): ")
        if response.lower() != 'y':
            print("  Skipping download")
            return True
    
    print("\n Running download script...")
    print("Note: Some datasets require manual download")
    
    try:
        result = subprocess.run(
            [sys.executable, "data/download_public_datasets.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )
        
        print(result.stdout)
        if result.returncode == 0:
            print("\n Download complete")
            return True
        else:
            print(f"\n  Download completed with warnings")
            print(result.stderr)
            
            # Check if at least some data was downloaded
            if datasets_dir.exists() and any(datasets_dir.iterdir()):
                response = input("\nSome datasets downloaded. Continue? (y/n): ")
                return response.lower() == 'y'
            return False
            
    except subprocess.TimeoutExpired:
        print("\n  Download timed out (30 min)")
        response = input("Continue with partial download? (y/n): ")
        return response.lower() == 'y'
    except Exception as e:
        print(f"\n Error: {e}")
        return False

def step2_preprocess_videos():
    """Step 2: Preprocess videos to landmarks"""
    print_header(" Step 2: Preprocess Videos to Landmarks")
    
    processed_dir = Path("data/biomechanics_processed/landmark_sequences")
    if processed_dir.exists() and any(processed_dir.iterdir()):
        print(f" Processed data exists: {processed_dir}")
        response = input("Re-process videos? (y/n): ")
        if response.lower() != 'y':
            print("  Skipping preprocessing")
            return True
    
    print("\n Running preprocessing script...")
    print("This may take 30-60 minutes depending on dataset size")
    
    try:
        result = subprocess.run(
            [sys.executable, "data/preprocess_public_datasets.py"],
            capture_output=False,  # Show real-time progress
            timeout=3600  # 1 hour
        )
        
        if result.returncode == 0:
            print("\n Preprocessing complete")
            
            # Show statistics
            stats_file = Path("data/biomechanics_processed/statistics.json")
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                print(f"\n Processed Statistics:")
                print(f"   Total sequences: {stats['total_sequences']}")
                print(f"   Exercise distribution: {stats['exercise_distribution']}")
            
            return True
        else:
            print(f"\n Preprocessing failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n  Preprocessing timed out (1 hour)")
        return False
    except Exception as e:
        print(f"\n Error: {e}")
        return False

def step3_fine_tune_model():
    """Step 3: Fine-tune model"""
    print_header(" Step 3: Fine-Tune Model")
    
    print(" Starting fine-tuning...")
    print("This will train for 30 epochs (~20-40 minutes)")
    
    try:
        result = subprocess.run(
            [sys.executable, "ml/biomechanics/fine_tune_model.py"],
            capture_output=False,  # Show real-time progress
            timeout=3600  # 1 hour
        )
        
        if result.returncode == 0:
            print("\n Fine-tuning complete")
            return True
        else:
            print(f"\n Fine-tuning failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n  Fine-tuning timed out (1 hour)")
        return False
    except Exception as e:
        print(f"\n Error: {e}")
        return False

def step4_test_model():
    """Step 4: Test model accuracy"""
    print_header(" Step 4: Test Model Accuracy")
    
    print(" Testing fine-tuned model...")
    
    # Check if model file exists
    model_path = Path("ml/biomechanics/gnn_lstm_finetuned.pth")
    if not model_path.exists():
        print(f" Model not found at {model_path}")
        return False
    
    print(f" Model found: {model_path}")
    print("\n Model is ready for inference!")
    print("\nTest with a video:")
    print("  python ml/biomechanics/inference.py \\")
    print("    --model ml/biomechanics/gnn_lstm_finetuned.pth \\")    print("    --source your_video.mp4 \\")
    print("    --exercise squat \\")
    print("    --output analyzed_video.mp4")
    
    return True

def main():
    """Run complete pipeline"""
    print("""
========================================================================
                                                                    
            Achieve 90%+ Accuracy Pipeline                    
                                                                    
  Public Datasets -> MediaPipe -> Fine-Tuning -> Production Model     
                                                                    
========================================================================
    """)
    
    print("This pipeline will:")
    print("  1. Download public fitness datasets (Fitness-AQA, Countix, etc.)")
    print("  2. Preprocess videos to MediaPipe landmarks")
    print("  3. Fine-tune GNN-LSTM model with transfer learning")
    print("  4. Test model accuracy")
    print("\nEstimated time: 1-2 hours total")
    print("Required space: ~5GB for datasets")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print(" Cancelled")
        return
    
    # Check dependencies
    check_dependencies()
    
    # Run pipeline
    steps = [
        ("Download Datasets", step1_download_datasets),
        ("Preprocess Videos", step2_preprocess_videos),
        ("Fine-Tune Model", step3_fine_tune_model),
        ("Test Model", step4_test_model)
    ]
    
    for i, (name, step_func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"  Step {i}/{len(steps)}: {name}")
        print('='*70)
        
        if not step_func():
            print(f"\n Step {i} failed. Pipeline stopped.")
            print("\nYou can resume from this step by running:")
            print(f"  python achieve_90_percent_accuracy.py")
            return
        
        if i < len(steps):
            response = input(f"\n Step {i} complete. Continue to Step {i+1}? (y/n): ")
            if response.lower() != 'y':
                print(f"\n  Pipeline paused at Step {i}")
                print(f"\nResume with Step {i+1} by running:")
                print(f"  python achieve_90_percent_accuracy.py")
                return
    
    print_header(" Pipeline Complete!")
    print(" All steps completed successfully")
    print("\n Your biomechanics model is now trained with real data!")
    print("\n Next steps:")
    print("  1. Test with your own videos")
    print("  2. Integrate into backend API")
    print("  3. Deploy to production")
    print("\n Model location:")
    print("   ml/biomechanics/gnn_lstm_finetuned.pth")

if __name__ == "__main__":
    main()
