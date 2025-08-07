#!/usr/bin/env python3
"""
Test script to verify FitBalance installation and basic functionality
"""

import sys
import os
import asyncio
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Test core dependencies
        import torch
        print("✅ PyTorch imported successfully")
        
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        from PIL import Image
        print("✅ Pillow imported successfully")
        
        # Test ML libraries
        from lifelines import CoxPHFitter
        print("✅ Lifelines imported successfully")
        
        from torch_geometric.nn import GCNConv
        print("✅ PyTorch Geometric imported successfully")
        
        # Test FastAPI
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_modules():
    """Test that our custom modules can be imported"""
    print("\n🔍 Testing FitBalance modules...")
    
    try:
        # Add backend to path
        backend_path = Path(__file__).parent / "backend"
        sys.path.insert(0, str(backend_path))
        
        # Test biomechanics module
        from modules.biomechanics import BiomechanicsCoach, GNNLSTMModel
        print("✅ Biomechanics module imported successfully")
        
        # Test nutrition module
        from modules.nutrition import ProteinOptimizer, CNNGRUModel
        print("✅ Nutrition module imported successfully")
        
        # Test burnout module
        from modules.burnout import BurnoutPredictor
        print("✅ Burnout module imported successfully")
        
        # Test config
        from config import Config
        print("✅ Config module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return False

def test_model_initialization():
    """Test that models can be initialized"""
    print("\n🔍 Testing model initialization...")
    
    try:
        from modules.biomechanics import BiomechanicsCoach
        from modules.nutrition import ProteinOptimizer
        from modules.burnout import BurnoutPredictor
        
        # Initialize models
        biomechanics = BiomechanicsCoach()
        print("✅ BiomechanicsCoach initialized successfully")
        
        nutrition = ProteinOptimizer()
        print("✅ ProteinOptimizer initialized successfully")
        
        burnout = BurnoutPredictor()
        print("✅ BurnoutPredictor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False

async def test_basic_functionality():
    """Test basic functionality of each module"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        from modules.biomechanics import BiomechanicsCoach
        from modules.nutrition import ProteinOptimizer
        from modules.burnout import BurnoutPredictor
        
        # Test burnout prediction
        burnout = BurnoutPredictor()
        analysis = await burnout.analyze_risk(
            user_id="test_user",
            workout_frequency=5,
            sleep_hours=7.5,
            stress_level=6,
            recovery_time=2,
            performance_trend="stable"
        )
        print(f"✅ Burnout analysis completed - Risk Level: {analysis.risk_level}")
        
        # Test nutrition recommendations
        nutrition = ProteinOptimizer()
        recommendations = await nutrition.get_recommendations(
            user_id="test_user",
            target_protein=120.0,
            activity_level="moderate"
        )
        print(f"✅ Nutrition recommendations generated - {len(recommendations.get('recommendations', []))} recommendations")
        
        # Test biomechanics heatmap
        biomechanics = BiomechanicsCoach()
        heatmap = await biomechanics.generate_heatmap("test_user", "squat")
        print(f"✅ Biomechanics heatmap generated - {'error' not in heatmap}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 FitBalance Installation Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test modules
    if not test_modules():
        print("\n❌ Module tests failed. Please check the backend directory structure.")
        return False
    
    # Test model initialization
    if not test_model_initialization():
        print("\n❌ Model initialization tests failed.")
        return False
    
    # Test basic functionality
    try:
        result = asyncio.run(test_basic_functionality())
        if not result:
            print("\n❌ Functionality tests failed.")
            return False
    except Exception as e:
        print(f"\n❌ Async test error: {e}")
        return False
    
    print("\n🎉 All tests passed! FitBalance is ready to use.")
    print("\n📋 Next steps:")
    print("1. Set up your environment variables in .env file")
    print("2. Configure your databases (Firebase, Neo4j)")
    print("3. Run the server: cd backend/api && python main.py")
    print("4. Access the API at http://localhost:8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 