#!/usr/bin/env python3
"""
Test script for Gemini AI integration in FitBalance nutrition system
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from modules.nutrition import ProteinOptimizer
    
    def test_gemini_integration():
        """Test the Gemini integration"""
        print("🧪 Testing Gemini AI Integration...")
        
        # Initialize the protein optimizer
        optimizer = ProteinOptimizer()
        
        # Check if Gemini is available
        if optimizer.use_gemini:
            print("✅ Gemini AI is properly configured and ready!")
            print(f"🤖 Model: {optimizer.gemini_model.model_name if hasattr(optimizer.gemini_model, 'model_name') else 'gemini-2.0-flash'}")
        else:
            print("⚠️  Gemini AI is not available. Reasons could be:")
            print("   - GEMINI_API_KEY environment variable not set")
            print("   - google-generativeai package not installed")
            print("   - API key is invalid")
            print("\n📖 See GEMINI_SETUP.md for setup instructions")
        
        print("\n🔄 System will use the following detection order:")
        if optimizer.use_gemini:
            print("   1. Gemini Vision API (Primary)")
            print("   2. CNN-GRU Model (Fallback)")
            print("   3. Mock Detection (Last Resort)")
        else:
            print("   1. CNN-GRU Model (Primary)")
            print("   2. Mock Detection (Fallback)")
        
        return optimizer.use_gemini

    if __name__ == "__main__":
        # Load environment variables if .env exists
        from dotenv import load_dotenv
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"📁 Loaded environment from {env_file}")
        
        gemini_available = test_gemini_integration()
        
        if gemini_available:
            print("\n🎉 Ready for enhanced food detection!")
        else:
            print("\n🔧 Complete Gemini setup for best results!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("   pip install google-generativeai")
except Exception as e:
    print(f"❌ Error: {e}")