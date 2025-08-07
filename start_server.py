#!/usr/bin/env python3
"""
Startup script for FitBalance AI Fitness Platform
"""

import os
import sys
import uvicorn
from pathlib import Path

def setup_environment():
    """Setup environment for running the server"""
    # Add backend to Python path
    backend_path = Path(__file__).parent / "backend"
    sys.path.insert(0, str(backend_path))
    
    # Set default environment variables if not set
    if not os.getenv("DEBUG"):
        os.environ["DEBUG"] = "True"
    
    if not os.getenv("HOST"):
        os.environ["HOST"] = "0.0.0.0"
    
    if not os.getenv("PORT"):
        os.environ["PORT"] = "8000"

def main():
    """Main function to start the server"""
    print("🚀 Starting FitBalance AI Fitness Platform...")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Import the FastAPI app
    try:
        from api.main import app
        print("✅ FastAPI application loaded successfully")
    except ImportError as e:
        print(f"❌ Error loading FastAPI application: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed.")
        return 1
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"🌐 Server will start on: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print("\n📋 Available endpoints:")
    print("  • GET  / - Welcome page")
    print("  • GET  /health - Health check")
    print("  • POST /biomechanics/analyze - Analyze movement")
    print("  • GET  /biomechanics/heatmap/{user_id} - Get torque heatmap")
    print("  • POST /nutrition/analyze-meal - Analyze meal photo")
    print("  • GET  /nutrition/recommendations/{user_id} - Get nutrition recommendations")
    print("  • POST /burnout/analyze - Analyze burnout risk")
    print("  • GET  /burnout/survival-curve/{user_id} - Get survival curve")
    print("  • GET  /burnout/recommendations/{user_id} - Get burnout recommendations")
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    try:
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if debug else "warning"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 