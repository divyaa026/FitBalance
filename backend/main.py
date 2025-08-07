#!/usr/bin/env python3
"""
FitBalance Backend Main Entry Point
This file serves as the main entry point for the FitBalance backend application.
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Import the FastAPI app from api/main.py
from api.main import app

def main():
    """Main function to run the FastAPI server"""
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print("🚀 Starting FitBalance Backend Server...")
    print(f"🌐 Server will run on: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )

if __name__ == "__main__":
    main() 