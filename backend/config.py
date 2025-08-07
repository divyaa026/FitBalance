"""
Configuration settings for FitBalance AI Fitness Platform
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Application settings
    APP_NAME = "FitBalance AI Fitness Platform"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Database settings
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # ML Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    BIOMECHANICS_MODEL_PATH = os.path.join(MODEL_PATH, "biomechanics_model.pth")
    NUTRITION_MODEL_PATH = os.path.join(MODEL_PATH, "nutrition_model.pth")
    BURNOUT_MODEL_PATH = os.path.join(MODEL_PATH, "burnout_model.pkl")
    
    # File upload settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/")
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/fitbalance.log")
    
    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Feature flags
    ENABLE_BIOMECHANICS = os.getenv("ENABLE_BIOMECHANICS", "True").lower() == "true"
    ENABLE_NUTRITION = os.getenv("ENABLE_NUTRITION", "True").lower() == "true"
    ENABLE_BURNOUT = os.getenv("ENABLE_BURNOUT", "True").lower() == "true"
    
    # ML Model parameters
    BIOMECHANICS_CONFIG = {
        "input_dim": 3,
        "hidden_dim": 64,
        "num_layers": 3,
        "num_classes": 10,
        "dropout_rate": 0.2
    }
    
    NUTRITION_CONFIG = {
        "num_classes": 100,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout_rate": 0.3
    }
    
    BURNOUT_CONFIG = {
        "risk_thresholds": {
            "low": 25,
            "medium": 50,
            "high": 75,
            "critical": 90
        },
        "prediction_horizon": 365  # days
    }
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "firebase": {
                "credentials_path": cls.FIREBASE_CREDENTIALS_PATH,
                "project_id": os.getenv("FIREBASE_PROJECT_ID", "")
            },
            "neo4j": {
                "uri": cls.NEO4J_URI,
                "user": cls.NEO4J_USER,
                "password": cls.NEO4J_PASSWORD
            }
        }
    
    @classmethod
    def get_ml_config(cls) -> Dict[str, Any]:
        """Get ML model configuration"""
        return {
            "biomechanics": cls.BIOMECHANICS_CONFIG,
            "nutrition": cls.NUTRITION_CONFIG,
            "burnout": cls.BURNOUT_CONFIG
        }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    # Use in-memory databases for testing
    NEO4J_URI = "bolt://localhost:7687"

# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "default")
    
    return config.get(config_name, config["default"]) 