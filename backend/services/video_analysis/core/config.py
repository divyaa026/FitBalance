"""
Configuration settings for FitBalance Biomechanics Backend.
============================================================
Defines supported exercises, model paths, and analysis parameters.
"""

import os
from typing import Dict, List, Tuple
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = "FitBalance Biomechanics API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # CORS Configuration
    allowed_origins: List[str] = ["*"]
    
    # File Upload Configuration
    max_upload_size_mb: int = 100
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    temp_upload_dir: str = "/tmp/uploads"
    
    # Model Configuration
    yolo_model_path: str = "yolov8n-pose.pt"  # Using YOLOv8 nano pose model
    pose_confidence_threshold: float = 0.5
    tracking_confidence_threshold: float = 0.5
    
    # Video Processing
    max_video_duration_seconds: int = 300  # 5 minutes max
    frame_skip: int = 1  # Process every Nth frame (1 = all frames)
    
    class Config:
        env_prefix = "FITBALANCE_"
        case_sensitive = False


# Supported Exercises Configuration
SUPPORTED_EXERCISES: List[str] = [
    "squat",
    "pushup", 
    "deadlift",
    "bicep_curl",
    "overhead_press"
]

# COCO Pose Keypoint Indices (17 keypoints)
# Reference: https://docs.ultralytics.com/datasets/pose/coco/
KEYPOINT_NAMES: List[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle"     # 16
]

# Keypoint indices mapping
KEYPOINT_INDICES: Dict[str, int] = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

# Exercise-specific keypoint configurations for AIGym analysis
# Each exercise defines: angle keypoints, up/down angles, and key joints to monitor
EXERCISE_CONFIGS: Dict[str, Dict] = {
    "squat": {
        "pose_type": "squat",
        "kpts_to_check": [11, 13, 15],  # hip -> knee -> ankle (left side)
        "angle_up": 170,  # Standing position angle
        "angle_down": 90,  # Bottom squat position angle
        "key_joints": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "form_rules": {
            "knee_over_toes": {"check": True, "feedback": "Keep your knees behind your toes"},
            "back_straight": {"check": True, "feedback": "Maintain a straight back throughout the movement"},
            "depth": {"check": True, "feedback": "Try to reach parallel or below for full range of motion"},
            "feet_width": {"check": True, "feedback": "Keep feet shoulder-width apart"}
        }
    },
    "pushup": {
        "pose_type": "pushup",
        "kpts_to_check": [5, 7, 9],  # shoulder -> elbow -> wrist (left side)
        "angle_up": 170,  # Arms extended
        "angle_down": 90,  # Bottom pushup position
        "key_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
        "form_rules": {
            "body_alignment": {"check": True, "feedback": "Keep your body in a straight line from head to heels"},
            "elbow_position": {"check": True, "feedback": "Keep elbows at 45 degrees to your body"},
            "full_extension": {"check": True, "feedback": "Fully extend arms at the top"},
            "chest_depth": {"check": True, "feedback": "Lower your chest close to the ground"}
        }
    },
    "deadlift": {
        "pose_type": "squat",  # Similar tracking to squat
        "kpts_to_check": [5, 11, 13],  # shoulder -> hip -> knee
        "angle_up": 170,  # Standing position
        "angle_down": 100,  # Bottom position (hip hinge)
        "key_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"],
        "form_rules": {
            "back_neutral": {"check": True, "feedback": "Maintain a neutral spine throughout the lift"},
            "hip_hinge": {"check": True, "feedback": "Hinge at the hips, not the lower back"},
            "bar_path": {"check": True, "feedback": "Keep the weight close to your body"},
            "lockout": {"check": True, "feedback": "Fully extend hips at the top"}
        }
    },
    "bicep_curl": {
        "pose_type": "pullup",  # Arm movement tracking
        "kpts_to_check": [5, 7, 9],  # shoulder -> elbow -> wrist
        "angle_up": 40,  # Curl up position (contracted)
        "angle_down": 160,  # Arms extended
        "key_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
        "form_rules": {
            "elbow_stable": {"check": True, "feedback": "Keep your elbows stationary at your sides"},
            "full_contraction": {"check": True, "feedback": "Squeeze at the top of the movement"},
            "controlled_descent": {"check": True, "feedback": "Lower the weight slowly and controlled"},
            "no_swinging": {"check": True, "feedback": "Avoid swinging your body for momentum"}
        }
    },
    "overhead_press": {
        "pose_type": "pullup",  # Vertical arm movement
        "kpts_to_check": [7, 5, 11],  # elbow -> shoulder -> hip
        "angle_up": 170,  # Arms overhead
        "angle_down": 90,  # Starting position
        "key_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"],
        "form_rules": {
            "core_engaged": {"check": True, "feedback": "Engage your core to protect your lower back"},
            "full_lockout": {"check": True, "feedback": "Fully extend arms at the top"},
            "bar_path": {"check": True, "feedback": "Press in a straight line overhead"},
            "no_lean_back": {"check": True, "feedback": "Avoid excessive leaning back"}
        }
    }
}

# Scoring weights for form analysis
SCORING_WEIGHTS: Dict[str, float] = {
    "rep_consistency": 0.25,      # How consistent are the reps
    "range_of_motion": 0.25,      # Full ROM achieved
    "posture_quality": 0.30,      # Overall form quality
    "tempo_control": 0.20         # Controlled movement speed
}

# Thresholds for validation
VALIDATION_THRESHOLDS: Dict[str, float] = {
    "min_pose_detection_rate": 0.7,   # At least 70% of frames must have pose
    "min_movement_variance": 0.02,     # Minimum movement to be considered exercise
    "min_periodicity_score": 0.3,      # Score for repetitive motion detection
    "min_frames_for_analysis": 30      # Minimum frames needed
}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
