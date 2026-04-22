# MediaPipe Fitness Trainer Module
# Based on AI Fitness Trainer from LearnOpenCV

from .thresholds import get_thresholds_beginner, get_thresholds_pro
from .utils import get_mediapipe_pose, find_angle, get_landmark_features, draw_text, draw_dotted_line
from .process_frame import ProcessFrame

__all__ = [
    'ProcessFrame',
    'get_thresholds_beginner',
    'get_thresholds_pro',
    'get_mediapipe_pose',
    'find_angle',
    'get_landmark_features',
    'draw_text',
    'draw_dotted_line',
]
