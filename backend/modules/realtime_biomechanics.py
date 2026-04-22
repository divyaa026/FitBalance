"""
Real-time Biomechanics Analysis Module
WebSocket-based real-time form analysis using YOLOv8-pose and GNN-LSTM

Enhanced with:
- Pose quality detection (NONE/PARTIAL/FULL)
- Movement detection (STATIC/MOVING)
- Rep counting
- Contextual feedback based on pose state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import logging
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, asdict, field
from collections import deque
from enum import Enum
import time

# Fix for PyTorch 2.6+ weights_only restriction
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

logger = logging.getLogger(__name__)


class PoseQualityType(str, Enum):
    NONE = "NONE"           # No person detected
    PARTIAL = "PARTIAL"     # Person detected but incomplete pose
    FULL = "FULL"           # Full workout pose visible


class MovementState(str, Enum):
    STATIC = "STATIC"       # No significant movement
    MOVING = "MOVING"       # Active movement detected


@dataclass
class Keypoint:
    """Single keypoint with coordinates and confidence"""
    x: float
    y: float
    confidence: float


@dataclass
class PoseQuality:
    """Assessment of pose detection quality"""
    type: PoseQualityType
    coverage: float  # 0-1, percentage of keypoints detected with conf > 0.5
    missing_joints: List[str]
    message: str
    avg_confidence: float


@dataclass
class FormIssue:
    """Detected form issue"""
    type: str
    severity: float  # 0-1
    message: str
    is_critical: bool = False


@dataclass
class RealtimeAnalysisResult:
    """Result of single-frame analysis"""
    form_score: float
    feedback: str
    keypoints: List[Dict]
    issues: List[Dict]
    joint_angles: Dict[str, float]
    processing_time_ms: float
    # New fields
    pose_quality: Dict = field(default_factory=dict)
    movement_state: str = "STATIC"
    reps: int = 0
    should_speak: bool = True


class SingleFrameGNN(nn.Module):
    """Simplified GNN for single-frame form analysis"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=10):
        super(SingleFrameGNN, self).__init__()
        
        if HAS_TORCH_GEOMETRIC:
            self.gnn1 = GCNConv(input_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
            self.gnn3 = GCNConv(hidden_dim, hidden_dim)
        
        # Output heads
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.issue_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        if not HAS_TORCH_GEOMETRIC:
            # Fallback: simple MLP if no torch_geometric
            x = x.mean(dim=0, keepdim=True)
            return torch.tensor([[0.75]]), torch.zeros(1, 10)
        
        # GNN layers
        x = F.relu(self.gnn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gnn2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gnn3(x, edge_index))
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        
        # Outputs
        score = self.score_head(x)
        issues = self.issue_head(x)
        
        return score, issues


class FeedbackGenerator:
    """Generates dynamic voice-friendly feedback based on actual analysis results"""
    
    @classmethod
    def generate_pose_feedback(cls, pose_quality: PoseQuality, exercise: str) -> str:
        """Generate feedback based on pose quality - uses actual detected state"""
        if pose_quality.type == PoseQualityType.NONE:
            return "No person detected. Please step into frame."
        
        # Use the specific message from pose quality assessment
        # This is already dynamically generated based on what body parts are visible
        if pose_quality.message:
            return pose_quality.message
        
        return "Adjust your position for better detection."
    
    @classmethod
    def generate_dynamic_feedback(cls, exercise: str, issues: List[FormIssue], 
                                   form_score: float, angles: Dict[str, float],
                                   reps: int, movement_state: MovementState) -> str:
        """Generate fully dynamic feedback based on actual measurements"""
        
        # If we have issues, report the most severe one with context
        if issues:
            sorted_issues = sorted(issues, key=lambda x: x.severity, reverse=True)
            top_issue = sorted_issues[0]
            
            # Critical issues (high severity)
            if top_issue.severity >= 0.8:
                return f"Stop! {top_issue.message}"
            elif top_issue.severity >= 0.6:
                return top_issue.message
            else:
                # Minor issue
                return f"Good, but {top_issue.message.lower()}"
        
        # No issues - give positive feedback based on score
        if form_score >= 90:
            return f"Excellent {exercise.replace('_', ' ')} form!"
        elif form_score >= 80:
            return f"Good {exercise.replace('_', ' ')}! Keep it up."
        elif form_score >= 70:
            return "Decent form. Focus on control."
        else:
            return "Keep practicing your technique."
    
    @classmethod
    def generate(cls, exercise: str, issues: List[FormIssue], form_score: float,
                 pose_quality: Optional[PoseQuality] = None,
                 movement_state: MovementState = MovementState.STATIC,
                 angles: Dict[str, float] = None,
                 reps: int = 0) -> str:
        """Generate voice-friendly feedback based on current state"""
        
        # First check pose quality
        if pose_quality and pose_quality.type != PoseQualityType.FULL:
            return cls.generate_pose_feedback(pose_quality, exercise)
        
        # If full pose but static (not moving)
        if movement_state == MovementState.STATIC:
            exercise_name = exercise.replace("_", " ")
            return f"Ready. Begin your {exercise_name}."
        
        # Active workout - generate dynamic feedback
        return cls.generate_dynamic_feedback(
            exercise, issues, form_score, angles or {}, reps, movement_state
        )
    
    @classmethod
    def get_detailed_feedback(cls, exercise: str, angles: Dict[str, float], 
                               issues: List[FormIssue], form_score: float,
                               pose_quality: Optional[PoseQuality] = None,
                               movement_state: MovementState = MovementState.STATIC,
                               reps: int = 0) -> str:
        """Generate detailed feedback for display (more verbose than voice)"""
        if pose_quality and pose_quality.type == PoseQualityType.NONE:
            return "No person detected. Step into frame."
        
        if pose_quality and pose_quality.type == PoseQualityType.PARTIAL:
            return f"Incomplete pose: {pose_quality.message}"
        
        if movement_state == MovementState.STATIC:
            return f"Ready position. Start your {exercise.replace('_', ' ')} to analyze form."
        
        # Build detailed feedback from angles and issues
        feedback_parts = []
        
        # Add score context
        if form_score >= 90:
            feedback_parts.append("Excellent form!")
        elif form_score >= 80:
            feedback_parts.append("Good form.")
        elif form_score >= 60:
            feedback_parts.append("Form needs improvement.")
        else:
            feedback_parts.append("⚠️ Poor form detected.")
        
        # Add specific issues
        if issues:
            for issue in sorted(issues, key=lambda x: x.severity, reverse=True)[:2]:
                feedback_parts.append(issue.message)
        
        # Add angle info for context
        if angles:
            knee_angle = (angles.get("left_knee", 0) + angles.get("right_knee", 0)) / 2
            if knee_angle > 0:
                if exercise == "squat":
                    if knee_angle > 140:
                        feedback_parts.append(f"Knee angle: {knee_angle:.0f}° - go deeper")
                    elif knee_angle < 80:
                        feedback_parts.append(f"Knee angle: {knee_angle:.0f}° - good depth")
        
        return " ".join(feedback_parts)


class RealtimeBiomechanicsAnalyzer:
    """Real-time biomechanics analysis using YOLOv8-pose"""
    
    # COCO keypoint names (indices 0-16)
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # COCO keypoint connections for skeleton drawing
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Torso
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]
    
    # Joint angle definitions
    ANGLE_JOINTS = {
        "left_knee": (11, 13, 15),  # hip, knee, ankle
        "right_knee": (12, 14, 16),
        "left_hip": (5, 11, 13),  # shoulder, hip, knee
        "right_hip": (6, 12, 14),
        "left_elbow": (5, 7, 9),  # shoulder, elbow, wrist
        "right_elbow": (6, 8, 10),
        "left_shoulder": (7, 5, 11),  # elbow, shoulder, hip
        "right_shoulder": (8, 6, 12),
    }
    
    # Body part groups for pose quality assessment
    BODY_PARTS = {
        "face": [0, 1, 2, 3, 4],           # nose, eyes, ears
        "upper_body": [5, 6, 7, 8, 9, 10], # shoulders, elbows, wrists
        "lower_body": [11, 12, 13, 14, 15, 16],  # hips, knees, ankles
    }
    
    # Required keypoints for workout detection (need most of these with conf > 0.5)
    REQUIRED_FOR_WORKOUT = [5, 6, 11, 12, 13, 14, 15, 16]  # shoulders, hips, knees, ankles
    
    # Exercise-specific critical thresholds
    EXERCISE_STANDARDS = {
        "squat": {
            "knee_angle_range": (70, 130),
            "hip_angle_range": (40, 100),
            "critical_issues": {
                "knee_valgus": {"threshold": 15, "severity": 0.9},
                "back_rounding": {"max_angle": 150, "severity": 0.85},
                "insufficient_depth": {"min_knee_angle": 60, "severity": 0.6},
            },
            "minor_issues": {
                "forward_lean": {"max_offset": 50, "severity": 0.4},
            }
        },
        "deadlift": {
            "hip_angle_range": (25, 70),
            "knee_angle_range": (50, 120),
            "back_angle_range": (10, 50),
            "critical_issues": {
                "back_rounding": {"max_angle": 25, "severity": 0.95},
            },
            "minor_issues": {
                "hips_rising_first": {"threshold": 0.5, "severity": 0.5},
            }
        },
        "bench_press": {
            "elbow_angle_range": (70, 120),
            "critical_issues": {
                "elbow_flare": {"max_angle": 75, "severity": 0.7},
            },
            "minor_issues": {
                "uneven_press": {"threshold": 10, "severity": 0.4},
            }
        },
        "pushup": {
            "elbow_angle_range": (80, 120),
            "critical_issues": {
                "elbow_flare": {"max_angle": 60, "severity": 0.7},
            },
            "minor_issues": {
                "incomplete_lockout": {"min_angle": 165, "severity": 0.4},
            }
        },
        "lunge": {
            "knee_angle_range": (70, 120),
            "hip_angle_range": (70, 130),
            "critical_issues": {
                "knee_past_toes": {"threshold": 0, "severity": 0.7},
            },
            "minor_issues": {}
        },
        "overhead_press": {
            "elbow_angle_range": (80, 180),
            "shoulder_angle_range": (90, 180),
            "critical_issues": {},
            "minor_issues": {
                "forward_lean": {"max_offset": 30, "severity": 0.5},
            }
        },
        "row": {
            "elbow_angle_range": (40, 150),
            "critical_issues": {
                "back_rounding": {"max_angle": 30, "severity": 0.8},
            },
            "minor_issues": {}
        },
    }
    
    # Movement detection thresholds (lowered for better sensitivity)
    MOVEMENT_VELOCITY_THRESHOLD = 3.0  # pixels per frame (was 5.0)
    STATIC_FRAME_COUNT = 10  # frames to consider as static
    
    def __init__(self):
        self.yolo_model = None
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_feedback_time = 0
        self.feedback_cooldown = 1.5  # seconds between voice feedback (was 2.0)
        self.last_feedback = ""
        
        # Per-user state for frame buffering and movement detection
        self.user_buffers: Dict[str, deque] = {}
        self.user_rep_state: Dict[str, Dict] = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load YOLO pose and GNN models"""
        # Load YOLO
        try:
            from ultralytics import YOLO
            model_paths = [
                'yolov8n-pose.pt',
                '../yolov8n-pose.pt',
                'backend/yolov8n-pose.pt',
            ]
            for path in model_paths:
                try:
                    self.yolo_model = YOLO(path)
                    logger.info(f"YOLO pose model loaded from: {path}")
                    break
                except:
                    continue
            if self.yolo_model is None:
                self.yolo_model = YOLO('yolov8n-pose.pt')
                logger.info("YOLO pose model downloaded")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
        
        # Load GNN model
        try:
            self.gnn_model = SingleFrameGNN().to(self.device)
            self.gnn_model.eval()
            logger.info("GNN model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GNN model: {e}")
    
    def decode_frame(self, base64_frame: str) -> Optional[np.ndarray]:
        """Decode base64 frame to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_frame:
                base64_frame = base64_frame.split(',')[1]
            
            img_bytes = base64.b64decode(base64_frame)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Extract pose keypoints from frame using YOLO"""
        if self.yolo_model is None:
            return None
        
        try:
            results = self.yolo_model(frame, verbose=False)
            
            if results and len(results) > 0 and results[0].keypoints is not None:
                kpts = results[0].keypoints
                
                if hasattr(kpts, 'xy') and kpts.xy is not None and len(kpts.xy) > 0:
                    xy = kpts.xy[0].cpu().numpy()
                    conf = kpts.conf[0].cpu().numpy() if hasattr(kpts, 'conf') else np.ones(17)
                    
                    keypoints = []
                    for i in range(len(xy)):
                        keypoints.append(Keypoint(
                            x=float(xy[i][0]),
                            y=float(xy[i][1]),
                            confidence=float(conf[i]) if i < len(conf) else 0.5
                        ))
                    return keypoints
        except Exception as e:
            logger.error(f"Keypoint extraction error: {e}")
        
        return None
    
    def calculate_angle(self, p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle)
    
    def calculate_joint_angles(self, keypoints: List[Keypoint]) -> Dict[str, float]:
        """Calculate all joint angles from keypoints"""
        angles = {}
        
        for joint_name, (i1, i2, i3) in self.ANGLE_JOINTS.items():
            if (i1 < len(keypoints) and i2 < len(keypoints) and i3 < len(keypoints)):
                p1, p2, p3 = keypoints[i1], keypoints[i2], keypoints[i3]
                
                # Only calculate if keypoints are confident enough
                if min(p1.confidence, p2.confidence, p3.confidence) > 0.3:
                    angles[joint_name] = self.calculate_angle(p1, p2, p3)
        
        return angles
    
    def detect_issues(self, keypoints: List[Keypoint], angles: Dict[str, float], 
                      exercise: str) -> List[FormIssue]:
        """Detect form issues based on angles and exercise type - returns dynamic messages with actual values"""
        issues = []
        standards = self.EXERCISE_STANDARDS.get(exercise, self.EXERCISE_STANDARDS["squat"])
        
        # Check knee angles with actual measurements
        left_knee = angles.get("left_knee", 180)
        right_knee = angles.get("right_knee", 180)
        knee_angle = (left_knee + right_knee) / 2
        
        if "knee_angle_range" in standards:
            min_angle, max_angle = standards["knee_angle_range"]
            target = (min_angle + max_angle) / 2
            
            if knee_angle > max_angle:
                depth_needed = knee_angle - target
                severity = min(1.0, (knee_angle - max_angle) / 30)
                issues.append(FormIssue(
                    type="insufficient_depth",
                    severity=severity,
                    message=f"Knee at {knee_angle:.0f}°. Lower {depth_needed:.0f}° more for proper depth."
                ))
            elif knee_angle < min_angle:
                issues.append(FormIssue(
                    type="too_deep",
                    severity=0.5,
                    message=f"Knee at {knee_angle:.0f}°. Don't go below {min_angle:.0f}° to protect joints."
                ))
        
        # Check hip angles with actual measurements
        left_hip_angle = angles.get("left_hip", 180)
        right_hip_angle = angles.get("right_hip", 180)
        hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        if "hip_angle_range" in standards:
            min_angle, max_angle = standards["hip_angle_range"]
            if hip_angle < min_angle:
                diff = min_angle - hip_angle
                severity = min(1.0, diff / 20)
                issues.append(FormIssue(
                    type="hips_rising_first",
                    severity=severity,
                    message=f"Hip angle {hip_angle:.0f}°. Open hips {diff:.0f}° more, lead with chest."
                ))
        
        # Check knee alignment (valgus) with position data
        if len(keypoints) >= 17:
            left_hip_kp, left_knee_kp, left_ankle = keypoints[11], keypoints[13], keypoints[15]
            right_hip_kp, right_knee_kp, right_ankle = keypoints[12], keypoints[14], keypoints[16]
            
            # Calculate knee deviation from hip-ankle line
            left_knee_offset = left_knee_kp.x - left_hip_kp.x
            left_ankle_offset = left_ankle.x - left_hip_kp.x
            right_knee_offset = right_hip_kp.x - right_knee_kp.x
            right_ankle_offset = right_hip_kp.x - right_ankle.x
            
            if left_ankle_offset != 0 and left_knee_offset < left_ankle_offset * 0.7:
                deviation = abs(left_ankle_offset * 0.7 - left_knee_offset)
                issues.append(FormIssue(
                    type="knee_valgus",
                    severity=0.7,
                    message=f"Left knee caving in. Push it outward over your toe."
                ))
            
            if right_ankle_offset != 0 and right_knee_offset < right_ankle_offset * 0.7:
                deviation = abs(right_ankle_offset * 0.7 - right_knee_offset)
                issues.append(FormIssue(
                    type="knee_valgus",
                    severity=0.7,
                    message=f"Right knee caving in. Push it outward over your toe."
                ))
        
        # Check for forward lean with torso angle
        if exercise in ["squat", "lunge"] and len(keypoints) >= 12:
            left_shoulder, left_hip_kp = keypoints[5], keypoints[11]
            
            # Calculate actual torso lean angle
            if left_hip_kp.confidence > 0.3 and left_shoulder.confidence > 0.3:
                dx = left_shoulder.x - left_hip_kp.x
                dy = left_hip_kp.y - left_shoulder.y  # Y is inverted
                torso_lean = np.degrees(np.arctan2(abs(dx), dy)) if dy > 0 else 0
                
                if torso_lean > 30:  # More than 30 degrees forward
                    issues.append(FormIssue(
                        type="forward_lean",
                        severity=min(1.0, torso_lean / 60),
                        message=f"Torso leaning {torso_lean:.0f}°. Stay more upright, chest up."
                    ))
        
        # Check back angle (spine alignment)
        if len(keypoints) >= 12:
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            left_hip_kp, right_hip_kp = keypoints[11], keypoints[12]
            
            # Check shoulder asymmetry
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            hip_diff = abs(left_hip_kp.y - right_hip_kp.y)
            
            if shoulder_diff > 30 and shoulder_diff > hip_diff + 15:
                issues.append(FormIssue(
                    type="uneven_shoulders",
                    severity=0.5,
                    message=f"Shoulders uneven by {shoulder_diff:.0f}px. Keep them level."
                ))
        
        return issues
    
    def calculate_form_score(self, keypoints: List[Keypoint], angles: Dict[str, float],
                             issues: List[FormIssue], exercise: str) -> float:
        """Calculate form score based on analysis"""
        base_score = 100.0
        
        # Deduct for issues
        for issue in issues:
            deduction = issue.severity * 15  # Max 15 points per issue
            base_score -= deduction
        
        # Bonus for being in ideal angle ranges
        standards = self.EXERCISE_STANDARDS.get(exercise, {})
        
        if "knee_angle_range" in standards:
            knee_angle = (angles.get("left_knee", 90) + angles.get("right_knee", 90)) / 2
            min_a, max_a = standards["knee_angle_range"]
            ideal = (min_a + max_a) / 2
            if abs(knee_angle - ideal) < 15:
                base_score += 5
        
        # Keypoint confidence penalty
        avg_conf = sum(kp.confidence for kp in keypoints) / len(keypoints)
        if avg_conf < 0.5:
            base_score -= (0.5 - avg_conf) * 20
        
        return max(0, min(100, base_score))

    def get_user_buffer(self, user_id: str) -> deque:
        """Get or create frame buffer for a user"""
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = deque(maxlen=10)  # Store last 10 frames
        return self.user_buffers[user_id]

    def get_user_rep_state(self, user_id: str) -> Dict:
        """Get or create rep counting state for a user"""
        if user_id not in self.user_rep_state:
            self.user_rep_state[user_id] = {
                "reps": 0,
                "phase": "neutral",  # neutral, down, up
                "last_hip_y": None,
                "min_hip_y": float('inf'),
                "max_hip_y": float('-inf'),
            }
        return self.user_rep_state[user_id]

    def assess_pose_quality(self, keypoints: Optional[List[Keypoint]], 
                           frame_height: int = 480) -> PoseQuality:
        """
        Assess the quality of pose detection.
        Returns NONE, PARTIAL, or FULL based on keypoint coverage.
        """
        # No keypoints at all - no person detected
        if keypoints is None or len(keypoints) == 0:
            return PoseQuality(
                type=PoseQualityType.NONE,
                coverage=0.0,
                missing_joints=[],
                message="No person detected. Step into frame.",
                avg_confidence=0.0
            )
        
        # Count confident keypoints (conf > 0.5)
        confident_kps = [kp for kp in keypoints if kp.confidence > 0.5]
        coverage = len(confident_kps) / 17.0
        avg_confidence = sum(kp.confidence for kp in keypoints) / len(keypoints)
        
        # Find missing joints (conf < 0.5)
        missing_joints = []
        for i, kp in enumerate(keypoints):
            if kp.confidence < 0.5:
                missing_joints.append(self.KEYPOINT_NAMES[i])
        
        # Check body part coverage
        face_visible = sum(1 for i in self.BODY_PARTS["face"] 
                          if i < len(keypoints) and keypoints[i].confidence > 0.5)
        upper_visible = sum(1 for i in self.BODY_PARTS["upper_body"] 
                           if i < len(keypoints) and keypoints[i].confidence > 0.5)
        lower_visible = sum(1 for i in self.BODY_PARTS["lower_body"] 
                           if i < len(keypoints) and keypoints[i].confidence > 0.5)
        
        # Check required workout keypoints
        required_visible = sum(1 for i in self.REQUIRED_FOR_WORKOUT 
                              if i < len(keypoints) and keypoints[i].confidence > 0.5)
        
        # First check: Do we have LOWER BODY? This is critical for workout detection
        # Prioritize body part presence over confidence scores
        
        # If lower body is mostly missing, can't do workout analysis
        if lower_visible < 3:  # Less than 50% of lower body (hips, knees, ankles)
            if face_visible >= 2 and upper_visible < 3:
                # Only face visible - too close
                return PoseQuality(
                    type=PoseQualityType.PARTIAL,
                    coverage=coverage,
                    missing_joints=missing_joints,
                    message="Step back and show your full body for workout analysis.",
                    avg_confidence=avg_confidence
                )
            elif upper_visible >= 3:
                # Upper body + face but no legs
                return PoseQuality(
                    type=PoseQualityType.PARTIAL,
                    coverage=coverage,
                    missing_joints=missing_joints,
                    message="Lower your camera or step back to show your full body including legs.",
                    avg_confidence=avg_confidence
                )
            else:
                # Very little visible
                return PoseQuality(
                    type=PoseQualityType.PARTIAL,
                    coverage=coverage,
                    missing_joints=missing_joints,
                    message="Position yourself so your full body is visible.",
                    avg_confidence=avg_confidence
                )
        
        # Lower body visible - check if we have enough for full analysis
        # FULL: At least 6/8 required keypoints visible (75%)
        if required_visible >= 6 and coverage >= 0.6:
            # Check if head is off-center (top 10% of frame)
            nose = keypoints[0]
            if nose.confidence > 0.5 and nose.y < frame_height * 0.1:
                return PoseQuality(
                    type=PoseQualityType.PARTIAL,
                    coverage=coverage,
                    missing_joints=missing_joints,
                    message="Center yourself in frame.",
                    avg_confidence=avg_confidence
                )
            
            # Good enough for workout!
            return PoseQuality(
                type=PoseQualityType.FULL,
                coverage=coverage,
                missing_joints=missing_joints,
                message="Full pose detected",
                avg_confidence=avg_confidence
            )
        
        # Have lower body but missing some key joints
        if upper_visible < 3:  # Lower body visible but not upper
            return PoseQuality(
                type=PoseQualityType.PARTIAL,
                coverage=coverage,
                missing_joints=missing_joints,
                message="Raise camera to capture your upper body too.",
                avg_confidence=avg_confidence
            )
        # Some of everything but confidence is low
        if avg_confidence < 0.35:
            return PoseQuality(
                type=PoseQualityType.PARTIAL,
                coverage=coverage,
                missing_joints=missing_joints,
                message="Improve lighting for better detection.",
                avg_confidence=avg_confidence
            )
        
        # Fallback - should have enough for basic analysis
        return PoseQuality(
            type=PoseQualityType.FULL,
            coverage=coverage,
            missing_joints=missing_joints,
            message="Pose detected",
            avg_confidence=avg_confidence
        )

    def detect_movement(self, keypoints: List[Keypoint], user_id: str) -> MovementState:
        """
        Detect if the user is actively moving by comparing keypoint positions
        across the last N frames in the buffer.
        """
        buffer = self.get_user_buffer(user_id)
        
        # Add current frame to buffer
        kp_positions = [(kp.x, kp.y) for kp in keypoints]
        buffer.append(kp_positions)
        
        # Need at least 5 frames to detect movement
        if len(buffer) < 5:
            return MovementState.STATIC
        
        # Calculate maximum velocity across key joints (hips, knees, shoulders)
        key_indices = [5, 6, 11, 12, 13, 14]  # shoulders, hips, knees
        max_velocity = 0.0
        
        # Compare last frame to 5 frames ago
        current_frame = buffer[-1]
        old_frame = buffer[-5]
        
        for idx in key_indices:
            if idx < len(current_frame) and idx < len(old_frame):
                dx = current_frame[idx][0] - old_frame[idx][0]
                dy = current_frame[idx][1] - old_frame[idx][1]
                velocity = np.sqrt(dx*dx + dy*dy) / 5.0  # velocity per frame
                max_velocity = max(max_velocity, velocity)
        
        if max_velocity > self.MOVEMENT_VELOCITY_THRESHOLD:
            return MovementState.MOVING
        return MovementState.STATIC

    def update_rep_count(self, keypoints: List[Keypoint], exercise: str, 
                        user_id: str) -> int:
        """
        Count repetitions based on hip movement pattern.
        For squats/lunges: track hip going down and up
        """
        rep_state = self.get_user_rep_state(user_id)
        
        # Get hip position (average of left and right hip)
        if len(keypoints) < 17:
            return rep_state["reps"]
        
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        if left_hip.confidence < 0.5 and right_hip.confidence < 0.5:
            return rep_state["reps"]
        
        # Use the more confident hip or average
        if left_hip.confidence > right_hip.confidence:
            hip_y = left_hip.y
        elif right_hip.confidence > left_hip.confidence:
            hip_y = right_hip.y
        else:
            hip_y = (left_hip.y + right_hip.y) / 2
        
        # Track min/max for movement range
        rep_state["min_hip_y"] = min(rep_state["min_hip_y"], hip_y)
        rep_state["max_hip_y"] = max(rep_state["max_hip_y"], hip_y)
        
        # Detect rep cycle based on exercise
        if exercise in ["squat", "lunge", "deadlift"]:
            # Hip goes down (y increases in screen coords) then back up (y decreases)
            last_y = rep_state["last_hip_y"]
            
            # Initialize start position on first frame
            if "start_hip_y" not in rep_state:
                rep_state["start_hip_y"] = hip_y
            
            if last_y is not None:
                movement_range = rep_state["max_hip_y"] - rep_state["min_hip_y"]
                
                # Need at least 30 pixels of movement to count as exercise (lowered from 50)
                if movement_range > 30:
                    # Going down (hip_y increases in screen coordinates)
                    if hip_y > last_y + 8 and rep_state["phase"] == "neutral":
                        rep_state["phase"] = "down"
                        logger.debug(f"Phase: DOWN, hip_y={hip_y:.0f}")
                    # At bottom of squat
                    elif rep_state["phase"] == "down" and hip_y >= rep_state["max_hip_y"] - 15:
                        rep_state["phase"] = "bottom"
                        logger.debug(f"Phase: BOTTOM, hip_y={hip_y:.0f}")
                    # Going up from bottom
                    elif hip_y < last_y - 8 and rep_state["phase"] == "bottom":
                        rep_state["phase"] = "up"
                        logger.debug(f"Phase: UP, hip_y={hip_y:.0f}")
                    # Completed rep (returned to near start position)
                    elif rep_state["phase"] == "up" and hip_y <= rep_state["start_hip_y"] + 20:
                        rep_state["reps"] += 1
                        rep_state["phase"] = "neutral"
                        # Reset start position for next rep
                        rep_state["start_hip_y"] = hip_y
                        logger.info(f"REP COUNTED! Total: {rep_state['reps']}")
        
        elif exercise in ["pushup", "bench_press"]:
            # For these, track elbow angle changes instead
            # Simplified: just count based on vertical movement of shoulders
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            
            if left_shoulder.confidence > 0.5:
                shoulder_y = left_shoulder.y
                last_y = rep_state.get("last_shoulder_y")
                
                if last_y is not None:
                    if shoulder_y > last_y + 15 and rep_state["phase"] == "neutral":
                        rep_state["phase"] = "down"
                    elif shoulder_y < last_y - 15 and rep_state["phase"] == "down":
                        rep_state["reps"] += 1
                        rep_state["phase"] = "neutral"
                
                rep_state["last_shoulder_y"] = shoulder_y
        
        rep_state["last_hip_y"] = hip_y
        return rep_state["reps"]
    
    def get_user_feedback_state(self, user_id: str) -> Dict:
        """Get or create feedback debounce state for a user"""
        if not hasattr(self, 'user_feedback_state'):
            self.user_feedback_state = {}
        if user_id not in self.user_feedback_state:
            self.user_feedback_state[user_id] = {
                "last_feedback": "",
                "last_feedback_time": 0
            }
        return self.user_feedback_state[user_id]
    
    def should_speak_feedback(self, feedback: str, user_id: str = "default") -> bool:
        """Determine if voice feedback should be spoken (per-user debouncing)"""
        current_time = time.time()
        state = self.get_user_feedback_state(user_id)
        
        # Don't repeat same feedback
        if feedback == state["last_feedback"]:
            return False
        
        # Respect cooldown
        if current_time - state["last_feedback_time"] < self.feedback_cooldown:
            return False
        
        state["last_feedback"] = feedback
        state["last_feedback_time"] = current_time
        return True
    
    def process_frame(self, base64_frame: str, exercise_type: str, 
                      user_id: str) -> Optional[RealtimeAnalysisResult]:
        """
        Process a single frame and return analysis with enhanced pose quality detection.
        
        Returns contextual feedback based on:
        1. Pose quality (NONE/PARTIAL/FULL)
        2. Movement state (STATIC/MOVING)
        3. Form issues (if actively exercising)
        """
        start_time = time.time()
        
        # Decode frame
        frame = self.decode_frame(base64_frame)
        if frame is None:
            return RealtimeAnalysisResult(
                form_score=0,
                feedback="Camera error. Check connection.",
                keypoints=[],
                issues=[],
                joint_angles={},
                processing_time_ms=0,
                pose_quality={"type": "NONE", "coverage": 0, "message": "Frame decode failed"},
                movement_state="STATIC",
                reps=0,
                should_speak=True
            )
        
        frame_height = frame.shape[0]
        
        # Extract keypoints
        keypoints = self.extract_keypoints(frame)
        
        # Assess pose quality
        pose_quality = self.assess_pose_quality(keypoints, frame_height)
        
        # If no pose or partial pose, return early with appropriate feedback
        if pose_quality.type == PoseQualityType.NONE:
            feedback = FeedbackGenerator.generate_pose_feedback(pose_quality, exercise_type)
            should_speak = self.should_speak_feedback(feedback, user_id)
            
            return RealtimeAnalysisResult(
                form_score=0,
                feedback=feedback,
                keypoints=[],
                issues=[],
                joint_angles={},
                processing_time_ms=(time.time() - start_time) * 1000,
                pose_quality={
                    "type": pose_quality.type.value,
                    "coverage": pose_quality.coverage,
                    "missing_joints": pose_quality.missing_joints,
                    "message": pose_quality.message,
                    "avg_confidence": pose_quality.avg_confidence
                },
                movement_state="STATIC",
                reps=0,
                should_speak=should_speak
            )
        
        if pose_quality.type == PoseQualityType.PARTIAL:
            # Partial pose - give score 20-40 based on coverage
            partial_score = 20 + (pose_quality.coverage * 20)
            feedback = FeedbackGenerator.generate_pose_feedback(pose_quality, exercise_type)
            should_speak = self.should_speak_feedback(feedback, user_id)
            
            return RealtimeAnalysisResult(
                form_score=round(partial_score, 1),
                feedback=feedback,
                keypoints=[{"x": kp.x, "y": kp.y, "confidence": kp.confidence} for kp in keypoints] if keypoints else [],
                issues=[],
                joint_angles={},
                processing_time_ms=(time.time() - start_time) * 1000,
                pose_quality={
                    "type": pose_quality.type.value,
                    "coverage": pose_quality.coverage,
                    "missing_joints": pose_quality.missing_joints,
                    "message": pose_quality.message,
                    "avg_confidence": pose_quality.avg_confidence
                },
                movement_state="STATIC",
                reps=0,
                should_speak=should_speak
            )
        
        # FULL pose detected - check movement
        movement_state = self.detect_movement(keypoints, user_id)
        
        # Calculate angles regardless of movement
        angles = self.calculate_joint_angles(keypoints)
        
        # If static (not actively exercising)
        if movement_state == MovementState.STATIC:
            feedback = FeedbackGenerator.POSE_FEEDBACK["static_ready"].format(
                exercise=exercise_type.replace("_", " ")
            )
            should_speak = self.should_speak_feedback(feedback, user_id)
            
            return RealtimeAnalysisResult(
                form_score=50,  # Neutral score for ready position
                feedback=feedback,
                keypoints=[{"x": kp.x, "y": kp.y, "confidence": kp.confidence} for kp in keypoints],
                issues=[],
                joint_angles=angles,
                processing_time_ms=(time.time() - start_time) * 1000,
                pose_quality={
                    "type": pose_quality.type.value,
                    "coverage": pose_quality.coverage,
                    "missing_joints": [],
                    "message": "Ready position",
                    "avg_confidence": pose_quality.avg_confidence
                },
                movement_state="STATIC",
                reps=self.get_user_rep_state(user_id)["reps"],
                should_speak=should_speak
            )
        
        # MOVING - Active exercise! Run full form analysis
        
        # Detect form issues
        issues = self.detect_issues(keypoints, angles, exercise_type)
        
        # Mark critical issues
        standards = self.EXERCISE_STANDARDS.get(exercise_type, {})
        critical_issue_types = set(standards.get("critical_issues", {}).keys())
        for issue in issues:
            if issue.type in critical_issue_types:
                issue.is_critical = True
        
        # Calculate form score
        form_score = self.calculate_form_score(keypoints, angles, issues, exercise_type)
        
        # Update rep count
        reps = self.update_rep_count(keypoints, exercise_type, user_id)
        
        # Generate contextual feedback with actual measurements
        feedback = FeedbackGenerator.generate(
            exercise_type, 
            issues, 
            form_score,
            pose_quality,
            movement_state,
            angles,
            reps
        )
        
        # Check if we should speak this feedback
        should_speak = self.should_speak_feedback(feedback, user_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RealtimeAnalysisResult(
            form_score=round(form_score, 1),
            feedback=feedback,
            keypoints=[{"x": kp.x, "y": kp.y, "confidence": kp.confidence} for kp in keypoints],
            issues=[{"type": issue.type, "severity": issue.severity, 
                    "message": issue.message, "is_critical": issue.is_critical} for issue in issues],
            joint_angles=angles,
            processing_time_ms=round(processing_time, 2),
            pose_quality={
                "type": pose_quality.type.value,
                "coverage": pose_quality.coverage,
                "missing_joints": pose_quality.missing_joints,
                "message": pose_quality.message,
                "avg_confidence": pose_quality.avg_confidence
            },
            movement_state=movement_state.value,
            reps=reps,
            should_speak=should_speak
        )


# Global analyzer instance (singleton)
_analyzer_instance = None

def get_analyzer() -> RealtimeBiomechanicsAnalyzer:
    """Get or create the global analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = RealtimeBiomechanicsAnalyzer()
    return _analyzer_instance
