"""
Real-time Biomechanics Analysis Module
WebSocket-based real-time form analysis using MediaPipe Pose

Based on AI Fitness Trainer from LearnOpenCV
"""

import numpy as np
import cv2
import base64
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import MediaPipe fitness module
from .mediapipe_fitness import (
    ProcessFrame,
    get_thresholds_beginner,
    get_thresholds_pro,
    get_mediapipe_pose
)

logger = logging.getLogger(__name__)


class PoseQualityType(str, Enum):
    NONE = "NONE"           # No person detected
    PARTIAL = "PARTIAL"     # Person detected but camera not aligned
    FULL = "FULL"           # Full workout pose visible


class MovementState(str, Enum):
    STATIC = "STATIC"       # No significant movement
    MOVING = "MOVING"       # Active workout


@dataclass
class PoseQuality:
    type: PoseQualityType
    coverage: float = 0.0
    missing_joints: List[str] = field(default_factory=list)
    message: str = ""
    avg_confidence: float = 0.0


@dataclass
class RealtimeAnalysisResult:
    """Result from real-time frame analysis"""
    form_score: float
    feedback: str
    keypoints: List[Dict]  # List of {x, y, confidence}
    issues: List[Dict]     # List of {type, severity, message}
    joint_angles: Dict[str, float]
    processing_time_ms: float
    pose_quality: Dict
    movement_state: str
    reps: int
    correct_reps: int
    incorrect_reps: int
    should_speak: bool
    processed_frame: Optional[str] = None  # Base64 encoded annotated frame


class RealtimeBiomechanicsAnalyzer:
    """
    Real-time biomechanics analyzer using MediaPipe Pose
    Based on AI Fitness Trainer
    """
    
    # Performance settings
    TARGET_WIDTH = 480  # Resize to this width for faster processing
    JPEG_QUALITY = 60   # Lower quality for faster encoding/transfer
    FRAME_SKIP = 2      # Process every Nth frame
    
    def __init__(self):
        """Initialize MediaPipe pose and frame processor"""
        logger.info("Initializing MediaPipe Realtime Biomechanics Analyzer")
        
        # Initialize MediaPipe pose with lighter model
        self.pose = get_mediapipe_pose(
            static_image_mode=False,
            model_complexity=0,  # Use lightest model for speed
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Per-user processors (for state tracking)
        self.user_processors: Dict[str, ProcessFrame] = {}
        
        # Difficulty mode per user
        self.user_modes: Dict[str, str] = {}
        
        # Voice feedback throttling per user
        self.last_feedback_time: Dict[str, float] = {}
        self.last_feedback_text: Dict[str, str] = {}
        self.FEEDBACK_COOLDOWN = 2.0  # seconds
        
        # Frame counter per user (for frame skipping)
        self.user_frame_count: Dict[str, int] = {}
        self.user_last_result: Dict[str, RealtimeAnalysisResult] = {}
        
        logger.info("MediaPipe analyzer initialized successfully")
    
    def get_user_processor(self, user_id: str, mode: str = "beginner") -> ProcessFrame:
        """Get or create processor for user"""
        key = f"{user_id}_{mode}"
        
        if key not in self.user_processors or self.user_modes.get(user_id) != mode:
            # Get thresholds based on mode
            if mode == "pro":
                thresholds = get_thresholds_pro()
            else:
                thresholds = get_thresholds_beginner()
            
            # Create new processor with flip_frame=True for webcam
            self.user_processors[key] = ProcessFrame(thresholds=thresholds, flip_frame=True)
            self.user_modes[user_id] = mode
            logger.info(f"Created processor for user {user_id} in {mode} mode")
        
        return self.user_processors[key]
    
    def should_speak_feedback(self, feedback: str, user_id: str) -> bool:
        """Check if we should speak this feedback (throttling)"""
        now = time.time()
        
        # Check cooldown
        last_time = self.last_feedback_time.get(user_id, 0)
        if now - last_time < self.FEEDBACK_COOLDOWN:
            return False
        
        # Check if same feedback
        last_text = self.last_feedback_text.get(user_id, "")
        if feedback == last_text:
            return False
        
        # Update tracking
        self.last_feedback_time[user_id] = now
        self.last_feedback_text[user_id] = feedback
        
        return True
    
    def process_frame(self, frame_data: str, exercise_type: str, user_id: str, 
                      mode: str = "beginner") -> RealtimeAnalysisResult:
        """
        Process a single frame and return analysis results
        
        Args:
            frame_data: Base64 encoded JPEG image
            exercise_type: Type of exercise (currently only 'squat' supported)
            user_id: Unique user identifier
            mode: 'beginner' or 'pro'
        
        Returns:
            RealtimeAnalysisResult with processed frame and analysis
        """
        start_time = time.time()
        
        # Frame skipping for performance
        frame_count = self.user_frame_count.get(user_id, 0) + 1
        self.user_frame_count[user_id] = frame_count
        
        # Return cached result if skipping this frame
        if frame_count % self.FRAME_SKIP != 0 and user_id in self.user_last_result:
            cached = self.user_last_result[user_id]
            cached.should_speak = False  # Don't repeat voice feedback
            return cached
        
        try:
            # Decode base64 frame
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return self._error_result("Failed to decode frame", start_time)
            
            # Resize for performance (maintain aspect ratio)
            h, w = frame.shape[:2]
            if w > self.TARGET_WIDTH:
                scale = self.TARGET_WIDTH / w
                frame = cv2.resize(frame, (self.TARGET_WIDTH, int(h * scale)), 
                                   interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get user's processor
            processor = self.get_user_processor(user_id, mode)
            
            # Process frame through AI Fitness Trainer
            processed_frame, play_sound = processor.process(frame_rgb, self.pose)
            
            # Convert back to BGR for encoding
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            # Encode processed frame to base64 (lower quality for speed)
            _, buffer = cv2.imencode('.jpg', processed_frame_bgr, 
                                     [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Extract state from processor
            state = processor.state_tracker
            correct_reps = state['SQUAT_COUNT']
            incorrect_reps = state['IMPROPER_SQUAT']
            total_reps = correct_reps + incorrect_reps
            
            # Determine pose quality based on processor state
            pose_quality = self._determine_pose_quality(state, processor)
            
            # Determine movement state
            if state['curr_state'] in ['s2', 's3']:
                movement_state = MovementState.MOVING
            else:
                movement_state = MovementState.STATIC
            
            # Calculate form score
            form_score = self._calculate_form_score(state, correct_reps, incorrect_reps)
            
            # Generate feedback message
            feedback = self._generate_feedback(state, processor, play_sound, pose_quality)
            
            # Generate issues list
            issues = self._generate_issues(state, processor)
            
            # Check if should speak
            should_speak = self.should_speak_feedback(feedback, user_id) if feedback else False
            
            processing_time = (time.time() - start_time) * 1000
            
            result = RealtimeAnalysisResult(
                form_score=round(form_score, 1),
                feedback=feedback,
                keypoints=[],  # Not sending keypoints - drawing on frame instead
                issues=issues,
                joint_angles={},
                processing_time_ms=round(processing_time, 2),
                pose_quality={
                    "type": pose_quality.type.value,
                    "coverage": pose_quality.coverage,
                    "missing_joints": pose_quality.missing_joints,
                    "message": pose_quality.message,
                    "avg_confidence": pose_quality.avg_confidence
                },
                movement_state=movement_state.value,
                reps=total_reps,
                correct_reps=correct_reps,
                incorrect_reps=incorrect_reps,
                should_speak=should_speak,
                processed_frame=processed_frame_b64
            )
            
            # Cache result for frame skipping
            self.user_last_result[user_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return self._error_result(str(e), start_time)
    
    def _determine_pose_quality(self, state: Dict, processor: ProcessFrame) -> PoseQuality:
        """Determine pose quality from processor state"""
        
        # Check if camera not aligned (facing front instead of side)
        if state.get('INACTIVE_TIME_FRONT', 0) > 0 or state['curr_state'] is None:
            # Check if detecting anything based on recent activity
            if state['INACTIVE_TIME'] > processor.thresholds['INACTIVE_THRESH']:
                return PoseQuality(
                    type=PoseQualityType.NONE,
                    coverage=0.0,
                    missing_joints=[],
                    message="No person detected. Step into frame.",
                    avg_confidence=0.0
                )
            else:
                return PoseQuality(
                    type=PoseQualityType.PARTIAL,
                    coverage=0.5,
                    missing_joints=[],
                    message="Turn sideways to the camera for squat analysis.",
                    avg_confidence=0.5
                )
        
        # Full pose detected and aligned
        return PoseQuality(
            type=PoseQualityType.FULL,
            coverage=1.0,
            missing_joints=[],
            message="Good position! Ready for analysis.",
            avg_confidence=0.9
        )
    
    def _calculate_form_score(self, state: Dict, correct: int, incorrect: int) -> float:
        """Calculate form score based on performance"""
        
        # If no reps yet, check posture state
        if correct + incorrect == 0:
            if state['INCORRECT_POSTURE']:
                return 50.0
            elif state['curr_state'] is not None:
                return 75.0  # Ready position
            else:
                return 0.0
        
        # Score based on correct/incorrect ratio
        total = correct + incorrect
        accuracy = correct / total
        
        # Base score from accuracy
        score = accuracy * 100
        
        # Penalty for current bad posture
        if state['INCORRECT_POSTURE']:
            score -= 15
        
        return max(0, min(100, score))
    
    def _generate_feedback(self, state: Dict, processor: ProcessFrame, 
                          play_sound: Optional[str], pose_quality: PoseQuality) -> str:
        """Generate voice feedback message"""
        
        # Priority 1: Only show camera alignment issue if REALLY misaligned (long time)
        if pose_quality.type == PoseQualityType.NONE:
            return "Step into frame and face sideways."
        
        # Don't immediately say "turn sideways" - allow some tolerance
        # Only say it if inactive for a while (handled by INACTIVE_THRESH)
        
        # Priority 2: Sound cue from processor (rep completed or incorrect)
        if play_sound:
            if play_sound == 'incorrect':
                # Get specific feedback about what's wrong
                feedback_messages = processor.get_current_feedback()
                if feedback_messages:
                    return feedback_messages[0]
                return "Incorrect form! Check your posture."
            elif play_sound == 'reset_counters':
                return "Counters reset due to inactivity."
            elif play_sound.isdigit():
                count = int(play_sound)
                return f"Rep {count}! Good job!"
        
        # Priority 3: Current form issues (speak these!)
        display_text = state['DISPLAY_TEXT']
        if len(display_text) > 0 and display_text[0]:
            return "Bend backwards! You're leaning too far forward."
        if len(display_text) > 1 and display_text[1]:
            return "Bend forward! Keep your torso more upright."
        if len(display_text) > 2 and display_text[2]:
            return "Knee falling over toe! Sit back more."
        if len(display_text) > 3 and display_text[3]:
            return "Squat too deep! Don't go below parallel."
        
        # Priority 4: Lower hips feedback
        if state['LOWER_HIPS']:
            return "Lower your hips more."
        
        # Priority 5: State-based feedback (less critical)
        curr_state = state['curr_state']
        if curr_state == 's1':
            return "Good! Ready for next rep."
        elif curr_state == 's2':
            return "Going down, keep form."
        elif curr_state == 's3':
            return "Good depth! Now push up."
        
        return ""
    
    def _generate_issues(self, state: Dict, processor: ProcessFrame) -> List[Dict]:
        """Generate list of current form issues"""
        issues = []
        
        # Check display text flags
        display_text = state['DISPLAY_TEXT']
        
        if len(display_text) > 0 and display_text[0]:
            issues.append({
                "type": "bend_backwards",
                "severity": 0.6,
                "message": "Bend backwards - leaning too far forward",
                "is_critical": False
            })
        
        if len(display_text) > 1 and display_text[1]:
            issues.append({
                "type": "bend_forward",
                "severity": 0.6,
                "message": "Bend forward - keep torso more upright",
                "is_critical": False
            })
        
        if len(display_text) > 2 and display_text[2]:
            issues.append({
                "type": "knee_over_toe",
                "severity": 0.8,
                "message": "Knee falling over toe - sit back more",
                "is_critical": True
            })
        
        if len(display_text) > 3 and display_text[3]:
            issues.append({
                "type": "too_deep",
                "severity": 0.7,
                "message": "Squat too deep - don't go below parallel",
                "is_critical": True
            })
        
        if state['LOWER_HIPS']:
            issues.append({
                "type": "insufficient_depth",
                "severity": 0.5,
                "message": "Lower your hips for proper depth",
                "is_critical": False
            })
        
        return issues
    
    def _error_result(self, error_msg: str, start_time: float) -> RealtimeAnalysisResult:
        """Return error result"""
        processing_time = (time.time() - start_time) * 1000
        
        return RealtimeAnalysisResult(
            form_score=0,
            feedback=error_msg,
            keypoints=[],
            issues=[],
            joint_angles={},
            processing_time_ms=round(processing_time, 2),
            pose_quality={
                "type": PoseQualityType.NONE.value,
                "coverage": 0,
                "missing_joints": [],
                "message": error_msg,
                "avg_confidence": 0
            },
            movement_state=MovementState.STATIC.value,
            reps=0,
            correct_reps=0,
            incorrect_reps=0,
            should_speak=False,
            processed_frame=None
        )
    
    def reset_user_state(self, user_id: str):
        """Reset a user's tracking state"""
        keys_to_remove = [k for k in self.user_processors.keys() if k.startswith(user_id)]
        for key in keys_to_remove:
            del self.user_processors[key]
        
        if user_id in self.user_modes:
            del self.user_modes[user_id]
        if user_id in self.last_feedback_time:
            del self.last_feedback_time[user_id]
        if user_id in self.last_feedback_text:
            del self.last_feedback_text[user_id]
        
        logger.info(f"Reset state for user {user_id}")


# Global analyzer instance (singleton)
_analyzer_instance = None

def get_analyzer() -> RealtimeBiomechanicsAnalyzer:
    """Get or create the global analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = RealtimeBiomechanicsAnalyzer()
    return _analyzer_instance
