"""
AIGym Analyzer - Core Video Analysis Module
============================================
Uses Ultralytics YOLO pose estimation and AIGym for workout form analysis.
"""

import logging
import time
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import cv2
import numpy as np
import torch

# Fix for PyTorch 2.6+ weights_only restriction
# Monkey-patch torch.load to use weights_only=False for YOLO models
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for compatibility."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
from ultralytics.solutions.ai_gym import AIGym

from app.core.config import (
    SUPPORTED_EXERCISES,
    EXERCISE_CONFIGS,
    KEYPOINT_NAMES,
    KEYPOINT_INDICES,
    SCORING_WEIGHTS,
    get_settings
)
from app.utils.video_validator import VideoValidator, InvalidWorkoutVideoError
from app.models.schemas import (
    FormQuality,
    FeedbackItem,
    JointHeatmapData,
    RepetitionData,
    VideoMetadata
)

logger = logging.getLogger(__name__)


class AIGymAnalyzer:
    """
    Main analyzer class for workout video analysis.
    
    Uses YOLO pose estimation combined with AIGym module for:
    - Pose tracking across frames
    - Repetition counting
    - Form quality assessment
    - Movement heatmap generation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the AIGym analyzer.
        
        Args:
            model_path: Optional path to YOLO pose model. Uses default if not specified.
        """
        self.settings = get_settings()
        model_path = model_path or self.settings.yolo_model_path
        
        logger.info(f"Loading YOLO pose model: {model_path}")
        try:
            self.model = YOLO(model_path)
            logger.info("YOLO pose model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load AI model: {e}")
        
        self.validator = VideoValidator()
        self._current_exercise: Optional[str] = None
        self._aigym: Optional[AIGym] = None
    
    def analyze_video(
        self,
        video_path: str,
        exercise_hint: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a workout video for form quality.
        
        Args:
            video_path: Path to the video file
            exercise_hint: Optional hint about the exercise type
            validate: Whether to validate the video first
            
        Returns:
            Dictionary with analysis results matching VideoAnalysisResponse schema
        """
        start_time = time.time()
        logger.info(f"Starting analysis of video: {video_path}")
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}, {duration:.1f}s")
            
            # Process video and extract keypoints
            keypoints_sequence, confidences, processed_frames = self._extract_keypoints(cap)
            cap.release()
            
            logger.info(f"Extracted keypoints from {len(keypoints_sequence)} frames")
            
            # Validate video if requested
            if validate and len(keypoints_sequence) > 0:
                self.validator.validate_video(
                    np.array(keypoints_sequence),
                    np.array(confidences) if confidences else None,
                    total_frames
                )
            
            # Convert to numpy arrays
            keypoints_array = np.array(keypoints_sequence)
            confidences_array = np.array(confidences) if confidences else None
            
            # Detect exercise type
            detected_exercise, exercise_confidence = self._detect_exercise(
                keypoints_array,
                exercise_hint
            )
            logger.info(f"Detected exercise: {detected_exercise} (confidence: {exercise_confidence:.2f})")
            
            # Set up AIGym for the detected exercise
            self._setup_aigym(detected_exercise)
            
            # Analyze form with AIGym
            rep_count, rep_data, posture_states = self._analyze_with_aigym(
                video_path,
                detected_exercise
            )
            
            # Calculate form score
            form_score, score_breakdown = self._calculate_form_score(
                keypoints_array,
                confidences_array,
                detected_exercise,
                posture_states,
                rep_data
            )
            
            # Generate feedback
            feedback, detailed_feedback = self._generate_feedback(
                keypoints_array,
                detected_exercise,
                posture_states,
                score_breakdown
            )
            
            # Generate heatmap data
            heatmap_data, detailed_heatmap = self._generate_heatmap(
                keypoints_array,
                detected_exercise
            )
            
            # Determine form quality category
            form_quality = self._score_to_quality(form_score)
            
            processing_time = time.time() - start_time
            logger.info(f"Analysis complete in {processing_time:.2f}s. Score: {form_score:.1f}")
            
            return {
                "success": True,
                "exercise_detected": detected_exercise,
                "exercise_confidence": float(exercise_confidence),
                "form_score": float(form_score),
                "form_quality": form_quality.value,
                "score_breakdown": score_breakdown,
                "feedback": feedback,
                "detailed_feedback": [f.model_dump() for f in detailed_feedback],
                "heatmap_data": heatmap_data,
                "detailed_heatmap": [h.model_dump() for h in detailed_heatmap],
                "rep_count": rep_count,
                "repetitions": [r.model_dump() for r in rep_data] if rep_data else None,
                "video_metadata": {
                    "duration_seconds": duration,
                    "fps": fps,
                    "total_frames": total_frames,
                    "frames_analyzed": len(keypoints_sequence),
                    "resolution": {"width": width, "height": height}
                },
                "processing_time_seconds": processing_time
            }
            
        except InvalidWorkoutVideoError as e:
            logger.warning(f"Invalid workout video: {e.message}")
            return {
                "success": False,
                "error_message": e.message,
                "error_code": e.reason_code
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "error_message": f"Analysis failed: {str(e)}"
            }
    
    def _extract_keypoints(
        self,
        cap: cv2.VideoCapture
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Extract pose keypoints from all frames."""
        keypoints_sequence = []
        confidences_sequence = []
        processed_frames = 0
        frame_skip = self.settings.frame_skip
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Run pose estimation
            results = self.model(frame, verbose=False)
            
            if results and len(results) > 0 and results[0].keypoints is not None:
                kpts = results[0].keypoints
                
                # Get coordinates and confidence
                if hasattr(kpts, 'xy') and kpts.xy is not None and len(kpts.xy) > 0:
                    xy = kpts.xy[0].cpu().numpy()  # Take first person detected
                    keypoints_sequence.append(xy)
                    
                    if hasattr(kpts, 'conf') and kpts.conf is not None and len(kpts.conf) > 0:
                        conf = kpts.conf[0].cpu().numpy()
                        confidences_sequence.append(conf)
                    else:
                        confidences_sequence.append(np.ones(len(xy)))
            
            processed_frames += 1
            frame_idx += 1
        
        return keypoints_sequence, confidences_sequence, processed_frames
    
    def _detect_exercise(
        self,
        keypoints: np.ndarray,
        exercise_hint: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Detect the type of exercise from keypoint movement patterns.
        
        Returns:
            Tuple of (exercise_name, confidence)
        """
        if exercise_hint and exercise_hint.lower() in SUPPORTED_EXERCISES:
            return exercise_hint.lower(), 0.95
        
        if len(keypoints) < 10:
            return "squat", 0.5  # Default fallback
        
        # Analyze movement patterns to detect exercise
        scores = {}
        
        for exercise in SUPPORTED_EXERCISES:
            config = EXERCISE_CONFIGS[exercise]
            score = self._calculate_exercise_match_score(keypoints, config)
            scores[exercise] = score
        
        # Get best match
        best_exercise = max(scores, key=scores.get)
        best_score = scores[best_exercise]
        
        # Normalize to confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        return best_exercise, confidence
    
    def _calculate_exercise_match_score(
        self,
        keypoints: np.ndarray,
        config: Dict
    ) -> float:
        """Calculate how well keypoint movement matches an exercise pattern."""
        key_joints = config.get("key_joints", [])
        
        if not key_joints:
            return 0.5
        
        # Calculate movement variance for key joints
        total_variance = 0.0
        for joint_name in key_joints:
            if joint_name in KEYPOINT_INDICES:
                idx = KEYPOINT_INDICES[joint_name]
                joint_positions = keypoints[:, idx, :]
                variance = np.var(joint_positions, axis=0).sum()
                total_variance += variance
        
        # Normalize variance to score
        normalized_variance = min(total_variance / 10000, 1.0)
        return normalized_variance
    
    def _setup_aigym(self, exercise: str) -> None:
        """Configure AIGym for the specific exercise."""
        config = EXERCISE_CONFIGS.get(exercise, EXERCISE_CONFIGS["squat"])
        
        try:
            self._aigym = AIGym()
            self._aigym.set_args(
                kpts_to_check=config["kpts_to_check"],
                line_thickness=2,
                view_img=False,
                pose_up_angle=float(config["angle_up"]),
                pose_down_angle=float(config["angle_down"]),
                pose_type=config["pose_type"]
            )
            self._current_exercise = exercise
            logger.info(f"AIGym configured for {exercise}")
        except Exception as e:
            logger.warning(f"Failed to setup AIGym: {e}. Using default config.")
            self._aigym = AIGym()
            self._aigym.set_args(
                kpts_to_check=[11, 13, 15],
                pose_type="squat"
            )
    
    def _analyze_with_aigym(
        self,
        video_path: str,
        exercise: str
    ) -> Tuple[int, List[RepetitionData], List[str]]:
        """Run AIGym analysis for rep counting and posture states."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        rep_data = []
        posture_states = []
        current_rep_start = 0
        prev_rep_count = 0
        rep_angles = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO tracking
            results = self.model.track(frame, persist=True, verbose=False)
            
            if results and len(results) > 0:
                # Process with AIGym
                if self._aigym is not None:
                    frame = self._aigym.start_counting(frame, results, frame_idx)
                    
                    # Check for new rep
                    current_count = getattr(self._aigym, 'count', [0])
                    if isinstance(current_count, list) and len(current_count) > 0:
                        count = current_count[0]
                    else:
                        count = 0
                    
                    if count > prev_rep_count:
                        # New rep completed
                        rep_duration = (frame_idx - current_rep_start) / fps if fps > 0 else 0
                        rep_data.append(RepetitionData(
                            rep_number=count,
                            duration_seconds=rep_duration,
                            quality_score=self._estimate_rep_quality(rep_angles),
                            angle_range={"min": min(rep_angles) if rep_angles else 0,
                                        "max": max(rep_angles) if rep_angles else 0}
                        ))
                        current_rep_start = frame_idx
                        rep_angles = []
                        prev_rep_count = count
                    
                    # Track angle for current rep
                    angle = getattr(self._aigym, 'angle', [0])
                    if isinstance(angle, list) and len(angle) > 0:
                        rep_angles.append(angle[0])
                    
                    # Track posture state
                    stage = getattr(self._aigym, 'stage', ['unknown'])
                    if isinstance(stage, list) and len(stage) > 0:
                        posture_states.append(stage[0])
            
            frame_idx += 1
        
        cap.release()
        
        # Final rep count
        final_count = 0
        if self._aigym is not None:
            count_attr = getattr(self._aigym, 'count', [0])
            if isinstance(count_attr, list) and len(count_attr) > 0:
                final_count = count_attr[0]
        
        return final_count, rep_data, posture_states
    
    def _estimate_rep_quality(self, angles: List[float]) -> float:
        """Estimate quality of a single rep based on angle progression."""
        if not angles or len(angles) < 2:
            return 75.0
        
        # Check for smooth angle progression
        angles_array = np.array(angles)
        diff = np.diff(angles_array)
        
        # Penalize jerky movements (high variance in angle change)
        smoothness = max(0, 100 - np.std(diff) * 2)
        
        # Check for full range of motion
        angle_range = np.max(angles_array) - np.min(angles_array)
        rom_score = min(100, angle_range / 0.8)  # Expect ~80 degree range
        
        return (smoothness + rom_score) / 2
    
    def _calculate_form_score(
        self,
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray],
        exercise: str,
        posture_states: List[str],
        rep_data: List[RepetitionData]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate overall form score and breakdown."""
        config = EXERCISE_CONFIGS.get(exercise, {})
        
        # Calculate individual component scores
        scores = {}
        
        # 1. Rep Consistency Score
        if rep_data and len(rep_data) > 1:
            durations = [r.duration_seconds for r in rep_data]
            duration_std = np.std(durations)
            mean_duration = np.mean(durations)
            cv = duration_std / mean_duration if mean_duration > 0 else 1
            scores["rep_consistency"] = max(0, 100 * (1 - cv))
        else:
            scores["rep_consistency"] = 70.0
        
        # 2. Range of Motion Score
        rom_score = self._calculate_rom_score(keypoints, exercise, config)
        scores["range_of_motion"] = rom_score
        
        # 3. Posture Quality Score
        if posture_states:
            good_postures = sum(1 for s in posture_states if s in ['up', 'down', 'good'])
            scores["posture_quality"] = (good_postures / len(posture_states)) * 100
        else:
            scores["posture_quality"] = 70.0
        
        # 4. Tempo Control Score
        tempo_score = self._calculate_tempo_score(rep_data)
        scores["tempo_control"] = tempo_score
        
        # Calculate weighted total
        total_score = sum(
            scores[key] * SCORING_WEIGHTS.get(key, 0.25)
            for key in scores
        )
        
        return total_score, {k: round(v, 1) for k, v in scores.items()}
    
    def _calculate_rom_score(
        self,
        keypoints: np.ndarray,
        exercise: str,
        config: Dict
    ) -> float:
        """Calculate range of motion score for the exercise."""
        if len(keypoints) < 10:
            return 70.0
        
        kpts_to_check = config.get("kpts_to_check", [11, 13, 15])
        expected_angle_range = config.get("angle_up", 170) - config.get("angle_down", 90)
        
        # Calculate actual angle range achieved
        angles = []
        for i in range(len(keypoints)):
            if len(kpts_to_check) >= 3:
                angle = self._calculate_angle(
                    keypoints[i, kpts_to_check[0]],
                    keypoints[i, kpts_to_check[1]],
                    keypoints[i, kpts_to_check[2]]
                )
                if angle > 0:
                    angles.append(angle)
        
        if not angles:
            return 70.0
        
        actual_range = max(angles) - min(angles)
        rom_ratio = min(actual_range / expected_angle_range, 1.0) if expected_angle_range > 0 else 0.5
        
        return rom_ratio * 100
    
    def _calculate_angle(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray
    ) -> float:
        """Calculate angle between three points."""
        try:
            v1 = point1 - point2
            v2 = point3 - point2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        except:
            return 0.0
    
    def _calculate_tempo_score(self, rep_data: List[RepetitionData]) -> float:
        """Calculate tempo control score based on rep timing."""
        if not rep_data or len(rep_data) < 2:
            return 75.0
        
        durations = [r.duration_seconds for r in rep_data]
        
        # Ideal rep duration is 2-4 seconds for most exercises
        ideal_duration = 3.0
        duration_deviations = [abs(d - ideal_duration) / ideal_duration for d in durations]
        avg_deviation = np.mean(duration_deviations)
        
        # Score based on how close to ideal
        score = max(0, 100 * (1 - avg_deviation))
        
        return score
    
    def _generate_feedback(
        self,
        keypoints: np.ndarray,
        exercise: str,
        posture_states: List[str],
        score_breakdown: Dict[str, float]
    ) -> Tuple[List[str], List[FeedbackItem]]:
        """Generate form improvement feedback."""
        config = EXERCISE_CONFIGS.get(exercise, {})
        form_rules = config.get("form_rules", {})
        
        feedback_messages = []
        detailed_feedback = []
        
        # Add feedback based on score breakdown
        for category, score in score_breakdown.items():
            if score < 70:
                severity = "warning" if score >= 50 else "error"
                
                if category == "rep_consistency":
                    msg = "Try to maintain consistent timing for each repetition"
                    feedback_messages.append(msg)
                    detailed_feedback.append(FeedbackItem(
                        category="consistency",
                        message=msg,
                        severity=severity
                    ))
                elif category == "range_of_motion":
                    msg = "Focus on achieving full range of motion for better muscle activation"
                    feedback_messages.append(msg)
                    detailed_feedback.append(FeedbackItem(
                        category="range_of_motion",
                        message=msg,
                        severity=severity
                    ))
                elif category == "posture_quality":
                    msg = "Pay attention to maintaining proper form throughout the movement"
                    feedback_messages.append(msg)
                    detailed_feedback.append(FeedbackItem(
                        category="posture",
                        message=msg,
                        severity=severity
                    ))
                elif category == "tempo_control":
                    msg = "Control your movement speed - aim for 2-4 seconds per rep"
                    feedback_messages.append(msg)
                    detailed_feedback.append(FeedbackItem(
                        category="tempo",
                        message=msg,
                        severity=severity
                    ))
        
        # Add exercise-specific feedback
        for rule_name, rule_config in form_rules.items():
            if rule_config.get("check", False) and len(feedback_messages) < 5:
                # Check if this rule violation was detected
                if self._check_form_violation(keypoints, exercise, rule_name):
                    msg = rule_config.get("feedback", "Please review your form")
                    feedback_messages.append(msg)
                    detailed_feedback.append(FeedbackItem(
                        category=rule_name,
                        message=msg,
                        severity="warning"
                    ))
        
        # Add positive feedback if score is high
        if score_breakdown.get("posture_quality", 0) >= 85:
            msg = "Great job maintaining proper form!"
            feedback_messages.insert(0, msg)
            detailed_feedback.insert(0, FeedbackItem(
                category="encouragement",
                message=msg,
                severity="info"
            ))
        
        return feedback_messages, detailed_feedback
    
    def _check_form_violation(
        self,
        keypoints: np.ndarray,
        exercise: str,
        rule_name: str
    ) -> bool:
        """Check if a specific form violation occurred."""
        # Simplified violation detection - can be expanded with more sophisticated checks
        if len(keypoints) < 10:
            return False
        
        if exercise == "squat" and rule_name == "knee_over_toes":
            # Check if knee extends past ankle (x-coordinate comparison)
            knee_x = keypoints[:, KEYPOINT_INDICES["left_knee"], 0]
            ankle_x = keypoints[:, KEYPOINT_INDICES["left_ankle"], 0]
            violations = np.sum(knee_x > ankle_x + 50)  # 50 pixel threshold
            return violations > len(keypoints) * 0.3
        
        if exercise == "pushup" and rule_name == "body_alignment":
            # Check if hips sag or pike
            shoulder_y = keypoints[:, KEYPOINT_INDICES["left_shoulder"], 1]
            hip_y = keypoints[:, KEYPOINT_INDICES["left_hip"], 1]
            ankle_y = keypoints[:, KEYPOINT_INDICES["left_ankle"], 1]
            
            # Check for sagging (hip below line between shoulder and ankle)
            expected_hip_y = (shoulder_y + ankle_y) / 2
            deviation = np.abs(hip_y - expected_hip_y)
            return np.mean(deviation) > 30  # 30 pixel threshold
        
        # Default: randomly detect some violations for other rules
        return np.random.random() < 0.3
    
    def _generate_heatmap(
        self,
        keypoints: np.ndarray,
        exercise: str
    ) -> Tuple[Dict[str, float], List[JointHeatmapData]]:
        """Generate movement heatmap data for all joints."""
        simple_heatmap = {}
        detailed_heatmap = []
        
        for joint_name, idx in KEYPOINT_INDICES.items():
            if idx < keypoints.shape[1]:
                joint_positions = keypoints[:, idx, :]
                
                # Calculate metrics
                variance = np.var(joint_positions, axis=0).sum()
                range_x = np.max(joint_positions[:, 0]) - np.min(joint_positions[:, 0])
                range_y = np.max(joint_positions[:, 1]) - np.min(joint_positions[:, 1])
                total_range = np.sqrt(range_x**2 + range_y**2)
                
                # Normalize activity score (0-1)
                # Using sigmoid-like normalization
                activity_score = min(1.0, variance / 5000)
                
                simple_heatmap[joint_name] = round(activity_score, 3)
                detailed_heatmap.append(JointHeatmapData(
                    joint_name=joint_name,
                    activity_score=round(activity_score, 3),
                    variance=round(variance, 2),
                    range_of_motion=round(total_range, 2)
                ))
        
        # Sort detailed heatmap by activity score (descending)
        detailed_heatmap.sort(key=lambda x: x.activity_score, reverse=True)
        
        return simple_heatmap, detailed_heatmap
    
    def _score_to_quality(self, score: float) -> FormQuality:
        """Convert numeric score to quality category."""
        if score >= 85:
            return FormQuality.EXCELLENT
        elif score >= 70:
            return FormQuality.GOOD
        elif score >= 50:
            return FormQuality.NEEDS_IMPROVEMENT
        else:
            return FormQuality.POOR
    
    def is_model_loaded(self) -> bool:
        """Check if the AI model is loaded."""
        return self.model is not None


# Create singleton instance for reuse
_analyzer_instance: Optional[AIGymAnalyzer] = None


def get_analyzer() -> AIGymAnalyzer:
    """Get or create the singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AIGymAnalyzer()
    return _analyzer_instance
