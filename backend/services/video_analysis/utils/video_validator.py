"""
Video Validation Utilities
==========================
Validates uploaded videos to ensure they contain workout content.
"""

import logging
import numpy as np
from typing import Tuple, Optional, List
from scipy import signal
from scipy.fft import fft

from app.core.config import VALIDATION_THRESHOLDS, KEYPOINT_INDICES

logger = logging.getLogger(__name__)


class InvalidWorkoutVideoError(Exception):
    """Raised when a video is determined to not be a workout video."""
    
    def __init__(self, message: str, reason_code: str = "INVALID_VIDEO"):
        self.message = message
        self.reason_code = reason_code
        super().__init__(self.message)


class VideoValidator:
    """
    Validates videos to ensure they contain actual workout content.
    
    Uses multiple heuristics:
    1. Pose detection rate - are humans consistently detected?
    2. Movement variance - is there meaningful movement?
    3. Periodicity detection - is the movement repetitive (like exercise)?
    """
    
    def __init__(self):
        self.min_detection_rate = VALIDATION_THRESHOLDS["min_pose_detection_rate"]
        self.min_movement_variance = VALIDATION_THRESHOLDS["min_movement_variance"]
        self.min_periodicity_score = VALIDATION_THRESHOLDS["min_periodicity_score"]
        self.min_frames = VALIDATION_THRESHOLDS["min_frames_for_analysis"]
    
    def validate_video(
        self,
        keypoints_sequence: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        frame_count: int = 0
    ) -> Tuple[bool, str]:
        """
        Validate if the video contains workout content.
        
        Args:
            keypoints_sequence: Array of shape (num_frames, num_keypoints, 2) with xy coordinates
            confidences: Optional array of shape (num_frames, num_keypoints) with confidence scores
            frame_count: Total number of frames in the video
            
        Returns:
            Tuple of (is_valid, message)
            
        Raises:
            InvalidWorkoutVideoError: If the video is not a valid workout video
        """
        logger.info(f"Validating video with {len(keypoints_sequence)} pose frames out of {frame_count} total")
        
        # Check 1: Minimum frames
        if len(keypoints_sequence) < self.min_frames:
            raise InvalidWorkoutVideoError(
                f"Video too short. Need at least {self.min_frames} frames with detected poses.",
                "VIDEO_TOO_SHORT"
            )
        
        # Check 2: Pose detection rate
        detection_rate = self._calculate_detection_rate(keypoints_sequence, confidences, frame_count)
        if detection_rate < self.min_detection_rate:
            raise InvalidWorkoutVideoError(
                f"Could not detect a person consistently in the video. Detection rate: {detection_rate:.1%}",
                "LOW_POSE_DETECTION"
            )
        logger.info(f"Pose detection rate: {detection_rate:.1%}")
        
        # Check 3: Movement variance
        movement_variance = self._calculate_movement_variance(keypoints_sequence)
        if movement_variance < self.min_movement_variance:
            raise InvalidWorkoutVideoError(
                "No significant movement detected. Please upload a video of you performing an exercise.",
                "NO_MOVEMENT"
            )
        logger.info(f"Movement variance: {movement_variance:.4f}")
        
        # Check 4: Periodicity (repetitive motion typical of exercise)
        periodicity_score = self._calculate_periodicity(keypoints_sequence)
        if periodicity_score < self.min_periodicity_score:
            logger.warning(f"Low periodicity score: {periodicity_score:.2f}. Video may not be an exercise.")
            # This is a soft warning, not a hard rejection
            # Some exercises (like holds) don't have periodic motion
        
        logger.info(f"Video validation passed. Periodicity score: {periodicity_score:.2f}")
        return True, "Video validated successfully"
    
    def _calculate_detection_rate(
        self,
        keypoints_sequence: np.ndarray,
        confidences: Optional[np.ndarray],
        total_frames: int
    ) -> float:
        """Calculate what percentage of frames have valid pose detection."""
        if total_frames == 0:
            total_frames = len(keypoints_sequence)
        
        if confidences is not None:
            # Use confidence scores to determine valid detections
            # A frame is valid if at least 8 keypoints have confidence > 0.3
            valid_frames = np.sum(np.sum(confidences > 0.3, axis=1) >= 8)
        else:
            # Without confidence, check if keypoints have non-zero values
            # A frame is valid if at least 8 keypoints have non-zero coordinates
            non_zero_kpts = np.sum(np.any(keypoints_sequence != 0, axis=2), axis=1)
            valid_frames = np.sum(non_zero_kpts >= 8)
        
        return valid_frames / total_frames if total_frames > 0 else 0.0
    
    def _calculate_movement_variance(self, keypoints_sequence: np.ndarray) -> float:
        """Calculate overall movement variance across all keypoints."""
        if len(keypoints_sequence) < 2:
            return 0.0
        
        # Calculate frame-to-frame displacement for all keypoints
        displacements = np.diff(keypoints_sequence, axis=0)
        
        # Calculate the magnitude of displacement for each keypoint per frame
        displacement_magnitudes = np.sqrt(np.sum(displacements ** 2, axis=2))
        
        # Normalize by image dimensions (assume 640x480 as standard)
        normalized_displacements = displacement_magnitudes / np.sqrt(640**2 + 480**2)
        
        # Return mean variance across all keypoints
        return float(np.var(normalized_displacements))
    
    def _calculate_periodicity(self, keypoints_sequence: np.ndarray) -> float:
        """
        Calculate periodicity score using Fourier analysis.
        
        Looks for repetitive motion patterns typical of exercise.
        Uses the y-coordinate of hip joints as the primary signal.
        """
        if len(keypoints_sequence) < 30:
            return 0.0
        
        try:
            # Extract y-coordinates of hip joints (indices 11 and 12)
            left_hip_y = keypoints_sequence[:, KEYPOINT_INDICES["left_hip"], 1]
            right_hip_y = keypoints_sequence[:, KEYPOINT_INDICES["right_hip"], 1]
            
            # Use average of both hips
            hip_y = (left_hip_y + right_hip_y) / 2
            
            # Remove DC component (mean)
            hip_y_centered = hip_y - np.mean(hip_y)
            
            # Handle zero or constant signal
            if np.std(hip_y_centered) < 1e-6:
                # Try shoulder movement instead
                left_shoulder_y = keypoints_sequence[:, KEYPOINT_INDICES["left_shoulder"], 1]
                right_shoulder_y = keypoints_sequence[:, KEYPOINT_INDICES["right_shoulder"], 1]
                shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                hip_y_centered = shoulder_y - np.mean(shoulder_y)
                
                if np.std(hip_y_centered) < 1e-6:
                    return 0.0
            
            # Apply FFT
            n = len(hip_y_centered)
            fft_result = fft(hip_y_centered)
            frequencies = np.fft.fftfreq(n)
            
            # Get power spectrum (magnitude squared)
            power = np.abs(fft_result[:n//2]) ** 2
            positive_freqs = frequencies[:n//2]
            
            # Ignore DC and very low frequencies (< 0.02 Hz)
            # Focus on typical exercise rep frequencies (0.1 - 1 Hz for most exercises)
            freq_mask = (positive_freqs > 0.02) & (positive_freqs < 0.5)
            
            if not np.any(freq_mask):
                return 0.0
            
            filtered_power = power[freq_mask]
            
            # Calculate periodicity score as ratio of peak to total energy
            if np.sum(filtered_power) > 0:
                peak_power = np.max(filtered_power)
                periodicity_score = peak_power / np.sum(filtered_power)
            else:
                periodicity_score = 0.0
            
            return float(min(periodicity_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating periodicity: {e}")
            return 0.0
    
    def get_validation_metrics(
        self,
        keypoints_sequence: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        frame_count: int = 0
    ) -> dict:
        """Get all validation metrics without raising exceptions."""
        if frame_count == 0:
            frame_count = len(keypoints_sequence)
            
        return {
            "detection_rate": self._calculate_detection_rate(keypoints_sequence, confidences, frame_count),
            "movement_variance": self._calculate_movement_variance(keypoints_sequence),
            "periodicity_score": self._calculate_periodicity(keypoints_sequence),
            "frame_count": len(keypoints_sequence),
            "total_frames": frame_count
        }


def is_workout_video(
    keypoints_sequence: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    frame_count: int = 0
) -> bool:
    """
    Simple function interface for video validation.
    
    Args:
        keypoints_sequence: Array of keypoint coordinates
        confidences: Optional confidence scores
        frame_count: Total frames in video
        
    Returns:
        True if video is a valid workout video
        
    Raises:
        InvalidWorkoutVideoError: If video is not valid
    """
    validator = VideoValidator()
    is_valid, _ = validator.validate_video(keypoints_sequence, confidences, frame_count)
    return is_valid


def quick_validate(
    pose_detection_rate: float,
    has_movement: bool
) -> bool:
    """
    Quick validation without full analysis.
    
    Args:
        pose_detection_rate: Rate of successful pose detection (0-1)
        has_movement: Whether significant movement was detected
        
    Returns:
        True if passes quick validation
    """
    min_detection = VALIDATION_THRESHOLDS["min_pose_detection_rate"]
    return pose_detection_rate >= min_detection and has_movement
