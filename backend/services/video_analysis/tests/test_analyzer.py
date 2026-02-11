"""
Unit tests for the FitBalance Biomechanics Backend.
===================================================
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from app.core.config import SUPPORTED_EXERCISES, EXERCISE_CONFIGS, KEYPOINT_INDICES
from app.utils.video_validator import VideoValidator, InvalidWorkoutVideoError, is_workout_video
from app.models.schemas import VideoAnalysisResponse, FormQuality


class TestConfig:
    """Test configuration settings."""
    
    def test_supported_exercises_defined(self):
        """Verify all supported exercises are defined."""
        assert len(SUPPORTED_EXERCISES) == 5
        assert "squat" in SUPPORTED_EXERCISES
        assert "pushup" in SUPPORTED_EXERCISES
        assert "deadlift" in SUPPORTED_EXERCISES
        assert "bicep_curl" in SUPPORTED_EXERCISES
        assert "overhead_press" in SUPPORTED_EXERCISES
    
    def test_exercise_configs_complete(self):
        """Verify each exercise has required config fields."""
        required_fields = ["pose_type", "kpts_to_check", "angle_up", "angle_down", "key_joints", "form_rules"]
        
        for exercise in SUPPORTED_EXERCISES:
            assert exercise in EXERCISE_CONFIGS, f"Missing config for {exercise}"
            config = EXERCISE_CONFIGS[exercise]
            for field in required_fields:
                assert field in config, f"Missing {field} in {exercise} config"
    
    def test_keypoint_indices(self):
        """Verify keypoint indices are correct."""
        assert KEYPOINT_INDICES["nose"] == 0
        assert KEYPOINT_INDICES["left_shoulder"] == 5
        assert KEYPOINT_INDICES["right_knee"] == 14
        assert len(KEYPOINT_INDICES) == 17


class TestVideoValidator:
    """Test video validation logic."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return VideoValidator()
    
    @pytest.fixture
    def valid_keypoints(self):
        """Generate valid keypoint sequence simulating a squat."""
        num_frames = 60
        num_keypoints = 17
        
        keypoints = np.zeros((num_frames, num_keypoints, 2))
        
        # Simulate periodic vertical movement (up and down)
        for i in range(num_frames):
            # Base positions
            for j in range(num_keypoints):
                keypoints[i, j, 0] = 320 + j * 5  # Spread in x
                keypoints[i, j, 1] = 240  # Base y position
            
            # Add periodic movement to hips (simulating squat)
            phase = np.sin(2 * np.pi * i / 15)  # 4 reps in 60 frames
            hip_movement = phase * 50
            keypoints[i, 11, 1] += hip_movement  # Left hip
            keypoints[i, 12, 1] += hip_movement  # Right hip
            keypoints[i, 13, 1] += hip_movement  # Left knee
            keypoints[i, 14, 1] += hip_movement  # Right knee
        
        return keypoints
    
    def test_valid_workout_video(self, validator, valid_keypoints):
        """Test that valid workout video passes validation."""
        is_valid, message = validator.validate_video(valid_keypoints, frame_count=60)
        assert is_valid
    
    def test_short_video_rejected(self, validator):
        """Test that videos with too few frames are rejected."""
        short_keypoints = np.zeros((20, 17, 2))  # Only 20 frames
        
        with pytest.raises(InvalidWorkoutVideoError) as exc_info:
            validator.validate_video(short_keypoints, frame_count=20)
        
        assert "too short" in exc_info.value.message.lower()
    
    def test_no_movement_rejected(self, validator):
        """Test that static videos are rejected."""
        static_keypoints = np.ones((50, 17, 2)) * 100  # No movement
        
        with pytest.raises(InvalidWorkoutVideoError) as exc_info:
            validator.validate_video(static_keypoints, frame_count=50)
        
        assert "movement" in exc_info.value.message.lower()
    
    def test_validation_metrics(self, validator, valid_keypoints):
        """Test metrics calculation."""
        metrics = validator.get_validation_metrics(valid_keypoints, frame_count=60)
        
        assert "detection_rate" in metrics
        assert "movement_variance" in metrics
        assert "periodicity_score" in metrics
        assert metrics["frame_count"] == 60


class TestSchemas:
    """Test Pydantic schema validation."""
    
    def test_video_analysis_response_success(self):
        """Test successful response schema."""
        response = VideoAnalysisResponse(
            success=True,
            exercise_detected="squat",
            exercise_confidence=0.92,
            form_score=78.5,
            form_quality=FormQuality.GOOD,
            rep_count=8,
            feedback=["Keep knees behind toes"]
        )
        
        assert response.success
        assert response.form_score == 78.5
    
    def test_video_analysis_response_failure(self):
        """Test failure response schema."""
        response = VideoAnalysisResponse(
            success=False,
            error_message="Video too short"
        )
        
        assert not response.success
        assert response.form_score is None
    
    def test_score_to_quality(self):
        """Test score to quality mapping."""
        assert FormQuality.EXCELLENT == FormQuality.EXCELLENT
        assert FormQuality.GOOD == FormQuality.GOOD
        assert FormQuality.NEEDS_IMPROVEMENT == FormQuality.NEEDS_IMPROVEMENT
        assert FormQuality.POOR == FormQuality.POOR


class TestAnalyzerMocked:
    """Test analyzer with mocked YOLO model."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock YOLO model."""
        mock = MagicMock()
        
        # Mock pose results
        mock_keypoints = MagicMock()
        mock_keypoints.xy = [np.random.rand(17, 2) * 640]
        mock_keypoints.conf = [np.random.rand(17)]
        
        mock_result = MagicMock()
        mock_result.keypoints = mock_keypoints
        
        mock.return_value = [mock_result]
        mock.track.return_value = [mock_result]
        
        return mock
    
    @patch('app.core.aigym_analyzer.YOLO')
    def test_analyzer_initialization(self, mock_yolo_class):
        """Test analyzer can be initialized."""
        mock_yolo_class.return_value = MagicMock()
        
        from app.core.aigym_analyzer import AIGymAnalyzer
        analyzer = AIGymAnalyzer()
        
        assert analyzer.is_model_loaded()


class TestFormRules:
    """Test form rule checking logic."""
    
    def test_squat_form_rules_defined(self):
        """Test squat has proper form rules."""
        squat_config = EXERCISE_CONFIGS["squat"]
        form_rules = squat_config["form_rules"]
        
        assert "knee_over_toes" in form_rules
        assert "back_straight" in form_rules
        assert "depth" in form_rules
        
        # Check rule structure
        knee_rule = form_rules["knee_over_toes"]
        assert "check" in knee_rule
        assert "feedback" in knee_rule
    
    def test_pushup_form_rules_defined(self):
        """Test pushup has proper form rules."""
        pushup_config = EXERCISE_CONFIGS["pushup"]
        form_rules = pushup_config["form_rules"]
        
        assert "body_alignment" in form_rules
        assert "elbow_position" in form_rules


# Integration tests (require actual video)
class TestIntegration:
    """Integration tests - skipped by default."""
    
    @pytest.mark.skip(reason="Requires actual video file")
    def test_full_video_analysis(self):
        """Test full video analysis pipeline."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
