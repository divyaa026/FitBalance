"""
Pydantic models for API request and response schemas.
======================================================
Defines data validation and serialization for the biomechanics API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ExerciseType(str, Enum):
    """Supported exercise types."""
    SQUAT = "squat"
    PUSHUP = "pushup"
    DEADLIFT = "deadlift"
    BICEP_CURL = "bicep_curl"
    OVERHEAD_PRESS = "overhead_press"


class FormQuality(str, Enum):
    """Form quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


class FeedbackItem(BaseModel):
    """Individual feedback item for form correction."""
    category: str = Field(..., description="Category of the feedback (e.g., 'posture', 'range_of_motion')")
    message: str = Field(..., description="Human-readable feedback message")
    severity: str = Field(default="info", description="Severity level: info, warning, error")
    timestamp_seconds: Optional[float] = Field(None, description="Video timestamp where issue was observed")


class JointHeatmapData(BaseModel):
    """Heatmap data for a single joint."""
    joint_name: str = Field(..., description="Name of the joint")
    activity_score: float = Field(..., ge=0, le=1, description="Normalized activity/deviation score (0-1)")
    variance: float = Field(..., ge=0, description="Movement variance for this joint")
    range_of_motion: float = Field(..., ge=0, description="Range of motion in pixels")


class RepetitionData(BaseModel):
    """Data for a single repetition."""
    rep_number: int = Field(..., ge=1, description="Repetition number")
    duration_seconds: float = Field(..., ge=0, description="Duration of the rep")
    quality_score: float = Field(..., ge=0, le=100, description="Quality score for this rep")
    angle_range: Optional[Dict[str, float]] = Field(None, description="Min/max angles observed")


class VideoMetadata(BaseModel):
    """Metadata about the processed video."""
    duration_seconds: float = Field(..., ge=0, description="Total video duration")
    fps: float = Field(..., ge=0, description="Video frames per second")
    total_frames: int = Field(..., ge=0, description="Total number of frames")
    frames_analyzed: int = Field(..., ge=0, description="Number of frames actually analyzed")
    resolution: Dict[str, int] = Field(..., description="Video resolution (width, height)")


class VideoAnalysisRequest(BaseModel):
    """Request schema for video analysis (for future metadata options)."""
    exercise_hint: Optional[ExerciseType] = Field(
        None, 
        description="Optional hint about which exercise is in the video"
    )
    analyze_all_frames: bool = Field(
        False, 
        description="Whether to analyze all frames (slower but more accurate)"
    )
    generate_annotated_video: bool = Field(
        False, 
        description="Whether to generate an annotated output video"
    )


class VideoAnalysisResponse(BaseModel):
    """Response schema for video analysis results."""
    success: bool = Field(..., description="Whether the analysis was successful")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    
    # Exercise Detection
    exercise_detected: Optional[str] = Field(
        None, 
        description="Type of exercise detected in the video"
    )
    exercise_confidence: Optional[float] = Field(
        None, 
        ge=0, 
        le=1, 
        description="Confidence in exercise detection"
    )
    
    # Form Scoring
    form_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=100, 
        description="Overall form score (0-100)"
    )
    form_quality: Optional[FormQuality] = Field(
        None, 
        description="Qualitative form assessment"
    )
    
    # Detailed Scoring Breakdown
    score_breakdown: Optional[Dict[str, float]] = Field(
        None, 
        description="Breakdown of score by category"
    )
    
    # Feedback
    feedback: Optional[List[str]] = Field(
        None, 
        description="List of feedback messages for form improvement"
    )
    detailed_feedback: Optional[List[FeedbackItem]] = Field(
        None, 
        description="Detailed feedback with categories and timestamps"
    )
    
    # Heatmap Data
    heatmap_data: Optional[Dict[str, float]] = Field(
        None, 
        description="Joint activity heatmap data (joint_name -> activity_score)"
    )
    detailed_heatmap: Optional[List[JointHeatmapData]] = Field(
        None, 
        description="Detailed heatmap data for each joint"
    )
    
    # Repetition Data
    rep_count: Optional[int] = Field(
        None, 
        ge=0, 
        description="Number of repetitions detected"
    )
    repetitions: Optional[List[RepetitionData]] = Field(
        None, 
        description="Detailed data for each repetition"
    )
    
    # Video Metadata
    video_metadata: Optional[VideoMetadata] = Field(
        None, 
        description="Metadata about the processed video"
    )
    
    # Processing Info
    processing_time_seconds: Optional[float] = Field(
        None, 
        ge=0, 
        description="Time taken to process the video"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "exercise_detected": "squat",
                "exercise_confidence": 0.92,
                "form_score": 78.5,
                "form_quality": "good",
                "score_breakdown": {
                    "rep_consistency": 82.0,
                    "range_of_motion": 75.0,
                    "posture_quality": 80.0,
                    "tempo_control": 76.0
                },
                "feedback": [
                    "Keep your knees behind your toes",
                    "Try to reach parallel for full range of motion"
                ],
                "heatmap_data": {
                    "left_knee": 0.85,
                    "right_knee": 0.82,
                    "left_hip": 0.76,
                    "right_hip": 0.74
                },
                "rep_count": 8,
                "processing_time_seconds": 2.34
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    ai_model_loaded: bool = Field(..., description="Whether the AI model is loaded")


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = Field(default=False)
    error_message: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
