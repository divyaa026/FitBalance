"""
FitBalance Biomechanics API
============================
FastAPI server for workout video analysis using Ultralytics AIGym.

Endpoints:
- POST /analyze - Analyze a workout video
- GET /health - Health check
- GET /exercises - List supported exercises

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles

from app.core.config import get_settings, SUPPORTED_EXERCISES, EXERCISE_CONFIGS
from app.core.aigym_analyzer import AIGymAnalyzer, get_analyzer
from app.models.schemas import (
    VideoAnalysisResponse,
    HealthResponse,
    ErrorResponse,
    ExerciseType
)
from app.utils.video_validator import InvalidWorkoutVideoError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Global analyzer instance
analyzer: Optional[AIGymAnalyzer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global analyzer
    
    # Startup
    logger.info("Starting FitBalance Biomechanics API...")
    
    # Create temp upload directory
    os.makedirs(settings.temp_upload_dir, exist_ok=True)
    
    # Pre-load the AI model
    try:
        logger.info("Loading AI model...")
        analyzer = get_analyzer()
        logger.info("AI model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AI model: {e}")
        # Continue without model - will fail on requests
    
    yield
    
    # Shutdown
    logger.info("Shutting down FitBalance Biomechanics API...")
    # Cleanup temp files
    try:
        for f in Path(settings.temp_upload_dir).glob("*"):
            f.unlink()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered workout video analysis for form assessment and feedback",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cleanup_temp_file(file_path: str):
    """Background task to cleanup temporary files."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs."""
    return {"message": "FitBalance Biomechanics API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and model loading state.
    Used by Cloud Run and load balancers for health monitoring.
    """
    global analyzer
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        ai_model_loaded=analyzer is not None and analyzer.is_model_loaded()
    )


@app.get("/exercises", tags=["Info"])
async def list_exercises():
    """
    List all supported exercises.
    
    Returns the list of exercises that can be analyzed, along with their
    configuration details like key joints to monitor.
    """
    exercises = []
    for exercise in SUPPORTED_EXERCISES:
        config = EXERCISE_CONFIGS.get(exercise, {})
        exercises.append({
            "name": exercise,
            "display_name": exercise.replace("_", " ").title(),
            "key_joints": config.get("key_joints", []),
            "form_rules": list(config.get("form_rules", {}).keys())
        })
    
    return {
        "exercises": exercises,
        "count": len(exercises)
    }


@app.post(
    "/analyze",
    response_model=VideoAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        422: {"model": ErrorResponse, "description": "Invalid workout video"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Analysis"],
    summary="Analyze workout video",
    description="Upload a workout video to analyze exercise form and get feedback."
)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file (MP4, AVI, MOV)"),
    exercise_hint: Optional[str] = Query(
        None,
        description="Optional hint about the exercise type",
        enum=SUPPORTED_EXERCISES
    ),
    validate: bool = Query(
        True,
        description="Whether to validate the video as a workout"
    )
):
    """
    Analyze a workout video for form quality.
    
    Upload a video of yourself performing an exercise. The AI will:
    - Detect the exercise type
    - Count repetitions
    - Analyze your form
    - Generate a form score (0-100)
    - Provide personalized feedback
    - Create a movement heatmap
    
    **Supported formats:** MP4, AVI, MOV, MKV, WebM  
    **Max file size:** 100MB  
    **Max duration:** 5 minutes
    
    Returns detailed analysis including:
    - Form score and quality rating
    - Specific feedback for improvement
    - Joint movement heatmap data
    - Repetition count and timing
    """
    global analyzer
    
    # Check if model is loaded
    if analyzer is None:
        try:
            analyzer = get_analyzer()
        except Exception as e:
            logger.error(f"Failed to load analyzer: {e}")
            raise HTTPException(
                status_code=500,
                detail="AI model not available. Please try again later."
            )
    
    # Validate file extension
    file_ext = Path(video.filename or "").suffix.lower()
    if file_ext not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(settings.allowed_video_extensions)}"
        )
    
    # Check file size (read first chunk to estimate)
    # Note: For production, use streaming and proper size limits
    content = await video.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum: {settings.max_upload_size_mb}MB"
        )
    
    # Save to temp file
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(settings.temp_upload_dir, temp_filename)
    
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"Saved uploaded video to {temp_path} ({file_size_mb:.1f}MB)")
        
        # Run analysis
        result = analyzer.analyze_video(
            video_path=temp_path,
            exercise_hint=exercise_hint,
            validate=validate
        )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Check for errors
        if not result.get("success", False):
            error_msg = result.get("error_message", "Analysis failed")
            error_code = result.get("error_code", "ANALYSIS_ERROR")
            
            if error_code in ["VIDEO_TOO_SHORT", "LOW_POSE_DETECTION", "NO_MOVEMENT"]:
                raise HTTPException(
                    status_code=422,
                    detail=error_msg
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
        
        return VideoAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except InvalidWorkoutVideoError as e:
        background_tasks.add_task(cleanup_temp_file, temp_path)
        raise HTTPException(
            status_code=422,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        background_tasks.add_task(cleanup_temp_file, temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/quick", tags=["Analysis"])
async def quick_analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """
    Quick analysis endpoint - faster but less detailed.
    
    Skips validation and provides basic metrics only.
    Useful for real-time feedback during workout.
    """
    global analyzer
    
    if analyzer is None:
        raise HTTPException(status_code=500, detail="AI model not available")
    
    file_ext = Path(video.filename or "").suffix.lower()
    if file_ext not in settings.allowed_video_extensions:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    content = await video.read()
    temp_filename = f"quick_{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(settings.temp_upload_dir, temp_filename)
    
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        result = analyzer.analyze_video(
            video_path=temp_path,
            validate=False  # Skip validation for speed
        )
        
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Return simplified response
        return {
            "success": result.get("success", False),
            "exercise": result.get("exercise_detected"),
            "score": result.get("form_score"),
            "reps": result.get("rep_count"),
            "feedback": result.get("feedback", [])[:3]  # Top 3 feedback items
        }
        
    except Exception as e:
        background_tasks.add_task(cleanup_temp_file, temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_message": "An unexpected error occurred. Please try again.",
            "error_code": "INTERNAL_ERROR"
        }
    )


# Main entry point for direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=settings.debug
    )
