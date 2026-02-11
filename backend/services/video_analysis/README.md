# FitBalance Biomechanics Backend

AI-powered workout video analysis using Ultralytics YOLOv8 pose estimation and AIGym module.

## Features

- **Video Upload**: Accept MP4, AVI, MOV, MKV, WebM video files
- **Exercise Detection**: Automatically detect 5 supported exercises
- **Form Analysis**: AI-powered form scoring (0-100) with detailed breakdown
- **Rep Counting**: Accurate repetition counting using AIGym
- **Feedback Generation**: Personalized corrective feedback
- **Movement Heatmap**: Joint activity visualization data
- **Video Validation**: Reject non-workout videos automatically

## Supported Exercises

1. **Squat** - Hip hinge and knee bend tracking
2. **Push-up** - Arm extension and body alignment
3. **Deadlift** - Hip hinge and back position
4. **Bicep Curl** - Elbow flexion and stability
5. **Overhead Press** - Shoulder and arm movement

## Quick Start

### Local Development

```bash
# Navigate to backend directory
cd fitbalance-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build image
docker build -t fitbalance-biomechanics .

# Run container
docker run -p 8000:8000 fitbalance-biomechanics
```

## API Endpoints

### Health Check
```
GET /health
```
Returns service status and model loading state.

### List Exercises
```
GET /exercises
```
Returns all supported exercises with configuration details.

### Analyze Video
```
POST /analyze
Content-Type: multipart/form-data

Parameters:
- video: Video file (required)
- exercise_hint: Optional exercise type hint
- validate: Whether to validate as workout (default: true)
```

**Example Response:**
```json
{
  "success": true,
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
    "left_hip": 0.76
  },
  "rep_count": 8,
  "processing_time_seconds": 2.34
}
```

### Quick Analyze
```
POST /analyze/quick
```
Faster analysis with minimal output - useful for real-time feedback.

## Project Structure

```
fitbalance-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and routes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Settings and exercise configs
│   │   └── aigym_analyzer.py # Main AI analysis logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── video_validator.py # Workout validation
├── configs/
│   └── exercises/           # Exercise-specific YAML configs
├── tests/
│   └── test_analyzer.py     # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FITBALANCE_DEBUG` | `false` | Enable debug mode |
| `FITBALANCE_MAX_UPLOAD_SIZE_MB` | `100` | Max video upload size |
| `FITBALANCE_YOLO_MODEL_PATH` | `yolov8n-pose.pt` | YOLO model path |
| `PORT` | `8000` | Server port |

### Exercise Configuration

Each exercise is configured in `app/core/config.py` with:
- **kpts_to_check**: Keypoint indices for angle calculation
- **angle_up/down**: Expected angle range
- **key_joints**: Joints to monitor
- **form_rules**: Specific form checks with feedback

## Form Scoring

The form score (0-100) is calculated from four components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Rep Consistency | 25% | Timing consistency across reps |
| Range of Motion | 25% | Full ROM achievement |
| Posture Quality | 30% | Overall form throughout movement |
| Tempo Control | 20% | Controlled movement speed |

### Score Quality Ratings

- **Excellent**: 85-100
- **Good**: 70-84
- **Needs Improvement**: 50-69
- **Poor**: 0-49

## Deployment to Google Cloud Run

1. **Build and push to Container Registry:**
```bash
# Set project ID
export PROJECT_ID=your-project-id

# Build image
docker build -t gcr.io/$PROJECT_ID/fitbalance-biomechanics .

# Push to registry
docker push gcr.io/$PROJECT_ID/fitbalance-biomechanics
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy fitbalance-biomechanics \
  --image gcr.io/$PROJECT_ID/fitbalance-biomechanics \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

## Video Validation

Videos are validated to ensure they contain workout content:

1. **Pose Detection Rate**: At least 70% of frames must have detected poses
2. **Movement Variance**: Minimum movement threshold must be exceeded
3. **Periodicity**: Repetitive motion pattern detection (FFT analysis)
4. **Minimum Frames**: At least 30 frames with poses

## Testing

```bash
# Run tests
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=app --cov-report=html
```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Model not loading
Ensure you have enough memory (at least 2GB) and the YOLO model is accessible.

### Video processing errors
- Check video format is supported
- Ensure video duration is under 5 minutes
- Verify video contains visible human poses

### Low accuracy
- Ensure good lighting in the video
- Film from a side angle for best pose detection
- Keep the full body in frame

## License

Proprietary - FitBalance Team

## Contributing

See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines.
