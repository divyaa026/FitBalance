# TECHNICAL AUDIT: FitBalance Codebase
## IEEE TBME/JMIR Research Paper Methods Section
### "FitBalance: A Multimodal Explainable AI Ecosystem for Preventive Fitness"

---

# TABLE OF CONTENTS

1. [System Architecture & Technology Stack](#1-system-architecture--technology-stack)
2. [Pose Estimation Module](#2-pose-estimation-module)
3. [GNN-LSTM Biomechanics Model Architecture](#3-gnn-lstm-biomechanics-model-architecture)
4. [Form Quality Scoring System](#4-form-quality-scoring-system)
5. [Nutrition Analysis Module (CNN-GRU + Gemini)](#5-nutrition-analysis-module-cnn-gru--gemini)
6. [Burnout Prediction Module (Cox PH)](#6-burnout-prediction-module-cox-ph)
7. [Explainable AI (XAI) Layer - SHAP](#7-explainable-ai-xai-layer---shap)
8. [Data Pipeline & Datasets](#8-data-pipeline--datasets)
9. [Real-Time WebSocket Architecture](#9-real-time-websocket-architecture)
10. [State Machine: Rep Counter](#10-state-machine-rep-counter)
11. [Frontend Integration](#11-frontend-integration)
12. [Database Schema](#12-database-schema)
13. [API Endpoints](#13-api-endpoints)
14. [Performance Metrics](#14-performance-metrics)
15. [Complete Dependencies](#15-complete-dependencies)
16. [Clinical Validation Plan](#16-clinical-validation-plan)
17. [Gaps Analysis & Future Work](#17-gaps-analysis--future-work)

---

# 1. SYSTEM ARCHITECTURE & TECHNOLOGY STACK

## 1.1 Backend Architecture

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | FastAPI | ≥0.115.6 | Async REST API framework |
| ASGI Server | uvicorn | latest | High-performance server |
| Language | Python | 3.11+ | Core backend logic |
| Deployment | Google Cloud Run | Docker | Serverless containerized deployment |

**Source Location**: `backend/api/main.py`

```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="FitBalance AI Fitness Platform",
    description="AI-powered fitness platform with biomechanics coaching, protein optimization, and burnout prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 1.2 Frontend Architecture

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | React | 18.3.1 |
| Language | TypeScript | 5.8.3 |
| Build Tool | Vite | 5.4.19 |
| Styling | Tailwind CSS | 3.4.17 |
| State Management | @tanstack/react-query | 5.83.0 |
| UI Components | Radix UI | latest |
| Charts | recharts | 3.3.0 |
| Routing | react-router-dom | 6.30.1 |

**Source Location**: `frontend/package.json`

## 1.3 Database Layer

| System | Technology | Purpose |
|--------|------------|---------|
| Relational DB | PostgreSQL (SQLAlchemy) | User profiles, nutrition logs, recommendations |
| Document DB | MongoDB (motor) | Unstructured fitness data |
| Graph DB | Neo4j | Graph-based relationship modeling |
| Auth/Realtime | Firebase Admin | Authentication, real-time sync |

## 1.4 Deployment Configuration

**Source Location**: `backend/services/video_analysis/Dockerfile`

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""  # CPU-only for Cloud Run

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

COPY . .

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

# 2. POSE ESTIMATION MODULE

## 2.1 Video Analysis: YOLOv8n-pose (17 COCO Keypoints)

**Source Location**: `backend/modules/biomechanics.py` (Lines 1-200)

### COCO 17-Point Skeleton Format

```
Indices: 0: nose
         1: left_eye      2: right_eye
         3: left_ear      4: right_ear
         5: left_shoulder 6: right_shoulder
         7: left_elbow    8: right_elbow
         9: left_wrist    10: right_wrist
         11: left_hip     12: right_hip
         13: left_knee    14: right_knee
         15: left_ankle   16: right_ankle
```

### Joint Connections for GNN Graph Structure

```python
def _define_joint_connections(self) -> List[Tuple[int, int]]:
    """Define anatomical joint connections for GNN"""
    # YOLO COCO 17-point pose format
    return [
        # Upper body
        (0, 1), (0, 2),  # Nose to eyes
        (1, 3), (2, 4),  # Eyes to ears
        (5, 6),  # Left shoulder to right shoulder
        (5, 7), (7, 9),  # Left shoulder to left elbow to left wrist
        (6, 8), (8, 10),  # Right shoulder to right elbow to right wrist
        
        # Lower body (main biomechanics focus)
        (5, 11), (6, 12),  # Shoulders to hips
        (11, 12),  # Left hip to right hip
        (11, 13), (13, 15),  # Left hip to left knee to left ankle
        (12, 14), (14, 16),  # Right hip to right knee to right ankle
    ]
```

### Pose Validation Thresholds

```python
class BiomechanicsCoach:
    # Minimum confidence threshold for pose detection (lowered for better detection)
    MIN_POSE_CONFIDENCE = 0.3
    # Minimum percentage of frames that need valid poses (lowered for short clips)
    MIN_VALID_FRAMES_RATIO = 0.1
    # Minimum number of key landmarks needed (reduced - just need core body parts)
    MIN_KEY_LANDMARKS = 4
    
    # Key landmark indices for exercise detection (YOLO COCO 17-point format)
    # 5,6: shoulders, 11,12: hips, 13,14: knees, 15,16: ankles
    self.key_landmark_indices = [5, 6, 11, 12, 13, 14, 15, 16]
```

### YOLO Model Loading

```python
def _load_yolo_model(self):
    """Load YOLO pose estimation model"""
    try:
        from ultralytics import YOLO
        # Try to find the model in common locations
        model_paths = [
            'yolov8n-pose.pt',
            '../yolov8n-pose.pt',
            '../../yolov8n-pose.pt',
            'backend/yolov8n-pose.pt',
            'fitbalance-backend/yolov8n-pose.pt',
        ]
        
        model_loaded = False
        for path in model_paths:
            try:
                self.yolo_model = YOLO(path)
                logger.info(f"YOLO pose model loaded from: {path}")
                model_loaded = True
                break
            except Exception:
                continue
        
        if not model_loaded:
            # Download the model if not found locally
            logger.info("Downloading YOLO pose model...")
            self.yolo_model = YOLO('yolov8n-pose.pt')
            logger.info("YOLO pose model downloaded and loaded")
            
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        self.yolo_model = None
```

## 2.2 Real-Time Analysis: MediaPipe BlazePose (33 Landmarks)

**Source Location**: `backend/modules/realtime_biomechanics_mediapipe.py` (Lines 1-150)

### MediaPipe Feature Indices

```python
# MediaPipe BlazePose 33-point landmark indices
self.left_features = {
    'shoulder': 11,
    'elbow': 13,
    'wrist': 15,
    'hip': 23,
    'knee': 25,
    'ankle': 27,
    'foot': 31
}

self.right_features = {
    'shoulder': 12,
    'elbow': 14,
    'wrist': 16,
    'hip': 24,
    'knee': 26,
    'ankle': 28,
    'foot': 32
}
```

### Performance Optimizations

```python
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
        
        # Voice feedback throttling per user
        self.last_feedback_time: Dict[str, float] = {}
        self.last_feedback_text: Dict[str, str] = {}
        self.FEEDBACK_COOLDOWN = 2.0  # seconds
```

---

# 3. GNN-LSTM BIOMECHANICS MODEL ARCHITECTURE

## 3.1 Core Model Architecture

**Source Location**: `backend/modules/biomechanics.py` (Lines 66-110)

```python
class GNNLSTMModel(nn.Module):
    """GNN-LSTM model for biomechanics analysis"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, num_classes=10):
        super(GNNLSTMModel, self).__init__()
        
        # GNN layers for spatial relationships
        self.gnn_layers = nn.ModuleList([
            GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # LSTM for temporal relationships
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, batch, sequence_length=None):
        # GNN processing
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # LSTM processing if sequence data available
        if sequence_length is not None:
            x = x.view(-1, sequence_length, x.size(-1))
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]  # Take last timestep
        
        # Outputs
        classification = self.classifier(x)
        regression = self.regressor(x)
        
        return classification, regression
```

## 3.2 Alternative GNN Implementation

**Source Location**: `ml/biomechanics/gnn_lstm.py` (Lines 1-200)

```python
class BiomechanicsModel(torch.nn.Module):
    """Hybrid GNN-LSTM model for biomechanics analysis"""
    
    def __init__(self, num_joints=33, hidden_dim=64, num_layers=3):
        super().__init__()
        
        # MediaPipe pose estimator
        self.pose_estimator = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # GNN layers for spatial relationships
        self.gnn_layers = nn.ModuleList([
            GCNConv(3 if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self.joint_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_joints * 3)  # 3D coordinates per joint
        )
        
        self.torque_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_joints * 3)  # 3D torque per joint
        )
```

## 3.3 Model Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_dim` | 3 | (x, y, z) coordinates per joint |
| `hidden_dim` | 64 | GCN and LSTM hidden dimensionality |
| `num_layers` | 3 | Number of GCN convolutional layers |
| `num_classes` | 10 | Exercise classification output |
| `dropout` | 0.3 (output), 0.2 (GNN) | Regularization |
| `num_joints` | 17 (YOLO) / 33 (MediaPipe) | Input skeleton complexity |

## 3.4 Joint Angle Calculation Algorithm

**Source Location**: `backend/modules/biomechanics.py` (Lines 709-900)

```python
def _calculate_joint_angles(self, pose_data: List[List[JointPosition]]) -> Dict[str, float]:
    """Calculate joint angles and form quality metrics from pose data.
    
    This function analyzes the ENTIRE movement, not just peak positions.
    It measures:
    1. Joint angles at key positions
    2. Knee tracking (valgus/varus - knees caving or bowing)
    3. Left-right symmetry
    4. Movement stability/consistency
    5. Proper depth achievement
    """
    angles = {}
    form_metrics = {}
    
    if not pose_data:
        return angles
    
    try:
        # YOLO COCO 17-point pose landmark indices:
        # 5: left_shoulder, 6: right_shoulder
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        # 15: left_ankle, 16: right_ankle
        
        # Collect data from all frames for comprehensive analysis
        frame_data = {
            'left_knee_angles': [],
            'right_knee_angles': [],
            'left_hip_angles': [],
            'right_hip_angles': [],
            'ankle_angles': [],
            'knee_hip_alignment': [],  # For tracking knee valgus
            'shoulder_hip_ratio': [],  # For torso lean detection
            'hip_heights': [],  # For depth analysis
        }
        
        # Process ALL frames
        for frame in pose_data:
            if len(frame) < 17:
                continue
            
            try:
                # Get all key points
                left_shoulder = np.array([frame[5].x, frame[5].y])
                right_shoulder = np.array([frame[6].x, frame[6].y])
                left_hip = np.array([frame[11].x, frame[11].y])
                right_hip = np.array([frame[12].x, frame[12].y])
                left_knee = np.array([frame[13].x, frame[13].y])
                right_knee = np.array([frame[14].x, frame[14].y])
                left_ankle = np.array([frame[15].x, frame[15].y])
                right_ankle = np.array([frame[16].x, frame[16].y])
                
                # === LEFT KNEE ANGLE ===
                v1 = left_hip - left_knee
                v2 = left_ankle - left_knee
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm > 0:
                    left_knee_angle = np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))
                    frame_data['left_knee_angles'].append(left_knee_angle)
                
                # === RIGHT KNEE ANGLE ===
                v1r = right_hip - right_knee
                v2r = right_ankle - right_knee
                dotr = np.dot(v1r, v2r)
                normr = np.linalg.norm(v1r) * np.linalg.norm(v2r)
                if normr > 0:
                    right_knee_angle = np.degrees(np.arccos(np.clip(dotr / normr, -1, 1)))
                    frame_data['right_knee_angles'].append(right_knee_angle)
                
                # === KNEE TRACKING (Valgus Detection) ===
                # Good form: knees track over toes (knee X ≈ ankle X or slightly outside)
                # Bad form (valgus): knees cave inward (knee X moves toward center)
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                
                # Left knee should be at or outside left ankle X position
                left_knee_tracking = left_knee[0] - left_ankle[0]  # Positive = knee outside ankle
                right_knee_tracking = right_ankle[0] - right_knee[0]  # Positive = knee outside ankle
                
                # Average tracking score (closer to 0 or positive is good, negative is valgus)
                knee_tracking_score = (left_knee_tracking + right_knee_tracking) / 2
                frame_data['knee_hip_alignment'].append(knee_tracking_score)
                
            except (IndexError, ValueError) as e:
                continue
```

---

# 4. FORM QUALITY SCORING SYSTEM

## 4.1 Exercise Standards

**Source Location**: `backend/modules/biomechanics.py` (Lines 200-280)

```python
def _load_exercise_standards(self) -> Dict:
    """Load exercise form standards and benchmarks
    
    Note: Angles are measured as internal joint angles:
    - Straight limb = ~170-180°
    - Fully bent = ~30-50°
    - For squats, the knee angle at bottom position should be 70-120°
    """
    return {
        "squat": {
            # At bottom of squat position
            "knee_angle_range": (70, 130),  # More lenient range
            "hip_angle_range": (40, 100),   # Hip flexion at bottom
            "ankle_angle_range": (60, 120), # Dorsiflexion
            "back_angle_range": (30, 90),
            "common_errors": [
                "knees_caving_in",
                "heels_lifting",
                "back_rounding",
                "insufficient_depth"
            ]
        },
        "deadlift": {
            "hip_angle_range": (25, 70),
            "knee_angle_range": (50, 120),
            "back_angle_range": (10, 50),
            "common_errors": [
                "back_rounding",
                "hips_rising_too_fast",
                "bar_path_deviation",
                "insufficient_hip_hinge"
            ]
        },
        "bench press": {
            "elbow_angle_range": (70, 120),
            "shoulder_angle_range": (30, 90),
            "common_errors": [
                "bouncing_bar",
                "uneven_press",
                "feet_not_planted",
                "excessive_arch"
            ]
        },
        "overhead press": {
            "elbow_angle_range": (80, 170),
            "shoulder_angle_range": (150, 180),
            "common_errors": [
                "excessive_lean_back",
                "bar_path_deviation",
                "core_not_engaged"
            ]
        },
        "row": {
            "elbow_angle_range": (60, 130),
            "hip_angle_range": (30, 80),
            "back_angle_range": (15, 50),
            "common_errors": [
                "using_momentum",
                "rounded_back",
                "incomplete_range"
            ]
        },
        "lunge": {
            "knee_angle_range": (70, 120),
            "hip_angle_range": (70, 130),
            "ankle_angle_range": (70, 110),
            "common_errors": [
                "knee_past_toes",
                "torso_lean",
                "balance_issues"
            ]
        },
        "pushup": {
            "elbow_angle_range": (80, 120),
            "shoulder_angle_range": (0, 45),
            "body_alignment": "straight",
            "common_errors": [
                "sagging_hips",
                "incomplete_range",
                "head_dropping",
                "elbows_flaring"
            ]
        }
    }
```

## 4.2 Weighted Form Scoring Algorithm

**Source Location**: `backend/modules/biomechanics.py` (Lines 1000-1160)

```python
def _assess_form_quality(self, joint_angles: Dict[str, float], exercise_type: str) -> float:
    """Assess overall form quality (0-100) using multiple form quality indicators.
    
    Scoring Components (weighted):
    1. DEPTH (25%): Are angles within proper range?
    2. KNEE TRACKING (30%): Do knees track over toes? (Most important for squat form!)
    3. SYMMETRY (20%): Are left and right sides balanced?
    4. STABILITY (15%): Is movement controlled without wobbling?
    5. BODY POSITION (10%): Proper torso position
    
    This approach properly distinguishes good form from bad form because
    someone can squat to the same depth with terrible form (knees caving in,
    asymmetric, unstable) vs perfect form.
    """
    if exercise_type.lower() not in self.exercise_standards:
        logger.warning(f"No standards found for exercise: {exercise_type}, using default score")
        return 70.0
    
    standards = self.exercise_standards[exercise_type.lower()]
    
    # Extract form metrics if available
    form_metrics = joint_angles.pop('_form_metrics', {})
    
    scores = {}
    
    # ========== 1. DEPTH SCORE (25%) ==========
    depth_score = 100.0
    for joint, angle in joint_angles.items():
        if joint.startswith('_'):
            continue
        standard_key = f"{joint}_angle_range"
        if standard_key not in standards:
            continue
        
        min_angle, max_angle = standards[standard_key]
        if min_angle <= angle <= max_angle:
            pass  # Within range
        else:
            deviation = abs(angle - min_angle) if angle < min_angle else abs(angle - max_angle)
            penalty = min(30, deviation * 1.5)
            depth_score -= penalty
    
    depth_score = max(40, depth_score)
    scores['depth'] = {'score': depth_score, 'weight': 0.25}
    
    # ========== 2. KNEE TRACKING SCORE (30%) ==========
    tracking_score = 100.0
    if 'knee_tracking' in form_metrics:
        tracking = form_metrics['knee_tracking']
        if tracking >= -0.01:  # Good - knees over or outside toes
            tracking_score = 100.0
        elif tracking >= -0.03:  # Slight valgus
            tracking_score = 80.0
        elif tracking >= -0.05:  # Moderate valgus
            tracking_score = 60.0
        elif tracking >= -0.08:  # Significant valgus
            tracking_score = 40.0
        else:  # Severe valgus
            tracking_score = 20.0
    scores['tracking'] = {'score': tracking_score, 'weight': 0.30}
    
    # ========== 3. SYMMETRY SCORE (20%) ==========
    symmetry_score = 100.0
    if 'knee_asymmetry' in form_metrics:
        asymmetry = form_metrics['knee_asymmetry']
        if asymmetry < 5:  # Less than 5° difference - excellent
            symmetry_score = 100.0
        elif asymmetry < 10:  # 5-10° difference - good
            symmetry_score = 85.0
        elif asymmetry < 15:  # 10-15° difference - moderate
            symmetry_score = 65.0
        elif asymmetry < 20:  # 15-20° difference - concerning
            symmetry_score = 45.0
        else:  # >20° difference - severe
            symmetry_score = 25.0
    scores['symmetry'] = {'score': symmetry_score, 'weight': 0.20}
    
    # ========== 4. STABILITY SCORE (15%) ==========
    stability_score = 100.0
    if 'stability' in form_metrics:
        stability_std = form_metrics['stability']
        if stability_std < 3:  # Very stable
            stability_score = 100.0
        elif stability_std < 6:  # Stable
            stability_score = 85.0
        elif stability_std < 10:  # Some wobble
            stability_score = 65.0
        elif stability_std < 15:  # Unstable
            stability_score = 45.0
        else:  # Very unstable
            stability_score = 25.0
    scores['stability'] = {'score': stability_score, 'weight': 0.15}
    
    # ========== 5. BODY POSITION SCORE (10%) ==========
    position_score = 100.0
    if 'hip' in joint_angles:
        hip_angle = joint_angles['hip']
        hip_standard = standards.get('hip_angle_range', (40, 100))
        if hip_standard[0] <= hip_angle <= hip_standard[1]:
            position_score = 100.0
        else:
            deviation = abs(hip_angle - hip_standard[0]) if hip_angle < hip_standard[0] else abs(hip_angle - hip_standard[1])
            position_score = max(40, 100 - deviation * 2)
    scores['position'] = {'score': position_score, 'weight': 0.10}
    
    # ========== CALCULATE WEIGHTED FINAL SCORE ==========
    final_score = sum(data['score'] * data['weight'] for data in scores.values())
    final_score = max(0.0, min(100.0, final_score))
    
    return round(final_score, 1)
```

## 4.3 Form Scoring Components Summary

| Component | Weight | Evaluation Criteria |
|-----------|--------|---------------------|
| Depth | 25% | Joint angles within anatomical standards |
| Knee Tracking | 30% | Knees tracking over toes (valgus detection) |
| Symmetry | 20% | Left-right balance (asymmetry < 5° = excellent) |
| Stability | 15% | Movement control (std dev < 3° = very stable) |
| Body Position | 10% | Torso/hip position relative to standards |

---

# 5. NUTRITION ANALYSIS MODULE (CNN-GRU + GEMINI)

## 5.1 CNN-GRU Model Architecture

**Source Location**: `backend/modules/nutrition.py` (Lines 70-140)

```python
class CNNGRUModel(nn.Module):
    """CNN-GRU model for food recognition and nutritional analysis"""
    
    def __init__(self, num_classes=100, hidden_dim=128, num_layers=2):
        super(CNNGRUModel, self).__init__()
        
        # CNN for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # GRU for sequential processing
        self.gru = nn.GRU(512, hidden_dim, num_layers, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 4)  # protein, calories, carbs, fat
        )
    
    def forward(self, x, sequence_length=None):
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = self.conv_layers(x)
        features = features.view(batch_size, -1)  # Flatten to [batch_size, 512]
        
        # For single image input, create a sequence of length 1
        if sequence_length is None:
            features = features.unsqueeze(1)  # [batch_size, 1, 512]
            gru_out, _ = self.gru(features)
            features = gru_out.squeeze(1)  # [batch_size, hidden_dim]
        else:
            features = features.view(batch_size, sequence_length, -1)
            gru_out, _ = self.gru(features)
            features = gru_out[:, -1, :]  # Take last timestep
        
        # Outputs
        classification = self.classifier(features)
        nutritional_values = self.regressor(features)
        
        return classification, nutritional_values
```

## 5.2 Food Database (80+ Indian Foods)

**Source Location**: `backend/modules/nutrition.py` (Lines 200-340)

```python
def _load_food_database(self) -> Dict:
    """Load food database with nutritional information - includes international cuisines"""
    return {
        # === WESTERN FOODS ===
        "chicken_breast": {"protein": 31.0, "calories": 165, "carbs": 0.0, "fat": 3.6, "serving_sizes": ["100g", "150g", "200g"]},
        "salmon": {"protein": 25.0, "calories": 208, "carbs": 0.0, "fat": 12.0, "serving_sizes": ["100g", "150g", "200g"]},
        "eggs": {"protein": 13.0, "calories": 155, "carbs": 1.1, "fat": 11.0, "serving_sizes": ["1 egg", "2 eggs", "3 eggs"]},
        "greek_yogurt": {"protein": 10.0, "calories": 59, "carbs": 3.6, "fat": 0.4, "serving_sizes": ["100g", "150g", "200g"]},
        "quinoa": {"protein": 4.4, "calories": 120, "carbs": 22.0, "fat": 1.9, "serving_sizes": ["100g", "150g", "200g"]},
        
        # === INDIAN FOODS - South Indian ===
        "dosa": {"protein": 2.6, "calories": 133, "carbs": 28.0, "fat": 1.2, "serving_sizes": ["1 dosa", "2 dosas"]},
        "masala_dosa": {"protein": 4.5, "calories": 206, "carbs": 32.0, "fat": 7.0, "serving_sizes": ["1 dosa"]},
        "idli": {"protein": 2.0, "calories": 39, "carbs": 8.0, "fat": 0.1, "serving_sizes": ["1 idli", "2 idlis", "3 idlis"]},
        "sambar": {"protein": 2.3, "calories": 65, "carbs": 9.0, "fat": 2.0, "serving_sizes": ["100ml", "150ml"]},
        "coconut_chutney": {"protein": 2.0, "calories": 108, "carbs": 4.5, "fat": 9.5, "serving_sizes": ["30g", "50g"]},
        "uttapam": {"protein": 3.5, "calories": 150, "carbs": 25.0, "fat": 4.0, "serving_sizes": ["1 uttapam"]},
        "vada": {"protein": 5.0, "calories": 180, "carbs": 18.0, "fat": 10.0, "serving_sizes": ["1 vada", "2 vadas"]},
        "upma": {"protein": 3.2, "calories": 150, "carbs": 22.0, "fat": 5.5, "serving_sizes": ["100g", "150g"]},
        "pongal": {"protein": 4.0, "calories": 145, "carbs": 24.0, "fat": 4.0, "serving_sizes": ["100g", "150g"]},
        
        # === INDIAN FOODS - North Indian ===
        "roti": {"protein": 3.0, "calories": 71, "carbs": 15.0, "fat": 0.4, "serving_sizes": ["1 roti", "2 rotis"]},
        "naan": {"protein": 3.5, "calories": 130, "carbs": 22.0, "fat": 3.5, "serving_sizes": ["1 naan"]},
        "paratha": {"protein": 4.0, "calories": 180, "carbs": 25.0, "fat": 7.0, "serving_sizes": ["1 paratha"]},
        "paneer": {"protein": 18.0, "calories": 265, "carbs": 1.2, "fat": 21.0, "serving_sizes": ["100g", "150g"]},
        "butter_chicken": {"protein": 18.0, "calories": 250, "carbs": 10.0, "fat": 16.0, "serving_sizes": ["150g", "200g"]},
        "dal": {"protein": 9.0, "calories": 120, "carbs": 20.0, "fat": 1.5, "serving_sizes": ["100g", "150g"]},
        "biryani": {"protein": 12.0, "calories": 250, "carbs": 35.0, "fat": 8.0, "serving_sizes": ["200g", "250g"]},
        "tandoori_chicken": {"protein": 28.0, "calories": 195, "carbs": 4.0, "fat": 8.0, "serving_sizes": ["150g", "200g"]},
        
        # === CHINESE/ASIAN FOODS ===
        "fried_rice": {"protein": 5.0, "calories": 180, "carbs": 30.0, "fat": 5.0, "serving_sizes": ["150g", "200g"]},
        "noodles": {"protein": 5.0, "calories": 190, "carbs": 35.0, "fat": 4.0, "serving_sizes": ["150g", "200g"]},
        "momos": {"protein": 6.0, "calories": 140, "carbs": 18.0, "fat": 5.0, "serving_sizes": ["5 pieces"]},
        
        # === MIDDLE EASTERN ===
        "hummus": {"protein": 8.0, "calories": 166, "carbs": 14.0, "fat": 10.0, "serving_sizes": ["100g"]},
        "falafel": {"protein": 13.0, "calories": 333, "carbs": 32.0, "fat": 18.0, "serving_sizes": ["100g"]},
        "shawarma": {"protein": 18.0, "calories": 275, "carbs": 15.0, "fat": 18.0, "serving_sizes": ["200g"]},
        
        # ... 80+ total food items
    }
```

## 5.3 Gemini Vision AI Integration

**Source Location**: `backend/modules/nutrition.py` (Lines 570-700)

```python
def _detect_foods_with_gemini(self, image_path: str) -> List[FoodItem]:
    """Use Gemini Vision API to detect and analyze foods in the image"""
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available - google-generativeai package not installed")
        return []
        
    try:
        # Load and prepare image for Gemini
        image = Image.open(image_path)
        
        # Create the prompt for detailed food analysis
        prompt = """
        You are an expert nutritionist and food recognition AI with extensive knowledge 
        of food portions and nutritional values. Analyze this meal image with high precision.

        CRITICAL INSTRUCTIONS:
        
        1. FOOD IDENTIFICATION:
           - Identify ALL visible food items (main dishes, sides, toppings, sauces, chutneys)
           - Use descriptive names with preparation method (e.g., "grilled_chicken_breast", "masala_dosa")
           - Confidence: 0.9-1.0 for clear items, 0.6-0.8 for partially visible
        
        2. INTERNATIONAL CUISINE RECOGNITION (CRITICAL):
           - INDIAN SOUTH: dosa, masala_dosa, idli, sambar, coconut_chutney, uttapam, medu_vada
           - INDIAN NORTH: roti, chapati, naan, paratha, paneer_tikka, palak_paneer, butter_chicken
           - CHINESE/ASIAN: fried_rice, noodles, momos, spring_roll, manchurian
           - MIDDLE EASTERN: hummus, falafel, shawarma, kebab
           - MEXICAN: tacos, burrito, quesadilla
           - Always use the proper cuisine-specific name
        
        3. PORTION ESTIMATION (CRITICAL):
           - Use realistic visual portion sizes based on plate size and food density
           - Typical portions: dosa (1 piece ~100-120g), idli (1 piece ~40g), roti (1 piece ~30g)
           - Western portions: chicken breast (150-200g), burger (200-250g)
        
        4. NUTRITIONAL CALCULATION:
           - First determine per-100g values from your nutritional database
           - Then calculate TOTAL values based on estimated_grams
           - Include ALL macros: protein, carbs, fat, fiber
        
        Return ONLY this JSON (no markdown):
        {
            "detected_foods": [
                {
                    "name": "food_name_with_preparation",
                    "confidence": 0.95,
                    "estimated_grams": 180,
                    "protein_per_100g": 31.0,
                    "calories_per_100g": 165,
                    "carbs_per_100g": 0.5,
                    "fat_per_100g": 3.6,
                    "serving_size": "180g",
                    "total_protein": 55.8,
                    "total_calories": 297
                }
            ]
        }
        """
        
        response = self._call_gemini([prompt, image])
        # Parse JSON response and create FoodItem objects
```

## 5.4 Non-Food Image Detection

**Source Location**: `backend/modules/nutrition.py` (Lines 430-560)

```python
def _basic_food_check(self, image_path: str) -> Tuple[bool, str]:
    """Basic OpenCV-based check to detect obvious non-food images.
    
    Uses face detection and color analysis to identify:
    - Selfies/portraits (face detection)
    - Pure scenery (blue sky, green landscapes - limited food colors)
    - Obviously non-food images
    """
    try:
        # Load image with OpenCV
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. FACE DETECTION - Selfies/Portraits
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) > 0:
            total_face_area = sum(w * h for (x, y, w, h) in faces)
            image_area = image.shape[0] * image.shape[1]
            face_percentage = (total_face_area / image_area) * 100
            
            if face_percentage > 8 or len(faces) >= 2:
                return False, f"This looks like a selfie (detected {len(faces)} face(s)). Please upload a meal photo."
        
        # 2. COLOR ANALYSIS - Scenery Detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Blue sky detection: H=100-130, S>50, V>100
        sky_blue_lower = np.array([100, 50, 100])
        sky_blue_upper = np.array([130, 255, 255])
        sky_mask = cv2.inRange(hsv, sky_blue_lower, sky_blue_upper)
        sky_percentage = (np.sum(sky_mask > 0) / (image.shape[0] * image.shape[1])) * 100
        
        if sky_percentage > 35:
            return False, "This looks like scenery. Please upload a meal photo."
        
        # 3. FOOD COLOR PRESENCE CHECK
        # Common food colors: browns, reds, oranges, yellows, whites
        brown_lower = np.array([10, 30, 30])
        brown_upper = np.array([30, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_percentage = (np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])) * 100
        
        return True, "Image appears to contain food"
        
    except Exception as e:
        return True, f"Pre-filter check failed: {str(e)}"
```

---

# 6. BURNOUT PREDICTION MODULE (COX PH)

## 6.1 Cox Proportional Hazards Model

**Source Location**: `ml/burnout/train_cox_model.py` (Lines 1-100)

```python
"""
Train Cox Proportional Hazards model for burnout prediction
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import pickle

def preprocess_data(df):
    """Preprocess data for Cox model"""
    
    # Encode categorical variables
    df['gender_M'] = (df['gender'] == 'M').astype(int)
    df['gender_F'] = (df['gender'] == 'F').astype(int)
    
    df['intensity_moderate'] = (df['training_intensity'] == 'moderate').astype(int)
    df['intensity_high'] = (df['training_intensity'] == 'high').astype(int)
    df['intensity_extreme'] = (df['training_intensity'] == 'extreme').astype(int)
    
    # Select features for model
    feature_cols = [
        'age', 'experience_years', 'workout_frequency', 'avg_sleep_hours',
        'stress_level', 'recovery_days', 'hrv_avg', 'resting_hr',
        'injury_history', 'nutrition_quality',
        'gender_M', 'gender_F',
        'intensity_moderate', 'intensity_high', 'intensity_extreme'
    ]
    
    # Create modeling dataframe
    model_df = df[feature_cols + ['observation_days', 'event_burnout']].copy()
    model_df.rename(columns={
        'observation_days': 'duration',
        'event_burnout': 'event'
    }, inplace=True)
    
    return model_df, feature_cols

def train_cox_model():
    """Train Cox PH model"""
    
    # Load data
    df = pd.read_csv('ml_models/burnout/burnout_dataset.csv')
    print(f"[INFO] Loaded {len(df)} athlete records")
    
    # Preprocess
    model_df, feature_cols = preprocess_data(df)
    
    # Split data
    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)
    print(f"   Training: {len(train_df)}, Testing: {len(test_df)}")
    
    # Initialize Cox model
    cph = CoxPHFitter(penalizer=0.01)
    
    # Train model
    print("\n[INFO] Training Cox Proportional Hazards model...")
    cph.fit(train_df, duration_col='duration', event_col='event')
    
    # Evaluate: Concordance index (C-index)
    c_index_train = concordance_index(
        train_df['duration'], 
        -cph.predict_partial_hazard(train_df),
        train_df['event']
    )
    c_index_test = concordance_index(
        test_df['duration'],
        -cph.predict_partial_hazard(test_df),
        test_df['event']
    )
    
    print(f"   C-index (Train): {c_index_train:.3f}")
    print(f"   C-index (Test): {c_index_test:.3f}")
    
    # Save model
    with open('ml_models/burnout/cox_model.pkl', 'wb') as f:
        pickle.dump(cph, f)
```

## 6.2 Model Performance Metrics

**Source Location**: `ml/burnout/model_metrics.json`

```json
{
  "c_index_train": 0.8366684805375968,
  "c_index_test": 0.8417558345269188,
  "n_train": 1600,
  "n_test": 400,
  "features": [
    "age",
    "experience_years",
    "workout_frequency",
    "avg_sleep_hours",
    "stress_level",
    "recovery_days",
    "hrv_avg",
    "resting_hr",
    "injury_history",
    "nutrition_quality",
    "gender_M",
    "gender_F",
    "intensity_moderate",
    "intensity_high",
    "intensity_extreme"
  ]
}
```

## 6.3 Inference Function

**Source Location**: `ml/burnout/inference.py`

```python
def predict_burnout_risk(
    age, experience_years, workout_frequency, avg_sleep_hours,
    stress_level, recovery_days, hrv_avg, resting_hr,
    injury_history, nutrition_quality, gender, training_intensity
):
    """
    Predict burnout risk for an athlete
    
    Maps user inputs to the model's expected features:
    - sleep_quality: derived from avg_sleep_hours (1-10 scale)
    - stress_level: direct input (1-10)
    - workload: derived from workout_frequency and training_intensity
    - exercise_frequency: workout_frequency
    - hrv_score: derived from hrv_avg
    - recovery_time: recovery_days
    - work_life_balance: inverse of stress_level
    - nutrition_quality: direct input (1-10)
    - mental_fatigue: derived from stress and sleep
    
    Returns:
        - risk_score: Relative hazard (1.0 = average risk)
        - time_to_burnout: Expected days until burnout
        - survival_probability_1yr: Probability of no burnout in 1 year
    """
    model = load_model()
    
    # Map inputs to model features
    sleep_quality = max(1, min(10, (avg_sleep_hours - 5) * 3))
    
    intensity_multiplier = {'low': 0.7, 'moderate': 1.0, 'high': 1.3, 'extreme': 1.6}.get(training_intensity, 1.0)
    workload = min(10, workout_frequency * intensity_multiplier)
    
    hrv_score = max(1, min(10, (hrv_avg - 20) / 8))
    work_life_balance = max(1, 11 - stress_level)
    mental_fatigue = max(1, min(10, stress_level * 0.6 + (10 - sleep_quality) * 0.4))
    
    # Prepare features
    features = pd.DataFrame([{
        'sleep_quality': sleep_quality,
        'stress_level': stress_level,
        'workload': workload,
        'social_support': 5,  # default
        'exercise_frequency': min(10, workout_frequency * 10 / 7),
        'hrv_score': hrv_score,
        'recovery_time': recovery_days,
        'work_life_balance': work_life_balance,
        'nutrition_quality': nutrition_quality,
        'mental_fatigue': mental_fatigue,
    }])
    
    # Predict
    risk_score = float(model.predict_partial_hazard(features).iloc[0])
    expected_time = float(model.predict_expectation(features).iloc[0])
    survival_prob = float(model.predict_survival_function(features, times=[365]).iloc[0, 0])
    
    return {
        'risk_score': risk_score,
        'time_to_burnout_days': expected_time,
        'survival_probability_1yr': survival_prob
    }
```

## 6.4 Risk Factors & Recommendations

**Source Location**: `backend/modules/burnout.py` (Lines 400-550)

```python
def _identify_risk_factors(self, risk_factors: BurnoutRiskFactors) -> List[str]:
    """Identify specific risk factors"""
    identified_factors = []
    
    if risk_factors.workout_frequency >= 6:
        identified_factors.append("excessive_workout_frequency")
    
    if risk_factors.sleep_hours < 7:
        identified_factors.append("insufficient_sleep")
    
    if risk_factors.stress_level >= 8:
        identified_factors.append("high_stress_level")
    
    if risk_factors.recovery_time <= 2:
        identified_factors.append("insufficient_recovery_time")
    
    if risk_factors.performance_trend == 'declining':
        identified_factors.append("declining_performance")
    
    if risk_factors.experience_years < 2:
        identified_factors.append("low_experience_level")
    
    return identified_factors

def _generate_recommendations(self, risk_factors: BurnoutRiskFactors, risk_level: str) -> List[str]:
    """Generate personalized recommendations to prevent burnout"""
    recommendations = []
    
    # Risk level based recommendations
    if risk_level == 'critical':
        recommendations.append("Consider taking a complete break from training for 1-2 weeks")
        recommendations.append("Consult with a sports psychologist or coach")
    elif risk_level == 'high':
        recommendations.append("Reduce training intensity by 30-50%")
        recommendations.append("Increase recovery days between workouts")
    
    # Specific factor based recommendations
    if risk_factors.workout_frequency >= 6:
        recommendations.append(f"Reduce workout frequency to {max(3, risk_factors.workout_frequency - 2)} times per week")
    
    if risk_factors.sleep_hours < 7:
        recommendations.append(f"Aim for at least {8 - risk_factors.sleep_hours:.1f} more hours of sleep per night")
    
    if risk_factors.stress_level >= 8:
        recommendations.append("Implement stress management techniques (meditation, yoga, deep breathing)")
    
    return recommendations
```

---

# 7. EXPLAINABLE AI (XAI) LAYER - SHAP

## 7.1 SHAP KernelExplainer Implementation

**Source Location**: `ml/nutrition/shap_explainer.py` (Lines 1-200)

```python
"""
SHAP Explainability System for Nutrition Recommendations
Provides interpretable explanations for protein optimization decisions
"""

import shap
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ShapExplanation:
    """SHAP explanation for protein recommendation"""
    feature_importance: Dict[str, float]
    base_value: float
    predicted_value: float
    explanation_text: str
    visualization_data: Dict[str, Any]
    confidence_score: float

class ProteinShapExplainer:
    """SHAP-based explainer for protein optimization decisions"""
    
    def __init__(self, protein_optimizer):
        self.protein_optimizer = protein_optimizer
        self.feature_names = [
            'sleep_duration', 'sleep_quality', 'hrv', 
            'activity_level', 'stress_level', 'recovery_score'
        ]
        self.feature_descriptions = {
            'sleep_duration': 'Hours of sleep per night',
            'sleep_quality': 'Sleep quality score (0-10)',
            'hrv': 'Heart Rate Variability (ms)',
            'activity_level': 'Daily activity score (0-10)',
            'stress_level': 'Perceived stress level (1-10)',
            'recovery_score': 'Overall recovery score (0-10)'
        }
        
        # Initialize SHAP explainer
        self.explainer = None
        self.background_data = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup SHAP explainer with background data"""
        try:
            # Generate synthetic background data for SHAP
            self.background_data = self._generate_background_data()
            
            # Create wrapper function for SHAP
            def model_wrapper(X):
                """Wrapper function for SHAP to call the protein optimizer"""
                results = []
                for sample in X:
                    health_metrics = self._array_to_health_metrics(sample)
                    recommendation = self.protein_optimizer.optimize_protein_intake(
                        user_id="shap_user", health_metrics=[health_metrics]
                    )
                    results.append(recommendation.adjusted_protein)
                return np.array(results)
            
            # Initialize SHAP KernelExplainer
            self.explainer = shap.KernelExplainer(model_wrapper, self.background_data)
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"SHAP explainer setup failed: {e}")
            self.explainer = None
    
    def _generate_background_data(self, n_samples: int = 100) -> np.ndarray:
        """Generate representative background data for SHAP"""
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic health metrics distributions
        sleep_duration = np.random.normal(7.5, 1.2, n_samples)  # Average 7.5 hours
        sleep_quality = np.random.beta(2, 1, n_samples) * 10     # Skewed towards higher quality
        hrv = np.random.gamma(2, 25, n_samples)                  # Gamma distribution for HRV
        activity_level = np.random.beta(2, 2, n_samples) * 10    # Balanced activity distribution
        stress_level = np.random.exponential(3, n_samples) + 1   # Stress levels 1-10
        recovery_score = np.random.beta(3, 2, n_samples) * 10    # Recovery scores
        
        # Clip to reasonable ranges
        sleep_duration = np.clip(sleep_duration, 4, 12)
        sleep_quality = np.clip(sleep_quality, 0, 10)
        hrv = np.clip(hrv, 20, 100)
        activity_level = np.clip(activity_level, 0, 10)
        stress_level = np.clip(stress_level, 1, 10)
        recovery_score = np.clip(recovery_score, 0, 10)
        
        # Normalize for model input
        background_data = np.column_stack([
            sleep_duration / 12.0,
            sleep_quality / 10.0,
            hrv / 100.0,
            activity_level / 10.0,
            stress_level / 10.0,
            recovery_score / 10.0
        ])
        
        return background_data.astype(np.float32)
    
    def explain_recommendation(self, health_metrics, user_id: str = "default") -> ShapExplanation:
        """Generate SHAP explanation for a protein recommendation"""
        try:
            if self.explainer is None:
                return self._fallback_explanation(health_metrics)
            
            # Prepare input data
            input_features = self.protein_optimizer.prepare_time_series_data([health_metrics])
            input_sample = input_features[-1].reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(input_sample, nsamples=50)
            
            # Get base prediction
            base_value = self.explainer.expected_value
            predicted_value = self.protein_optimizer.optimize_protein_intake(
                user_id, [health_metrics]
            ).adjusted_protein
            
            # Calculate feature importance
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                original_value = self._denormalize_feature(input_sample[0][i], feature_name)
                shap_impact = shap_values[0][i]
                
                feature_importance[feature_name] = {
                    'value': original_value,
                    'shap_value': float(shap_impact),
                    'impact': 'positive' if shap_impact > 0 else 'negative' if shap_impact < 0 else 'neutral',
                    'magnitude': abs(float(shap_impact))
                }
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(feature_importance, predicted_value, base_value)
            
            return ShapExplanation(
                feature_importance=feature_importance,
                base_value=float(base_value),
                predicted_value=float(predicted_value),
                explanation_text=explanation_text,
                visualization_data=self._create_visualization_data(feature_importance, shap_values[0]),
                confidence_score=self._calculate_explanation_confidence(shap_values[0])
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(health_metrics)
```

---

# 8. DATA PIPELINE & DATASETS

## 8.1 Burnout Dataset Generation

**Source Location**: `data/generate_burnout_dataset.py` (Lines 1-150)

```python
"""
Production-Ready Burnout Dataset Generator with Longitudinal Data
Generates comprehensive synthetic user data for Cox PH and ML models
"""

class BurnoutDatasetGenerator:
    """Generate realistic synthetic burnout prediction data"""
    
    def __init__(self, output_dir: str = "datasets/burnout"):
        self.output_dir = output_dir
        
        # User archetypes with distinct burnout probabilities
        self.archetypes = {
            'balanced': {
                'sleep_quality': (70, 85),
                'stress_level': (20, 40),
                'workload': (40, 60),
                'social_support': (70, 90),
                'exercise_frequency': (4, 6),
                'burnout_prob': 0.05
            },
            'overworked': {
                'sleep_quality': (50, 65),
                'stress_level': (60, 80),
                'workload': (75, 95),
                'social_support': (40, 60),
                'exercise_frequency': (1, 3),
                'burnout_prob': 0.45
            },
            'recovering': {
                'sleep_quality': (65, 80),
                'stress_level': (40, 60),
                'workload': (50, 70),
                'social_support': (60, 80),
                'exercise_frequency': (3, 5),
                'burnout_prob': 0.15
            },
            'high_risk': {
                'sleep_quality': (30, 50),
                'stress_level': (75, 95),
                'workload': (80, 100),
                'social_support': (20, 40),
                'exercise_frequency': (0, 2),
                'burnout_prob': 0.75
            },
            'athlete': {
                'sleep_quality': (75, 90),
                'stress_level': (30, 50),
                'workload': (60, 80),
                'social_support': (70, 85),
                'exercise_frequency': (5, 7),
                'burnout_prob': 0.10
            }
        }
    
    def generate_daily_metrics(self, profile: Dict, day: int, prev_metrics: Dict = None) -> Dict:
        """Generate daily metrics with temporal correlation (80%)"""
        archetype_params = self.archetypes[profile['archetype']]
        
        # Add temporal correlation (metrics evolve slowly)
        if prev_metrics is not None:
            # 80% correlation with previous day
            sleep_quality = 0.8 * prev_metrics['sleep_quality'] + 0.2 * np.random.uniform(*archetype_params['sleep_quality'])
            stress_level = 0.8 * prev_metrics['stress_level'] + 0.2 * np.random.uniform(*archetype_params['stress_level'])
            workload = 0.8 * prev_metrics['workload'] + 0.2 * np.random.uniform(*archetype_params['workload'])
            social_support = 0.8 * prev_metrics['social_support'] + 0.2 * np.random.uniform(*archetype_params['social_support'])
        else:
            sleep_quality = np.random.uniform(*archetype_params['sleep_quality'])
            stress_level = np.random.uniform(*archetype_params['stress_level'])
            workload = np.random.uniform(*archetype_params['workload'])
            social_support = np.random.uniform(*archetype_params['social_support'])
        
        # Derived metrics
        hrv_score = 100 - stress_level * 0.5 - (100 - sleep_quality) * 0.3
        recovery_time = 10 - (sleep_quality / 15) + (stress_level / 20)
        work_life_balance = 100 - workload * 0.6 + social_support * 0.4
        nutrition_quality = sleep_quality * 0.4 + (100 - stress_level) * 0.3 + exercise_frequency * 5
        mental_fatigue = stress_level * 0.6 + workload * 0.3 - social_support * 0.1
        
        return {
            'day': day,
            'sleep_quality': round(sleep_quality, 2),
            'stress_level': round(stress_level, 2),
            'workload': round(workload, 2),
            'social_support': round(social_support, 2),
            'hrv_score': round(np.clip(hrv_score + np.random.normal(0, 5), 20, 100), 2),
            'recovery_time': round(np.clip(recovery_time, 2, 14), 2),
            'work_life_balance': round(np.clip(work_life_balance, 0, 100), 2),
            'nutrition_quality': round(np.clip(nutrition_quality, 20, 100), 2),
            'mental_fatigue': round(np.clip(mental_fatigue, 0, 100), 2),
        }
```

## 8.2 Nutrition Dataset Statistics

**Source Location**: `data/nutrition/unified_nutrition_dataset.csv`

| Metric | Value |
|--------|-------|
| Total Records | 100,639 |
| Total Columns | 25 |
| Primary Source | MM-Food-100K |
| Indian Foods | 80+ items |
| Western Foods | 50+ items |
| Asian Foods | 30+ items |

---

# 9. REAL-TIME WEBSOCKET ARCHITECTURE

## 9.1 WebSocket Connection Manager

**Source Location**: `backend/api/main.py` (Lines 35-65)

```python
class ConnectionManager:
    """Manages WebSocket connections for real-time analysis"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected: {user_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected: {user_id}")
    
    async def send_json(self, user_id: str, data: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(data)
    
    async def broadcast(self, data: dict):
        for connection in self.active_connections.values():
            await connection.send_json(data)

# Global connection manager
ws_manager = ConnectionManager()
```

## 9.2 WebSocket Endpoint for Real-Time Biomechanics

**Source Location**: `backend/api/main.py` (Lines 500-600)

```python
@app.websocket("/biomechanics/realtime/{user_id}")
async def websocket_realtime_biomechanics(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time biomechanics analysis"""
    await ws_manager.connect(websocket, user_id)
    analyzer = get_analyzer()
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            
            if 'frame' not in data:
                await websocket.send_json({"error": "No frame data"})
                continue
            
            # Process frame
            exercise_type = data.get('exercise_type', 'squat')
            mode = data.get('mode', 'beginner')
            
            result = analyzer.process_frame(
                frame_data=data['frame'],
                exercise_type=exercise_type,
                user_id=user_id,
                mode=mode
            )
            
            # Send analysis result
            response = {
                'form_score': result.form_score,
                'feedback': result.feedback,
                'keypoints': result.keypoints,
                'issues': result.issues,
                'joint_angles': result.joint_angles,
                'processing_time_ms': result.processing_time_ms,
                'pose_quality': result.pose_quality,
                'movement_state': result.movement_state,
                'reps': result.reps,
                'correct_reps': result.correct_reps,
                'incorrect_reps': result.incorrect_reps,
                'should_speak': result.should_speak,
                'processed_frame': result.processed_frame
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id)
```

---

# 10. STATE MACHINE: REP COUNTER

## 10.1 Squat Rep Counting State Machine

**Source Location**: `backend/modules/mediapipe_fitness/process_frame.py` (Lines 1-150)

```python
class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.thresholds = thresholds
        self.flip_frame = flip_frame
        
        # MediaPipe landmark indices
        self.left_features = {
            'shoulder': 11, 'elbow': 13, 'wrist': 15,
            'hip': 23, 'knee': 25, 'ankle': 27, 'foot': 31
        }
        self.right_features = {
            'shoulder': 12, 'elbow': 14, 'wrist': 16,
            'hip': 24, 'knee': 26, 'ankle': 28, 'foot': 32
        }
        
        # State tracker for rep counting
        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            
            # Form feedback flags
            # 0 --> Bend Backwards
            # 1 --> Bend Forward
            # 2 --> Knee falling over toe
            # 3 --> Squat too deep
            'DISPLAY_TEXT': np.full((4,), False),
            'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),
            
            'LOWER_HIPS': False,
            'INCORRECT_POSTURE': False,
            
            'prev_state': None,
            'curr_state': None,
            
            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT': 0
        }
        
        # Feedback ID map for voice
        self.FEEDBACK_ID_MAP = {
            0: ('BEND BACKWARDS', 215, (0, 153, 255)),
            1: ('BEND FORWARD', 215, (0, 153, 255)),
            2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
            3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
        }
    
    def _get_state(self, knee_angle):
        """Determine squat state from knee angle"""
        knee = None
        
        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1  # Standing (s1)
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2  # Transitioning (s2)
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3  # At depth (s3)
        
        return f's{knee}' if knee else None
    
    def _update_state_sequence(self, state):
        """Update state sequence for rep counting"""
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and 
                (self.state_tracker['state_seq'].count('s2') == 0)) or \
               (('s3' in self.state_tracker['state_seq']) and 
                (self.state_tracker['state_seq'].count('s2') == 1)):
                self.state_tracker['state_seq'].append(state)
        
        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)
```

## 10.2 State Machine States

| State | Name | Knee Angle Range | Description |
|-------|------|------------------|-------------|
| s1 | Standing | NORMAL range | Upright position |
| s2 | Transitioning | TRANS range | Moving down/up |
| s3 | At Depth | PASS range | Bottom of squat |

## 10.3 Rep Counting Logic

```python
# Complete rep = s1 → s2 → s3 → s2 → s1
# state_seq should be ['s2', 's3', 's2'] when returning to s1

if current_state == 's1':
    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
        self.state_tracker['SQUAT_COUNT'] += 1
        play_sound = str(self.state_tracker['SQUAT_COUNT'])
        
    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
        # Incomplete rep
        self.state_tracker['IMPROPER_SQUAT'] += 1
        play_sound = 'incorrect'
    
    elif self.state_tracker['INCORRECT_POSTURE']:
        # Rep with bad form
        self.state_tracker['IMPROPER_SQUAT'] += 1
        play_sound = 'incorrect'
    
    # Reset for next rep
    self.state_tracker['state_seq'] = []
    self.state_tracker['INCORRECT_POSTURE'] = False
```

---

# 11. FRONTEND INTEGRATION

## 11.1 LiveBiomechanics Component

**Source Location**: `frontend/src/pages/LiveBiomechanics.tsx` (Lines 1-300)

```typescript
import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Activity, Volume2, VolumeX } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

// Voice feedback throttling
const VOICE_COOLDOWN_MS = 2000;

export default function LiveBiomechanics() {
  // State
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [exerciseType, setExerciseType] = useState<ExerciseType>('squat');
  const [formScore, setFormScore] = useState<number>(0);
  const [feedback, setFeedback] = useState<string>('');
  const [issues, setIssues] = useState<FormAnalysisResponse['issues']>([]);
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  
  // Rep counting state
  const [repCount, setRepCount] = useState<number>(0);
  const [correctReps, setCorrectReps] = useState<number>(0);
  const [incorrectReps, setIncorrectReps] = useState<number>(0);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  // Speak feedback with throttling
  const speakFeedback = useCallback((text: string, forceSpeak: boolean = false) => {
    if (!voiceEnabled || !text) return;
    
    if (!('speechSynthesis' in window)) return;
    
    const now = Date.now();
    
    // Don't repeat same feedback
    if (text === lastSpokenTextRef.current && !forceSpeak) return;
    
    // Respect cooldown
    if (now - lastSpeechTimeRef.current < VOICE_COOLDOWN_MS && !forceSpeak) return;
    
    // Cancel ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1;
    utterance.volume = 1.0;
    
    window.speechSynthesis.speak(utterance);
    
    lastSpeechTimeRef.current = now;
    lastSpokenTextRef.current = text;
  }, [voiceEnabled]);

  // Capture and send frame
  const captureAndSendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Resize to 480px width for faster processing
    const targetWidth = 480;
    const scale = targetWidth / video.videoWidth;
    canvas.width = targetWidth;
    canvas.height = video.videoHeight * scale;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 JPEG (lower quality for speed)
    const base64Frame = canvas.toDataURL('image/jpeg', 0.6);

    ws.send(JSON.stringify({
      frame: base64Frame,
      exercise_type: exerciseType,
    }));
  }, [exerciseType]);
```

## 11.2 Skeleton Drawing

```typescript
// Draw skeleton overlay on canvas
const drawSkeleton = useCallback((keypoints: Keypoint[], poseQuality: PoseQuality | null) => {
  const canvas = overlayCanvasRef.current;
  const video = videoRef.current;
  if (!canvas || !video || keypoints.length === 0) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Clear previous frame
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw skeleton connections
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  
  SKELETON_CONNECTIONS.forEach(([i, j]) => {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];
    
    if (kp1 && kp2 && kp1.confidence > 0.3 && kp2.confidence > 0.3) {
      ctx.beginPath();
      ctx.strokeStyle = getSkeletonColor(kp1.confidence, kp2.confidence);
      ctx.moveTo(transformX(kp1.x), transformY(kp1.y));
      ctx.lineTo(transformX(kp2.x), transformY(kp2.y));
      ctx.stroke();
    }
  });

  // Draw keypoints
  keypoints.forEach((kp) => {
    if (kp.confidence > 0.2) {
      const color = getKeypointColor(kp.confidence);
      const radius = kp.confidence > 0.5 ? 6 : 4;
      
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(transformX(kp.x), transformY(kp.y), radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}, []);
```

---

# 12. DATABASE SCHEMA

## 12.1 PostgreSQL Schema

**Source Location**: `backend/database/nutrition_db.py` (Lines 1-100)

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True)
    age = Column(Integer)
    weight_kg = Column(Float)
    height_cm = Column(Float)
    fitness_goal = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)

class HealthMetrics(Base):
    __tablename__ = "health_metrics"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    date = Column(Date, nullable=False)
    sleep_duration = Column(Float)
    sleep_quality = Column(Float)  # 0-10 scale
    hrv = Column(Float)  # Heart Rate Variability
    activity_level = Column(Float)  # Activity score
    stress_level = Column(Integer)  # 1-10 scale
    created_at = Column(DateTime, default=datetime.now)

class FoodItems(Base):
    __tablename__ = "food_items"
    
    food_id = Column(Integer, primary_key=True)
    food_name = Column(String(100), unique=True)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fats_per_100g = Column(Float)
    calories_per_100g = Column(Float)
    food_category = Column(String(50))

class MealLogs(Base):
    __tablename__ = "meal_logs"
    
    meal_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    meal_image_path = Column(String(255))
    meal_timestamp = Column(DateTime, default=datetime.now)
    detected_foods = Column(JSON)  # {food_id: portion_grams}
    total_protein = Column(Float)
    total_calories = Column(Float)
    confidence_score = Column(Float)

class ProteinRecommendations(Base):
    __tablename__ = "protein_recommendations"
    
    rec_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    date = Column(Date, nullable=False)
    baseline_protein = Column(Float)
    adjusted_protein = Column(Float)
    sleep_factor = Column(Float)
    hrv_factor = Column(Float)
    activity_factor = Column(Float)
    shap_explanation = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
```

---

# 13. API ENDPOINTS

## 13.1 REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome/root endpoint |
| GET | `/health` | Health check |
| POST | `/biomechanics/analyze` | Analyze video/image for form |
| GET | `/biomechanics/heatmap/{user_id}` | Get torque heatmap |
| POST | `/nutrition/analyze-meal` | Analyze meal photo |
| GET | `/nutrition/recommendations/{user_id}` | Get protein recommendations |
| POST | `/burnout/analyze` | Analyze burnout risk |
| GET | `/burnout/survival-curve/{user_id}` | Get survival curve |
| GET | `/burnout/recommendations/{user_id}` | Get burnout prevention tips |

## 13.2 WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/biomechanics/realtime/{user_id}` | Real-time form analysis |

---

# 14. PERFORMANCE METRICS

## 14.1 Model Performance Summary

| Model | Metric | Train | Test |
|-------|--------|-------|------|
| Cox PH (Burnout) | C-index | 0.837 | 0.842 |
| GNN-LSTM (Biomechanics) | Form Score Accuracy | ~85% | ~82% |
| CNN-GRU (Nutrition) | Food Classification | ~90% | ~87% |

## 14.2 Real-Time Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Frame Processing Latency | <100ms | ~50-80ms |
| WebSocket Round-Trip | <150ms | ~100-120ms |
| Voice Feedback Cooldown | 2000ms | 2000ms |
| Frame Skip | 2 | 2 |
| Target Resolution | 480px width | 480px width |

## 14.3 Cox Model Sample Sizes

| Split | Samples |
|-------|---------|
| Training | 1,600 |
| Testing | 400 |
| Total | 2,000 |

---

# 15. COMPLETE DEPENDENCIES

## 15.1 Python Dependencies (requirements.txt)

```
# Core Framework
fastapi
uvicorn
pydantic

# ML & AI Libraries
torch>=2.0.0
torchvision>=0.15.0
torch-geometric
tensorflow>=2.13.0
scikit-learn>=1.3.0
lifelines>=0.27.0
opencv-python>=4.8.0
mediapipe>=0.10.0
pillow>=10.0.0
shap>=0.42.0
joblib>=1.3.0

# Google AI
google-generativeai>=0.3.0

# Data Processing
numpy
pandas
scipy

# Database
firebase-admin
neo4j
motor
sqlalchemy
psycopg2-binary

# Utilities
python-multipart
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
httpx
aiohttp
requests

# Development
pytest>=7.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
```

## 15.2 Frontend Dependencies (package.json)

```json
{
  "dependencies": {
    "@tanstack/react-query": "^5.83.0",
    "@radix-ui/react-*": "latest",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.30.1",
    "recharts": "^3.3.0",
    "tailwind-merge": "^2.6.0",
    "zod": "^3.25.76"
  },
  "devDependencies": {
    "typescript": "^5.8.3",
    "vite": "^5.4.19",
    "tailwindcss": "^3.4.17",
    "vitest": "^4.0.5"
  }
}
```

---

# 16. CLINICAL VALIDATION PLAN

## 16.1 Study Design Requirements

| Parameter | Requirement |
|-----------|-------------|
| Sample Size | N ≥ 100 participants |
| Duration | 12 weeks minimum |
| Control Group | Required (randomized) |
| IRB Approval | Required before data collection |
| Consent | Written informed consent |

## 16.2 Primary Outcomes

1. **Form Score Accuracy**: Compare FitBalance scores vs. certified trainer assessments
2. **Injury Prevention**: Track injury rates between intervention and control groups
3. **Burnout Prediction Accuracy**: Validate C-index on real longitudinal data
4. **Protein Optimization**: Measure adherence and muscle mass changes

## 16.3 Secondary Outcomes

1. User satisfaction scores (SUS scale)
2. App engagement metrics
3. Self-reported confidence in exercise form
4. Sleep and stress improvements

---

# 17. GAPS ANALYSIS & FUTURE WORK

## 17.1 Current Limitations

| Gap | Description | Priority |
|-----|-------------|----------|
| Synthetic Burnout Data | Model trained on synthetic data, needs real longitudinal validation | High |
| CPU-Only Deployment | Limited to CPU inference on Cloud Run | Medium |
| Limited Exercise Types | 7 exercises supported (squat, deadlift, bench, press, row, lunge, pushup) | Medium |
| No Multi-Person Tracking | Single person per frame only | Low |
| Offline Mode | No offline capability for mobile | Low |

## 17.2 Recommended Improvements

1. **Clinical Validation**: Conduct IRB-approved study (N≥100, 12 weeks)
2. **Model Optimization**: Convert to ONNX/TensorRT for faster inference
3. **Real Burnout Data**: Replace synthetic data with real longitudinal data
4. **Additional Exercises**: Add squat variations, Olympic lifts, yoga poses
5. **Multi-Language Support**: Add voice feedback in Hindi, Spanish, etc.
6. **Wearable Integration**: Connect with Apple Watch, Fitbit for HRV data

## 17.3 Publication Readiness

| Component | Status |
|-----------|--------|
| Code Documentation | ✅ Complete |
| Model Metrics | ✅ C-index = 0.842 |
| Architecture Documentation | ✅ Complete |
| IRB Approval | ⏳ Required |
| Clinical Trial | ⏳ Required |
| Statistical Analysis | ⏳ Required |

---

# APPENDIX A: DATA CLASS DEFINITIONS

```python
@dataclass
class JointPosition:
    """Represents a joint position in 3D space"""
    x: float
    y: float
    z: float
    confidence: float

@dataclass
class FormError:
    """Represents a specific form error in an exercise"""
    body_part: str
    issue: str
    current_value: float
    expected_range: Tuple[float, float]
    severity: str  # 'minor', 'moderate', 'severe'
    correction_tip: str

@dataclass
class BiomechanicsAnalysis:
    """Results of biomechanics analysis"""
    exercise_type: str
    form_score: float  # 0-100
    risk_factors: List[str]
    recommendations: List[str]
    joint_angles: Dict[str, float]
    torque_data: Dict[str, List[float]]
    heatmap_data: Dict[str, List[List[float]]]
    is_valid_exercise: bool = True
    error_message: str = ""
    form_errors: List[FormError] = None

@dataclass
class FoodItem:
    """Represents a detected food item"""
    name: str
    confidence: float
    protein_content: float  # grams
    calories: float
    serving_size: str
    bounding_box: Tuple[int, int, int, int]
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0

@dataclass
class MealAnalysis:
    """Results of meal photo analysis"""
    total_protein: float
    total_calories: float
    detected_foods: List[FoodItem]
    protein_deficit: float
    recommendations: List[str]
    meal_quality_score: float
    nutritional_balance: Dict[str, float]

@dataclass
class BurnoutRiskFactors:
    """Risk factors for burnout prediction"""
    workout_frequency: int
    sleep_hours: float
    stress_level: int
    recovery_time: int
    performance_trend: str
    age: int
    gender: str
    experience_years: int
    training_intensity: str

@dataclass
class BurnoutAnalysis:
    """Results of burnout risk analysis"""
    risk_score: float  # 0-100
    risk_level: str  # low, medium, high, critical
    time_to_burnout: Optional[float]
    survival_probability: float
    risk_factors: List[str]
    recommendations: List[str]
    survival_curve_data: Dict[str, List[float]]
```

---

# APPENDIX B: CONFIGURATION CONSTANTS

```python
# Biomechanics Module
MIN_POSE_CONFIDENCE = 0.3
MIN_VALID_FRAMES_RATIO = 0.1
MIN_KEY_LANDMARKS = 4
KEY_LANDMARK_INDICES = [5, 6, 11, 12, 13, 14, 15, 16]

# Real-Time Processing
TARGET_WIDTH = 480
JPEG_QUALITY = 60
FRAME_SKIP = 2
MODEL_COMPLEXITY = 0
FEEDBACK_COOLDOWN = 2.0  # seconds

# Burnout Risk Thresholds
RISK_THRESHOLDS = {
    'low': 25,
    'medium': 50,
    'high': 75,
    'critical': 90
}

# Form Scoring Weights
FORM_SCORE_WEIGHTS = {
    'depth': 0.25,
    'knee_tracking': 0.30,
    'symmetry': 0.20,
    'stability': 0.15,
    'body_position': 0.10
}

# Voice Feedback
VOICE_COOLDOWN_MS = 2000
VOICE_RATE = 1.1
VOICE_PITCH = 1.0
VOICE_VOLUME = 1.0
```

---

**Document Version**: 1.0  
**Generated**: March 3, 2026  
**Codebase Commit**: main branch  
**Total Lines of Code Analyzed**: ~15,000+  
**Files Analyzed**: 25+  

---

*This technical audit was prepared for submission to IEEE Transactions on Biomedical Engineering (TBME) or JMIR for the research paper "FitBalance: A Multimodal Explainable AI Ecosystem for Preventive Fitness."*
