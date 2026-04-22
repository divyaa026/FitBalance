"""
Biomechanics Coaching Module
Real-time biomechanics analysis with GNN-LSTM architecture and torque heatmaps
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Fix for PyTorch 2.6+ weights_only restriction
    # Patch torch.load to use weights_only=False for YOLO model compatibility
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        """Patched torch.load that defaults to weights_only=False for compatibility."""
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    GCNConv = None
    global_mean_pool = None
    Data = None
    DataLoader = None
    TORCH_AVAILABLE = False

import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from fastapi import UploadFile
import json

logger = logging.getLogger(__name__)

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
    
    def __post_init__(self):
        if self.form_errors is None:
            self.form_errors = []

if TORCH_AVAILABLE:
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
else:
    # Fallback when PyTorch is not available
    class GNNLSTMModel:
        """Placeholder GNN-LSTM model when PyTorch is not available"""
        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch not available - GNNLSTMModel is a placeholder")

class BiomechanicsCoach:
    """Main class for biomechanics coaching and analysis"""
    
    # Minimum confidence threshold for pose detection (lowered for better detection)
    MIN_POSE_CONFIDENCE = 0.3
    # Minimum percentage of frames that need valid poses (lowered for short clips)
    MIN_VALID_FRAMES_RATIO = 0.1
    # Minimum number of key landmarks needed (reduced - just need core body parts)
    MIN_KEY_LANDMARKS = 4
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.gnn_model = GNNLSTMModel()
        else:
            self.gnn_model = None
            logger.warning("PyTorch not available - model-based features will be disabled")
        
        self.joint_connections = self._define_joint_connections()
        self.exercise_standards = self._load_exercise_standards()
        self.user_data = {}  # In production, use database
        
        # Key landmark indices for exercise detection (YOLO COCO 17-point format)
        # 5,6: shoulders, 11,12: hips, 13,14: knees, 15,16: ankles
        self.key_landmark_indices = [5, 6, 11, 12, 13, 14, 15, 16]

        # Pose backends (YOLO preferred, MediaPipe fallback)
        self.yolo_model = None
        self.mediapipe_pose = None
        self.pose_backend = "none"
        
        # Load YOLO pose model
        self._load_yolo_model()
        
        # Load pre-trained GNN model (placeholder)
        self._load_model()
    
    def _load_yolo_model(self):
        """Load YOLO pose estimation model"""
        try:
            from ultralytics import YOLO
            model_loaded = False

            # Prefer explicit absolute paths to avoid cwd-dependent failures.
            model_paths = self._resolve_yolo_model_paths()
            for path in model_paths:
                try:
                    self.yolo_model = YOLO(path)
                    logger.info(f"YOLO pose model loaded from: {path}")
                    self.pose_backend = "yolo"
                    model_loaded = True
                    break
                except Exception as load_err:
                    logger.debug(f"Failed loading YOLO model from {path}: {load_err}")
                    continue

            if not model_loaded:
                # Download the model if not found locally
                logger.info("Local YOLO pose model not found. Attempting fallback download...")
                try:
                    self.yolo_model = YOLO('yolov8n-pose.pt')
                    self.pose_backend = "yolo"
                    logger.info("YOLO pose model downloaded and loaded")
                except Exception as download_err:
                    logger.warning(f"Failed to download YOLO pose model: {download_err}")
                    self.yolo_model = None

        except ImportError as e:
            logger.warning(f"Ultralytics is not installed; YOLO backend unavailable: {e}")
            self.yolo_model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None

        if self.yolo_model is None:
            self._load_mediapipe_pose_model()

    def _resolve_yolo_model_paths(self) -> List[str]:
        """Resolve candidate YOLO model paths as absolute paths first."""
        module_dir = Path(__file__).resolve().parent
        backend_dir = module_dir.parent
        repo_root = backend_dir.parent

        candidate_paths = [
            backend_dir / "yolov8n-pose.pt",
            repo_root / "yolov8n-pose.pt",
            backend_dir / "backend" / "yolov8n-pose.pt",
            backend_dir / "fitbalance-backend" / "yolov8n-pose.pt",
            Path.cwd() / "yolov8n-pose.pt",
        ]

        existing_paths = []
        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                existing_paths.append(str(candidate))

        # Keep the default name as a final fallback for ultralytics cache/download logic.
        existing_paths.append("yolov8n-pose.pt")
        return existing_paths

    def _load_mediapipe_pose_model(self):
        """Load MediaPipe PoseLandmarker (Tasks API) as a fallback when YOLO is unavailable.
        
        MediaPipe 0.10+ removed the legacy `solutions` API. The Tasks API
        (mp.tasks.vision.PoseLandmarker) is the supported replacement.
        """
        try:
            import mediapipe as mp
            from pathlib import Path as _Path

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode

            # Resolve model file — look next to this module, then the backend dir
            module_dir = _Path(__file__).resolve().parent
            backend_dir = module_dir.parent
            model_candidates = [
                module_dir / 'pose_landmarker_lite.task',
                backend_dir / 'pose_landmarker_lite.task',
                _Path.cwd() / 'pose_landmarker_lite.task',
            ]
            model_path = None
            for candidate in model_candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    break

            if model_path is None:
                logger.warning(
                    "pose_landmarker_lite.task not found — attempting auto-download..."
                )
                import urllib.request
                dest = backend_dir / 'pose_landmarker_lite.task'
                url = ('https://storage.googleapis.com/mediapipe-models/'
                       'pose_landmarker/pose_landmarker_lite/float16/latest/'
                       'pose_landmarker_lite.task')
                urllib.request.urlretrieve(url, dest)
                model_path = str(dest)
                logger.info(f"Downloaded pose model to {dest}")

            # Use IMAGE mode: stateless per-frame detection, no monotonically
            # increasing timestamp requirement — correct for uploaded video files.
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
            )
            self.mediapipe_pose = PoseLandmarker.create_from_options(options)
            self.pose_backend = 'mediapipe'
            logger.info(f"MediaPipe PoseLandmarker (IMAGE mode) initialized from {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe fallback: {e}")
            self.mediapipe_pose = None
            self.pose_backend = 'none'

    def get_pose_backend_name(self) -> str:
        """Get a display name for the active pose detection backend."""
        if self.pose_backend == "yolo":
            return "YOLO"
        if self.pose_backend == "mediapipe":
            return "MediaPipe"
        return "Unavailable"

    def _extract_mediapipe_frame_joints(
        self, frame: np.ndarray, timestamp_ms: int = 0
    ) -> Tuple[List[JointPosition], int, float]:
        """Extract 17-point COCO-like joints from a frame using MediaPipe PoseLandmarker.

        Uses IMAGE running mode — stateless, works across multiple requests.
        The timestamp_ms parameter is kept for signature compatibility but unused.
        """
        if self.mediapipe_pose is None:
            return [], 0, 0.0

        import mediapipe as mp
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.mediapipe_pose.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return [], 0, 0.0

        # Tasks API returns NormalizedLandmark objects (no 'visibility', use 'presence')
        landmarks = result.pose_landmarks[0]
        frame_joints = [JointPosition(x=0.0, y=0.0, z=0.0, confidence=0.0) for _ in range(17)]

        # Map MediaPipe 33-point landmarks → COCO-17 indices
        mp_to_coco = {
            0:  0,   # nose
            11: 5,   # left shoulder
            12: 6,   # right shoulder
            13: 7,   # left elbow
            14: 8,   # right elbow
            15: 9,   # left wrist
            16: 10,  # right wrist
            23: 11,  # left hip
            24: 12,  # right hip
            25: 13,  # left knee
            26: 14,  # right knee
            27: 15,  # left ankle
            28: 16,  # right ankle
        }

        key_landmarks_detected = 0
        total_confidence = 0.0

        for mp_idx, coco_idx in mp_to_coco.items():
            if mp_idx >= len(landmarks):
                continue
            lm = landmarks[mp_idx]
            # NormalizedLandmark Tasks API: visibility = in-frame likelihood (primary)
            # presence = in-scene likelihood (secondary fallback)
            vis = getattr(lm, 'visibility', None)
            pres = getattr(lm, 'presence', None)
            confidence = float(vis if vis is not None else (pres if pres is not None else 0.0))
            frame_joints[coco_idx] = JointPosition(
                x=float(lm.x),
                y=float(lm.y),
                z=float(getattr(lm, 'z', 0.0)),
                confidence=confidence,
            )
            if coco_idx in self.key_landmark_indices and confidence >= self.MIN_POSE_CONFIDENCE:
                key_landmarks_detected += 1
                total_confidence += confidence

        return frame_joints, key_landmarks_detected, total_confidence
    
    def _define_joint_connections(self) -> List[Tuple[int, int]]:
        """Define anatomical joint connections for GNN"""
        # YOLO COCO 17-point pose format
        # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists
        # 11-12: hips, 13-14: knees, 15-16: ankles
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
    
    def _load_model(self):
        """Load pre-trained model weights"""
        try:
            # Set self.model to the GNN model for inference
            self.model = self.gnn_model
            
            # Try to load pre-trained weights if available
            import os
            model_paths = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'biomechanics', 'models', 'gnn_lstm_model.pth'),
                'ml/biomechanics/models/gnn_lstm_model.pth',
                '../ml/biomechanics/models/gnn_lstm_model.pth',
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        checkpoint = torch.load(path, weights_only=False)
                        
                        # Handle different saved formats
                        if isinstance(checkpoint, dict):
                            if 'model_state_dict' in checkpoint:
                                # Model was saved with extra metadata
                                state_dict = checkpoint['model_state_dict']
                            elif 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                # Assume the dict is the state dict itself
                                state_dict = checkpoint
                        else:
                            state_dict = checkpoint
                        
                        # Try to load the state dict
                        try:
                            self.model.load_state_dict(state_dict)
                            logger.info(f"Loaded pre-trained GNN model from: {path}")
                            break
                        except Exception as load_error:
                            # If keys don't match, log and continue with untrained model
                            logger.warning(f"Model architecture mismatch, using untrained model: {load_error}")
                    except Exception as e:
                        logger.warning(f"Could not load weights from {path}: {e}")
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Biomechanics model initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
            self.model = self.gnn_model  # Use untrained model as fallback
    
    async def analyze_movement(self, video_file: UploadFile, exercise_type: str, user_id: str) -> BiomechanicsAnalysis:
        """Analyze movement from uploaded video or image"""
        try:
            logger.info(f"Starting biomechanics analysis for user {user_id}, exercise: {exercise_type}")
            
            # Determine file type (image or video)
            is_image = False
            file_extension = '.mp4'  # default
            if video_file.filename:
                lower_name = video_file.filename.lower()
                if lower_name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                    file_extension = '.jpg'
                    is_image = True
                elif lower_name.endswith('.mp4'):
                    file_extension = '.mp4'
                elif lower_name.endswith('.mov'):
                    file_extension = '.mov'
                elif lower_name.endswith('.webm'):
                    file_extension = '.webm'
                elif lower_name.endswith('.avi'):
                    file_extension = '.avi'
            
            # Also check content type
            if video_file.content_type and video_file.content_type.startswith('image/'):
                is_image = True
                file_extension = '.jpg'
            
            logger.info(f"File type detected: {'image' if is_image else 'video'}, extension: {file_extension}")
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                content = await video_file.read()
                tmp_file.write(content)
                file_path = tmp_file.name
            
            logger.info(f"Saved temporary file: {file_path} ({len(content)} bytes)")
            
            # Extract pose data - use different method for images vs videos
            if is_image:
                pose_data, validation_info = self._extract_pose_from_image(file_path)
            else:
                pose_data, validation_info = self._extract_pose_data_with_validation(file_path)
            
            # Clean up temp file early
            try:
                os.unlink(file_path)
            except:
                pass
            
            # Check if valid exercise was detected
            if not validation_info['is_valid_exercise']:
                logger.warning(f"No valid exercise detected: {validation_info['error_message']}")
                return BiomechanicsAnalysis(
                    exercise_type=exercise_type,
                    form_score=0.0,
                    risk_factors=[],
                    recommendations=[],
                    joint_angles={},
                    torque_data={},
                    heatmap_data={},
                    is_valid_exercise=False,
                    error_message=validation_info['error_message'],
                    form_errors=[]
                )
            
            # Detect what exercise is actually in the video
            detected_exercise = self._detect_exercise_type(pose_data)
            logger.info(f"User selected: {exercise_type}, Detected exercise: {detected_exercise}")
            
            # Check if detected exercise matches the selected exercise type
            exercise_mismatch = not self._are_exercises_compatible(exercise_type, detected_exercise)
            
            # If there's a mismatch, REJECT the video with a clear error message
            if exercise_mismatch:
                logger.warning(f"Exercise mismatch: selected '{exercise_type}' but detected '{detected_exercise}'")
                return BiomechanicsAnalysis(
                    exercise_type=exercise_type,
                    form_score=0.0,
                    risk_factors=["exercise_type_mismatch"],
                    recommendations=[
                        f"❌ Wrong exercise detected! You selected '{exercise_type}' but uploaded a '{detected_exercise}' video.",
                        f"Please upload a video of you performing a {exercise_type}.",
                        f"💡 Tip: Make sure the entire {exercise_type} movement is visible in the video."
                    ],
                    joint_angles={},
                    torque_data={},
                    heatmap_data={},
                    is_valid_exercise=False,
                    error_message=f"Exercise mismatch: You selected '{exercise_type}' but the video shows '{detected_exercise}'. Please upload a {exercise_type} video.",
                    form_errors=[]
                )
            
            # Analyze movement using pose data with the selected exercise type
            analysis = self._analyze_pose_sequence(pose_data, exercise_type)
            
            # Generate form errors with specific details
            analysis.form_errors = self._identify_form_errors(analysis.joint_angles, exercise_type)
            
            # Generate recommendations based on form analysis
            analysis.recommendations = self._generate_recommendations(analysis, exercise_type)
            
            # Keep the exercise type as what the user selected
            analysis.exercise_type = exercise_type
            
            # Store user data
            if user_id not in self.user_data:
                self.user_data[user_id] = []
            self.user_data[user_id].append(analysis)
            
            logger.info(f"Analysis complete: Form score = {analysis.form_score}, Risks = {len(analysis.risk_factors)}, Form errors = {len(analysis.form_errors)}")
            return analysis
            
        except Exception as e:
            logger.error(f"Movement analysis error: {str(e)}", exc_info=True)
            raise
    
    def _extract_pose_from_image(self, image_path: str) -> Tuple[List[List[JointPosition]], Dict]:
        """Extract pose data from a single image using YOLO pose estimation"""
        pose_data = []
        validation_info = {
            'is_valid_exercise': False,
            'error_message': '',
            'total_frames': 1,
            'valid_pose_frames': 0,
            'avg_confidence': 0.0
        }
        
        try:
            if self.yolo_model is None and self.mediapipe_pose is None:
                validation_info['error_message'] = "Pose detection model is not available. Please try again later."
                return pose_data, validation_info
            
            logger.info(f"Processing image file: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                validation_info['error_message'] = "Could not read image file. Please upload a valid image (JPG, PNG)."
                return pose_data, validation_info
            
            logger.info(f"Image loaded: {image.shape}")
            
            if self.yolo_model is not None:
                # Run YOLO pose estimation
                results = self.yolo_model(image, verbose=False)

                if results and len(results) > 0 and results[0].keypoints is not None:
                    kpts = results[0].keypoints

                    if hasattr(kpts, 'xy') and kpts.xy is not None and len(kpts.xy) > 0:
                        # Get first person's keypoints
                        xy = kpts.xy[0].cpu().numpy()  # Shape: (17, 2)
                        conf = kpts.conf[0].cpu().numpy() if hasattr(kpts, 'conf') and kpts.conf is not None else np.ones(17)

                        frame_joints = []
                        key_landmarks_detected = 0
                        total_confidence = 0.0

                        for idx in range(len(xy)):
                            joint = JointPosition(
                                x=float(xy[idx][0]) / image.shape[1],  # Normalize to 0-1
                                y=float(xy[idx][1]) / image.shape[0],
                                z=0.0,  # YOLO doesn't provide depth
                                confidence=float(conf[idx]) if idx < len(conf) else 0.5
                            )
                            frame_joints.append(joint)

                            # Check key landmarks
                            if idx in self.key_landmark_indices and joint.confidence >= 0.2:
                                key_landmarks_detected += 1
                                total_confidence += joint.confidence

                        logger.info(f"Image analysis (YOLO): {key_landmarks_detected} key landmarks detected out of {len(self.key_landmark_indices)}")

                        # For images, we only need a few key landmarks
                        if key_landmarks_detected >= 3:  # At least 3 key body parts visible
                            pose_data.append(frame_joints)
                            # Duplicate the frame to create a mini "sequence" for analysis
                            pose_data.append(frame_joints)
                            pose_data.append(frame_joints)

                            validation_info['valid_pose_frames'] = 1
                            validation_info['avg_confidence'] = total_confidence / max(key_landmarks_detected, 1)
                            validation_info['is_valid_exercise'] = True
                            logger.info(f"Valid pose detected in image with {validation_info['avg_confidence']:.2f} confidence")
                        else:
                            validation_info['error_message'] = f"Could not detect enough body parts in the image. Only {key_landmarks_detected} key points found. Please upload a clearer image showing your full body or exercise form."
                    else:
                        validation_info['error_message'] = "No human body detected in the image. Please upload an image showing a person performing an exercise or their workout form."
                else:
                    validation_info['error_message'] = "No human body detected in the image. Please upload an image showing a person performing an exercise or their workout form."
            else:
                frame_joints, key_landmarks_detected, total_confidence = self._extract_mediapipe_frame_joints(image, timestamp_ms=0)
                logger.info(f"Image analysis (MediaPipe): {key_landmarks_detected} key landmarks detected out of {len(self.key_landmark_indices)}")

                if key_landmarks_detected >= 3 and frame_joints:
                    pose_data.append(frame_joints)
                    pose_data.append(frame_joints)
                    pose_data.append(frame_joints)

                    validation_info['valid_pose_frames'] = 1
                    validation_info['avg_confidence'] = total_confidence / max(key_landmarks_detected, 1)
                    validation_info['is_valid_exercise'] = True
                    logger.info(f"Valid pose detected in image with {validation_info['avg_confidence']:.2f} confidence")
                else:
                    validation_info['error_message'] = "No human body detected in the image. Please upload an image showing a person performing an exercise or their workout form."
                    
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            validation_info['error_message'] = f"Error analyzing image. Please try a different image."
        
        return pose_data, validation_info
    
    def _extract_pose_data_with_validation(self, video_path: str) -> Tuple[List[List[JointPosition]], Dict]:
        """Extract pose data from video using YOLO pose estimation with validation"""
        pose_data = []
        validation_info = {
            'is_valid_exercise': False,
            'error_message': '',
            'total_frames': 0,
            'valid_pose_frames': 0,
            'avg_confidence': 0.0
        }
        
        try:
            if self.yolo_model is None and self.mediapipe_pose is None:
                validation_info['error_message'] = "Pose detection model is not available. Please try again later."
                return pose_data, validation_info
            
            logger.info(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error("Failed to open video file")
                validation_info['error_message'] = "Could not open video file. Please upload a valid video."
                return pose_data, validation_info
            
            frame_count = 0
            valid_pose_frames = 0
            total_confidence = 0.0
            confidence_samples = 0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                # Compute monotonically increasing timestamp in milliseconds for MediaPipe Tasks API
                timestamp_ms = int((frame_count - 1) * (1000.0 / fps))
                
                if self.yolo_model is not None:
                    # Run YOLO pose estimation
                    results = self.yolo_model(frame, verbose=False)

                    if results and len(results) > 0 and results[0].keypoints is not None:
                        kpts = results[0].keypoints

                        if hasattr(kpts, 'xy') and kpts.xy is not None and len(kpts.xy) > 0:
                            # Get first person's keypoints
                            xy = kpts.xy[0].cpu().numpy()  # Shape: (17, 2)
                            conf = kpts.conf[0].cpu().numpy() if hasattr(kpts, 'conf') and kpts.conf is not None else np.ones(17)

                            frame_joints = []
                            key_landmarks_detected = 0
                            frame_confidence = 0.0

                            for idx in range(len(xy)):
                                joint = JointPosition(
                                    x=float(xy[idx][0]) / frame_width if frame_width > 0 else 0,
                                    y=float(xy[idx][1]) / frame_height if frame_height > 0 else 0,
                                    z=0.0,
                                    confidence=float(conf[idx]) if idx < len(conf) else 0.5
                                )
                                frame_joints.append(joint)

                                # Check if this is a key landmark for exercise detection
                                if idx in self.key_landmark_indices and joint.confidence >= self.MIN_POSE_CONFIDENCE:
                                    key_landmarks_detected += 1
                                    frame_confidence += joint.confidence

                            # Only count frame as valid if enough key landmarks detected
                            if key_landmarks_detected >= self.MIN_KEY_LANDMARKS:
                                valid_pose_frames += 1
                                pose_data.append(frame_joints)
                                total_confidence += frame_confidence / key_landmarks_detected
                                confidence_samples += 1

                            # Log every 10 frames
                            if frame_count % 10 == 0:
                                logger.info(f"Frame {frame_count}: {key_landmarks_detected} key landmarks detected")
                else:
                    frame_joints, key_landmarks_detected, frame_confidence = self._extract_mediapipe_frame_joints(frame, timestamp_ms)
                    if key_landmarks_detected >= self.MIN_KEY_LANDMARKS and frame_joints:
                        valid_pose_frames += 1
                        pose_data.append(frame_joints)
                        total_confidence += frame_confidence / max(key_landmarks_detected, 1)
                        confidence_samples += 1

                    if frame_count % 10 == 0:
                        logger.info(f"Frame {frame_count}: {key_landmarks_detected} key landmarks detected (MediaPipe)")
            
            cap.release()
            
            # Update validation info
            validation_info['total_frames'] = frame_count
            validation_info['valid_pose_frames'] = valid_pose_frames
            validation_info['avg_confidence'] = total_confidence / confidence_samples if confidence_samples > 0 else 0.0
            
            logger.info(f"Extracted pose data: {frame_count} total frames, {valid_pose_frames} valid poses")
            
            # Validate if this is a valid exercise video
            if frame_count == 0:
                validation_info['error_message'] = "Could not read the video file. Please upload a valid workout video (MP4, MOV, WebM)."
            elif valid_pose_frames == 0:
                validation_info['error_message'] = "No workout or exercise form detected. Please upload a video showing a person performing an exercise."
            elif valid_pose_frames / frame_count < self.MIN_VALID_FRAMES_RATIO:
                validation_info['error_message'] = f"Could not clearly detect exercise form. Please upload a video with better lighting and a clear view of your body performing the exercise."
            else:
                validation_info['is_valid_exercise'] = True
                logger.info(f"Valid exercise detected with {validation_info['avg_confidence']:.2f} average confidence")
            
        except Exception as e:
            logger.error(f"Error extracting pose data: {e}", exc_info=True)
            validation_info['error_message'] = f"Error processing video. Please try a different video file."
        
        return pose_data, validation_info
    
    def _extract_pose_data(self, video_path: str) -> List[List[JointPosition]]:
        """Extract pose data from video using YOLO (legacy method)"""
        pose_data, _ = self._extract_pose_data_with_validation(video_path)
        return pose_data if pose_data else self._generate_mock_pose_data()
    
    def _generate_mock_pose_data(self) -> List[List[JointPosition]]:
        """Generate mock pose data for testing"""
        frames = 30
        joints_per_frame = 17  # YOLO COCO pose landmarks
        
        pose_data = []
        for frame in range(frames):
            frame_joints = []
            for joint in range(joints_per_frame):
                joint_pos = JointPosition(
                    x=np.random.uniform(0, 1),
                    y=np.random.uniform(0, 1),
                    z=np.random.uniform(-0.5, 0.5),
                    confidence=np.random.uniform(0.7, 1.0)
                )
                frame_joints.append(joint_pos)
            pose_data.append(frame_joints)
        
        return pose_data
    
    def _analyze_pose_sequence(self, pose_data: List[List[JointPosition]], exercise_type: str) -> BiomechanicsAnalysis:
        """Analyze pose sequence using GNN-LSTM model"""
        
        # Get the number of landmarks detected in the first frame
        num_landmarks = len(pose_data[0]) if pose_data else 33
        
        # Convert pose data to tensor format
        features = []
        for frame in pose_data:
            frame_features = []
            for joint in frame:
                frame_features.extend([joint.x, joint.y, joint.z])
            features.append(frame_features)
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Filter joint connections to only include landmarks that were detected
        valid_connections = [
            (i, j) for i, j in self.joint_connections 
            if i < num_landmarks and j < num_landmarks
        ]
        
        logger.info(f"Using {len(valid_connections)} valid connections out of {len(self.joint_connections)} for {num_landmarks} landmarks")
        
        # Create edge index for GNN
        if valid_connections:
            edge_index = torch.tensor(valid_connections, dtype=torch.long).t().contiguous()
        else:
            # Fallback: create minimal connections if none are valid
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        
        # Create graph data
        data = Data(x=features, edge_index=edge_index)
        
        # Model inference
        try:
            with torch.no_grad():
                classification, regression = self.model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
        except Exception as e:
            logger.warning(f"Model inference failed: {e}, continuing with angle-based analysis")
        
        # Calculate joint angles
        joint_angles = self._calculate_joint_angles(pose_data)
        
        # Calculate torque data
        torque_data = self._calculate_torque_data(pose_data, joint_angles)
        
        # Generate heatmap data
        heatmap_data = self._generate_heatmap_data(torque_data)
        
        # Assess form quality
        form_score = self._assess_form_quality(joint_angles, exercise_type)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(joint_angles, exercise_type)
        
        return BiomechanicsAnalysis(
            exercise_type=exercise_type,
            form_score=form_score,
            risk_factors=risk_factors,
            recommendations=[],  # Will be filled by _generate_recommendations
            joint_angles=joint_angles,
            torque_data=torque_data,
            heatmap_data=heatmap_data
        )
    
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
                    
                    # Skip low confidence detections
                    min_conf = min(frame[5].confidence, frame[11].confidence, 
                                   frame[13].confidence, frame[15].confidence)
                    if min_conf < 0.3:
                        continue
                    
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
                    
                    # === HIP ANGLE (left side) ===
                    v3 = left_shoulder - left_hip
                    v4 = left_knee - left_hip
                    doth = np.dot(v3, v4)
                    normh = np.linalg.norm(v3) * np.linalg.norm(v4)
                    if normh > 0:
                        hip_angle = np.degrees(np.arccos(np.clip(doth / normh, -1, 1)))
                        frame_data['left_hip_angles'].append(hip_angle)
                    
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
                    
                    # === TORSO LEAN (shoulder vs hip position) ===
                    shoulder_center = (left_shoulder + right_shoulder) / 2
                    hip_center = (left_hip + right_hip) / 2
                    torso_lean = abs(shoulder_center[0] - hip_center[0])  # Lateral lean
                    frame_data['shoulder_hip_ratio'].append(torso_lean)
                    
                    # === HIP DEPTH ===
                    hip_height = (left_hip[1] + right_hip[1]) / 2
                    knee_height = (left_knee[1] + right_knee[1]) / 2
                    frame_data['hip_heights'].append(hip_height - knee_height)  # Negative = hip below knee
                    
                except (IndexError, ValueError) as e:
                    continue
            
            # === ANALYZE COLLECTED DATA ===
            
            # 1. KNEE ANGLE - use 10th percentile (near deepest position, but not extreme outliers)
            if frame_data['left_knee_angles']:
                left_knee_min = np.percentile(frame_data['left_knee_angles'], 10)
                right_knee_min = np.percentile(frame_data['right_knee_angles'], 10) if frame_data['right_knee_angles'] else left_knee_min
                
                avg_knee = (left_knee_min + right_knee_min) / 2
                angles['knee'] = round(float(avg_knee), 1)
                
                # Symmetry penalty: large difference between left and right
                knee_asymmetry = abs(left_knee_min - right_knee_min)
                form_metrics['knee_asymmetry'] = knee_asymmetry
                logger.info(f"Knee angles - L: {left_knee_min:.1f}°, R: {right_knee_min:.1f}°, avg: {avg_knee:.1f}°, asymmetry: {knee_asymmetry:.1f}°")
            
            # 2. HIP ANGLE
            if frame_data['left_hip_angles']:
                hip_10th = np.percentile(frame_data['left_hip_angles'], 10)
                angles['hip'] = round(float(hip_10th), 1)
                logger.info(f"Hip angle - selected: {hip_10th:.1f}°")
            
            # 3. KNEE TRACKING SCORE (critical for form quality)
            if frame_data['knee_hip_alignment']:
                # Get tracking during the deepest part of movement (when it matters most)
                # Lower percentile = during bent position
                tracking_scores = frame_data['knee_hip_alignment']
                
                # Find frames where knees are most bent (lowest knee angle)
                if frame_data['left_knee_angles'] and len(tracking_scores) == len(frame_data['left_knee_angles']):
                    knee_angles_arr = np.array(frame_data['left_knee_angles'])
                    bent_threshold = np.percentile(knee_angles_arr, 25)  # Bottom 25% of positions
                    bent_indices = knee_angles_arr < bent_threshold
                    
                    tracking_during_bent = [tracking_scores[i] for i in range(len(tracking_scores)) if bent_indices[i]]
                    
                    if tracking_during_bent:
                        avg_tracking = np.mean(tracking_during_bent)
                        form_metrics['knee_tracking'] = avg_tracking
                        logger.info(f"Knee tracking during bent position: {avg_tracking:.4f} (positive = good, negative = valgus)")
            
            # 4. STABILITY - standard deviation of angles during bottom position
            if frame_data['left_knee_angles']:
                knee_angles_arr = np.array(frame_data['left_knee_angles'])
                if len(knee_angles_arr) > 10:
                    bent_mask = knee_angles_arr < np.percentile(knee_angles_arr, 40)
                    stability = np.std(knee_angles_arr[bent_mask]) if np.sum(bent_mask) > 3 else 30
                    form_metrics['stability'] = stability
                    logger.info(f"Movement stability (lower is better): {stability:.1f}°")
            
            # 5. DEPTH ACHIEVEMENT
            if frame_data['hip_heights']:
                min_hip_height = min(frame_data['hip_heights'])
                form_metrics['depth'] = min_hip_height  # Negative = below knees (good depth)
                logger.info(f"Hip depth relative to knees: {min_hip_height:.3f} (negative = below knees)")
            
            # Store metrics in angles dict for scoring
            angles['_form_metrics'] = form_metrics
            
            logger.info(f"Final calculated angles and metrics: {angles}")
            
        except Exception as e:
            logger.error(f"Error calculating joint angles: {e}", exc_info=True)
            angles = {}
        
        return angles
    
    def _calculate_torque_data(self, pose_data: List[List[JointPosition]], joint_angles: Dict[str, float]) -> Dict[str, List[float]]:
        """Calculate torque data for visualization based on actual movement.
        
        Torque estimates are based on:
        - Joint angle deviation from neutral/optimal positions
        - Position in movement cycle
        - Simplified biomechanical model
        """
        torque_data = {
            'knee_torque': [],
            'hip_torque': [],
            'ankle_torque': [],
            'back_torque': []
        }
        
        # Get baseline angles (or use defaults)
        base_knee = joint_angles.get('knee', 120)
        base_hip = joint_angles.get('hip', 90)
        base_ankle = joint_angles.get('ankle', 90)
        
        # Calculate torque for each frame based on actual pose
        for i, frame in enumerate(pose_data):
            progress = i / max(len(pose_data) - 1, 1)
            
            if len(frame) >= 17:
                try:
                    # Extract positions
                    left_hip = np.array([frame[11].x, frame[11].y])
                    left_knee = np.array([frame[13].x, frame[13].y])
                    left_ankle = np.array([frame[15].x, frame[15].y])
                    
                    # Calculate segment lengths
                    thigh_len = np.linalg.norm(left_knee - left_hip)
                    shin_len = np.linalg.norm(left_ankle - left_knee)
                    
                    # Torque increases with joint flexion (lower angles = more torque)
                    # Also consider distance from center of mass
                    knee_torque = max(0, (180 - base_knee) / 180 * 100 * (1 + 0.3 * np.sin(progress * np.pi * 2)))
                    hip_torque = max(0, (180 - base_hip) / 180 * 120 * (1 + 0.2 * np.cos(progress * np.pi * 2)))
                    ankle_torque = max(0, (180 - base_ankle) / 180 * 60 * (1 + 0.25 * np.sin(progress * np.pi)))
                    back_torque = max(0, (180 - base_hip) / 180 * 80 * (1 + 0.15 * np.cos(progress * np.pi)))
                    
                except (IndexError, ValueError):
                    knee_torque = 50
                    hip_torque = 60
                    ankle_torque = 30
                    back_torque = 40
            else:
                # Fallback values
                knee_torque = 50
                hip_torque = 60
                ankle_torque = 30
                back_torque = 40
            
            torque_data['knee_torque'].append(float(knee_torque))
            torque_data['hip_torque'].append(float(hip_torque))
            torque_data['ankle_torque'].append(float(ankle_torque))
            torque_data['back_torque'].append(float(back_torque))
        
        return torque_data
    
    def _generate_heatmap_data(self, torque_data: Dict[str, List[float]]) -> Dict[str, List[List[float]]]:
        """Generate heatmap data for visualization"""
        heatmap_data = {}
        
        for joint, torques in torque_data.items():
            # Create 2D heatmap grid
            grid_size = 20
            heatmap = np.zeros((grid_size, grid_size))
            
            # Distribute torque values across grid
            for i, torque in enumerate(torques):
                row = int((i / len(torques)) * grid_size)
                col = int((torque + 200) / 400 * grid_size)  # Normalize to 0-20
                col = max(0, min(grid_size - 1, col))
                
                if 0 <= row < grid_size:
                    heatmap[row, col] += abs(torque)
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            heatmap_data[joint] = heatmap.tolist()
        
        return heatmap_data
    
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
        
        if not joint_angles:
            logger.warning(f"No joint angles provided for assessment")
            return 65.0
        
        standards = self.exercise_standards[exercise_type.lower()]
        
        logger.info(f"=== FORM QUALITY ASSESSMENT for {exercise_type} ===")
        logger.info(f"Input angles and metrics: {joint_angles}")
        
        # Extract form metrics if available
        form_metrics = joint_angles.pop('_form_metrics', {})
        
        scores = {}
        
        # ========== 1. DEPTH SCORE (25%) ==========
        # Check if joint angles are within proper range
        depth_score = 100.0
        angles_checked = 0
        depth_details = []
        
        for joint, angle in joint_angles.items():
            if joint.startswith('_'):  # Skip internal metrics
                continue
            standard_key = f"{joint}_angle_range"
            if standard_key not in standards:
                continue
            
            angles_checked += 1
            min_angle, max_angle = standards[standard_key]
            
            if min_angle <= angle <= max_angle:
                depth_details.append(f"  {joint}: {angle:.1f}° ✓ (range: {min_angle}°-{max_angle}°)")
            else:
                deviation = abs(angle - min_angle) if angle < min_angle else abs(angle - max_angle)
                penalty = min(30, deviation * 1.5)  # Less harsh penalty for depth
                depth_score -= penalty
                depth_details.append(f"  {joint}: {angle:.1f}° ✗ {deviation:.1f}° off (range: {min_angle}°-{max_angle}°)")
        
        depth_score = max(40, depth_score)
        scores['depth'] = {'score': depth_score, 'weight': 0.25}
        logger.info(f"DEPTH SCORE: {depth_score:.1f}/100")
        for d in depth_details:
            logger.info(d)
        
        # ========== 2. KNEE TRACKING SCORE (30%) ==========
        # CRITICAL: Knees should track over toes, not cave inward (valgus)
        # Positive tracking = good (knees outside ankles)
        # Negative tracking = bad (knees caving in)
        tracking_score = 100.0
        
        if 'knee_tracking' in form_metrics:
            tracking = form_metrics['knee_tracking']
            
            # Tracking thresholds (normalized coordinates)
            # Positive values = knee outside ankle (good)
            # Negative values = knee inside ankle (valgus/bad)
            # Near zero = knee directly over ankle (perfect)
            
            if tracking >= -0.01:  # Good - knees over or outside toes
                tracking_score = 100.0
                logger.info(f"KNEE TRACKING: {tracking:.4f} ✓ Excellent (knees tracking over toes)")
            elif tracking >= -0.03:  # Slight valgus
                tracking_score = 80.0
                logger.info(f"KNEE TRACKING: {tracking:.4f} ⚠ Minor valgus tendency")
            elif tracking >= -0.05:  # Moderate valgus
                tracking_score = 60.0
                logger.info(f"KNEE TRACKING: {tracking:.4f} ✗ Moderate knee valgus")
            elif tracking >= -0.08:  # Significant valgus
                tracking_score = 40.0
                logger.info(f"KNEE TRACKING: {tracking:.4f} ✗ Significant knee valgus")
            else:  # Severe valgus
                tracking_score = 20.0
                logger.info(f"KNEE TRACKING: {tracking:.4f} ✗ Severe knee valgus (dangerous!)")
        else:
            tracking_score = 70.0  # Default if can't measure
            logger.info("KNEE TRACKING: Could not measure, using default 70")
        
        scores['tracking'] = {'score': tracking_score, 'weight': 0.30}
        logger.info(f"TRACKING SCORE: {tracking_score:.1f}/100")
        
        # ========== 3. SYMMETRY SCORE (20%) ==========
        # Left-right balance is crucial for injury prevention
        symmetry_score = 100.0
        
        if 'knee_asymmetry' in form_metrics:
            asymmetry = form_metrics['knee_asymmetry']
            
            # Asymmetry in degrees between left and right knee angles
            if asymmetry < 5:  # Less than 5° difference - excellent
                symmetry_score = 100.0
                logger.info(f"SYMMETRY: {asymmetry:.1f}° difference ✓ Excellent balance")
            elif asymmetry < 10:  # 5-10° difference - good
                symmetry_score = 85.0
                logger.info(f"SYMMETRY: {asymmetry:.1f}° difference - Good balance")
            elif asymmetry < 15:  # 10-15° difference - moderate
                symmetry_score = 65.0
                logger.info(f"SYMMETRY: {asymmetry:.1f}° difference ⚠ Moderate imbalance")
            elif asymmetry < 20:  # 15-20° difference - concerning
                symmetry_score = 45.0
                logger.info(f"SYMMETRY: {asymmetry:.1f}° difference ✗ Significant imbalance")
            else:  # >20° difference - severe
                symmetry_score = 25.0
                logger.info(f"SYMMETRY: {asymmetry:.1f}° difference ✗ Severe imbalance!")
        else:
            symmetry_score = 75.0
            logger.info("SYMMETRY: Could not measure, using default 75")
        
        scores['symmetry'] = {'score': symmetry_score, 'weight': 0.20}
        logger.info(f"SYMMETRY SCORE: {symmetry_score:.1f}/100")
        
        # ========== 4. STABILITY SCORE (15%) ==========
        # Movement should be controlled without wobbling
        stability_score = 100.0
        
        if 'stability' in form_metrics:
            stability_std = form_metrics['stability']
            
            # Standard deviation of knee angle during bottom position
            if stability_std < 3:  # Very stable
                stability_score = 100.0
                logger.info(f"STABILITY: {stability_std:.1f}° std dev ✓ Very controlled")
            elif stability_std < 6:  # Stable
                stability_score = 85.0
                logger.info(f"STABILITY: {stability_std:.1f}° std dev - Stable movement")
            elif stability_std < 10:  # Some wobble
                stability_score = 65.0
                logger.info(f"STABILITY: {stability_std:.1f}° std dev ⚠ Some instability")
            elif stability_std < 15:  # Unstable
                stability_score = 45.0
                logger.info(f"STABILITY: {stability_std:.1f}° std dev ✗ Unstable")
            else:  # Very unstable
                stability_score = 25.0
                logger.info(f"STABILITY: {stability_std:.1f}° std dev ✗ Very unstable!")
        else:
            stability_score = 70.0
            logger.info("STABILITY: Could not measure, using default 70")
        
        scores['stability'] = {'score': stability_score, 'weight': 0.15}
        logger.info(f"STABILITY SCORE: {stability_score:.1f}/100")
        
        # ========== 5. BODY POSITION SCORE (10%) ==========
        # Proper torso position (not leaning excessively)
        position_score = 100.0
        
        if 'hip' in joint_angles:
            hip_angle = joint_angles['hip']
            hip_standard = standards.get('hip_angle_range', (40, 100))
            
            if hip_standard[0] <= hip_angle <= hip_standard[1]:
                position_score = 100.0
                logger.info(f"BODY POSITION: Hip angle {hip_angle:.1f}° ✓ Good torso position")
            else:
                deviation = abs(hip_angle - hip_standard[0]) if hip_angle < hip_standard[0] else abs(hip_angle - hip_standard[1])
                position_score = max(40, 100 - deviation * 2)
                logger.info(f"BODY POSITION: Hip angle {hip_angle:.1f}° ⚠ {deviation:.1f}° off ideal")
        else:
            position_score = 70.0
            logger.info("BODY POSITION: Could not measure, using default 70")
        
        scores['position'] = {'score': position_score, 'weight': 0.10}
        logger.info(f"POSITION SCORE: {position_score:.1f}/100")
        
        # ========== CALCULATE WEIGHTED FINAL SCORE ==========
        final_score = 0.0
        total_weight = 0.0
        
        logger.info("\n=== FINAL SCORE CALCULATION ===")
        for component, data in scores.items():
            weighted = data['score'] * data['weight']
            final_score += weighted
            total_weight += data['weight']
            logger.info(f"  {component}: {data['score']:.1f} * {data['weight']:.0%} = {weighted:.1f}")
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            final_score = final_score / total_weight
        
        final_score = max(0.0, min(100.0, final_score))
        
        logger.info(f"\n=== FINAL FORM SCORE: {final_score:.1f}/100 ===")
        
        return round(final_score, 1)
    
    def _identify_risk_factors(self, joint_angles: Dict[str, float], exercise_type: str) -> List[str]:
        """Identify potential risk factors based on joint angles, form metrics, and exercise type"""
        risk_factors = []
        
        if exercise_type.lower() not in self.exercise_standards:
            return risk_factors
        
        standards = self.exercise_standards[exercise_type.lower()]
        
        logger.info(f"Identifying risk factors for {exercise_type}")
        
        # Extract form metrics if present
        form_metrics = {}
        if '_form_metrics' in joint_angles:
            form_metrics = joint_angles['_form_metrics']
        
        # ========== CHECK KNEE TRACKING (VALGUS) - CRITICAL ==========
        if 'knee_tracking' in form_metrics:
            tracking = form_metrics['knee_tracking']
            if tracking < -0.03:  # Knees caving inward
                risk_factors.append("knees_caving_in")
                logger.info(f"Risk: Knees caving in (valgus), tracking={tracking:.4f}")
        
        # ========== CHECK SYMMETRY ==========
        if 'knee_asymmetry' in form_metrics:
            asymmetry = form_metrics['knee_asymmetry']
            if asymmetry > 15:  # Significant asymmetry
                risk_factors.append("asymmetric_movement")
                logger.info(f"Risk: Asymmetric movement, {asymmetry:.1f}° difference")
        
        # ========== CHECK STABILITY ==========
        if 'stability' in form_metrics:
            stability = form_metrics['stability']
            if stability > 10:  # High variation = unstable
                risk_factors.append("unstable_movement")
                logger.info(f"Risk: Unstable movement, std dev={stability:.1f}°")
        
        # ========== CHECK JOINT ANGLES ==========
        # Check for common errors based on joint angles
        if 'knee' in joint_angles:
            knee_angle = joint_angles['knee']
            
            if knee_angle < 65:  # Extremely deep bend - risky
                risk_factors.append("excessive_knee_flexion")
                logger.info(f"Risk: Excessive knee flexion ({knee_angle:.1f}°)")
            elif knee_angle > 150:  # Insufficient knee flexion
                risk_factors.append("insufficient_knee_flexion")
                logger.info(f"Risk: Insufficient knee flexion ({knee_angle:.1f}°)")
        
        if 'hip' in joint_angles:
            hip_angle = joint_angles['hip']
            
            if hip_angle < 35:  # Excessive forward lean
                risk_factors.append("excessive_forward_lean")
                logger.info(f"Risk: Excessive forward lean ({hip_angle:.1f}°)")
            elif hip_angle > 150:  # Too upright
                risk_factors.append("insufficient_hip_hinge")
                logger.info(f"Risk: Insufficient hip hinge ({hip_angle:.1f}°)")
        
        # Exercise-specific risk assessment
        if exercise_type.lower() == "squat":
            if 'knee' in joint_angles and 'hip' in joint_angles:
                # Check for proper depth
                if joint_angles['knee'] > 120:
                    risk_factors.append("insufficient_depth")
                    logger.info("Risk: Squat depth insufficient")
        
        elif exercise_type.lower() == "deadlift":
            if 'hip' in joint_angles:
                # Check for proper hip hinge
                if joint_angles['hip'] > 100:
                    risk_factors.append("insufficient_hip_hinge")
                    logger.info("Risk: Insufficient hip hinge in deadlift")
        
        logger.info(f"Identified {len(risk_factors)} risk factors: {risk_factors}")
        return risk_factors
    
    def _identify_form_errors(self, joint_angles: Dict[str, float], exercise_type: str) -> List[FormError]:
        """Identify specific form errors with detailed information using form metrics.
        
        Provides EXERCISE-SPECIFIC correction tips based on the detected exercise.
        """
        form_errors = []
        
        exercise_lower = exercise_type.lower()
        if exercise_lower not in self.exercise_standards:
            logger.warning(f"No standards for {exercise_type}, defaulting to squat")
            exercise_lower = "squat"
        
        standards = self.exercise_standards[exercise_lower]
        
        # Extract form metrics if present
        form_metrics = {}
        if '_form_metrics' in joint_angles and isinstance(joint_angles.get('_form_metrics'), dict):
            form_metrics = joint_angles.pop('_form_metrics')
        
        # EXERCISE-SPECIFIC correction tips
        correction_tips = {
            'squat': {
                'knee': {
                    'too_low': "Reduce squat depth slightly. Don't go so deep that form breaks down.",
                    'too_high': "Go deeper in your squat. Aim for thighs parallel to the ground.",
                    'valgus': "⚠️ CRITICAL: Your knees are caving inward! Push knees OUT over your toes."
                },
                'hip': {
                    'too_low': "Keep your chest up more. You're leaning too far forward.",
                    'too_high': "Hinge more at the hips. Push your hips back as you descend."
                },
                'general': {
                    'symmetry': "Focus on equal weight distribution between both legs.",
                    'stability': "Control the descent. Brace your core throughout the movement."
                }
            },
            'deadlift': {
                'knee': {
                    'too_low': "Your knees are bending too much. Deadlift requires less knee bend than squat.",
                    'too_high': "Bend your knees slightly more at the start position.",
                    'valgus': "Keep your knees tracking outward as you lift. Don't let them cave in."
                },
                'hip': {
                    'too_low': "Keep your hips higher. You're squatting the deadlift instead of hinging.",
                    'too_high': "Hinge at the hips more. Push your hips back to load the hamstrings."
                },
                'back': {
                    'too_low': "⚠️ Your back is too horizontal. Keep chest up and maintain neutral spine.",
                    'too_high': "Hinge forward more from the hips while keeping back neutral."
                },
                'general': {
                    'symmetry': "Keep the bar path straight and centered. Equal force through both legs.",
                    'stability': "Brace core hard before lifting. The bar should move in a straight line."
                }
            },
            'lunge': {
                'knee': {
                    'too_low': "Don't let your knee go too far past your toes.",
                    'too_high': "Lunge deeper for full range of motion.",
                    'valgus': "Keep front knee tracking over your toes, not caving inward."
                },
                'hip': {
                    'too_low': "Keep torso more upright during the lunge.",
                    'too_high': "Allow a slight forward lean while keeping back straight."
                },
                'general': {
                    'symmetry': "Even out strength between left and right legs.",
                    'stability': "Focus on balance. Use a narrower stance if wobbling."
                }
            },
            'bench press': {
                'general': {
                    'symmetry': "Press evenly with both arms. The bar should stay level.",
                    'stability': "Keep feet planted and maintain back arch."
                }
            },
            'overhead press': {
                'general': {
                    'symmetry': "Press evenly overhead. Bar should travel in a straight line.",
                    'stability': "Brace core. Don't lean back excessively."
                }
            },
            'row': {
                'hip': {
                    'too_low': "Keep back more horizontal for better lat engagement.",
                    'too_high': "Hinge forward more. Back should be near parallel to ground."
                },
                'general': {
                    'symmetry': "Pull evenly with both arms.",
                    'stability': "Keep torso stable. Don't use momentum."
                }
            },
            'pushup': {
                'general': {
                    'symmetry': "Keep body straight from head to heels.",
                    'stability': "Engage core. Don't let hips sag."
                }
            }
        }
        
        # Get exercise-specific tips, default to squat if not found
        tips = correction_tips.get(exercise_lower, correction_tips['squat'])
        
        logger.info(f"Identifying form errors for {exercise_type}")
        
        # ========== CHECK KNEE TRACKING (VALGUS) ==========
        if 'knee_tracking' in form_metrics:
            tracking = form_metrics['knee_tracking']
            if tracking < -0.03:  # Moderate valgus
                severity = 'severe' if tracking < -0.06 else 'moderate'
                valgus_tip = tips.get('knee', {}).get('valgus', "Keep knees tracking over toes.")
                form_errors.append(FormError(
                    body_part="Knees",
                    issue="Knees caving inward (valgus)",
                    current_value=tracking,
                    expected_range=(0.0, 0.1),
                    severity=severity,
                    correction_tip=valgus_tip
                ))
        
        # ========== CHECK SYMMETRY ==========
        if 'knee_asymmetry' in form_metrics:
            asymmetry = form_metrics['knee_asymmetry']
            if asymmetry > 10:
                severity = 'severe' if asymmetry > 20 else ('moderate' if asymmetry > 15 else 'minor')
                sym_tip = tips.get('general', {}).get('symmetry', "Focus on even weight distribution.")
                form_errors.append(FormError(
                    body_part="Body Balance",
                    issue=f"Left-right imbalance ({asymmetry:.1f}° difference)",
                    current_value=asymmetry,
                    expected_range=(0, 5),
                    severity=severity,
                    correction_tip=sym_tip
                ))
        
        # ========== CHECK STABILITY ==========
        if 'stability' in form_metrics:
            stability_std = form_metrics['stability']
            if stability_std > 8:
                severity = 'severe' if stability_std > 15 else ('moderate' if stability_std > 10 else 'minor')
                stab_tip = tips.get('general', {}).get('stability', "Control the movement. Brace your core.")
                form_errors.append(FormError(
                    body_part="Movement Control",
                    issue=f"Unstable movement ({stability_std:.1f}° variation)",
                    current_value=stability_std,
                    expected_range=(0, 5),
                    severity=severity,
                    correction_tip=stab_tip
                ))
        
        # ========== CHECK JOINT ANGLES ==========
        for joint, angle in joint_angles.items():
            if joint.startswith('_'):
                continue
            standard_key = f"{joint}_angle_range"
            if standard_key in standards:
                min_angle, max_angle = standards[standard_key]
                
                if angle < min_angle:
                    deviation = min_angle - angle
                    severity = 'minor' if deviation < 10 else ('moderate' if deviation < 20 else 'severe')
                    tip = tips.get(joint, {}).get('too_low', f"Adjust your {joint} position for {exercise_type}.")
                    form_errors.append(FormError(
                        body_part=joint.capitalize(),
                        issue=f"{joint.capitalize()} angle too low ({angle:.1f}°)",
                        current_value=angle,
                        expected_range=(min_angle, max_angle),
                        severity=severity,
                        correction_tip=tip
                    ))
                    
                elif angle > max_angle:
                    deviation = angle - max_angle
                    severity = 'minor' if deviation < 10 else ('moderate' if deviation < 20 else 'severe')
                    tip = tips.get(joint, {}).get('too_high', f"Adjust your {joint} position for {exercise_type}.")
                    form_errors.append(FormError(
                        body_part=joint.capitalize(),
                        issue=f"{joint.capitalize()} angle too high ({angle:.1f}°)",
                        current_value=angle,
                        expected_range=(min_angle, max_angle),
                        severity=severity,
                        correction_tip=tip
                    ))
        
        # Sort by severity (severe first)
        severity_order = {'severe': 0, 'moderate': 1, 'minor': 2}
        form_errors.sort(key=lambda e: severity_order.get(e.severity, 3))
        
        logger.info(f"Identified {len(form_errors)} form errors for {exercise_type}")
        for error in form_errors:
            logger.info(f"  [{error.severity}] {error.body_part}: {error.issue}")
        
        return form_errors
    
    def _are_exercises_compatible(self, selected: str, detected: str) -> bool:
        """Check if the selected exercise is compatible with the detected exercise.
        
        This ensures users upload videos of the CORRECT exercise type.
        Exercises are only compatible if they are the same or in the same category.
        
        Categories:
        - Lower body hip hinge: deadlift, row (similar posture)
        - Lower body squat: squat, lunge (similar depth patterns)
        - Upper body push: pushup, bench press, overhead press
        
        Returns True if the exercises are compatible, False otherwise.
        """
        selected_lower = selected.lower().strip()
        detected_lower = detected.lower().strip()
        
        # Exact match is always compatible
        if selected_lower == detected_lower:
            logger.info(f"Exercise match: {selected} = {detected}")
            return True
        
        # Handle variations of the same exercise
        # e.g., "back squat" matches "squat", "romanian deadlift" matches "deadlift"
        if selected_lower in detected_lower or detected_lower in selected_lower:
            logger.info(f"Exercise variation match: {selected} ~ {detected}")
            return True
        
        # STRICT MISMATCH: These exercises are fundamentally different
        # Squat and deadlift are NEVER compatible - they are different movement patterns
        incompatible_pairs = [
            ("squat", "deadlift"),
            ("squat", "row"),
            ("squat", "overhead press"),
            ("squat", "pushup"),
            ("squat", "bench press"),
            ("deadlift", "lunge"),
            ("deadlift", "overhead press"),
            ("deadlift", "pushup"),
            ("deadlift", "bench press"),
            ("lunge", "row"),
            ("lunge", "overhead press"),
            ("lunge", "pushup"),
            ("lunge", "bench press"),
            ("row", "overhead press"),
            ("row", "pushup"),
            ("row", "bench press"),
            # Note: pushup, bench press, and overhead press are all upper body push exercises
            # and ARE compatible with each other - they share similar movement patterns
        ]
        
        for ex1, ex2 in incompatible_pairs:
            if (selected_lower == ex1 and detected_lower == ex2) or \
               (selected_lower == ex2 and detected_lower == ex1):
                logger.warning(f"Exercise INCOMPATIBLE: {selected} != {detected}")
                return False
        
        # If not in the incompatible list, check if they're in the same category
        # These are exercises that have similar movement patterns
        compatible_categories = {
            "hip_hinge": ["deadlift", "row"],
            "squat_pattern": ["squat", "lunge"],
            "upper_push": ["pushup", "bench press", "overhead press"],
        }
        
        for category, exercises in compatible_categories.items():
            if selected_lower in exercises and detected_lower in exercises:
                logger.info(f"Exercise compatible (category: {category}): {selected} ~ {detected}")
                return True
        
        # Default: Not compatible if we can't verify
        logger.warning(f"Exercise not verified compatible: {selected} != {detected}")
        return False
    
    def _detect_exercise_type(self, pose_data: List[List[JointPosition]]) -> str:
        """Detect what type of exercise is being performed using DEFINITIVE differentiators.
        
        THE KEY DIFFERENTIATOR (UNMISTAKABLE):
        =====================================
        SQUAT: Hips DROP DOWN to knee level or BELOW
        DEADLIFT: Hips stay ABOVE knee level throughout
        
        This works because:
        - Even a "bad squat" with forward lean still has hips going DOWN
        - A deadlift NEVER has hips at knee level - that's physically impossible for the movement
        
        Secondary differentiators:
        - KNEE FORWARD TRAVEL: Squats have knees moving forward over toes
        - SHIN ANGLE: Squats have forward-leaning shins, deadlifts have vertical shins
        """
        if not pose_data or len(pose_data) < 3:
            logger.warning("Not enough pose data for exercise detection, defaulting to squat")
            return "squat"
        
        try:
            # Collect movement data for analysis
            knee_angles = []
            hip_angles = []
            torso_angles = []
            
            # CRITICAL: Track hip height relative to knee height
            hip_to_knee_ratios = []  # Positive = hip above knee, Negative = hip below knee
            
            # Track knee forward travel (horizontal displacement from ankle)
            knee_forward_travel = []
            
            # Track shin angles from vertical
            shin_angles = []
            
            # Track shoulder movement for overhead press detection
            shoulder_y_positions = []
            
            for frame in pose_data:
                if len(frame) < 17:
                    continue
                    
                try:
                    # Get all key joint positions
                    left_shoulder = np.array([frame[5].x, frame[5].y])
                    right_shoulder = np.array([frame[6].x, frame[6].y])
                    left_hip = np.array([frame[11].x, frame[11].y])
                    right_hip = np.array([frame[12].x, frame[12].y])
                    left_knee = np.array([frame[13].x, frame[13].y])
                    right_knee = np.array([frame[14].x, frame[14].y])
                    left_ankle = np.array([frame[15].x, frame[15].y])
                    right_ankle = np.array([frame[16].x, frame[16].y])
                    
                    # Skip low confidence frames
                    if frame[11].confidence < 0.3 or frame[13].confidence < 0.3:
                        continue
                    
                    # ========== PRIMARY METRIC: HIP HEIGHT RELATIVE TO KNEE ==========
                    # In image coordinates, Y increases downward
                    # If hip_y > knee_y, hip is BELOW knee
                    # If hip_y < knee_y, hip is ABOVE knee
                    hip_center_y = (left_hip[1] + right_hip[1]) / 2
                    knee_center_y = (left_knee[1] + right_knee[1]) / 2
                    
                    # Positive = hip above knee, Negative = hip at/below knee
                    hip_knee_diff = knee_center_y - hip_center_y
                    hip_to_knee_ratios.append(hip_knee_diff)
                    
                    # ========== SECONDARY METRIC: KNEE FORWARD TRAVEL ==========
                    # How far knees move forward from ankles (horizontal)
                    ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
                    knee_center_x = (left_knee[0] + right_knee[0]) / 2
                    knee_forward = abs(knee_center_x - ankle_center_x)
                    knee_forward_travel.append(knee_forward)
                    
                    # ========== SECONDARY METRIC: SHIN ANGLE ==========
                    # Angle of shin from vertical (knee to ankle vector vs vertical)
                    shin_vec = left_ankle - left_knee
                    vertical = np.array([0, 1])  # Down direction
                    shin_dot = np.dot(shin_vec, vertical)
                    shin_norm = np.linalg.norm(shin_vec) * np.linalg.norm(vertical)
                    if shin_norm > 0:
                        shin_angle = np.degrees(np.arccos(np.clip(shin_dot / shin_norm, -1, 1)))
                        shin_angles.append(shin_angle)
                    
                    # ========== KNEE ANGLE ==========
                    v1 = left_hip - left_knee
                    v2 = left_ankle - left_knee
                    dot = np.dot(v1, v2)
                    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                    if norm > 0:
                        knee_angle = np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))
                        knee_angles.append(knee_angle)
                    
                    # ========== HIP ANGLE ==========
                    v3 = left_shoulder - left_hip
                    v4 = left_knee - left_hip
                    dot_hip = np.dot(v3, v4)
                    norm_hip = np.linalg.norm(v3) * np.linalg.norm(v4)
                    if norm_hip > 0:
                        hip_angle = np.degrees(np.arccos(np.clip(dot_hip / norm_hip, -1, 1)))
                        hip_angles.append(hip_angle)
                    
                    # ========== TORSO LEAN ==========
                    shoulder_center = (left_shoulder + right_shoulder) / 2
                    hip_center = (left_hip + right_hip) / 2
                    torso_vec = shoulder_center - hip_center
                    up_vec = np.array([0, -1])
                    torso_dot = np.dot(torso_vec, up_vec)
                    torso_norm = np.linalg.norm(torso_vec)
                    if torso_norm > 0:
                        torso_angle = np.degrees(np.arccos(np.clip(torso_dot / torso_norm, -1, 1)))
                        torso_angles.append(torso_angle)
                    
                    shoulder_y_positions.append(shoulder_center[1])
                    
                except (IndexError, ValueError):
                    continue
            
            if not knee_angles or not hip_to_knee_ratios:
                logger.warning("Could not calculate metrics, defaulting to squat")
                return "squat"
            
            # ========== ANALYZE COLLECTED DATA ==========
            
            # PRIMARY: Hip position at deepest point
            # Find MINIMUM hip-to-knee ratio (when hip is lowest relative to knee)
            min_hip_knee_diff = min(hip_to_knee_ratios)
            
            # Knee analysis
            min_knee = min(knee_angles)
            max_knee = max(knee_angles)
            knee_range = max_knee - min_knee
            
            # Hip analysis
            min_hip = min(hip_angles) if hip_angles else 90
            max_hip = max(hip_angles) if hip_angles else 90
            hip_range = max_hip - min_hip
            
            # Torso lean
            max_torso_lean = max(torso_angles) if torso_angles else 0
            
            # Knee forward travel
            max_knee_forward = max(knee_forward_travel) if knee_forward_travel else 0
            
            # Shin angle
            max_shin_angle = max(shin_angles) if shin_angles else 0
            
            # Shoulder movement (for overhead press)
            shoulder_movement = 0
            if shoulder_y_positions:
                shoulder_movement = max(shoulder_y_positions) - min(shoulder_y_positions)
            
            logger.info(f"=== EXERCISE DETECTION (IMPROVED) ===")
            logger.info(f"  PRIMARY - Min hip-knee diff: {min_hip_knee_diff:.4f}")
            logger.info(f"    (Negative/Near-zero = hip at/below knee = SQUAT)")
            logger.info(f"    (Positive > 0.05 = hip above knee = DEADLIFT)")
            logger.info(f"  Knee angle: {min_knee:.1f}° - {max_knee:.1f}° (range: {knee_range:.1f}°)")
            logger.info(f"  Hip angle: {min_hip:.1f}° - {max_hip:.1f}° (range: {hip_range:.1f}°)")
            logger.info(f"  Max torso lean: {max_torso_lean:.1f}°")
            logger.info(f"  Max knee forward travel: {max_knee_forward:.4f}")
            logger.info(f"  Max shin angle from vertical: {max_shin_angle:.1f}°")
            logger.info(f"  Shoulder movement: {shoulder_movement:.4f}")
            
            # ========== CLASSIFICATION LOGIC ==========
            # PRIORITY: Hip-knee relationship is THE PRIMARY differentiator
            # This is checked FIRST before any other indicators
            
            # OVERHEAD PRESS: Significant shoulder/arm movement, minimal lower body
            if shoulder_movement > 0.10 and knee_range < 20:
                detected = "overhead press"
                logger.info(f"→ Detected: {detected} (arms moving overhead, legs static)")
            
            # ROW: Bent over static position (minimal movement)
            elif max_torso_lean > 50 and knee_range < 25 and hip_range < 25:
                detected = "row"
                logger.info(f"→ Detected: {detected} (static bent-over position)")
            
            # ========== PRIMARY SQUAT vs DEADLIFT DISTINCTION ==========
            # THE UNMISTAKABLE DIFFERENTIATOR: HIP HEIGHT RELATIVE TO KNEE
            # This check MUST come before any other squat/deadlift indicators!
            
            # DEADLIFT: Hip stays ABOVE knee level throughout the entire movement
            # If min_hip_knee_diff > 0.04, the hip never went down to knee level
            # This is THE defining characteristic of a deadlift - checked FIRST
            elif min_hip_knee_diff > 0.04:
                detected = "deadlift"
                logger.info(f"→ Detected: {detected} (PRIMARY: hip stayed above knee, diff={min_hip_knee_diff:.4f})")
                logger.info(f"   Even with knee bend ({knee_range:.1f}°), hip position confirms deadlift")
            
            # SQUAT: Hip drops DOWN to knee level or below
            # If min_hip_knee_diff <= 0.04, the hip went down to near/at/below knee level
            # This is THE defining characteristic of a squat
            elif min_hip_knee_diff <= 0.04:
                detected = "squat"
                logger.info(f"→ Detected: {detected} (PRIMARY: hip dropped to knee level, diff={min_hip_knee_diff:.4f})")
            
            # FALLBACK: Use movement patterns if hip-knee data is unreliable
            # Large knee bend with minimal hip hinge suggests squat
            elif knee_range > 50 and hip_range < 40:
                detected = "squat"
                logger.info(f"→ Detected: {detected} (fallback: knee-dominant movement)")
            
            # Large hip hinge with minimal knee bend suggests deadlift
            elif hip_range > 50 and knee_range < 40:
                detected = "deadlift"
                logger.info(f"→ Detected: {detected} (fallback: hip-dominant movement)")
            
            else:
                # Default to squat for any lower body exercise
                detected = "squat"
                logger.info(f"→ Detected: {detected} (default fallback)")
            
            logger.info(f"=== FINAL: {detected.upper()} ===")
            return detected
            
        except Exception as e:
            logger.error(f"Exercise detection failed: {e}", exc_info=True)
            return "squat"
    
    def _generate_recommendations(self, analysis: BiomechanicsAnalysis, exercise_type: str, selected_exercise: str = None) -> List[str]:
        """Generate personalized recommendations based on form analysis.
        
        Recommendations are EXERCISE-SPECIFIC to ensure relevant advice.
        """
        recommendations = []
        exercise_lower = exercise_type.lower()
        
        logger.info(f"Generating recommendations for {exercise_type}, form score: {analysis.form_score}")
        
        # Handle exercise mismatch warning first
        if selected_exercise and selected_exercise.lower() != exercise_lower:
            recommendations.append(f"⚠️ Note: You selected '{selected_exercise}' but we detected '{exercise_type}'. Analysis is based on the detected exercise.")
        
        # Form score based recommendations
        if analysis.form_score >= 90:
            recommendations.append(f"✅ Excellent {exercise_type} form! Keep up the great work.")
        elif analysis.form_score >= 75:
            recommendations.append(f"👍 Good {exercise_type} form overall. See below for refinements.")
        elif analysis.form_score >= 60:
            recommendations.append(f"⚠️ Your {exercise_type} needs improvement. Pay attention to corrections below.")
        else:
            recommendations.append(f"❌ {exercise_type.capitalize()} form needs significant work. Consider working with a trainer.")
        
        # EXERCISE-SPECIFIC risk recommendations
        risk_recommendations = {
            'squat': {
                "excessive_knee_flexion": "Reduce squat depth to protect knee joints. Focus on parallel depth.",
                "insufficient_knee_flexion": "Increase squat depth. Aim for thighs parallel to ground.",
                "knees_caving_in": "⚠️ Keep knees tracking over toes. Push knees outward.",
                "excessive_forward_lean": "Keep chest up. You're leaning too far forward.",
                "insufficient_hip_hinge": "Push hips back more as you descend.",
                "insufficient_depth": "Go deeper. Aim for hips at or below knee level.",
                "asymmetric_movement": "Focus on equal weight on both legs.",
                "unstable_movement": "Control the descent. Brace core throughout."
            },
            'deadlift': {
                "excessive_knee_flexion": "Keep knees straighter. You're squatting the deadlift.",
                "insufficient_knee_flexion": "Bend knees slightly more at the start position.",
                "knees_caving_in": "Keep knees tracking outward as you lift.",
                "excessive_forward_lean": "⚠️ Your back is too horizontal at start. Keep hips lower initially.",
                "insufficient_hip_hinge": "Hinge at hips more. Push hips back to load hamstrings.",
                "asymmetric_movement": "Keep bar centered. Push evenly through both feet.",
                "unstable_movement": "Brace core hard. Bar should travel in straight line."
            },
            'lunge': {
                "excessive_knee_flexion": "Don't let front knee go too far past toes.",
                "insufficient_knee_flexion": "Lunge deeper for full range of motion.",
                "knees_caving_in": "Keep front knee tracking over toes.",
                "asymmetric_movement": "Work on balance between left and right legs.",
                "unstable_movement": "Use a narrower stance if wobbling. Focus on balance."
            },
            'row': {
                "excessive_forward_lean": "Good! Stay bent over for proper lat engagement.",
                "insufficient_hip_hinge": "Hinge forward more. Back should be near parallel to ground.",
                "unstable_movement": "Keep torso stable. Don't use momentum to pull."
            },
            'overhead press': {
                "excessive_forward_lean": "Don't lean back excessively. Keep core braced.",
                "unstable_movement": "Brace core. Press in a straight line overhead."
            },
            'pushup': {
                "unstable_movement": "Keep body straight from head to heels."
            }
        }
        
        # Get exercise-specific risk recommendations
        risk_recs = risk_recommendations.get(exercise_lower, risk_recommendations['squat'])
        
        for risk in analysis.risk_factors:
            if risk in risk_recs and risk_recs[risk]:
                recommendations.append(risk_recs[risk])
        
        # EXERCISE-SPECIFIC tips
        exercise_tips = {
            'squat': [
                "💡 Keep knees aligned with toes throughout the movement.",
                "💡 Push through your heels, not your toes.",
                "💡 Brace your core before descending."
            ],
            'deadlift': [
                "💡 Keep the bar close to your body throughout the lift.",
                "💡 Drive through your hips, not your lower back.",
                "💡 Start with hips higher than in a squat.",
                "💡 Pull bar back into your legs as you lift.",
                "💡 Maintain neutral spine - don't round your back."
            ],
            'lunge': [
                "💡 Keep front knee tracking over ankle.",
                "💡 Push through heel of front foot to return.",
                "💡 Keep torso upright throughout movement."
            ],
            'row': [
                "💡 Pull elbows back, not up.",
                "💡 Squeeze shoulder blades together at top.",
                "💡 Keep torso stable - no jerking."
            ],
            'overhead press': [
                "💡 Press bar in straight line overhead.",
                "💡 Move head back as bar passes face.",
                "💡 Lock out arms fully at top."
            ],
            'pushup': [
                "💡 Keep body in straight line from head to heels.",
                "💡 Lower chest to just above the ground.",
                "💡 Keep elbows at 45° angle to body."
            ],
            'bench press': [
                "💡 Keep feet planted firmly on ground.",
                "💡 Maintain slight arch in lower back.",
                "💡 Touch bar to mid-chest, then press up."
            ]
        }
        
        # Add exercise-specific tips
        tips = exercise_tips.get(exercise_lower, exercise_tips['squat'])
        
        # Add angle-based recommendations
        if 'knee' in analysis.joint_angles:
            knee_angle = analysis.joint_angles['knee']
            if exercise_lower == 'squat':
                if 85 <= knee_angle <= 105:
                    recommendations.append("Good squat depth achieved - knees at optimal angle.")
            elif exercise_lower == 'deadlift':
                if 90 <= knee_angle <= 130:
                    recommendations.append("Good knee position for deadlift.")
        
        # Add 2-3 tips (don't overwhelm user)
        for tip in tips[:3]:
            if len(recommendations) < 8:  # Leave room but don't overflow
                recommendations.append(tip)
        
        # Limit to top 6 recommendations
        logger.info(f"Generated {len(recommendations)} recommendations for {exercise_type}")
        return recommendations[:6]
    
    async def generate_heatmap(self, user_id: str, exercise_type: str) -> Dict:
        """Generate torque heatmap for user's exercise"""
        if user_id not in self.user_data or not self.user_data[user_id]:
            return {"error": "No data available for user"}
        
        # Get latest analysis
        latest_analysis = self.user_data[user_id][-1]
        
        return {
            "heatmap_data": latest_analysis.heatmap_data,
            "torque_data": latest_analysis.torque_data,
            "exercise_type": exercise_type,
            "timestamp": "2024-01-01T00:00:00Z"  # In production, use actual timestamp
        } 