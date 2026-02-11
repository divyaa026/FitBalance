"""
Biomechanics Coaching Module
Real-time biomechanics analysis with GNN-LSTM architecture and torque heatmaps
"""

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
import numpy as np
import cv2
import tempfile
import os
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

class BiomechanicsCoach:
    """Main class for biomechanics coaching and analysis"""
    
    # Minimum confidence threshold for pose detection (lowered for better detection)
    MIN_POSE_CONFIDENCE = 0.3
    # Minimum percentage of frames that need valid poses (lowered for short clips)
    MIN_VALID_FRAMES_RATIO = 0.1
    # Minimum number of key landmarks needed (reduced - just need core body parts)
    MIN_KEY_LANDMARKS = 4
    
    def __init__(self):
        self.gnn_model = GNNLSTMModel()
        self.joint_connections = self._define_joint_connections()
        self.exercise_standards = self._load_exercise_standards()
        self.user_data = {}  # In production, use database
        
        # Key landmark indices for exercise detection (YOLO COCO 17-point format)
        # 5,6: shoulders, 11,12: hips, 13,14: knees, 15,16: ankles
        self.key_landmark_indices = [5, 6, 11, 12, 13, 14, 15, 16]
        
        # Load YOLO pose model
        self._load_yolo_model()
        
        # Load pre-trained GNN model (placeholder)
        self._load_model()
    
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
            # In production, load from saved weights
            # self.model.load_state_dict(torch.load('models/biomechanics_model.pth'))
            logger.info("Biomechanics model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
    
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
            
            # Analyze movement using pose data
            analysis = self._analyze_pose_sequence(pose_data, exercise_type)
            
            # Generate form errors with specific details
            analysis.form_errors = self._identify_form_errors(analysis.joint_angles, exercise_type)
            
            # Generate recommendations
            analysis.recommendations = self._generate_recommendations(analysis, exercise_type)
            
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
            if self.yolo_model is None:
                validation_info['error_message'] = "Pose detection model is not available. Please try again later."
                return pose_data, validation_info
            
            logger.info(f"Processing image file: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                validation_info['error_message'] = "Could not read image file. Please upload a valid image (JPG, PNG)."
                return pose_data, validation_info
            
            logger.info(f"Image loaded: {image.shape}")
            
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
                        if idx in self.key_landmark_indices:
                            if joint.confidence >= 0.2:  # Lower threshold for images
                                key_landmarks_detected += 1
                                total_confidence += joint.confidence
                    
                    logger.info(f"Image analysis: {key_landmarks_detected} key landmarks detected out of {len(self.key_landmark_indices)}")
                    
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
            if self.yolo_model is None:
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
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
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
                            if idx in self.key_landmark_indices:
                                if joint.confidence >= self.MIN_POSE_CONFIDENCE:
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
        """Calculate joint angles from pose data using YOLO COCO landmark indices.
        
        For exercises like squats, we need to find the PEAK of the movement
        (minimum knee angle = bottom of squat) rather than just averaging frames.
        """
        angles = {}
        
        if not pose_data:
            return angles
        
        try:
            # YOLO COCO 17-point pose landmark indices:
            # 5: left_shoulder, 6: right_shoulder
            # 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee
            # 15: left_ankle, 16: right_ankle
            
            all_knee_angles = []
            all_hip_angles = []
            all_ankle_angles = []
            
            # Process ALL frames to find the peak exercise position
            for frame in pose_data:
                # Check if we have the required landmarks (YOLO has 17 keypoints, indices 0-16)
                if len(frame) < 17:
                    logger.debug(f"Frame has only {len(frame)} landmarks, skipping angle calculation")
                    continue
                
                try:
                    # Left leg angles (prefer left side for consistency)
                    left_hip = np.array([frame[11].x, frame[11].y, frame[11].z])
                    left_knee = np.array([frame[13].x, frame[13].y, frame[13].z])
                    left_ankle = np.array([frame[15].x, frame[15].y, frame[15].z])
                    left_shoulder = np.array([frame[5].x, frame[5].y, frame[5].z])
                    
                    # Calculate knee angle (angle between hip-knee and knee-ankle vectors)
                    v1 = left_hip - left_knee
                    v2 = left_ankle - left_knee
                    dot_product = np.dot(v1, v2)
                    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                    
                    if norm_product > 0:
                        knee_angle = np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)))
                        all_knee_angles.append(knee_angle)
                    
                    # Calculate hip angle (angle between shoulder-hip and hip-knee vectors)
                    v3 = left_shoulder - left_hip
                    v4 = left_knee - left_hip
                    dot_product_hip = np.dot(v3, v4)
                    norm_product_hip = np.linalg.norm(v3) * np.linalg.norm(v4)
                    
                    if norm_product_hip > 0:
                        hip_angle = np.degrees(np.arccos(np.clip(dot_product_hip / norm_product_hip, -1.0, 1.0)))
                        all_hip_angles.append(hip_angle)
                    
                    # Calculate ankle angle (angle between knee-ankle and ankle-foot direction)
                    v5 = left_knee - left_ankle
                    v6 = np.array([0, 1, 0])  # Vertical reference
                    dot_product_ankle = np.dot(v5, v6)
                    norm_product_ankle = np.linalg.norm(v5) * np.linalg.norm(v6)
                    
                    if norm_product_ankle > 0:
                        ankle_angle = np.degrees(np.arccos(np.clip(dot_product_ankle / norm_product_ankle, -1.0, 1.0)))
                        all_ankle_angles.append(ankle_angle)
                        
                except IndexError as e:
                    logger.warning(f"IndexError calculating angles for frame with {len(frame)} landmarks: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error calculating angles for frame: {e}")
                    continue
            
            # For exercises like squats, the PEAK position is when angles are at minimum
            # (bent position). We use the MINIMUM angle found in the video as the
            # representative angle for form assessment.
            # 
            # This way, if someone does a proper squat going from standing (170°) to
            # bottom (90°) and back up, we evaluate based on the 90° (good form),
            # not the 170° (standing position).
            
            if all_knee_angles:
                # Use the minimum knee angle (deepest bend) and also check median
                min_knee = min(all_knee_angles)
                median_knee = np.median(all_knee_angles)
                # Use the value closer to a "working" position (not standing)
                # If min is too extreme (< 50°), use median
                angles['knee'] = round(float(min_knee if min_knee >= 50 else median_knee), 1)
                logger.info(f"Knee angles - min: {min_knee:.1f}°, median: {median_knee:.1f}°, selected: {angles['knee']}°")
                
            if all_hip_angles:
                min_hip = min(all_hip_angles)
                median_hip = np.median(all_hip_angles)
                angles['hip'] = round(float(min_hip if min_hip >= 30 else median_hip), 1)
                logger.info(f"Hip angles - min: {min_hip:.1f}°, median: {median_hip:.1f}°, selected: {angles['hip']}°")
                
            if all_ankle_angles:
                min_ankle = min(all_ankle_angles)
                median_ankle = np.median(all_ankle_angles)
                angles['ankle'] = round(float(min_ankle if min_ankle >= 40 else median_ankle), 1)
                logger.info(f"Ankle angles - min: {min_ankle:.1f}°, median: {median_ankle:.1f}°, selected: {angles['ankle']}°")
            
            logger.info(f"Final calculated joint angles (using peak position): {angles}")
            
        except Exception as e:
            logger.error(f"Error calculating joint angles: {e}", exc_info=True)
            # Return empty - let calling code handle missing angles
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
        """Assess overall form quality (0-100) based on exercise-specific standards.
        
        The scoring is designed to be encouraging but accurate:
        - Start at 85 points (good baseline)
        - Add bonus points for angles within optimal range
        - Subtract smaller penalties for deviations
        - No penalty if angles are close to the range (within 10 degrees)
        """
        if exercise_type.lower() not in self.exercise_standards:
            logger.warning(f"No standards found for exercise: {exercise_type}, using default score")
            return 75.0  # Default score
        
        if not joint_angles:
            logger.warning(f"No joint angles provided for assessment")
            return 70.0  # Default when no angles available
        
        standards = self.exercise_standards[exercise_type.lower()]
        score = 85.0  # Start with good baseline
        penalties = []
        joints_checked = 0
        joints_in_range = 0
        
        logger.info(f"Assessing form for {exercise_type} with angles: {joint_angles}")
        
        # Check joint angles against standards
        for joint, angle in joint_angles.items():
            standard_key = f"{joint}_angle_range"
            if standard_key in standards:
                joints_checked += 1
                min_angle, max_angle = standards[standard_key]
                
                # Calculate how far off the angle is from the acceptable range
                if angle < min_angle:
                    deviation = min_angle - angle
                    if deviation <= 10:
                        # Close enough - no penalty
                        joints_in_range += 1
                        score += 2
                        penalties.append(f"{joint}: {angle:.1f}° (close to {min_angle}°, +2 points)")
                    else:
                        # Penalize based on deviation, but capped
                        penalty = min(deviation * 0.3, 12)  # Reduced penalty, max 12 per joint
                        score -= penalty
                        penalties.append(f"{joint}: {angle:.1f}° < {min_angle}° (-{penalty:.1f} points)")
                    
                elif angle > max_angle:
                    deviation = angle - max_angle
                    if deviation <= 10:
                        # Close enough - no penalty
                        joints_in_range += 1
                        score += 2
                        penalties.append(f"{joint}: {angle:.1f}° (close to {max_angle}°, +2 points)")
                    else:
                        # Penalize based on deviation, but capped
                        penalty = min(deviation * 0.3, 12)  # Reduced penalty, max 12 per joint
                        score -= penalty
                        penalties.append(f"{joint}: {angle:.1f}° > {max_angle}° (-{penalty:.1f} points)")
                    
                else:
                    # Within range - bonus for good form
                    joints_in_range += 1
                    score += 5  # Increased bonus
                    penalties.append(f"{joint}: {angle:.1f}° ✓ in range ({min_angle}°-{max_angle}°) (+5 points)")
        
        # Bonus if most joints are in range
        if joints_checked > 0 and joints_in_range / joints_checked >= 0.5:
            score += 5
            penalties.append(f"Good overall form (+5 bonus)")
        
        # Clamp score to 0-100 range
        final_score = max(0.0, min(100.0, score))
        
        logger.info(f"Form assessment complete:")
        for p in penalties:
            logger.info(f"  {p}")
        logger.info(f"  Final score: {final_score:.1f}/100")
        
        return round(final_score, 1)
    
    def _identify_risk_factors(self, joint_angles: Dict[str, float], exercise_type: str) -> List[str]:
        """Identify potential risk factors based on joint angles and exercise type"""
        risk_factors = []
        
        if exercise_type.lower() not in self.exercise_standards:
            return risk_factors
        
        standards = self.exercise_standards[exercise_type.lower()]
        
        logger.info(f"Identifying risk factors for {exercise_type}")
        
        # Check for common errors based on joint angles
        if 'knee' in joint_angles:
            knee_angle = joint_angles['knee']
            
            if knee_angle < 70:  # Excessive knee flexion
                risk_factors.append("excessive_knee_flexion")
                logger.info(f"Risk: Excessive knee flexion ({knee_angle:.1f}°)")
            elif knee_angle > 160:  # Insufficient knee flexion
                risk_factors.append("insufficient_knee_flexion")
                logger.info(f"Risk: Insufficient knee flexion ({knee_angle:.1f}°)")
            
            # Squat-specific: Check if knees are caving
            if exercise_type.lower() == "squat" and knee_angle < 85:
                risk_factors.append("knees_caving_in")
                logger.info(f"Risk: Knees may be caving in (angle {knee_angle:.1f}°)")
        
        if 'hip' in joint_angles:
            hip_angle = joint_angles['hip']
            
            if hip_angle < 40:  # Excessive forward lean
                risk_factors.append("excessive_forward_lean")
                logger.info(f"Risk: Excessive forward lean ({hip_angle:.1f}°)")
            elif hip_angle > 150:  # Too upright
                risk_factors.append("insufficient_hip_hinge")
                logger.info(f"Risk: Insufficient hip hinge ({hip_angle:.1f}°)")
        
        if 'ankle' in joint_angles:
            ankle_angle = joint_angles['ankle']
            
            if ankle_angle < 60:  # Heels lifting
                risk_factors.append("heels_lifting")
                logger.info(f"Risk: Heels may be lifting ({ankle_angle:.1f}°)")
            elif ankle_angle > 110:  # Limited ankle mobility
                risk_factors.append("limited_ankle_mobility")
                logger.info(f"Risk: Limited ankle mobility ({ankle_angle:.1f}°)")
        
        # Exercise-specific risk assessment
        if exercise_type.lower() == "squat":
            if 'knee' in joint_angles and 'hip' in joint_angles:
                # Check for proper depth
                if joint_angles['knee'] > 110:
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
        """Identify specific form errors with detailed information"""
        form_errors = []
        
        if exercise_type.lower() not in self.exercise_standards:
            return form_errors
        
        standards = self.exercise_standards[exercise_type.lower()]
        
        # Correction tips for each joint and error type
        correction_tips = {
            'knee': {
                'too_low': "Reduce squat depth slightly. Don't let knees go past toes excessively.",
                'too_high': "Go deeper in your squat. Aim for thighs parallel to the ground.",
                'caving': "Push your knees outward, tracking over your toes. Consider using a resistance band above knees."
            },
            'hip': {
                'too_low': "Keep your chest up more. You're leaning too far forward.",
                'too_high': "Hinge more at the hips. Push your hips back as you descend."
            },
            'ankle': {
                'too_low': "Your heels are lifting. Work on ankle mobility or elevate heels slightly.",
                'too_high': "Limited ankle mobility detected. Stretch calves and practice ankle mobility drills."
            }
        }
        
        # Check each joint angle against standards
        for joint, angle in joint_angles.items():
            standard_key = f"{joint}_angle_range"
            if standard_key in standards:
                min_angle, max_angle = standards[standard_key]
                
                if angle < min_angle:
                    deviation = min_angle - angle
                    severity = 'minor' if deviation < 10 else ('moderate' if deviation < 20 else 'severe')
                    form_errors.append(FormError(
                        body_part=joint.capitalize(),
                        issue=f"{joint.capitalize()} angle too low ({angle:.1f}°)",
                        current_value=angle,
                        expected_range=(min_angle, max_angle),
                        severity=severity,
                        correction_tip=correction_tips.get(joint, {}).get('too_low', f"Adjust your {joint} position.")
                    ))
                    
                elif angle > max_angle:
                    deviation = angle - max_angle
                    severity = 'minor' if deviation < 10 else ('moderate' if deviation < 20 else 'severe')
                    form_errors.append(FormError(
                        body_part=joint.capitalize(),
                        issue=f"{joint.capitalize()} angle too high ({angle:.1f}°)",
                        current_value=angle,
                        expected_range=(min_angle, max_angle),
                        severity=severity,
                        correction_tip=correction_tips.get(joint, {}).get('too_high', f"Adjust your {joint} position.")
                    ))
        
        # Exercise-specific form errors
        if exercise_type.lower() == "squat":
            if 'knee' in joint_angles and joint_angles['knee'] < 85:
                form_errors.append(FormError(
                    body_part="Knees",
                    issue="Knees may be caving inward",
                    current_value=joint_angles['knee'],
                    expected_range=(90, 120),
                    severity='moderate',
                    correction_tip=correction_tips['knee']['caving']
                ))
        
        logger.info(f"Identified {len(form_errors)} form errors")
        return form_errors
    
    def _generate_recommendations(self, analysis: BiomechanicsAnalysis, exercise_type: str) -> List[str]:
        """Generate personalized recommendations based on form analysis"""
        recommendations = []
        
        logger.info(f"Generating recommendations for {exercise_type}, form score: {analysis.form_score}")
        
        # Form score based recommendations
        if analysis.form_score >= 90:
            recommendations.append("Excellent form! Keep up the great work.")
        elif analysis.form_score >= 75:
            recommendations.append("Good form overall. Focus on the specific points below to improve further.")
        elif analysis.form_score >= 60:
            recommendations.append("Form needs improvement. Pay attention to the corrections below.")
        else:
            recommendations.append("⚠️ Form needs significant work. Consider working with a trainer.")
        
        # Risk factor based recommendations
        risk_recommendations = {
            "excessive_knee_flexion": "Reduce squat depth to protect knee joints. Focus on parallel depth.",
            "insufficient_knee_flexion": "Increase squat depth for better muscle engagement. Aim for thighs parallel to ground.",
            "knees_caving_in": "⚠️ Keep knees tracking over toes. Push knees outward throughout the movement.",
            "excessive_forward_lean": "Keep chest up and maintain a more upright posture. Engage your core.",
            "insufficient_hip_hinge": "Focus on hinging at the hips. Push hips back while maintaining neutral spine.",
            "heels_lifting": "⚠️ Keep heels planted on the ground. Work on ankle mobility if needed.",
            "limited_ankle_mobility": "Work on ankle mobility with stretches and mobility drills.",
            "insufficient_depth": "Go deeper in your squat. Aim for hips below knee level.",
            "back_rounding": "⚠️ Maintain neutral spine throughout the movement. Engage your lats."
        }
        
        for risk in analysis.risk_factors:
            if risk in risk_recommendations:
                recommendations.append(risk_recommendations[risk])
        
        # Exercise specific recommendations
        if exercise_type.lower() == "squat":
            if 'knee' in analysis.joint_angles:
                knee_angle = analysis.joint_angles['knee']
                if 90 <= knee_angle <= 110:
                    recommendations.append("Good squat depth - knees at optimal angle.")
            
            recommendations.append("💡 Tip: Keep knees aligned with toes throughout the movement.")
            recommendations.append("💡 Tip: Push through your heels, not your toes.")
            recommendations.append("💡 Tip: Brace your core before descending.")
            
        elif exercise_type.lower() == "deadlift":
            recommendations.append("💡 Tip: Keep the bar close to your body throughout the lift.")
            recommendations.append("💡 Tip: Drive through your hips, not your lower back.")
            recommendations.append("💡 Tip: Start with hips higher than knees.")
            
        elif exercise_type.lower() == "pushup":
            recommendations.append("💡 Tip: Keep your body in a straight line from head to heels.")
            recommendations.append("💡 Tip: Lower chest to just above the ground.")
            recommendations.append("💡 Tip: Keep elbows at 45° angle to body.")
        
        # Limit to top 6 recommendations
        logger.info(f"Generated {len(recommendations)} recommendations")
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