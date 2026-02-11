"""
Biomechanics Engine - Hybrid GNN-LSTM Model with MediaPipe Integration
Real-time pose estimation, joint angle analysis, and torque heatmap generation
"""

import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class JointAngle:
    """Represents a joint angle measurement"""
    joint_name: str
    angle: float  # degrees
    confidence: float
    normal_range: Tuple[float, float]
    is_abnormal: bool

@dataclass
class TorqueAnalysis:
    """Represents torque analysis for a joint"""
    joint_name: str
    torque_magnitude: float  # N⋅m
    torque_direction: Tuple[float, float, float]  # 3D vector
    risk_level: str  # low, medium, high, critical
    heatmap_coordinates: Tuple[int, int]

@dataclass
class BiomechanicsResult:
    """Complete biomechanics analysis result"""
    frame_number: int
    timestamp: float
    joint_angles: List[JointAngle]
    torques: List[TorqueAnalysis]
    form_score: float  # 0-100
    risk_factors: List[str]
    heatmap_data: np.ndarray
    recommendations: List[str]

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
        
        # Joint connections for GNN (MediaPipe pose landmarks)
        self.joint_connections = self._define_joint_connections()
        
        # Physics constraints
        self.physics_constraints = self._load_physics_constraints()
        
        # Exercise standards
        self.exercise_standards = self._load_exercise_standards()
    
    def _define_joint_connections(self) -> List[Tuple[int, int]]:
        """Define anatomical joint connections for GNN"""
        return [
            # Head and neck
            (0, 1), (1, 2), (2, 3), (3, 7),
            # Left arm
            (0, 4), (4, 5), (5, 6),
            # Right arm
            (0, 8), (8, 9), (9, 10),
            # Left leg
            (11, 12), (12, 14), (14, 16),
            # Right leg
            (11, 13), (13, 15), (15, 17),
            # Left leg extended
            (11, 23), (23, 24), (24, 26), (26, 28),
            # Right leg extended
            (12, 25), (25, 27), (27, 29), (29, 31),
        ]
    
    def _load_physics_constraints(self) -> Dict:
        """Load physics-based constraints for joint movements"""
        return {
            "knee": {
                "max_angle": 170,  # degrees
                "min_angle": 0,
                "max_torque": 200,  # N⋅m
                "normal_range": (90, 120)
            },
            "hip": {
                "max_angle": 180,
                "min_angle": 0,
                "max_torque": 300,
                "normal_range": (45, 90)
            },
            "ankle": {
                "max_angle": 150,
                "min_angle": 0,
                "max_torque": 100,
                "normal_range": (70, 110)
            },
            "shoulder": {
                "max_angle": 180,
                "min_angle": 0,
                "max_torque": 150,
                "normal_range": (0, 30)
            },
            "elbow": {
                "max_angle": 160,
                "min_angle": 0,
                "max_torque": 80,
                "normal_range": (80, 100)
            }
        }
    
    def _load_exercise_standards(self) -> Dict:
        """Load exercise form standards"""
        return {
            "squat": {
                "knee_angle_range": (90, 120),
                "hip_angle_range": (45, 90),
                "ankle_angle_range": (70, 110),
                "back_angle_range": (45, 75),
                "common_errors": [
                    "knees_caving_in",
                    "heels_lifting",
                    "back_rounding",
                    "insufficient_depth"
                ]
            },
            "deadlift": {
                "hip_angle_range": (30, 60),
                "knee_angle_range": (60, 100),
                "back_angle_range": (15, 45),
                "common_errors": [
                    "back_rounding",
                    "hips_rising_too_fast",
                    "bar_path_deviation"
                ]
            },
            "pushup": {
                "elbow_angle_range": (80, 100),
                "shoulder_angle_range": (0, 30),
                "body_alignment": "straight",
                "common_errors": [
                    "sagging_hips",
                    "incomplete_range",
                    "head_dropping"
                ]
            }
        }
    
    def analyze_frame(self, frame: np.ndarray, exercise_type: str = "squat") -> BiomechanicsResult:
        """Analyze a single frame for biomechanics"""
        try:
            # 1. Run MediaPipe pose detection
            pose_landmarks = self._extract_pose_landmarks(frame)
            
            if pose_landmarks is None:
                return self._create_empty_result()
            
            # 2. Extract joint angles
            joint_angles = self._calculate_joint_angles(pose_landmarks)
            
            # 3. Apply physics constraints and calculate torques
            torques = self._calculate_torques(pose_landmarks, joint_angles)
            
            # 4. Generate real-time heatmap
            heatmap_data = self._generate_heatmap(torques, frame.shape[:2])
            
            # 5. Assess form and generate recommendations
            form_score, risk_factors, recommendations = self._assess_form(
                joint_angles, torques, exercise_type
            )
            
            return BiomechanicsResult(
                frame_number=0,  # Will be set by caller
                timestamp=0.0,   # Will be set by caller
                joint_angles=joint_angles,
                torques=torques,
                form_score=form_score,
                risk_factors=risk_factors,
                heatmap_data=heatmap_data,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return self._create_empty_result()
    
    def _extract_pose_landmarks(self, frame: np.ndarray) -> Optional[List]:
        """Extract pose landmarks using MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose_estimator.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                return landmarks
            
            return None
            
        except Exception as e:
            logger.error(f"Pose extraction error: {e}")
            return None
    
    def _calculate_joint_angles(self, landmarks: List[List[float]]) -> List[JointAngle]:
        """Calculate joint angles from pose landmarks"""
        joint_angles = []
        
        try:
            # Convert landmarks to numpy array
            landmarks = np.array(landmarks)
            
            # Calculate key joint angles
            angles_to_calculate = [
                ("knee", [23, 25, 27]),      # Left knee: hip-knee-ankle
                ("knee", [24, 26, 28]),      # Right knee: hip-knee-ankle
                ("hip", [11, 23, 25]),       # Left hip: shoulder-hip-knee
                ("hip", [12, 24, 26]),       # Right hip: shoulder-hip-knee
                ("ankle", [25, 27, 29]),     # Left ankle: knee-ankle-toe
                ("ankle", [26, 28, 30]),     # Right ankle: knee-ankle-toe
                ("shoulder", [11, 12, 14]),  # Left shoulder: neck-shoulder-elbow
                ("shoulder", [11, 12, 16]),  # Right shoulder: neck-shoulder-elbow
                ("elbow", [12, 14, 16]),     # Left elbow: shoulder-elbow-wrist
                ("elbow", [11, 13, 15]),     # Right elbow: shoulder-elbow-wrist
            ]
            
            for joint_name, joint_indices in angles_to_calculate:
                if all(idx < len(landmarks) for idx in joint_indices):
                    angle = self._calculate_angle_3d(
                        landmarks[joint_indices[0]],
                        landmarks[joint_indices[1]],
                        landmarks[joint_indices[2]]
                    )
                    
                    # Get normal range from physics constraints
                    normal_range = self.physics_constraints.get(joint_name, {}).get("normal_range", (0, 180))
                    is_abnormal = angle < normal_range[0] or angle > normal_range[1]
                    
                    joint_angles.append(JointAngle(
                        joint_name=joint_name,
                        angle=angle,
                        confidence=0.8,  # Placeholder confidence
                        normal_range=normal_range,
                        is_abnormal=is_abnormal
                    ))
        
        except Exception as e:
            logger.error(f"Joint angle calculation error: {e}")
        
        return joint_angles
    
    def _calculate_angle_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three 3D points"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Calculate angle
            cos_angle = np.dot(v1_norm, v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
            
        except Exception as e:
            logger.error(f"3D angle calculation error: {e}")
            return 90.0  # Default angle
    
    def _calculate_torques(self, landmarks: List[List[float]], joint_angles: List[JointAngle]) -> List[TorqueAnalysis]:
        """Calculate torques using physics-based modeling"""
        torques = []
        
        try:
            landmarks = np.array(landmarks)
            
            # Simplified torque calculation based on joint angles and forces
            for joint_angle in joint_angles:
                joint_name = joint_angle.joint_name
                angle = joint_angle.angle
                
                # Get physics constraints
                constraints = self.physics_constraints.get(joint_name, {})
                max_torque = constraints.get("max_torque", 100)
                normal_range = constraints.get("normal_range", (0, 180))
                
                # Calculate torque based on angle deviation from normal range
                if normal_range[0] <= angle <= normal_range[1]:
                    # Normal range - low torque
                    torque_magnitude = max_torque * 0.2
                else:
                    # Abnormal range - higher torque
                    deviation = min(abs(angle - normal_range[0]), abs(angle - normal_range[1]))
                    torque_magnitude = max_torque * (0.2 + 0.8 * (deviation / 90))
                
                # Determine risk level
                if torque_magnitude < max_torque * 0.4:
                    risk_level = "low"
                elif torque_magnitude < max_torque * 0.7:
                    risk_level = "medium"
                elif torque_magnitude < max_torque * 0.9:
                    risk_level = "high"
                else:
                    risk_level = "critical"
                
                # Calculate torque direction (simplified)
                torque_direction = (0.0, 0.0, 1.0)  # Placeholder
                
                # Calculate heatmap coordinates
                heatmap_x = int((torque_magnitude / max_torque) * 100)
                heatmap_y = int((angle / 180) * 100)
                heatmap_coordinates = (heatmap_x, heatmap_y)
                
                torques.append(TorqueAnalysis(
                    joint_name=joint_name,
                    torque_magnitude=torque_magnitude,
                    torque_direction=torque_direction,
                    risk_level=risk_level,
                    heatmap_coordinates=heatmap_coordinates
                ))
        
        except Exception as e:
            logger.error(f"Torque calculation error: {e}")
        
        return torques
    
    def _generate_heatmap(self, torques: List[TorqueAnalysis], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Generate real-time heatmap from torque data"""
        try:
            height, width = frame_shape
            heatmap = np.zeros((height, width), dtype=np.float32)
            
            for torque in torques:
                x, y = torque.heatmap_coordinates
                
                # Scale coordinates to frame size
                x_scaled = int((x / 100) * width)
                y_scaled = int((y / 100) * height)
                
                # Create heatmap blob around the point
                radius = int(torque.torque_magnitude / 10)  # Scale radius by torque
                intensity = min(torque.torque_magnitude / 200, 1.0)  # Normalize intensity
                
                # Draw circle on heatmap
                cv2.circle(heatmap, (x_scaled, y_scaled), radius, intensity, -1)
            
            # Apply Gaussian blur for smooth heatmap
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return np.zeros(frame_shape, dtype=np.float32)
    
    def _assess_form(self, joint_angles: List[JointAngle], torques: List[TorqueAnalysis], exercise_type: str) -> Tuple[float, List[str], List[str]]:
        """Assess overall form quality and generate recommendations"""
        try:
            form_score = 100.0
            risk_factors = []
            recommendations = []
            
            # Get exercise standards
            standards = self.exercise_standards.get(exercise_type, {})
            
            # Assess joint angles
            for joint_angle in joint_angles:
                if joint_angle.is_abnormal:
                    form_score -= 10
                    risk_factors.append(f"abnormal_{joint_angle.joint_name}_angle")
                    
                    if joint_angle.angle < joint_angle.normal_range[0]:
                        recommendations.append(f"Increase {joint_angle.joint_name} angle")
                    else:
                        recommendations.append(f"Decrease {joint_angle.joint_name} angle")
            
            # Assess torques
            for torque in torques:
                if torque.risk_level in ["high", "critical"]:
                    form_score -= 15
                    risk_factors.append(f"high_{torque.joint_name}_torque")
                    recommendations.append(f"Reduce {torque.joint_name} stress")
            
            # Exercise-specific assessments
            if exercise_type == "squat":
                if any(ta.joint_name == "knee" and ta.risk_level == "high" for ta in torques):
                    recommendations.append("Keep knees aligned with toes")
                if any(ta.joint_name == "hip" and ta.risk_level == "high" for ta in torques):
                    recommendations.append("Maintain proper hip hinge")
            
            # Ensure form score is within bounds
            form_score = max(0.0, min(100.0, form_score))
            
            return form_score, risk_factors, recommendations
            
        except Exception as e:
            logger.error(f"Form assessment error: {e}")
            return 75.0, [], ["Unable to assess form"]
    
    def _create_empty_result(self) -> BiomechanicsResult:
        """Create empty result when analysis fails"""
        return BiomechanicsResult(
            frame_number=0,
            timestamp=0.0,
            joint_angles=[],
            torques=[],
            form_score=0.0,
            risk_factors=["analysis_failed"],
            heatmap_data=np.zeros((480, 640), dtype=np.float32),
            recommendations=["Unable to analyze frame"]
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training"""
        # GNN processing
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # LSTM processing (if sequence data available)
        if len(x.shape) > 2:
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]  # Take last timestep
        
        # Outputs
        joint_predictions = self.joint_classifier(x)
        torque_predictions = self.torque_regressor(x)
        
        return joint_predictions, torque_predictions 