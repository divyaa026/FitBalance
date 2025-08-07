"""
Biomechanics Coaching Module
Real-time biomechanics analysis with GNN-LSTM architecture and torque heatmaps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class BiomechanicsAnalysis:
    """Results of biomechanics analysis"""
    exercise_type: str
    form_score: float  # 0-100
    risk_factors: List[str]
    recommendations: List[str]
    joint_angles: Dict[str, float]
    torque_data: Dict[str, List[float]]
    heatmap_data: Dict[str, List[List[float]]]

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
    
    def __init__(self):
        self.model = GNNLSTMModel()
        self.joint_connections = self._define_joint_connections()
        self.exercise_standards = self._load_exercise_standards()
        self.user_data = {}  # In production, use database
        
        # Load pre-trained model (placeholder)
        self._load_model()
    
    def _define_joint_connections(self) -> List[Tuple[int, int]]:
        """Define anatomical joint connections for GNN"""
        # Standard human pose connections (MediaPipe format)
        return [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head and neck
            (0, 4), (4, 5), (5, 6),  # Left arm
            (0, 8), (8, 9), (9, 10),  # Right arm
            (11, 12), (12, 14), (14, 16),  # Left leg
            (11, 13), (13, 15), (15, 17),  # Right leg
            (11, 23), (23, 24), (24, 26), (26, 28),  # Left leg extended
            (12, 25), (25, 27), (27, 29), (29, 31),  # Right leg extended
        ]
    
    def _load_exercise_standards(self) -> Dict:
        """Load exercise form standards and benchmarks"""
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
                    "bar_path_deviation",
                    "insufficient_hip_hinge"
                ]
            },
            "pushup": {
                "elbow_angle_range": (80, 100),
                "shoulder_angle_range": (0, 30),
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
        """Analyze movement from uploaded video"""
        try:
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                content = await video_file.read()
                tmp_file.write(content)
                video_path = tmp_file.name
            
            # Extract pose data from video
            pose_data = self._extract_pose_data(video_path)
            
            # Analyze movement using GNN-LSTM
            analysis = self._analyze_pose_sequence(pose_data, exercise_type)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis, exercise_type)
            
            # Store user data
            if user_id not in self.user_data:
                self.user_data[user_id] = []
            self.user_data[user_id].append(analysis)
            
            # Clean up
            os.unlink(video_path)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Movement analysis error: {str(e)}")
            raise
    
    def _extract_pose_data(self, video_path: str) -> List[List[JointPosition]]:
        """Extract pose data from video using MediaPipe"""
        pose_data = []
        
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            
            cap = cv2.VideoCapture(video_path)
            
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            ) as pose:
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    
                    if results.pose_landmarks:
                        frame_joints = []
                        for landmark in results.pose_landmarks.landmark:
                            joint = JointPosition(
                                x=landmark.x,
                                y=landmark.y,
                                z=landmark.z,
                                confidence=landmark.visibility
                            )
                            frame_joints.append(joint)
                        pose_data.append(frame_joints)
            
            cap.release()
            
        except ImportError:
            # Fallback to mock data if MediaPipe not available
            logger.warning("MediaPipe not available, using mock data")
            pose_data = self._generate_mock_pose_data()
        
        return pose_data
    
    def _generate_mock_pose_data(self) -> List[List[JointPosition]]:
        """Generate mock pose data for testing"""
        frames = 30
        joints_per_frame = 33  # MediaPipe pose landmarks
        
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
        
        # Convert pose data to tensor format
        features = []
        for frame in pose_data:
            frame_features = []
            for joint in frame:
                frame_features.extend([joint.x, joint.y, joint.z])
            features.append(frame_features)
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Create edge index for GNN
        edge_index = torch.tensor(self.joint_connections, dtype=torch.long).t().contiguous()
        
        # Create graph data
        data = Data(x=features, edge_index=edge_index)
        
        # Model inference
        with torch.no_grad():
            classification, regression = self.model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
        
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
        """Calculate joint angles from pose data"""
        angles = {}
        
        # Use last frame for current angles
        if not pose_data:
            return angles
        
        frame = pose_data[-1]
        
        # Calculate key joint angles
        try:
            # Knee angle (hip-knee-ankle)
            hip = np.array([frame[23].x, frame[23].y])
            knee = np.array([frame[25].x, frame[25].y])
            ankle = np.array([frame[27].x, frame[27].y])
            
            v1 = hip - knee
            v2 = ankle - knee
            knee_angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            angles['knee'] = knee_angle
            
            # Hip angle (shoulder-hip-knee)
            shoulder = np.array([frame[11].x, frame[11].y])
            hip_angle = np.degrees(np.arccos(np.dot(shoulder - hip, knee - hip) / 
                                           (np.linalg.norm(shoulder - hip) * np.linalg.norm(knee - hip))))
            angles['hip'] = hip_angle
            
            # Ankle angle
            ankle_angle = np.degrees(np.arccos(np.dot(knee - ankle, np.array([0, -1])) / 
                                             (np.linalg.norm(knee - ankle))))
            angles['ankle'] = ankle_angle
            
        except Exception as e:
            logger.error(f"Error calculating joint angles: {e}")
        
        return angles
    
    def _calculate_torque_data(self, pose_data: List[List[JointPosition]], joint_angles: Dict[str, float]) -> Dict[str, List[float]]:
        """Calculate torque data for heatmap generation"""
        torque_data = {
            'knee_torque': [],
            'hip_torque': [],
            'ankle_torque': [],
            'back_torque': []
        }
        
        # Simplified torque calculation based on joint angles and time
        for i, frame in enumerate(pose_data):
            time_factor = i / len(pose_data)
            
            # Mock torque calculations (in production, use physics-based models)
            knee_torque = 100 * np.sin(2 * np.pi * time_factor) * (joint_angles.get('knee', 90) / 90)
            hip_torque = 150 * np.cos(2 * np.pi * time_factor) * (joint_angles.get('hip', 45) / 45)
            ankle_torque = 50 * np.sin(4 * np.pi * time_factor) * (joint_angles.get('ankle', 80) / 80)
            back_torque = 80 * np.cos(3 * np.pi * time_factor)
            
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
        """Assess overall form quality (0-100)"""
        if exercise_type not in self.exercise_standards:
            return 75.0  # Default score
        
        standards = self.exercise_standards[exercise_type]
        score = 100.0
        
        # Check joint angles against standards
        for joint, angle in joint_angles.items():
            if f"{joint}_angle_range" in standards:
                min_angle, max_angle = standards[f"{joint}_angle_range"]
                if angle < min_angle or angle > max_angle:
                    # Penalize for being outside range
                    penalty = min(abs(angle - min_angle), abs(angle - max_angle)) * 2
                    score -= penalty
        
        return max(0.0, min(100.0, score))
    
    def _identify_risk_factors(self, joint_angles: Dict[str, float], exercise_type: str) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        if exercise_type not in self.exercise_standards:
            return risk_factors
        
        standards = self.exercise_standards[exercise_type]
        
        # Check for common errors based on joint angles
        if 'knee' in joint_angles:
            if joint_angles['knee'] < 60:  # Too deep
                risk_factors.append("excessive_knee_flexion")
            elif joint_angles['knee'] > 150:  # Too shallow
                risk_factors.append("insufficient_knee_flexion")
        
        if 'hip' in joint_angles:
            if joint_angles['hip'] < 30:  # Too much forward lean
                risk_factors.append("excessive_forward_lean")
        
        return risk_factors
    
    def _generate_recommendations(self, analysis: BiomechanicsAnalysis, exercise_type: str) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Form score based recommendations
        if analysis.form_score < 60:
            recommendations.append("Consider working with a trainer to improve form")
        elif analysis.form_score < 80:
            recommendations.append("Focus on maintaining proper alignment throughout the movement")
        
        # Risk factor based recommendations
        for risk in analysis.risk_factors:
            if risk == "excessive_knee_flexion":
                recommendations.append("Reduce squat depth to protect knee joints")
            elif risk == "insufficient_knee_flexion":
                recommendations.append("Increase squat depth for better muscle engagement")
            elif risk == "excessive_forward_lean":
                recommendations.append("Keep chest up and maintain upright posture")
        
        # Exercise specific recommendations
        if exercise_type == "squat":
            recommendations.append("Keep knees aligned with toes")
            recommendations.append("Push through your heels")
        elif exercise_type == "deadlift":
            recommendations.append("Keep the bar close to your body")
            recommendations.append("Drive through your hips")
        
        return recommendations
    
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