"""
Create mock biomechanics models for initial deployment
These can be replaced with real trained models later
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from datetime import datetime

class MockBiomechanicsModel(nn.Module):
    """Mock GNN-LSTM model that generates reasonable outputs"""
    
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple feed-forward network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 33, hidden_dim),  # 33 landmarks
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.joint_predictor = nn.Linear(hidden_dim, output_dim)
        self.torque_predictor = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        joint_angles = self.joint_predictor(features)
        torques = self.torque_predictor(features)
        return joint_angles, torques

def create_biomechanics_models():
    """Create mock biomechanics models"""
    
    output_dir = "ml_models/biomechanics/models"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating mock biomechanics models...")
    
    # 1. GNN-LSTM Model
    print("  - Creating GNN-LSTM model...")
    gnn_lstm_model = MockBiomechanicsModel(input_dim=3, hidden_dim=128, output_dim=10)
    
    # Initialize with reasonable weights
    for param in gnn_lstm_model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    torch.save({
        'model_state_dict': gnn_lstm_model.state_dict(),
        'input_dim': 3,
        'hidden_dim': 128,
        'output_dim': 10,
        'created_date': datetime.now().isoformat(),
        'model_type': 'mock_gnn_lstm',
        'version': '1.0.0'
    }, f"{output_dir}/gnn_lstm_model.pth")
    
    print(f"    ✓ Saved to {output_dir}/gnn_lstm_model.pth")
    
    # 2. Joint Angle Model
    print("  - Creating joint angle model...")
    joint_angle_model = MockBiomechanicsModel(input_dim=3, hidden_dim=64, output_dim=10)
    
    torch.save({
        'model_state_dict': joint_angle_model.state_dict(),
        'input_dim': 3,
        'hidden_dim': 64,
        'output_dim': 10,
        'created_date': datetime.now().isoformat(),
        'model_type': 'mock_joint_angle',
        'version': '1.0.0'
    }, f"{output_dir}/joint_angle_model.pth")
    
    print(f"    ✓ Saved to {output_dir}/joint_angle_model.pth")
    
    # 3. Torque Heatmap Model
    print("  - Creating torque heatmap model...")
    torque_model = MockBiomechanicsModel(input_dim=3, hidden_dim=64, output_dim=10)
    
    torch.save({
        'model_state_dict': torque_model.state_dict(),
        'input_dim': 3,
        'hidden_dim': 64,
        'output_dim': 10,
        'created_date': datetime.now().isoformat(),
        'model_type': 'mock_torque_heatmap',
        'version': '1.0.0'
    }, f"{output_dir}/torque_heatmap_model.pth")
    
    print(f"    ✓ Saved to {output_dir}/torque_heatmap_model.pth")
    
    # 4. Create metadata
    metadata = {
        'models': {
            'gnn_lstm': {
                'file': 'gnn_lstm_model.pth',
                'type': 'mock',
                'description': 'Mock GNN-LSTM for pose estimation and movement analysis'
            },
            'joint_angle': {
                'file': 'joint_angle_model.pth',
                'type': 'mock',
                'description': 'Mock joint angle calculation model'
            },
            'torque_heatmap': {
                'file': 'torque_heatmap_model.pth',
                'type': 'mock',
                'description': 'Mock torque heatmap generation model'
            }
        },
        'created_date': datetime.now().isoformat(),
        'version': '1.0.0',
        'note': 'These are mock models for initial deployment. Replace with trained models for production.'
    }
    
    with open(f"{output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"    ✓ Saved metadata to {output_dir}/metadata.pkl")
    
    print("\n✅ Biomechanics mock models created successfully!")
    print(f"📁 Location: {output_dir}")
    print("📝 Note: These are placeholder models. Team can replace with trained models later.")

if __name__ == "__main__":
    create_biomechanics_models()
