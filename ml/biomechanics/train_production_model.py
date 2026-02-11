"""
Production GNN-LSTM Training Pipeline for Biomechanics Analysis
Uses comprehensive synthetic dataset with realistic exercise sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm

class BiomechanicsDataset(Dataset):
    """PyTorch dataset for biomechanics sequences"""
    
    def __init__(self, landmarks_dir: str, metadata_df: pd.DataFrame, labels_path: str):
        self.landmarks_dir = Path(landmarks_dir)
        self.metadata = metadata_df
        
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Load landmark sequence
        sequence_file = self.landmarks_dir / f"sequence_{idx}.npy"
        landmarks = np.load(sequence_file)  # (frames, 33, 3)
        
        # Get label
        label_data = self.labels[f"sequence_{idx}"]
        
        # Form quality score (0-100)
        quality_map = {'excellent': 95, 'good': 75, 'fair': 55, 'poor': 35}
        form_score = quality_map[label_data['form_quality']]
        
        # Joint angles ground truth
        joint_angles = np.array([
            label_data['avg_joint_angles']['left_knee'],
            label_data['avg_joint_angles']['right_knee'],
            label_data['avg_joint_angles']['left_hip'],
            label_data['avg_joint_angles']['right_hip'],
            label_data['avg_joint_angles']['left_shoulder'],
            label_data['avg_joint_angles']['right_shoulder']
        ])
        
        # Exercise type (one-hot)
        exercise_types = ['squat', 'deadlift', 'bench_press', 'overhead_press', 'lunge']
        exercise_idx = exercise_types.index(label_data['exercise_type'])
        
        # Risk score
        risk_score = label_data['injury_risk_score']
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'form_score': torch.FloatTensor([form_score]),
            'joint_angles': torch.FloatTensor(joint_angles),
            'exercise_type': torch.LongTensor([exercise_idx]),
            'risk_score': torch.FloatTensor([risk_score])
        }

class GraphConvolution(nn.Module):
    """Graph convolution layer for skeletal structure"""
    
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        x: (batch, nodes, features)
        adj: (nodes, nodes) adjacency matrix
        """
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj, support)  # Aggregate from neighbors
        return output + self.bias

class GNNLSTM(nn.Module):
    """GNN-LSTM model for biomechanics analysis"""
    
    def __init__(self, num_joints: int = 33, joint_dim: int = 3, 
                 gcn_hidden: int = 64, lstm_hidden: int = 128, 
                 num_exercises: int = 5):
        super(GNNLSTM, self).__init__()
        
        # Graph structure (33 MediaPipe landmarks)
        self.num_joints = num_joints
        self.adjacency = self.create_skeleton_adjacency()
        
        # GCN layers
        self.gcn1 = GraphConvolution(joint_dim, gcn_hidden)
        self.gcn2 = GraphConvolution(gcn_hidden, gcn_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=num_joints * gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Prediction heads
        self.form_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.joint_angle_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)  # 6 key joint angles
        )
        
        self.exercise_classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_exercises)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def create_skeleton_adjacency(self):
        """Create adjacency matrix for MediaPipe skeleton"""
        adj = torch.zeros(33, 33)
        
        # Define connections (MediaPipe pose landmark connections)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head
            (0, 4), (4, 5), (5, 6), (6, 8),  # Head
            (9, 10),  # Mouth
            (11, 12),  # Shoulders
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
        ]
        
        # Make adjacency symmetric
        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Add self-loops
        adj += torch.eye(33)
        
        # Normalize
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / degree
        
        return adj
    
    def forward(self, x):
        """
        x: (batch, frames, joints, 3)
        """
        batch_size, frames, joints, dims = x.size()
        
        # Process each frame with GCN
        frame_features = []
        for t in range(frames):
            frame = x[:, t, :, :]  # (batch, joints, 3)
            
            # GCN layers
            h = self.gcn1(frame, self.adjacency.to(x.device))
            h = self.relu(h)
            h = self.dropout(h)
            h = self.gcn2(h, self.adjacency.to(x.device))
            h = self.relu(h)
            
            # Flatten joint features
            h = h.view(batch_size, -1)  # (batch, joints * gcn_hidden)
            frame_features.append(h)
        
        # Stack frames
        sequence = torch.stack(frame_features, dim=1)  # (batch, frames, joints * gcn_hidden)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(sequence)
        
        # Use last timestep
        final_state = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        
        # Prediction heads
        form_score = self.form_head(final_state) * 100  # Scale to 0-100
        joint_angles = self.joint_angle_head(final_state)
        exercise_type = self.exercise_classifier(final_state)
        risk_score = self.risk_head(final_state) * 100  # Scale to 0-100
        
        return {
            'form_score': form_score,
            'joint_angles': joint_angles,
            'exercise_type': exercise_type,
            'risk_score': risk_score
        }

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    form_losses = []
    angle_losses = []
    exercise_losses = []
    risk_losses = []
    
    for batch in tqdm(dataloader, desc="Training"):
        landmarks = batch['landmarks'].to(device)
        form_score = batch['form_score'].to(device)
        joint_angles = batch['joint_angles'].to(device)
        exercise_type = batch['exercise_type'].squeeze().to(device)
        risk_score = batch['risk_score'].to(device)
        
        # Forward pass
        outputs = model(landmarks)
        
        # Multi-task losses
        form_loss = nn.MSELoss()(outputs['form_score'], form_score)
        angle_loss = nn.MSELoss()(outputs['joint_angles'], joint_angles)
        exercise_loss = nn.CrossEntropyLoss()(outputs['exercise_type'], exercise_type)
        risk_loss = nn.MSELoss()(outputs['risk_score'], risk_score)
        
        # Combined loss with weights
        loss = 0.3 * form_loss + 0.3 * angle_loss + 0.2 * exercise_loss + 0.2 * risk_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        form_losses.append(form_loss.item())
        angle_losses.append(angle_loss.item())
        exercise_losses.append(exercise_loss.item())
        risk_losses.append(risk_loss.item())
    
    return {
        'total_loss': total_loss / len(dataloader),
        'form_loss': np.mean(form_losses),
        'angle_loss': np.mean(angle_losses),
        'exercise_loss': np.mean(exercise_losses),
        'risk_loss': np.mean(risk_losses)
    }

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    
    all_form_preds = []
    all_form_true = []
    all_angle_preds = []
    all_angle_true = []
    all_exercise_preds = []
    all_exercise_true = []
    all_risk_preds = []
    all_risk_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            landmarks = batch['landmarks'].to(device)
            form_score = batch['form_score'].to(device)
            joint_angles = batch['joint_angles'].to(device)
            exercise_type = batch['exercise_type'].squeeze().to(device)
            risk_score = batch['risk_score'].to(device)
            
            outputs = model(landmarks)
            
            all_form_preds.extend(outputs['form_score'].cpu().numpy())
            all_form_true.extend(form_score.cpu().numpy())
            
            all_angle_preds.extend(outputs['joint_angles'].cpu().numpy())
            all_angle_true.extend(joint_angles.cpu().numpy())
            
            all_exercise_preds.extend(outputs['exercise_type'].argmax(dim=1).cpu().numpy())
            all_exercise_true.extend(exercise_type.cpu().numpy())
            
            all_risk_preds.extend(outputs['risk_score'].cpu().numpy())
            all_risk_true.extend(risk_score.cpu().numpy())
    
    # Calculate metrics
    form_mae = mean_absolute_error(all_form_true, all_form_preds)
    form_rmse = np.sqrt(mean_squared_error(all_form_true, all_form_preds))
    
    angle_mae = mean_absolute_error(all_angle_true, all_angle_preds)
    angle_rmse = np.sqrt(mean_squared_error(all_angle_true, all_angle_preds))
    
    exercise_acc = accuracy_score(all_exercise_true, all_exercise_preds)
    
    risk_mae = mean_absolute_error(all_risk_true, all_risk_preds)
    risk_rmse = np.sqrt(mean_squared_error(all_risk_true, all_risk_preds))
    
    return {
        'form_mae': form_mae,
        'form_rmse': form_rmse,
        'angle_mae': angle_mae,
        'angle_rmse': angle_rmse,
        'exercise_accuracy': exercise_acc,
        'risk_mae': risk_mae,
        'risk_rmse': risk_rmse
    }

def main():
    """Train production GNN-LSTM model"""
    
    print("🏋️ Training Production Biomechanics GNN-LSTM Model")
    print("=" * 60)
    
    # Paths
    data_dir = Path("data/biomechanics")
    metadata_path = data_dir / "biomechanics_dataset.csv"
    labels_path = data_dir / "biomechanics_labels.json"
    landmarks_dir = data_dir / "landmark_sequences"
    
    # Check if dataset exists
    if not metadata_path.exists():
        print("⚠️  Dataset not found. Generating...")
        import subprocess
        subprocess.run(["python", "data/generate_biomechanics_dataset.py"])
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"✅ Loaded {len(metadata)} sequences")
    
    # Split dataset
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = BiomechanicsDataset(landmarks_dir, train_meta.reset_index(drop=True), labels_path)
    val_dataset = BiomechanicsDataset(landmarks_dir, val_meta.reset_index(drop=True), labels_path)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    model = GNNLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_metrics['total_loss'])
        
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"    Form: {train_metrics['form_loss']:.4f}, Angle: {train_metrics['angle_loss']:.4f}")
        print(f"    Exercise: {train_metrics['exercise_loss']:.4f}, Risk: {train_metrics['risk_loss']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        val_metrics_history.append(val_metrics)
        
        print(f"  Val Metrics:")
        print(f"    Form MAE: {val_metrics['form_mae']:.2f}, RMSE: {val_metrics['form_rmse']:.2f}")
        print(f"    Angle MAE: {val_metrics['angle_mae']:.2f}°, RMSE: {val_metrics['angle_rmse']:.2f}°")
        print(f"    Exercise Accuracy: {val_metrics['exercise_accuracy']:.1%}")
        print(f"    Risk MAE: {val_metrics['risk_mae']:.2f}, RMSE: {val_metrics['risk_rmse']:.2f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['form_rmse'])
        
        # Save best model
        if val_metrics['form_rmse'] < best_val_loss:
            best_val_loss = val_metrics['form_rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, 'ml/biomechanics/gnn_lstm_best.pth')
            print(f"  ✅ Saved best model (Form RMSE: {best_val_loss:.2f})")
    
    print("\n🎉 Training complete!")
    print(f"📁 Model saved to: ml/biomechanics/gnn_lstm_best.pth")

if __name__ == "__main__":
    main()
