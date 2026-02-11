"""
Training script for Biomechanics GNN-LSTM Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import json
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the model
from gnn_lstm import BiomechanicsModel, BiomechanicsResult

logger = logging.getLogger(__name__)

class BiomechanicsDataset(Dataset):
    """Dataset for biomechanics training data"""
    
    def __init__(self, data_dir: str, sequence_length: int = 30):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load training samples from data directory"""
        samples = []
        
        # Look for video files and annotations
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mp4') or file.endswith('.avi'):
                    video_path = os.path.join(root, file)
                    annotation_path = video_path.replace('.mp4', '.json').replace('.avi', '.json')
                    
                    if os.path.exists(annotation_path):
                        samples.append({
                            'video_path': video_path,
                            'annotation_path': annotation_path
                        })
        
        logger.info(f"Loaded {len(samples)} training samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        sample = self.samples[idx]
        
        # Load video frames
        frames = self._load_video_frames(sample['video_path'])
        
        # Load annotations
        annotations = self._load_annotations(sample['annotation_path'])
        
        # Create sequence data
        sequence_data = self._create_sequence_data(frames, annotations)
        
        return sequence_data
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _load_annotations(self, annotation_path: str) -> Dict:
        """Load annotations from JSON file"""
        with open(annotation_path, 'r') as f:
            return json.load(f)
    
    def _create_sequence_data(self, frames: List[np.ndarray], annotations: Dict) -> Dict:
        """Create sequence data for training"""
        # Extract pose landmarks for each frame
        model = BiomechanicsModel()
        landmarks_sequence = []
        joint_angles_sequence = []
        torques_sequence = []
        
        for i, frame in enumerate(frames[:self.sequence_length]):
            # Analyze frame
            result = model.analyze_frame(frame)
            
            # Extract features
            landmarks = self._extract_landmarks_from_result(result)
            joint_angles = self._extract_joint_angles_from_result(result)
            torques = self._extract_torques_from_result(result)
            
            landmarks_sequence.append(landmarks)
            joint_angles_sequence.append(joint_angles)
            torques_sequence.append(torques)
        
        # Pad sequences if needed
        while len(landmarks_sequence) < self.sequence_length:
            landmarks_sequence.append(np.zeros_like(landmarks_sequence[0]))
            joint_angles_sequence.append(np.zeros_like(joint_angles_sequence[0]))
            torques_sequence.append(np.zeros_like(torques_sequence[0]))
        
        return {
            'landmarks': torch.tensor(landmarks_sequence, dtype=torch.float32),
            'joint_angles': torch.tensor(joint_angles_sequence, dtype=torch.float32),
            'torques': torch.tensor(torques_sequence, dtype=torch.float32),
            'form_score': torch.tensor(annotations.get('form_score', 75.0), dtype=torch.float32)
        }
    
    def _extract_landmarks_from_result(self, result: BiomechanicsResult) -> np.ndarray:
        """Extract landmarks from analysis result"""
        # Placeholder - in real implementation, this would extract from MediaPipe
        return np.random.rand(33, 3)  # 33 landmarks, 3D coordinates
    
    def _extract_joint_angles_from_result(self, result: BiomechanicsResult) -> np.ndarray:
        """Extract joint angles from analysis result"""
        angles = []
        for joint_angle in result.joint_angles:
            angles.append(joint_angle.angle)
        
        # Pad to fixed size
        while len(angles) < 10:  # 10 major joints
            angles.append(0.0)
        
        return np.array(angles[:10])
    
    def _extract_torques_from_result(self, result: BiomechanicsResult) -> np.ndarray:
        """Extract torques from analysis result"""
        torques = []
        for torque in result.torques:
            torques.append(torque.torque_magnitude)
        
        # Pad to fixed size
        while len(torques) < 10:  # 10 major joints
            torques.append(0.0)
        
        return np.array(torques[:10])

class BiomechanicsTrainer:
    """Trainer for biomechanics model"""
    
    def __init__(self, model: BiomechanicsModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.form_scores = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move data to device
            landmarks = batch['landmarks'].to(self.device)
            joint_angles = batch['joint_angles'].to(self.device)
            torques = batch['torques'].to(self.device)
            form_scores = batch['form_score'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Create edge index for GNN
            edge_index = self._create_edge_index().to(self.device)
            batch_idx = torch.zeros(landmarks.size(1), dtype=torch.long, device=self.device)
            
            # Reshape for GNN-LSTM
            batch_size, seq_len, num_joints, features = landmarks.shape
            landmarks_flat = landmarks.view(-1, features)
            batch_idx_flat = batch_idx.repeat(batch_size * seq_len)
            
            # Forward pass
            joint_pred, torque_pred = self.model(landmarks_flat, edge_index, batch_idx_flat)
            
            # Reshape predictions
            joint_pred = joint_pred.view(batch_size, seq_len, -1)
            torque_pred = torque_pred.view(batch_size, seq_len, -1)
            
            # Calculate losses
            joint_loss = self.criterion(joint_pred, joint_angles)
            torque_loss = self.criterion(torque_pred, torques)
            
            total_loss = joint_loss + torque_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            self.train_losses.append(total_loss.item())
        
        self.scheduler.step()
        return np.mean(self.train_losses[-len(dataloader):])
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move data to device
                landmarks = batch['landmarks'].to(self.device)
                joint_angles = batch['joint_angles'].to(self.device)
                torques = batch['torques'].to(self.device)
                
                # Create edge index for GNN
                edge_index = self._create_edge_index().to(self.device)
                batch_idx = torch.zeros(landmarks.size(1), dtype=torch.long, device=self.device)
                
                # Reshape for GNN-LSTM
                batch_size, seq_len, num_joints, features = landmarks.shape
                landmarks_flat = landmarks.view(-1, features)
                batch_idx_flat = batch_idx.repeat(batch_size * seq_len)
                
                # Forward pass
                joint_pred, torque_pred = self.model(landmarks_flat, edge_index, batch_idx_flat)
                
                # Reshape predictions
                joint_pred = joint_pred.view(batch_size, seq_len, -1)
                torque_pred = torque_pred.view(batch_size, seq_len, -1)
                
                # Calculate losses
                joint_loss = self.criterion(joint_pred, joint_angles)
                torque_loss = self.criterion(torque_pred, torques)
                
                total_loss = joint_loss + torque_loss
                self.val_losses.append(total_loss.item())
        
        return np.mean(self.val_losses[-len(dataloader):])
    
    def _create_edge_index(self) -> torch.Tensor:
        """Create edge index for GNN"""
        # Define joint connections (MediaPipe pose landmarks)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head and neck
            (0, 4), (4, 5), (5, 6),  # Left arm
            (0, 8), (8, 9), (9, 10),  # Right arm
            (11, 12), (12, 14), (14, 16),  # Left leg
            (11, 13), (13, 15), (15, 17),  # Right leg
            (11, 23), (23, 24), (24, 26), (26, 28),  # Left leg extended
            (12, 25), (25, 27), (27, 29), (29, 31),  # Right leg extended
        ]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.form_scores, label='Form Score')
        plt.title('Form Score Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Form Score')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def main():
    """Main training function"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    model = BiomechanicsModel()
    logger.info("Model created successfully")
    
    # Create datasets
    train_dataset = BiomechanicsDataset('data/train', sequence_length=30)
    val_dataset = BiomechanicsDataset('data/val', sequence_length=30)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create trainer
    trainer = BiomechanicsTrainer(model, device)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model('models/biomechanics/best_model.pth')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_model(f'models/biomechanics/checkpoint_epoch_{epoch+1}.pth')
    
    # Plot training history
    trainer.plot_training_history('models/biomechanics/training_history.png')
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 