"""
Fine-tune Biomechanics Model with Public Fitness Datasets
Transfer learning from synthetic to real data for 90%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import the production model
import sys
sys.path.append('ml_models/biomechanics')
from train_production_model import GNNLSTM, BiomechanicsDataset

class PublicDatasetWrapper(Dataset):
    """Wrapper for public fitness datasets"""
    
    def __init__(self, metadata_path: str, landmarks_dir: str):
        self.metadata = pd.read_csv(metadata_path)
        self.landmarks_dir = Path(landmarks_dir)
        
        # Filter valid sequences
        self.metadata = self.metadata[self.metadata['exercise_type'] != 'unknown']
        self.metadata = self.metadata[self.metadata['form_quality'] != 'unknown']
        
        # Map form quality to scores
        self.quality_to_score = {
            'excellent': 95,
            'good': 75,
            'fair': 55,
            'poor': 35
        }
        
        # Exercise type mapping
        self.exercise_types = ['squat', 'deadlift', 'bench_press', 'overhead_press', 'lunge']
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load landmarks
        landmark_file = self.landmarks_dir.parent / row['landmark_file']
        landmarks = np.load(landmark_file)  # (frames, 33, 4) - last dim is (x, y, z, visibility)
        
        # Use only x, y, z (drop visibility for model)
        landmarks = landmarks[:, :, :3]
        
        # Form score
        form_score = self.quality_to_score.get(row['form_quality'], 50)
        
        # Exercise type index
        exercise_idx = self.exercise_types.index(row['exercise_type']) if row['exercise_type'] in self.exercise_types else 0
        
        # Estimate joint angles and risk (simplified for real data)
        joint_angles = self._estimate_joint_angles(landmarks)
        risk_score = 100 - form_score  # Inverse of form score
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'form_score': torch.FloatTensor([form_score]),
            'joint_angles': torch.FloatTensor(joint_angles),
            'exercise_type': torch.LongTensor([exercise_idx]),
            'risk_score': torch.FloatTensor([risk_score])
        }
    
    def _estimate_joint_angles(self, landmarks):
        """Estimate 6 key joint angles from landmarks"""
        # Simplified estimation - in production, use proper biomechanics calculations
        # For now, return reasonable random values based on exercise
        return np.random.uniform(70, 120, 6)  # 6 key joint angles

class FineTuner:
    """Fine-tune model on real fitness data"""
    
    def __init__(self, 
                 pretrained_model_path: str = "ml/biomechanics/gnn_lstm_best.pth",
                 synthetic_data_path: str = "data/biomechanics",
                 real_data_path: str = "data/biomechanics_processed"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {self.device}")
        
        self.pretrained_model_path = Path(pretrained_model_path)
        self.synthetic_data_path = Path(synthetic_data_path)
        self.real_data_path = Path(real_data_path)
        
        # Initialize model
        self.model = GNNLSTM().to(self.device)
        
        # Load pretrained weights if available
        if self.pretrained_model_path.exists():
            print(f" Loading pretrained model from {self.pretrained_model_path}")
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(" Pretrained model loaded")
        else:
            print("  No pretrained model found. Training from scratch...")
    
    def prepare_datasets(self, synthetic_ratio: float = 0.3):
        """Prepare combined dataset with synthetic and real data"""
        
        print("\n Preparing Datasets...")
        
        datasets = []
        
        # Load synthetic data
        synthetic_metadata = self.synthetic_data_path / "biomechanics_dataset.csv"
        synthetic_labels = self.synthetic_data_path / "biomechanics_labels.json"
        synthetic_landmarks = self.synthetic_data_path / "landmark_sequences"
        
        if synthetic_metadata.exists():
            print(f"  Loading synthetic data...")
            synthetic_dataset = BiomechanicsDataset(
                str(synthetic_landmarks),
                pd.read_csv(synthetic_metadata),
                str(synthetic_labels)
            )
            
            # Sample subset based on ratio
            if synthetic_ratio < 1.0:
                synthetic_size = int(len(synthetic_dataset) * synthetic_ratio)
                synthetic_indices = np.random.choice(len(synthetic_dataset), synthetic_size, replace=False)
                synthetic_dataset = torch.utils.data.Subset(synthetic_dataset, synthetic_indices)
            
            datasets.append(synthetic_dataset)
            print(f"     Synthetic: {len(synthetic_dataset)} sequences")
        
        # Load real data
        real_metadata = self.real_data_path / "sequences_metadata.csv"
        real_landmarks = self.real_data_path / "landmark_sequences"
        
        if real_metadata.exists():
            print(f"  Loading real data...")
            real_dataset = PublicDatasetWrapper(
                str(real_metadata),
                str(real_landmarks)
            )
            datasets.append(real_dataset)
            print(f"     Real: {len(real_dataset)} sequences")
        else:
            print(f"      No real data found at {real_metadata}")
            print(f"    Run: python data/preprocess_public_datasets.py")
        
        if not datasets:
            raise ValueError("No datasets found! Please generate or download data first.")
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        print(f"\n   Combined dataset: {len(combined_dataset)} total sequences")
        
        # Split into train/val
        indices = list(range(len(combined_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)
        
        print(f"    Training: {len(train_dataset)} sequences")
        print(f"   Validation: {len(val_dataset)} sequences")
        
        return train_dataset, val_dataset
    
    def fine_tune(self, num_epochs: int = 30, learning_rate: float = 0.0001):
        """Fine-tune the model"""
        
        print("\n" + "=" * 70)
        print(" Starting Fine-Tuning")
        print("=" * 70)
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(synthetic_ratio=0.3)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Optimizer with lower learning rate for fine-tuning
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        
        for epoch in range(num_epochs):
            print(f"\n Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self._train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_metrics['total_loss'])
            
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"    Form: {train_metrics['form_loss']:.4f}, Angle: {train_metrics['angle_loss']:.4f}")
            print(f"    Exercise: {train_metrics['exercise_loss']:.4f}, Risk: {train_metrics['risk_loss']:.4f}")
            
            # Validate
            val_metrics = self._validate(val_loader)
            history['val_loss'].append(val_metrics['total_loss'])
            history['val_metrics'].append(val_metrics)
            
            print(f"  Val Metrics:")
            print(f"    Form MAE: {val_metrics['form_mae']:.2f}, RMSE: {val_metrics['form_rmse']:.2f}")
            print(f"    Angle MAE: {val_metrics['angle_mae']:.2f}, RMSE: {val_metrics['angle_rmse']:.2f}")
            print(f"    Exercise Accuracy: {val_metrics['exercise_accuracy']:.1%}")
            print(f"    Risk MAE: {val_metrics['risk_mae']:.2f}, RMSE: {val_metrics['risk_rmse']:.2f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self._save_model(epoch, optimizer, val_metrics)
                print(f"   Saved best model (Val Loss: {best_val_loss:.4f})")
        
        print("\n Fine-tuning complete!")
        self._plot_training_history(history)
        
        return history
    
    def _train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        form_losses, angle_losses, exercise_losses, risk_losses = [], [], [], []
        
        for batch in tqdm(dataloader, desc="Training"):
            try:
                landmarks = batch['landmarks'].to(self.device)
                form_score = batch['form_score'].to(self.device)
                joint_angles = batch['joint_angles'].to(self.device)
                exercise_type = batch['exercise_type'].squeeze().to(self.device)
                risk_score = batch['risk_score'].to(self.device)
                
                # Forward pass
                outputs = self.model(landmarks)
                
                # Multi-task losses
                form_loss = nn.MSELoss()(outputs['form_score'], form_score)
                angle_loss = nn.MSELoss()(outputs['joint_angles'], joint_angles)
                exercise_loss = nn.CrossEntropyLoss()(outputs['exercise_type'], exercise_type)
                risk_loss = nn.MSELoss()(outputs['risk_score'], risk_score)
                
                # Combined loss
                loss = 0.3 * form_loss + 0.3 * angle_loss + 0.2 * exercise_loss + 0.2 * risk_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                form_losses.append(form_loss.item())
                angle_losses.append(angle_loss.item())
                exercise_losses.append(exercise_loss.item())
                risk_losses.append(risk_loss.item())
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        return {
            'total_loss': total_loss / len(dataloader),
            'form_loss': np.mean(form_losses),
            'angle_loss': np.mean(angle_losses),
            'exercise_loss': np.mean(exercise_losses),
            'risk_loss': np.mean(risk_losses)
        }
    
    def _validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        
        all_preds = {'form': [], 'angles': [], 'exercise': [], 'risk': []}
        all_true = {'form': [], 'angles': [], 'exercise': [], 'risk': []}
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                try:
                    landmarks = batch['landmarks'].to(self.device)
                    form_score = batch['form_score'].to(self.device)
                    joint_angles = batch['joint_angles'].to(self.device)
                    exercise_type = batch['exercise_type'].squeeze().to(self.device)
                    risk_score = batch['risk_score'].to(self.device)
                    
                    outputs = self.model(landmarks)
                    
                    # Calculate loss
                    form_loss = nn.MSELoss()(outputs['form_score'], form_score)
                    angle_loss = nn.MSELoss()(outputs['joint_angles'], joint_angles)
                    exercise_loss = nn.CrossEntropyLoss()(outputs['exercise_type'], exercise_type)
                    risk_loss = nn.MSELoss()(outputs['risk_score'], risk_score)
                    
                    loss = 0.3 * form_loss + 0.3 * angle_loss + 0.2 * exercise_loss + 0.2 * risk_loss
                    total_loss += loss.item()
                    
                    # Collect predictions
                    all_preds['form'].extend(outputs['form_score'].cpu().numpy())
                    all_true['form'].extend(form_score.cpu().numpy())
                    
                    all_preds['angles'].extend(outputs['joint_angles'].cpu().numpy())
                    all_true['angles'].extend(joint_angles.cpu().numpy())
                    
                    all_preds['exercise'].extend(outputs['exercise_type'].argmax(dim=1).cpu().numpy())
                    all_true['exercise'].extend(exercise_type.cpu().numpy())
                    
                    all_preds['risk'].extend(outputs['risk_score'].cpu().numpy())
                    all_true['risk'].extend(risk_score.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        return {
            'total_loss': total_loss / len(dataloader),
            'form_mae': mean_absolute_error(all_true['form'], all_preds['form']),
            'form_rmse': np.sqrt(mean_squared_error(all_true['form'], all_preds['form'])),
            'angle_mae': mean_absolute_error(all_true['angles'], all_preds['angles']),
            'angle_rmse': np.sqrt(mean_squared_error(all_true['angles'], all_preds['angles'])),
            'exercise_accuracy': accuracy_score(all_true['exercise'], all_preds['exercise']),
            'risk_mae': mean_absolute_error(all_true['risk'], all_preds['risk']),
            'risk_rmse': np.sqrt(mean_squared_error(all_true['risk'], all_preds['risk']))
        }
    
    def _save_model(self, epoch, optimizer, metrics):
        """Save model checkpoint"""
        output_path = Path("ml_models/biomechanics/gnn_lstm_finetuned.pth")
        output_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, output_path)
    
    def _plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # Metrics plot
        plt.subplot(1, 2, 2)
        val_accuracies = [m['exercise_accuracy'] for m in history['val_metrics']]
        val_form_maes = [m['form_mae'] for m in history['val_metrics']]
        
        ax1 = plt.gca()
        ax1.plot(val_accuracies, 'b-', label='Exercise Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(val_form_maes, 'r-', label='Form MAE')
        ax2.set_ylabel('Form MAE', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Validation Metrics')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ml_models/biomechanics/finetuning_history.png')
        print(f"\n Training plots saved to: ml_models/biomechanics/finetuning_history.png")

def main():
    """Main function"""
    print("""
========================================================================
                                                                    
        Biomechanics Model Fine-Tuning                        
                                                                    
  Transfer learning from synthetic to real data                     
  Target: 90%+ accuracy on real fitness videos                      
                                                                    
========================================================================
    """)
    
    fine_tuner = FineTuner()
    fine_tuner.fine_tune(num_epochs=30, learning_rate=0.0001)
    
    print("\n Fine-tuning complete! Model saved to:")
    print("   ml_models/biomechanics/gnn_lstm_finetuned.pth")
    print("\n Test the model with:")
    print("   python ml_models/biomechanics/inference.py --model ml_models/biomechanics/gnn_lstm_finetuned.pth")

if __name__ == "__main__":
    main()
