# Biomechanics ML Models

This directory contains the trained models for biomechanics analysis.

## Models

- `gnn_lstm_model.pth` - GNN-LSTM model for pose estimation and movement analysis
- `joint_angle_model.pth` - Joint angle calculation model
- `torque_heatmap_model.pth` - Torque heatmap generation model

## Training Data

- `training_data/` - Training datasets
- `validation_data/` - Validation datasets
- `test_data/` - Test datasets

## Usage

Models are automatically loaded by the `BiomechanicsCoach` class in `backend/modules/biomechanics.py`.

## Model Architecture

- **GNN-LSTM**: Graph Neural Networks for spatial relationships + LSTM for temporal sequences
- **Input**: Pose landmarks from MediaPipe
- **Output**: Form assessment, risk factors, recommendations 