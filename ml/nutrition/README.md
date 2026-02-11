# Nutrition ML Models

This directory contains the trained models for food recognition and nutritional analysis.

## Models

- `cnn_gru_model.pth` - CNN-GRU model for food recognition
- `protein_estimation_model.pth` - Protein content estimation model
- `nutritional_balance_model.pth` - Nutritional balance assessment model

## Training Data

- `food_images/` - Food image datasets
- `nutritional_data/` - Nutritional information databases
- `validation_data/` - Validation datasets

## Usage

Models are automatically loaded by the `ProteinOptimizer` class in `backend/modules/nutrition.py`.

## Model Architecture

- **CNN-GRU**: Convolutional Neural Networks for feature extraction + GRU for sequential analysis
- **Input**: Food images
- **Output**: Food recognition, protein content, nutritional recommendations 