# Burnout Prediction ML Models

This directory contains the trained models for burnout prediction and survival analysis.

## Models

- `cox_ph_model.pkl` - Cox Proportional Hazards model for survival analysis
- `risk_assessment_model.pkl` - Risk factor assessment model
- `trend_analysis_model.pkl` - Trend analysis model

## Training Data

- `user_metrics/` - User training and lifestyle metrics
- `burnout_events/` - Historical burnout event data
- `validation_data/` - Validation datasets

## Usage

Models are automatically loaded by the `BurnoutPredictor` class in `backend/modules/burnout.py`.

## Model Architecture

- **Cox Proportional Hazards**: Survival analysis for time-to-burnout prediction
- **Input**: User metrics (workout frequency, sleep, stress, etc.)
- **Output**: Risk scores, survival curves, preventive recommendations 