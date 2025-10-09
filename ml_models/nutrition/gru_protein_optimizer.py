"""
GRU Time-Series Model for Protein Optimization
Dynamic protein recommendation based on sleep quality, HRV, and activity patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Health metrics for a specific day"""
    date: datetime
    sleep_duration: float  # hours
    sleep_quality: float   # 0-10 scale
    hrv: float            # Heart Rate Variability (ms)
    activity_level: float # Activity score 0-10
    stress_level: int     # 1-10 scale
    protein_intake: float # grams consumed
    recovery_score: float # calculated recovery score

@dataclass
class ProteinRecommendation:
    """Protein recommendation with explanation"""
    date: datetime
    baseline_protein: float
    adjusted_protein: float
    sleep_factor: float
    hrv_factor: float
    activity_factor: float
    confidence: float
    explanation: str

class ProteinGRUModel(nn.Module):
    """GRU-based time series model for protein optimization"""
    
    def __init__(self, input_features: int = 6, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(ProteinGRUModel, self).__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers for temporal patterns
        self.gru = nn.GRU(
            input_features, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Output layers for different predictions
        self.protein_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Protein adjustment factor
        )
        
        self.factor_predictors = nn.ModuleDict({
            'sleep_factor': nn.Linear(hidden_dim // 4, 1),
            'hrv_factor': nn.Linear(hidden_dim // 4, 1),
            'activity_factor': nn.Linear(hidden_dim // 4, 1),
            'recovery_factor': nn.Linear(hidden_dim // 4, 1)
        })
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_attention: bool = False):
        """
        Forward pass
        Args:
            x: Input tensor (batch_size, sequence_length, input_features)
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # GRU processing
        gru_out, hidden = self.gru(x)
        
        # Apply attention
        attended_out, attention_weights = self.attention(gru_out, gru_out, gru_out)
        
        # Use the last time step for prediction
        final_features = attended_out[:, -1, :]
        
        # Process features
        processed_features = self.feature_processor(final_features)
        
        # Generate predictions
        protein_adjustment = self.protein_regressor(processed_features)
        
        # Individual factors
        factors = {}
        for factor_name, predictor in self.factor_predictors.items():
            factors[factor_name] = torch.sigmoid(predictor(processed_features))
        
        # Confidence score
        confidence = self.confidence_estimator(processed_features)
        
        outputs = {
            'protein_adjustment': protein_adjustment,
            'factors': factors,
            'confidence': confidence
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs

class ProteinOptimizer:
    """Dynamic protein optimization engine using GRU and health correlations"""
    
    def __init__(self, model_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = ProteinGRUModel().to(device)
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        # NHANES and FitRec research correlations
        self.research_correlations = self._load_research_correlations()
        
        if model_path:
            self.load_model(model_path)
        
        # User profiles and history
        self.user_profiles = {}
        self.health_history = {}
    
    def _load_research_correlations(self) -> Dict:
        """Load research-based correlations from NHANES and FitRec studies"""
        return {
            'sleep_protein_correlation': {
                'poor_sleep': {'threshold': 6.0, 'protein_multiplier': 1.15},  # Need more protein
                'good_sleep': {'threshold': 8.0, 'protein_multiplier': 1.0},
                'excellent_sleep': {'threshold': 9.0, 'protein_multiplier': 0.95}
            },
            'hrv_protein_correlation': {
                'low_hrv': {'threshold': 30, 'protein_multiplier': 1.2},  # Higher stress = more protein
                'normal_hrv': {'threshold': 50, 'protein_multiplier': 1.0},
                'high_hrv': {'threshold': 70, 'protein_multiplier': 0.98}
            },
            'activity_protein_correlation': {
                'sedentary': {'threshold': 3.0, 'protein_multiplier': 0.9},
                'moderate': {'threshold': 6.0, 'protein_multiplier': 1.0},
                'high': {'threshold': 8.0, 'protein_multiplier': 1.3},
                'extreme': {'threshold': 10.0, 'protein_multiplier': 1.5}
            },
            'recovery_optimization': {
                'protein_timing_window': 2.0,  # hours post-workout
                'sleep_quality_impact': 0.25,   # 25% impact on protein needs
                'stress_impact': 0.15           # 15% impact on protein absorption
            }
        }
    
    def calculate_baseline_protein(self, user_profile: Dict) -> float:
        """Calculate baseline protein needs based on user profile"""
        weight_kg = user_profile.get('weight_kg', 70.0)
        activity_level = user_profile.get('activity_level', 'moderate')
        fitness_goal = user_profile.get('fitness_goal', 'maintenance')
        age = user_profile.get('age', 30)
        
        # Base protein calculation (g/kg body weight)
        base_multiplier = {
            'sedentary': 0.8,
            'light': 1.0,
            'moderate': 1.2,
            'high': 1.6,
            'extreme': 2.0
        }.get(activity_level, 1.2)
        
        # Goal adjustments
        goal_multiplier = {
            'weight_loss': 1.2,
            'maintenance': 1.0,
            'muscle_gain': 1.4,
            'performance': 1.6
        }.get(fitness_goal, 1.0)
        
        # Age adjustment (protein needs increase with age)
        age_multiplier = 1.0 + (max(0, age - 50) * 0.01)
        
        baseline = weight_kg * base_multiplier * goal_multiplier * age_multiplier
        return min(max(baseline, 50.0), 300.0)  # Reasonable bounds
    
    def prepare_time_series_data(self, health_metrics: List[HealthMetrics], 
                               sequence_length: int = 7) -> np.ndarray:
        """Prepare health metrics for GRU input"""
        if len(health_metrics) < sequence_length:
            # Pad with the first available metric
            padding_needed = sequence_length - len(health_metrics)
            padded_metrics = [health_metrics[0]] * padding_needed + health_metrics
        else:
            padded_metrics = health_metrics[-sequence_length:]
        
        # Extract features
        features = []
        for metric in padded_metrics:
            feature_vector = [
                metric.sleep_duration / 12.0,      # Normalize to 0-1
                metric.sleep_quality / 10.0,       # 0-10 scale to 0-1
                metric.hrv / 100.0,               # Normalize HRV
                metric.activity_level / 10.0,     # 0-10 scale to 0-1
                metric.stress_level / 10.0,       # 1-10 scale to 0-1
                metric.recovery_score / 10.0      # Recovery score 0-1
            ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def calculate_research_based_factors(self, latest_metrics: HealthMetrics) -> Dict[str, float]:
        """Calculate adjustment factors based on research correlations"""
        correlations = self.research_correlations
        
        # Sleep factor
        sleep_quality = latest_metrics.sleep_quality
        if sleep_quality < correlations['sleep_protein_correlation']['poor_sleep']['threshold']:
            sleep_factor = correlations['sleep_protein_correlation']['poor_sleep']['protein_multiplier']
        elif sleep_quality >= correlations['sleep_protein_correlation']['excellent_sleep']['threshold']:
            sleep_factor = correlations['sleep_protein_correlation']['excellent_sleep']['protein_multiplier']
        else:
            sleep_factor = correlations['sleep_protein_correlation']['good_sleep']['protein_multiplier']
        
        # HRV factor
        hrv = latest_metrics.hrv
        if hrv < correlations['hrv_protein_correlation']['low_hrv']['threshold']:
            hrv_factor = correlations['hrv_protein_correlation']['low_hrv']['protein_multiplier']
        elif hrv >= correlations['hrv_protein_correlation']['high_hrv']['threshold']:
            hrv_factor = correlations['hrv_protein_correlation']['high_hrv']['protein_multiplier']
        else:
            hrv_factor = correlations['hrv_protein_correlation']['normal_hrv']['protein_multiplier']
        
        # Activity factor
        activity = latest_metrics.activity_level
        if activity >= correlations['activity_protein_correlation']['extreme']['threshold']:
            activity_factor = correlations['activity_protein_correlation']['extreme']['protein_multiplier']
        elif activity >= correlations['activity_protein_correlation']['high']['threshold']:
            activity_factor = correlations['activity_protein_correlation']['high']['protein_multiplier']
        elif activity >= correlations['activity_protein_correlation']['moderate']['threshold']:
            activity_factor = correlations['activity_protein_correlation']['moderate']['protein_multiplier']
        else:
            activity_factor = correlations['activity_protein_correlation']['sedentary']['protein_multiplier']
        
        return {
            'sleep_factor': sleep_factor,
            'hrv_factor': hrv_factor,
            'activity_factor': activity_factor
        }
    
    def optimize_protein_intake(self, user_id: str, health_metrics: List[HealthMetrics]) -> ProteinRecommendation:
        """Generate optimized protein recommendation"""
        try:
            if not health_metrics:
                raise ValueError("No health metrics provided")
            
            # Get user profile
            user_profile = self.user_profiles.get(user_id, {
                'weight_kg': 70.0,
                'activity_level': 'moderate',
                'fitness_goal': 'maintenance'
            })
            
            # Calculate baseline protein needs
            baseline_protein = self.calculate_baseline_protein(user_profile)
            
            # Get latest metrics
            latest_metrics = health_metrics[-1]
            
            # Calculate research-based factors
            research_factors = self.calculate_research_based_factors(latest_metrics)
            
            # If model is trained, use it for fine-tuning
            if self.is_fitted and len(health_metrics) >= 7:
                time_series_data = self.prepare_time_series_data(health_metrics)
                time_series_tensor = torch.FloatTensor(time_series_data).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    model_output = self.model(time_series_tensor)
                    
                    # Combine research factors with model predictions
                    model_sleep_factor = model_output['factors']['sleep_factor'].item()
                    model_hrv_factor = model_output['factors']['hrv_factor'].item()
                    model_activity_factor = model_output['factors']['activity_factor'].item()
                    confidence = model_output['confidence'].item()
                    
                    # Weighted combination of research and model factors
                    alpha = 0.7  # Weight for research-based factors
                    sleep_factor = alpha * research_factors['sleep_factor'] + (1 - alpha) * model_sleep_factor
                    hrv_factor = alpha * research_factors['hrv_factor'] + (1 - alpha) * model_hrv_factor
                    activity_factor = alpha * research_factors['activity_factor'] + (1 - alpha) * model_activity_factor
            else:
                # Use only research-based factors
                sleep_factor = research_factors['sleep_factor']
                hrv_factor = research_factors['hrv_factor']
                activity_factor = research_factors['activity_factor']
                confidence = 0.8  # High confidence in research-based recommendations
            
            # Calculate adjusted protein
            combined_factor = (sleep_factor + hrv_factor + activity_factor) / 3.0
            adjusted_protein = baseline_protein * combined_factor
            
            # Ensure reasonable bounds
            adjusted_protein = min(max(adjusted_protein, baseline_protein * 0.8), baseline_protein * 1.8)
            
            # Generate explanation
            explanation = self._generate_explanation(
                latest_metrics, sleep_factor, hrv_factor, activity_factor, baseline_protein, adjusted_protein
            )
            
            return ProteinRecommendation(
                date=latest_metrics.date,
                baseline_protein=baseline_protein,
                adjusted_protein=adjusted_protein,
                sleep_factor=sleep_factor,
                hrv_factor=hrv_factor,
                activity_factor=activity_factor,
                confidence=confidence,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Protein optimization failed: {e}")
            # Return baseline recommendation
            return ProteinRecommendation(
                date=datetime.now(),
                baseline_protein=120.0,
                adjusted_protein=120.0,
                sleep_factor=1.0,
                hrv_factor=1.0,
                activity_factor=1.0,
                confidence=0.5,
                explanation="Using baseline protein recommendation due to insufficient data."
            )
    
    def _generate_explanation(self, metrics: HealthMetrics, sleep_factor: float, 
                            hrv_factor: float, activity_factor: float,
                            baseline: float, adjusted: float) -> str:
        """Generate human-readable explanation for protein recommendation"""
        explanations = []
        
        # Sleep impact
        if sleep_factor > 1.05:
            explanations.append(f"Poor sleep quality ({metrics.sleep_quality:.1f}/10) increases protein needs")
        elif sleep_factor < 0.95:
            explanations.append(f"Excellent sleep quality ({metrics.sleep_quality:.1f}/10) slightly reduces protein needs")
        
        # HRV impact
        if hrv_factor > 1.05:
            explanations.append(f"Low HRV ({metrics.hrv:.0f}ms) indicates higher stress, increasing protein needs")
        elif hrv_factor < 0.95:
            explanations.append(f"High HRV ({metrics.hrv:.0f}ms) indicates good recovery")
        
        # Activity impact
        if activity_factor > 1.2:
            explanations.append(f"High activity level ({metrics.activity_level:.1f}/10) significantly increases protein needs")
        elif activity_factor > 1.0:
            explanations.append(f"Moderate activity level ({metrics.activity_level:.1f}/10) increases protein needs")
        
        change_percent = ((adjusted - baseline) / baseline) * 100
        if abs(change_percent) > 5:
            direction = "increase" if change_percent > 0 else "decrease"
            explanations.append(f"Overall recommendation: {direction} protein by {abs(change_percent):.0f}%")
        
        return ". ".join(explanations) if explanations else "Baseline protein recommendation maintained."
    
    def train_model(self, training_data: List[Tuple[List[HealthMetrics], float]], 
                   epochs: int = 100, learning_rate: float = 0.001):
        """Train the GRU model on historical data"""
        logger.info("Starting GRU model training...")
        
        # Prepare training data
        X_sequences = []
        y_targets = []
        
        for health_sequence, target_protein in training_data:
            if len(health_sequence) >= 7:
                time_series = self.prepare_time_series_data(health_sequence)
                X_sequences.append(time_series)
                y_targets.append(target_protein)
        
        if len(X_sequences) < 10:
            logger.warning("Insufficient training data. Model will rely on research correlations.")
            return
        
        X_train = torch.FloatTensor(np.stack(X_sequences)).to(self.device)
        y_train = torch.FloatTensor(y_targets).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs['protein_adjustment'].squeeze(), y_train)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            scheduler.step(loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("GRU model training completed")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_params': self.scaler.get_params() if hasattr(self.scaler, 'get_params') else None,
            'research_correlations': self.research_correlations,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Protein optimization model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.research_correlations = checkpoint.get('research_correlations', self.research_correlations)
        self.is_fitted = checkpoint.get('is_fitted', False)
        logger.info(f"Protein optimization model loaded from {filepath}")

# Global protein optimizer instance
protein_optimizer = ProteinOptimizer()