"""
SHAP Explainability System for Nutrition Recommendations
Provides interpretable explanations for protein optimization decisions
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import torch
import json
from datetime import datetime
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

@dataclass
class ShapExplanation:
    """SHAP explanation for protein recommendation"""
    feature_importance: Dict[str, float]
    base_value: float
    predicted_value: float
    explanation_text: str
    visualization_data: Dict[str, Any]
    confidence_score: float

class ProteinShapExplainer:
    """SHAP-based explainer for protein optimization decisions"""
    
    def __init__(self, protein_optimizer):
        self.protein_optimizer = protein_optimizer
        self.feature_names = [
            'sleep_duration', 'sleep_quality', 'hrv', 
            'activity_level', 'stress_level', 'recovery_score'
        ]
        self.feature_descriptions = {
            'sleep_duration': 'Hours of sleep per night',
            'sleep_quality': 'Sleep quality score (0-10)',
            'hrv': 'Heart Rate Variability (ms)',
            'activity_level': 'Daily activity score (0-10)',
            'stress_level': 'Perceived stress level (1-10)',
            'recovery_score': 'Overall recovery score (0-10)'
        }
        
        # Initialize SHAP explainer
        self.explainer = None
        self.background_data = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup SHAP explainer with background data"""
        try:
            # Generate synthetic background data for SHAP
            self.background_data = self._generate_background_data()
            
            # Create wrapper function for SHAP
            def model_wrapper(X):
                """Wrapper function for SHAP to call the protein optimizer"""
                results = []
                
                for sample in X:
                    # Convert to health metrics format
                    health_metrics = self._array_to_health_metrics(sample)
                    
                    # Get protein recommendation
                    recommendation = self.protein_optimizer.optimize_protein_intake(
                        user_id="shap_user", health_metrics=[health_metrics]
                    )
                    
                    # Return adjusted protein as the target
                    results.append(recommendation.adjusted_protein)
                
                return np.array(results)
            
            # Initialize SHAP explainer
            self.explainer = shap.KernelExplainer(model_wrapper, self.background_data)
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"SHAP explainer setup failed: {e}")
            self.explainer = None
    
    def _generate_background_data(self, n_samples: int = 100) -> np.ndarray:
        """Generate representative background data for SHAP"""
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic health metrics distributions
        sleep_duration = np.random.normal(7.5, 1.2, n_samples)  # Average 7.5 hours
        sleep_quality = np.random.beta(2, 1, n_samples) * 10     # Skewed towards higher quality
        hrv = np.random.gamma(2, 25, n_samples)                  # Gamma distribution for HRV
        activity_level = np.random.beta(2, 2, n_samples) * 10    # Balanced activity distribution
        stress_level = np.random.exponential(3, n_samples) + 1   # Stress levels 1-10
        recovery_score = np.random.beta(3, 2, n_samples) * 10    # Recovery scores
        
        # Clip to reasonable ranges
        sleep_duration = np.clip(sleep_duration, 4, 12)
        sleep_quality = np.clip(sleep_quality, 0, 10)
        hrv = np.clip(hrv, 20, 100)
        activity_level = np.clip(activity_level, 0, 10)
        stress_level = np.clip(stress_level, 1, 10)
        recovery_score = np.clip(recovery_score, 0, 10)
        
        # Normalize for model input
        background_data = np.column_stack([
            sleep_duration / 12.0,
            sleep_quality / 10.0,
            hrv / 100.0,
            activity_level / 10.0,
            stress_level / 10.0,
            recovery_score / 10.0
        ])
        
        return background_data.astype(np.float32)
    
    def _array_to_health_metrics(self, array: np.ndarray):
        """Convert normalized array back to health metrics object"""
        from ml_models.nutrition.gru_protein_optimizer import HealthMetrics
        
        return HealthMetrics(
            date=datetime.now(),
            sleep_duration=array[0] * 12.0,
            sleep_quality=array[1] * 10.0,
            hrv=array[2] * 100.0,
            activity_level=array[3] * 10.0,
            stress_level=int(array[4] * 10.0),
            protein_intake=0.0,  # Placeholder
            recovery_score=array[5] * 10.0
        )
    
    def explain_recommendation(self, health_metrics, user_id: str = "default") -> ShapExplanation:
        """Generate SHAP explanation for a protein recommendation"""
        try:
            if self.explainer is None:
                return self._fallback_explanation(health_metrics)
            
            # Prepare input data
            input_features = self.protein_optimizer.prepare_time_series_data([health_metrics])
            input_sample = input_features[-1].reshape(1, -1)  # Take last timestep
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(input_sample, nsamples=50)
            
            # Get base prediction
            base_value = self.explainer.expected_value
            predicted_value = self.protein_optimizer.optimize_protein_intake(
                user_id, [health_metrics]
            ).adjusted_protein
            
            # Calculate feature importance
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                # Convert back to original scale for interpretation
                original_value = self._denormalize_feature(input_sample[0][i], feature_name)
                shap_impact = shap_values[0][i]
                
                feature_importance[feature_name] = {
                    'value': original_value,
                    'shap_value': float(shap_impact),
                    'impact': 'positive' if shap_impact > 0 else 'negative' if shap_impact < 0 else 'neutral',
                    'magnitude': abs(float(shap_impact))
                }
            
            # Generate explanation text
            explanation_text = self._generate_explanation_text(feature_importance, predicted_value, base_value)
            
            # Create visualization data
            visualization_data = self._create_visualization_data(feature_importance, shap_values[0])
            
            # Calculate confidence based on SHAP value consistency
            confidence_score = self._calculate_explanation_confidence(shap_values[0])
            
            return ShapExplanation(
                feature_importance=feature_importance,
                base_value=float(base_value),
                predicted_value=float(predicted_value),
                explanation_text=explanation_text,
                visualization_data=visualization_data,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(health_metrics)
    
    def _denormalize_feature(self, normalized_value: float, feature_name: str) -> float:
        """Convert normalized feature back to original scale"""
        denorm_map = {
            'sleep_duration': normalized_value * 12.0,
            'sleep_quality': normalized_value * 10.0,
            'hrv': normalized_value * 100.0,
            'activity_level': normalized_value * 10.0,
            'stress_level': normalized_value * 10.0,
            'recovery_score': normalized_value * 10.0
        }
        return denorm_map.get(feature_name, normalized_value)
    
    def _generate_explanation_text(self, feature_importance: Dict, predicted_value: float, base_value: float) -> str:
        """Generate human-readable explanation text"""
        explanations = []
        
        # Sort features by absolute SHAP value impact
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1]['magnitude'], 
            reverse=True
        )
        
        # Overall prediction
        protein_change = predicted_value - base_value
        if protein_change > 5:
            explanations.append(f"Your protein needs are increased by {protein_change:.1f}g above the baseline.")
        elif protein_change < -5:
            explanations.append(f"Your protein needs are reduced by {abs(protein_change):.1f}g below the baseline.")
        else:
            explanations.append("Your protein needs are close to the baseline recommendation.")
        
        # Top contributing factors
        for feature_name, feature_data in sorted_features[:3]:
            if feature_data['magnitude'] > 0.5:  # Only significant impacts
                value = feature_data['value']
                impact = feature_data['impact']
                description = self.feature_descriptions[feature_name]
                
                if feature_name == 'sleep_quality':
                    if impact == 'positive':
                        explanations.append(f"Your sleep quality ({value:.1f}/10) is poor, increasing protein needs for recovery.")
                    else:
                        explanations.append(f"Your excellent sleep quality ({value:.1f}/10) allows for efficient protein utilization.")
                
                elif feature_name == 'activity_level':
                    if impact == 'positive':
                        explanations.append(f"Your high activity level ({value:.1f}/10) increases protein requirements for muscle repair.")
                    else:
                        explanations.append(f"Your lower activity level ({value:.1f}/10) reduces protein needs.")
                
                elif feature_name == 'hrv':
                    if impact == 'positive':
                        explanations.append(f"Your HRV ({value:.0f}ms) indicates stress, requiring more protein for recovery.")
                    else:
                        explanations.append(f"Your good HRV ({value:.0f}ms) shows optimal recovery status.")
                
                elif feature_name == 'stress_level':
                    if impact == 'positive':
                        explanations.append(f"Your stress level ({value:.1f}/10) increases protein needs for immune support.")
        
        return " ".join(explanations)
    
    def _create_visualization_data(self, feature_importance: Dict, shap_values: np.ndarray) -> Dict:
        """Create data for SHAP visualizations"""
        return {
            'waterfall_data': {
                'features': list(feature_importance.keys()),
                'shap_values': [feature_importance[f]['shap_value'] for f in feature_importance.keys()],
                'feature_values': [feature_importance[f]['value'] for f in feature_importance.keys()],
                'base_value': self.explainer.expected_value if self.explainer else 120.0
            },
            'force_plot_data': {
                'shap_values': shap_values.tolist(),
                'feature_names': self.feature_names,
                'feature_values': [feature_importance[f]['value'] for f in self.feature_names]
            },
            'summary_stats': {
                'total_positive_impact': sum(f['shap_value'] for f in feature_importance.values() if f['shap_value'] > 0),
                'total_negative_impact': sum(f['shap_value'] for f in feature_importance.values() if f['shap_value'] < 0),
                'most_important_feature': max(feature_importance.keys(), key=lambda x: feature_importance[x]['magnitude'])
            }
        }
    
    def _calculate_explanation_confidence(self, shap_values: np.ndarray) -> float:
        """Calculate confidence score for the explanation"""
        # Confidence based on the magnitude and consistency of SHAP values
        total_magnitude = np.sum(np.abs(shap_values))
        
        if total_magnitude == 0:
            return 0.5  # Neutral confidence
        
        # Higher total magnitude indicates clearer feature impacts
        magnitude_score = min(total_magnitude / 20.0, 1.0)  # Normalize to 0-1
        
        # Consistency score: prefer fewer strong features over many weak ones
        consistency_score = 1.0 - (np.std(np.abs(shap_values)) / (np.mean(np.abs(shap_values)) + 1e-8))
        consistency_score = max(0, min(consistency_score, 1.0))
        
        return (magnitude_score + consistency_score) / 2.0
    
    def _fallback_explanation(self, health_metrics) -> ShapExplanation:
        """Fallback explanation when SHAP is not available"""
        # Simple rule-based explanation
        feature_importance = {
            'sleep_quality': {
                'value': health_metrics.sleep_quality,
                'shap_value': (7.0 - health_metrics.sleep_quality) * 2.0,
                'impact': 'positive' if health_metrics.sleep_quality < 7.0 else 'negative',
                'magnitude': abs((7.0 - health_metrics.sleep_quality) * 2.0)
            },
            'activity_level': {
                'value': health_metrics.activity_level,
                'shap_value': (health_metrics.activity_level - 5.0) * 3.0,
                'impact': 'positive' if health_metrics.activity_level > 5.0 else 'negative',
                'magnitude': abs((health_metrics.activity_level - 5.0) * 3.0)
            },
            'hrv': {
                'value': health_metrics.hrv,
                'shap_value': (50.0 - health_metrics.hrv) * 0.3,
                'impact': 'positive' if health_metrics.hrv < 50.0 else 'negative',
                'magnitude': abs((50.0 - health_metrics.hrv) * 0.3)
            }
        }
        
        explanation_text = "Basic explanation based on sleep quality, activity level, and HRV patterns."
        
        return ShapExplanation(
            feature_importance=feature_importance,
            base_value=120.0,
            predicted_value=120.0,
            explanation_text=explanation_text,
            visualization_data={'fallback': True},
            confidence_score=0.6
        )
    
    def generate_summary_report(self, explanations: List[ShapExplanation]) -> Dict:
        """Generate a summary report from multiple explanations"""
        if not explanations:
            return {}
        
        # Aggregate feature importance across explanations
        feature_aggregates = {}
        for feature in self.feature_names:
            impacts = [exp.feature_importance.get(feature, {}).get('shap_value', 0) for exp in explanations]
            feature_aggregates[feature] = {
                'mean_impact': np.mean(impacts),
                'std_impact': np.std(impacts),
                'frequency_positive': sum(1 for x in impacts if x > 0) / len(impacts),
                'frequency_negative': sum(1 for x in impacts if x < 0) / len(impacts)
            }
        
        # Find most consistently important features
        consistency_scores = {}
        for feature, stats in feature_aggregates.items():
            consistency_scores[feature] = abs(stats['mean_impact']) / (stats['std_impact'] + 1e-8)
        
        most_consistent = max(consistency_scores.keys(), key=lambda x: consistency_scores[x])
        
        return {
            'feature_aggregates': feature_aggregates,
            'most_consistent_feature': most_consistent,
            'average_confidence': np.mean([exp.confidence_score for exp in explanations]),
            'prediction_variance': np.std([exp.predicted_value for exp in explanations]),
            'total_explanations': len(explanations)
        }
    
    def create_visualization(self, explanation: ShapExplanation, plot_type: str = "waterfall") -> str:
        """Create visualization for SHAP explanation"""
        try:
            plt.figure(figsize=(10, 6))
            
            if plot_type == "waterfall":
                self._create_waterfall_plot(explanation)
            elif plot_type == "bar":
                self._create_bar_plot(explanation)
            elif plot_type == "force":
                self._create_force_plot(explanation)
            
            # Save to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return ""
    
    def _create_waterfall_plot(self, explanation: ShapExplanation):
        """Create waterfall plot showing feature contributions"""
        feature_data = explanation.feature_importance
        features = list(feature_data.keys())
        values = [feature_data[f]['shap_value'] for f in features]
        
        # Sort by absolute value
        sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
        features, values = zip(*sorted_pairs)
        
        colors = ['red' if v < 0 else 'green' for v in values]
        
        plt.barh(features, values, color=colors, alpha=0.7)
        plt.xlabel('Impact on Protein Recommendation (g)')
        plt.title('Feature Contributions to Protein Recommendation')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(v + (0.1 if v >= 0 else -0.1), i, f'{v:.1f}g', 
                    ha='left' if v >= 0 else 'right', va='center')
    
    def _create_bar_plot(self, explanation: ShapExplanation):
        """Create bar plot of feature importance"""
        feature_data = explanation.feature_importance
        features = list(feature_data.keys())
        values = [abs(feature_data[f]['shap_value']) for f in features]
        
        plt.bar(features, values, alpha=0.7, color='steelblue')
        plt.xlabel('Features')
        plt.ylabel('Absolute Impact (g)')
        plt.title('Feature Importance for Protein Recommendation')
        plt.xticks(rotation=45)
    
    def _create_force_plot(self, explanation: ShapExplanation):
        """Create force plot visualization"""
        # Simplified force plot representation
        feature_data = explanation.feature_importance
        base_value = explanation.base_value
        predicted_value = explanation.predicted_value
        
        cumulative = base_value
        x_pos = 0
        
        plt.axhline(y=base_value, color='gray', linestyle='--', alpha=0.5, label='Base value')
        plt.axhline(y=predicted_value, color='black', linestyle='-', label='Prediction')
        
        for feature, data in feature_data.items():
            impact = data['shap_value']
            if abs(impact) > 0.1:  # Only show significant impacts
                color = 'red' if impact < 0 else 'green'
                plt.arrow(x_pos, cumulative, 0, impact, 
                         head_width=0.3, head_length=0.5, fc=color, ec=color, alpha=0.7)
                plt.text(x_pos + 0.1, cumulative + impact/2, f'{feature}\n{impact:.1f}g', 
                        ha='left', va='center', fontsize=8)
                cumulative += impact
                x_pos += 1
        
        plt.xlabel('Features')
        plt.ylabel('Protein Recommendation (g)')
        plt.title('Force Plot: How Features Push the Prediction')
        plt.legend()

# Global SHAP explainer instance
shap_explainer = None

def initialize_shap_explainer(protein_optimizer):
    """Initialize global SHAP explainer"""
    global shap_explainer
    shap_explainer = ProteinShapExplainer(protein_optimizer)
    return shap_explainer