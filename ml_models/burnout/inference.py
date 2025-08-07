"""
Real-time Inference for Burnout Prediction Model
API integration and visualization for burnout risk assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging
import json
import requests
from datetime import datetime, timedelta
import argparse
import os

from cox_model import BurnoutCoxModel, BurnoutRiskFactors, BurnoutPrediction, SurvivalCurve

logger = logging.getLogger(__name__)

class BurnoutPredictor:
    """Real-time burnout prediction system"""
    
    def __init__(self, model_path: Optional[str] = None, api_url: str = "http://localhost:8000"):
        self.model = BurnoutCoxModel(model_path)
        self.api_url = api_url
        self.prediction_history = []
        
        # Initialize model if not loaded
        if not self.model.is_trained:
            logger.info("Training model with synthetic data...")
            self.model.train_model()
    
    def predict_from_api(self, user_id: str) -> BurnoutPrediction:
        """Get burnout prediction from API"""
        try:
            # Fetch user data from API
            response = requests.get(f"{self.api_url}/burnout/analyze", params={'user_id': user_id})
            
            if response.status_code == 200:
                user_data = response.json()
                
                # Convert to BurnoutRiskFactors
                risk_factors = BurnoutRiskFactors(
                    sleep_quality=user_data.get('sleep_quality', 70),
                    stress_level=user_data.get('stress_level', 50),
                    workload=user_data.get('workload', 60),
                    social_support=user_data.get('social_support', 70),
                    exercise_frequency=user_data.get('exercise_frequency', 3),
                    hrv_score=user_data.get('hrv_score', 50),
                    recovery_time=user_data.get('recovery_time', 6),
                    work_life_balance=user_data.get('work_life_balance', 60),
                    nutrition_quality=user_data.get('nutrition_quality', 70),
                    mental_fatigue=user_data.get('mental_fatigue', 50)
                )
                
                # Make prediction
                prediction = self.model.predict_burnout(risk_factors)
                
                # Store prediction history
                self.prediction_history.append({
                    'timestamp': datetime.now(),
                    'user_id': user_id,
                    'prediction': prediction,
                    'risk_factors': risk_factors
                })
                
                return prediction
            else:
                raise Exception(f"API request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching prediction from API: {e}")
            # Return mock prediction
            return self._create_mock_prediction()
    
    def predict_from_data(self, risk_factors: Dict[str, float]) -> BurnoutPrediction:
        """Predict burnout from provided risk factors"""
        try:
            # Convert to BurnoutRiskFactors
            burnout_risk = BurnoutRiskFactors(
                sleep_quality=risk_factors.get('sleep_quality', 70),
                stress_level=risk_factors.get('stress_level', 50),
                workload=risk_factors.get('workload', 60),
                social_support=risk_factors.get('social_support', 70),
                exercise_frequency=risk_factors.get('exercise_frequency', 3),
                hrv_score=risk_factors.get('hrv_score', 50),
                recovery_time=risk_factors.get('recovery_time', 6),
                work_life_balance=risk_factors.get('work_life_balance', 60),
                nutrition_quality=risk_factors.get('nutrition_quality', 70),
                mental_fatigue=risk_factors.get('mental_fatigue', 50)
            )
            
            # Make prediction
            prediction = self.model.predict_burnout(burnout_risk)
            
            # Store prediction history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'risk_factors': burnout_risk
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._create_mock_prediction()
    
    def _create_mock_prediction(self) -> BurnoutPrediction:
        """Create mock prediction for testing"""
        return BurnoutPrediction(
            risk_score=65.0,
            risk_level="high",
            time_to_burnout=45.0,
            survival_probability=0.35,
            risk_factors=["High stress levels", "Poor sleep quality", "Excessive workload"],
            recommendations=[
                "Implement stress management techniques",
                "Improve sleep hygiene",
                "Consider workload reduction"
            ],
            confidence_interval=(0.25, 0.45)
        )
    
    def generate_survival_curve(self, risk_factors: Dict[str, float]) -> SurvivalCurve:
        """Generate survival curve for visualization"""
        burnout_risk = BurnoutRiskFactors(
            sleep_quality=risk_factors.get('sleep_quality', 70),
            stress_level=risk_factors.get('stress_level', 50),
            workload=risk_factors.get('workload', 60),
            social_support=risk_factors.get('social_support', 70),
            exercise_frequency=risk_factors.get('exercise_frequency', 3),
            hrv_score=risk_factors.get('hrv_score', 50),
            recovery_time=risk_factors.get('recovery_time', 6),
            work_life_balance=risk_factors.get('work_life_balance', 60),
            nutrition_quality=risk_factors.get('nutrition_quality', 70),
            mental_fatigue=risk_factors.get('mental_fatigue', 50)
        )
        
        return self.model.generate_survival_curve(burnout_risk)
    
    def create_dashboard(self, prediction: BurnoutPrediction, 
                        risk_factors: Dict[str, float], 
                        save_path: Optional[str] = None):
        """Create comprehensive dashboard visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # Set up the grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Risk Score Gauge
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_risk_gauge(ax1, prediction.risk_score, prediction.risk_level)
        
        # 2. Survival Curve
        ax2 = fig.add_subplot(gs[0, 1:])
        survival_curve = self.generate_survival_curve(risk_factors)
        self._plot_survival_curve(ax2, survival_curve)
        
        # 3. Risk Factors Radar
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        self._plot_risk_radar(ax3, risk_factors)
        
        # 4. Time to Burnout
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_time_to_burnout(ax4, prediction.time_to_burnout)
        
        # 5. Risk Factors Bar Chart
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_risk_factors_bar(ax5, risk_factors)
        
        # 6. Recommendations
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_recommendations(ax6, prediction.recommendations)
        
        plt.suptitle('Burnout Risk Assessment Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        else:
            plt.show()
    
    def _plot_risk_gauge(self, ax, risk_score: float, risk_level: str):
        """Plot risk score gauge"""
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        radius = 1
        
        # Background arc
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-', linewidth=3)
        
        # Risk level colors
        colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
        color = colors.get(risk_level, 'gray')
        
        # Risk indicator
        risk_angle = (risk_score / 100) * np.pi
        ax.plot([0, radius * np.cos(risk_angle)], [0, radius * np.sin(risk_angle)], 
                color=color, linewidth=4)
        
        # Add text
        ax.text(0, -0.3, f'{risk_score:.1f}', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.6, risk_level.upper(), ha='center', va='center', fontsize=12, color=color)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Risk Score', fontweight='bold')
    
    def _plot_survival_curve(self, ax, survival_curve: SurvivalCurve):
        """Plot survival curve"""
        ax.plot(survival_curve.time_points, survival_curve.survival_probabilities, 
                'b-', linewidth=2, label='Survival Probability')
        ax.fill_between(survival_curve.time_points, survival_curve.survival_probabilities, 
                       alpha=0.3, color='blue')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Burnout Survival Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_risk_radar(self, ax, risk_factors: Dict[str, float]):
        """Plot risk factors radar chart"""
        categories = ['Sleep', 'Stress', 'Workload', 'Support', 'Exercise', 'HRV']
        values = [
            risk_factors.get('sleep_quality', 70),
            risk_factors.get('stress_level', 50),
            risk_factors.get('workload', 60),
            risk_factors.get('social_support', 70),
            risk_factors.get('exercise_frequency', 3) * 14.3,  # Scale to 0-100
            risk_factors.get('hrv_score', 50)
        ]
        
        # Normalize values
        values = [min(100, max(0, v)) for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Risk Factors Profile')
    
    def _plot_time_to_burnout(self, ax, time_to_burnout: float):
        """Plot time to burnout visualization"""
        # Create countdown-style visualization
        days_remaining = int(time_to_burnout)
        
        if days_remaining > 365:
            months = days_remaining // 30
            ax.text(0.5, 0.6, f'{months} months', ha='center', va='center', 
                   fontsize=24, fontweight='bold', color='green')
            ax.text(0.5, 0.4, 'until burnout', ha='center', va='center', 
                   fontsize=12, color='gray')
        else:
            ax.text(0.5, 0.6, f'{days_remaining} days', ha='center', va='center', 
                   fontsize=24, fontweight='bold', color='red')
            ax.text(0.5, 0.4, 'until burnout', ha='center', va='center', 
                   fontsize=12, color='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Time to Burnout', fontweight='bold')
    
    def _plot_risk_factors_bar(self, ax, risk_factors: Dict[str, float]):
        """Plot risk factors bar chart"""
        factors = ['Sleep', 'Stress', 'Workload', 'Support', 'Exercise', 'HRV']
        values = [
            risk_factors.get('sleep_quality', 70),
            risk_factors.get('stress_level', 50),
            risk_factors.get('workload', 60),
            risk_factors.get('social_support', 70),
            risk_factors.get('exercise_frequency', 3) * 14.3,
            risk_factors.get('hrv_score', 50)
        ]
        
        colors = ['green' if v < 50 else 'orange' if v < 70 else 'red' for v in values]
        
        bars = ax.barh(factors, values, color=colors, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_title('Risk Factors Assessment')
        ax.set_xlabel('Score')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.0f}', va='center', fontsize=10)
    
    def _plot_recommendations(self, ax, recommendations: List[str]):
        """Plot recommendations"""
        ax.axis('off')
        ax.set_title('Recommendations', fontweight='bold', fontsize=14)
        
        y_pos = 0.9
        for i, rec in enumerate(recommendations):
            ax.text(0.05, y_pos, f'{i+1}. {rec}', fontsize=11, 
                   transform=ax.transAxes, verticalalignment='top')
            y_pos -= 0.15
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analyze prediction trends over time"""
        if len(self.prediction_history) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Filter recent predictions
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            p for p in self.prediction_history 
            if p['timestamp'] > cutoff_date
        ]
        
        if len(recent_predictions) < 2:
            return {"error": "Insufficient recent data"}
        
        # Calculate trends
        risk_scores = [p['prediction'].risk_score for p in recent_predictions]
        timestamps = [p['timestamp'] for p in recent_predictions]
        
        # Linear trend
        x = np.array([(t - timestamps[0]).days for t in timestamps])
        y = np.array(risk_scores)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        else:
            slope = 0
            trend_direction = "stable"
        
        return {
            "trend_direction": trend_direction,
            "slope": slope,
            "average_risk_score": np.mean(risk_scores),
            "risk_score_change": risk_scores[-1] - risk_scores[0],
            "prediction_count": len(recent_predictions),
            "time_period_days": days
        }
    
    def export_predictions(self, filepath: str, format: str = 'json'):
        """Export prediction history"""
        if format == 'json':
            # Convert to serializable format
            export_data = []
            for pred in self.prediction_history:
                export_data.append({
                    'timestamp': pred['timestamp'].isoformat(),
                    'user_id': pred.get('user_id', 'unknown'),
                    'risk_score': pred['prediction'].risk_score,
                    'risk_level': pred['prediction'].risk_level,
                    'time_to_burnout': pred['prediction'].time_to_burnout,
                    'survival_probability': pred['prediction'].survival_probability,
                    'risk_factors': pred['prediction'].risk_factors,
                    'recommendations': pred['prediction'].recommendations
                })
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'csv':
            # Convert to DataFrame and save
            df_data = []
            for pred in self.prediction_history:
                df_data.append({
                    'timestamp': pred['timestamp'],
                    'user_id': pred.get('user_id', 'unknown'),
                    'risk_score': pred['prediction'].risk_score,
                    'risk_level': pred['prediction'].risk_level,
                    'time_to_burnout': pred['prediction'].time_to_burnout,
                    'survival_probability': pred['prediction'].survival_probability
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Predictions exported to {filepath}")

def main():
    """Main function for burnout prediction"""
    parser = argparse.ArgumentParser(description='Burnout Prediction Inference')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', 
                       help='API URL for user data')
    parser.add_argument('--user-id', type=str, help='User ID for API prediction')
    parser.add_argument('--risk-factors', type=str, help='JSON string of risk factors')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--export', type=str, help='Export predictions to file')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = BurnoutPredictor(args.model_path, args.api_url)
    
    # Make prediction
    if args.user_id:
        prediction = predictor.predict_from_api(args.user_id)
        print(f"API Prediction for user {args.user_id}:")
    elif args.risk_factors:
        risk_factors = json.loads(args.risk_factors)
        prediction = predictor.predict_from_data(risk_factors)
        print("Prediction from provided risk factors:")
    else:
        # Use default risk factors
        risk_factors = {
            'sleep_quality': 65,
            'stress_level': 75,
            'workload': 80,
            'social_support': 60,
            'exercise_frequency': 2,
            'hrv_score': 45,
            'recovery_time': 5,
            'work_life_balance': 55,
            'nutrition_quality': 70,
            'mental_fatigue': 70
        }
        prediction = predictor.predict_from_data(risk_factors)
        print("Prediction with default risk factors:")
    
    # Print results
    print(f"Risk Score: {prediction.risk_score:.1f}")
    print(f"Risk Level: {prediction.risk_level}")
    print(f"Time to Burnout: {prediction.time_to_burnout:.1f} days")
    print(f"Survival Probability: {prediction.survival_probability:.3f}")
    print(f"Risk Factors: {', '.join(prediction.risk_factors)}")
    print("Recommendations:")
    for i, rec in enumerate(prediction.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Generate dashboard if requested
    if args.dashboard:
        output_path = args.output or 'burnout_dashboard.png'
        predictor.create_dashboard(prediction, risk_factors, output_path)
    
    # Export predictions if requested
    if args.export:
        predictor.export_predictions(args.export)
    
    # Show trend analysis
    trend_analysis = predictor.get_trend_analysis()
    if 'error' not in trend_analysis:
        print(f"\nTrend Analysis (30 days):")
        print(f"Direction: {trend_analysis['trend_direction']}")
        print(f"Average Risk Score: {trend_analysis['average_risk_score']:.1f}")
        print(f"Risk Score Change: {trend_analysis['risk_score_change']:.1f}")

if __name__ == "__main__":
    main() 