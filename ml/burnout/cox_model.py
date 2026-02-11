"""
Burnout Prediction - Cox Proportional Hazards Model
Survival analysis for predicting time to burnout and risk assessment
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import joblib
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class BurnoutRiskFactors:
    """Risk factors for burnout prediction"""
    sleep_quality: float  # 0-100 scale
    stress_level: float   # 0-100 scale
    workload: float       # 0-100 scale
    social_support: float # 0-100 scale
    exercise_frequency: float  # days per week
    hrv_score: float      # Heart Rate Variability score
    recovery_time: float  # hours of recovery per day
    work_life_balance: float  # 0-100 scale
    nutrition_quality: float  # 0-100 scale
    mental_fatigue: float     # 0-100 scale

@dataclass
class BurnoutPrediction:
    """Burnout prediction results"""
    risk_score: float  # 0-100 scale
    risk_level: str    # low, medium, high, critical
    time_to_burnout: float  # days until predicted burnout
    survival_probability: float  # probability of not burning out
    risk_factors: List[str]  # contributing risk factors
    recommendations: List[str]  # preventive recommendations
    confidence_interval: Tuple[float, float]  # confidence bounds

@dataclass
class SurvivalCurve:
    """Survival curve data"""
    time_points: np.ndarray
    survival_probabilities: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    risk_groups: Optional[Dict[str, np.ndarray]] = None

class BurnoutCoxModel:
    """Cox Proportional Hazards model for burnout prediction"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.cph_model = CoxPHFitter()
        self.km_model = KaplanMeierFitter()
        self.scaler = StandardScaler()
        self.feature_names = [
            'sleep_quality', 'stress_level', 'workload', 'social_support',
            'exercise_frequency', 'hrv_score', 'recovery_time', 
            'work_life_balance', 'nutrition_quality', 'mental_fatigue'
        ]
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data for burnout prediction"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Generate realistic risk factors
            sleep_quality = np.random.normal(70, 15)
            stress_level = np.random.normal(60, 20)
            workload = np.random.normal(65, 18)
            social_support = np.random.normal(75, 12)
            exercise_frequency = np.random.normal(3.5, 1.5)
            hrv_score = np.random.normal(50, 10)
            recovery_time = np.random.normal(6, 2)
            work_life_balance = np.random.normal(60, 15)
            nutrition_quality = np.random.normal(70, 12)
            mental_fatigue = np.random.normal(55, 18)
            
            # Calculate risk score (higher = more likely to burnout)
            risk_score = (
                (100 - sleep_quality) * 0.15 +
                stress_level * 0.20 +
                workload * 0.18 +
                (100 - social_support) * 0.12 +
                (7 - exercise_frequency) * 0.08 +
                (100 - hrv_score) * 0.10 +
                (24 - recovery_time) * 0.05 +
                (100 - work_life_balance) * 0.12
            )
            
            # Determine time to burnout based on risk
            if risk_score < 30:
                time_to_burnout = np.random.exponential(365)  # Low risk
            elif risk_score < 50:
                time_to_burnout = np.random.exponential(180)  # Medium risk
            elif risk_score < 70:
                time_to_burnout = np.random.exponential(90)   # High risk
            else:
                time_to_burnout = np.random.exponential(30)   # Critical risk
            
            # Censor some data (not everyone burns out)
            if np.random.random() < 0.3:  # 30% don't burn out
                time_to_burnout = np.random.uniform(365, 730)  # 1-2 years
                burnout_event = 0
            else:
                burnout_event = 1
            
            # Ensure time is positive
            time_to_burnout = max(1, time_to_burnout)
            
            data.append({
                'sleep_quality': np.clip(sleep_quality, 0, 100),
                'stress_level': np.clip(stress_level, 0, 100),
                'workload': np.clip(workload, 0, 100),
                'social_support': np.clip(social_support, 0, 100),
                'exercise_frequency': np.clip(exercise_frequency, 0, 7),
                'hrv_score': np.clip(hrv_score, 0, 100),
                'recovery_time': np.clip(recovery_time, 0, 24),
                'work_life_balance': np.clip(work_life_balance, 0, 100),
                'nutrition_quality': np.clip(nutrition_quality, 0, 100),
                'mental_fatigue': np.clip(mental_fatigue, 0, 100),
                'training_days': time_to_burnout,
                'burnout_event': burnout_event,
                'risk_score': risk_score
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, data: Optional[pd.DataFrame] = None, 
                   validation_split: float = 0.2) -> Dict[str, float]:
        """Train the Cox Proportional Hazards model"""
        if data is None:
            data = self.prepare_training_data()
        
        logger.info(f"Training Cox model with {len(data)} samples")
        
        # Split data
        train_data, val_data = train_test_split(
            data, test_size=validation_split, random_state=42
        )
        
        # Prepare features
        X_train = train_data[self.feature_names]
        X_val = val_data[self.feature_names]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create training dataframe
        train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        train_df['training_days'] = train_data['training_days']
        train_df['burnout_event'] = train_data['burnout_event']
        
        # Fit Cox model
        self.cph_model.fit(
            train_df, 
            duration_col='training_days', 
            event_col='burnout_event'
        )
        
        # Fit Kaplan-Meier for comparison
        self.km_model.fit(
            train_data['training_days'], 
            train_data['burnout_event']
        )
        
        # Evaluate model
        metrics = self._evaluate_model(X_val_scaled, val_data)
        
        self.is_trained = True
        logger.info(f"Model training completed. Concordance: {metrics['concordance']:.3f}")
        
        return metrics
    
    def _evaluate_model(self, X_val: np.ndarray, val_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        # Predict survival functions
        val_df = pd.DataFrame(X_val, columns=self.feature_names)
        val_df['training_days'] = val_data['training_days']
        val_df['burnout_event'] = val_data['burnout_event']
        
        # Calculate concordance index
        predicted_survival = self.cph_model.predict_partial_hazard(val_df)
        concordance = concordance_index(
            val_data['training_days'], 
            -predicted_survival, 
            val_data['burnout_event']
        )
        
        # Calculate other metrics
        log_likelihood = self.cph_model.log_likelihood_
        aic = self.cph_model.AIC_
        
        return {
            'concordance': concordance,
            'log_likelihood': log_likelihood,
            'aic': aic
        }
    
    def predict_burnout(self, user_data: BurnoutRiskFactors) -> BurnoutPrediction:
        """Predict burnout risk for a user"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to feature vector
        features = np.array([
            user_data.sleep_quality,
            user_data.stress_level,
            user_data.workload,
            user_data.social_support,
            user_data.exercise_frequency,
            user_data.hrv_score,
            user_data.recovery_time,
            user_data.work_life_balance,
            user_data.nutrition_quality,
            user_data.mental_fatigue
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame(features_scaled, columns=self.feature_names)
        
        # Predict survival function
        survival_function = self.cph_model.predict_survival_function(pred_df)
        
        # Calculate risk metrics
        risk_score = self._calculate_risk_score(user_data)
        risk_level = self._determine_risk_level(risk_score)
        time_to_burnout = self._predict_time_to_burnout(survival_function)
        survival_probability = self._calculate_survival_probability(survival_function)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(user_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(user_data, risk_level)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(survival_function)
        
        return BurnoutPrediction(
            risk_score=risk_score,
            risk_level=risk_level,
            time_to_burnout=time_to_burnout,
            survival_probability=survival_probability,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence_interval=confidence_interval
        )
    
    def _calculate_risk_score(self, user_data: BurnoutRiskFactors) -> float:
        """Calculate overall risk score"""
        risk_score = (
            (100 - user_data.sleep_quality) * 0.15 +
            user_data.stress_level * 0.20 +
            user_data.workload * 0.18 +
            (100 - user_data.social_support) * 0.12 +
            (7 - user_data.exercise_frequency) * 0.08 +
            (100 - user_data.hrv_score) * 0.10 +
            (24 - user_data.recovery_time) * 0.05 +
            (100 - user_data.work_life_balance) * 0.12
        )
        
        return np.clip(risk_score, 0, 100)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < 25:
            return "low"
        elif risk_score < 50:
            return "medium"
        elif risk_score < 75:
            return "high"
        else:
            return "critical"
    
    def _predict_time_to_burnout(self, survival_function: pd.DataFrame) -> float:
        """Predict time to burnout from survival function"""
        # Find time when survival probability drops below 0.5
        survival_curve = survival_function.iloc[0]
        time_points = survival_function.index
        
        # Find median survival time
        median_survival_idx = np.argmax(survival_curve < 0.5)
        
        if median_survival_idx == 0:
            return time_points[-1]  # No burnout predicted
        else:
            return time_points[median_survival_idx]
    
    def _calculate_survival_probability(self, survival_function: pd.DataFrame) -> float:
        """Calculate survival probability at 6 months"""
        survival_curve = survival_function.iloc[0]
        time_points = survival_function.index
        
        # Find survival probability at 180 days (6 months)
        six_month_idx = np.argmin(np.abs(time_points - 180))
        return survival_curve.iloc[six_month_idx]
    
    def _identify_risk_factors(self, user_data: BurnoutRiskFactors) -> List[str]:
        """Identify contributing risk factors"""
        risk_factors = []
        
        if user_data.sleep_quality < 60:
            risk_factors.append("Poor sleep quality")
        if user_data.stress_level > 70:
            risk_factors.append("High stress levels")
        if user_data.workload > 75:
            risk_factors.append("Excessive workload")
        if user_data.social_support < 50:
            risk_factors.append("Low social support")
        if user_data.exercise_frequency < 2:
            risk_factors.append("Insufficient exercise")
        if user_data.hrv_score < 40:
            risk_factors.append("Poor heart rate variability")
        if user_data.recovery_time < 4:
            risk_factors.append("Inadequate recovery time")
        if user_data.work_life_balance < 50:
            risk_factors.append("Poor work-life balance")
        if user_data.nutrition_quality < 60:
            risk_factors.append("Poor nutrition")
        if user_data.mental_fatigue > 70:
            risk_factors.append("High mental fatigue")
        
        return risk_factors
    
    def _generate_recommendations(self, user_data: BurnoutRiskFactors, 
                                risk_level: str) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Sleep recommendations
        if user_data.sleep_quality < 70:
            recommendations.append(
                "Improve sleep quality: Aim for 7-9 hours, maintain consistent schedule"
            )
        
        # Stress management
        if user_data.stress_level > 60:
            recommendations.append(
                "Implement stress management techniques: meditation, deep breathing, or therapy"
            )
        
        # Workload management
        if user_data.workload > 70:
            recommendations.append(
                "Consider workload reduction or better task prioritization"
            )
        
        # Social support
        if user_data.social_support < 60:
            recommendations.append(
                "Strengthen social connections and seek support from friends/family"
            )
        
        # Exercise
        if user_data.exercise_frequency < 3:
            recommendations.append(
                "Increase exercise frequency to 3-5 days per week"
            )
        
        # Recovery
        if user_data.recovery_time < 6:
            recommendations.append(
                "Increase recovery time and incorporate rest days"
            )
        
        # Work-life balance
        if user_data.work_life_balance < 60:
            recommendations.append(
                "Improve work-life balance: set boundaries and take regular breaks"
            )
        
        # Risk-level specific recommendations
        if risk_level == "critical":
            recommendations.append(
                "Consider professional help: consult with a mental health professional"
            )
        elif risk_level == "high":
            recommendations.append(
                "Take immediate action: prioritize self-care and stress reduction"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence_interval(self, survival_function: pd.DataFrame) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simplified confidence interval calculation
        survival_curve = survival_function.iloc[0]
        mean_survival = survival_curve.mean()
        std_survival = survival_curve.std()
        
        lower_bound = max(0, mean_survival - 1.96 * std_survival)
        upper_bound = min(1, mean_survival + 1.96 * std_survival)
        
        return (lower_bound, upper_bound)
    
    def generate_survival_curve(self, user_data: BurnoutRiskFactors, 
                              time_horizon: int = 365) -> SurvivalCurve:
        """Generate survival curve for visualization"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating survival curves")
        
        # Prepare features
        features = np.array([
            user_data.sleep_quality,
            user_data.stress_level,
            user_data.workload,
            user_data.social_support,
            user_data.exercise_frequency,
            user_data.hrv_score,
            user_data.recovery_time,
            user_data.work_life_balance,
            user_data.nutrition_quality,
            user_data.mental_fatigue
        ]).reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        pred_df = pd.DataFrame(features_scaled, columns=self.feature_names)
        
        # Generate survival function
        survival_function = self.cph_model.predict_survival_function(pred_df)
        
        # Extract time points and survival probabilities
        time_points = survival_function.index.values
        survival_probabilities = survival_function.iloc[0].values
        
        # Limit to time horizon
        mask = time_points <= time_horizon
        time_points = time_points[mask]
        survival_probabilities = survival_probabilities[mask]
        
        return SurvivalCurve(
            time_points=time_points,
            survival_probabilities=survival_probabilities
        )
    
    def plot_survival_curves(self, user_data: BurnoutRiskFactors, 
                           save_path: Optional[str] = None):
        """Plot survival curves for visualization"""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting")
        
        # Generate survival curve
        survival_curve = self.generate_survival_curve(user_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Main survival curve
        plt.subplot(2, 2, 1)
        plt.plot(survival_curve.time_points, survival_curve.survival_probabilities, 
                'b-', linewidth=2, label='Survival Probability')
        plt.xlabel('Days')
        plt.ylabel('Survival Probability')
        plt.title('Burnout Survival Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Risk factors radar chart
        plt.subplot(2, 2, 2)
        self._plot_risk_radar(user_data)
        
        # Risk level distribution
        plt.subplot(2, 2, 3)
        self._plot_risk_distribution()
        
        # Model performance metrics
        plt.subplot(2, 2, 4)
        self._plot_model_metrics()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_risk_radar(self, user_data: BurnoutRiskFactors):
        """Plot risk factors radar chart"""
        categories = ['Sleep', 'Stress', 'Workload', 'Support', 'Exercise', 'HRV']
        values = [
            user_data.sleep_quality,
            user_data.stress_level,
            user_data.workload,
            user_data.social_support,
            user_data.exercise_frequency * 14.3,  # Scale to 0-100
            user_data.hrv_score
        ]
        
        # Normalize values to 0-100 scale
        values = [min(100, max(0, v)) for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 2, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Risk Factors Profile')
    
    def _plot_risk_distribution(self):
        """Plot risk level distribution"""
        risk_levels = ['Low', 'Medium', 'High', 'Critical']
        # Mock distribution data
        counts = [40, 35, 20, 5]
        
        plt.bar(risk_levels, counts, color=['green', 'yellow', 'orange', 'red'])
        plt.title('Risk Level Distribution')
        plt.ylabel('Percentage of Users')
        plt.xticks(rotation=45)
    
    def _plot_model_metrics(self):
        """Plot model performance metrics"""
        metrics = ['Concordance', 'Log-Likelihood', 'AIC']
        # Mock metric values
        values = [0.85, -1200, 2400]
        
        plt.bar(metrics, values, color=['blue', 'green', 'red'])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'cph_model': self.cph_model,
            'km_model': self.km_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        model_data = joblib.load(model_path)
        
        self.cph_model = model_data['cph_model']
        self.km_model = model_data['km_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {model_path}")

def predict_burnout(user_data: BurnoutRiskFactors, 
                   model_path: Optional[str] = None) -> BurnoutPrediction:
    """Convenience function for burnout prediction"""
    model = BurnoutCoxModel(model_path)
    
    if not model.is_trained:
        # Train model if not already trained
        model.train_model()
    
    return model.predict_burnout(user_data)

def main():
    """Main function for training and testing the model"""
    # Create model
    model = BurnoutCoxModel()
    
    # Train model
    metrics = model.train_model()
    print(f"Training completed. Metrics: {metrics}")
    
    # Test prediction
    test_user = BurnoutRiskFactors(
        sleep_quality=65,
        stress_level=75,
        workload=80,
        social_support=60,
        exercise_frequency=2,
        hrv_score=45,
        recovery_time=5,
        work_life_balance=55,
        nutrition_quality=70,
        mental_fatigue=70
    )
    
    prediction = model.predict_burnout(test_user)
    print(f"Prediction: {prediction}")
    
    # Generate survival curve
    survival_curve = model.generate_survival_curve(test_user)
    print(f"Survival curve generated with {len(survival_curve.time_points)} time points")
    
    # Plot results
    model.plot_survival_curves(test_user, 'burnout_prediction.png')
    
    # Save model
    model.save_model('burnout_model.pkl')

if __name__ == "__main__":
    main() 