"""
Burnout Prediction Module
Burnout prediction with survival curves using Cox Proportional Hazards model
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import io
import base64

logger = logging.getLogger(__name__)

@dataclass
class BurnoutRiskFactors:
    """Risk factors for burnout prediction"""
    workout_frequency: int  # workouts per week
    sleep_hours: float  # average hours per night
    stress_level: int  # 1-10 scale
    recovery_time: int  # days between intense workouts
    performance_trend: str  # improving, stable, declining
    age: int
    gender: str
    experience_years: int
    training_intensity: str  # low, moderate, high

@dataclass
class BurnoutAnalysis:
    """Results of burnout risk analysis"""
    risk_score: float  # 0-100
    risk_level: str  # low, medium, high, critical
    time_to_burnout: Optional[float]  # days, if predicted
    survival_probability: float  # probability of not burning out
    risk_factors: List[str]
    recommendations: List[str]
    survival_curve_data: Dict[str, List[float]]

class BurnoutPredictor:
    """Main class for burnout prediction and survival analysis"""
    
    def __init__(self):
        self.cph_model = None
        self.km_model = None
        self.user_data = {}  # In production, use database
        self.risk_thresholds = {
            'low': 25,
            'medium': 50,
            'high': 75,
            'critical': 90
        }
        
        # Load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained Cox PH and Kaplan-Meier models"""
        try:
            # In production, load from saved models
            # self.cph_model = CoxPHFitter()
            # self.cph_model.load_model('models/burnout_cph_model.pkl')
            
            # For demo, create mock model
            self._create_mock_models()
            logger.info("Burnout prediction models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            self._create_mock_models()
    
    def _create_mock_models(self):
        """Create mock models for demonstration"""
        # Generate mock training data
        np.random.seed(42)
        n_samples = 1000
        
        # Mock features
        workout_freq = np.random.randint(1, 8, n_samples)
        sleep_hours = np.random.uniform(5, 9, n_samples)
        stress_level = np.random.randint(1, 11, n_samples)
        recovery_time = np.random.randint(1, 8, n_samples)
        age = np.random.randint(18, 65, n_samples)
        experience_years = np.random.randint(0, 20, n_samples)
        
        # Mock survival times (days until burnout)
        # Higher risk factors should lead to shorter survival times
        base_survival = 365  # 1 year baseline
        risk_factor = (
            (8 - workout_freq) * 30 +  # More workouts = higher risk
            (7 - sleep_hours) * 20 +   # Less sleep = higher risk
            stress_level * 15 +        # Higher stress = higher risk
            (7 - recovery_time) * 25 + # Less recovery = higher risk
            (65 - age) * 0.5 +         # Younger = slightly higher risk
            (20 - experience_years) * 2 # Less experience = higher risk
        )
        
        survival_times = np.maximum(30, base_survival - risk_factor + np.random.normal(0, 30, n_samples))
        
        # Mock event indicator (1 = burnout occurred, 0 = censored)
        event_occurred = np.random.binomial(1, 0.3, n_samples)  # 30% burnout rate
        
        # Create DataFrame
        self.training_data = pd.DataFrame({
            'workout_frequency': workout_freq,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'recovery_time': recovery_time,
            'age': age,
            'experience_years': experience_years,
            'duration': survival_times,
            'event': event_occurred
        })
        
        # Fit Cox PH model
        self.cph_model = CoxPHFitter()
        self.cph_model.fit(
            self.training_data, 
            duration_col='duration', 
            event_col='event'
        )
        
        # Fit Kaplan-Meier model
        self.km_model = KaplanMeierFitter()
        self.km_model.fit(
            self.training_data['duration'], 
            self.training_data['event']
        )
    
    async def analyze_risk(self, user_id: str, workout_frequency: int, sleep_hours: float,
                          stress_level: int, recovery_time: int, performance_trend: str) -> BurnoutAnalysis:
        """Analyze burnout risk based on user metrics"""
        try:
            # Get user profile
            user_profile = self._get_user_profile(user_id)
            
            # Create risk factors object
            risk_factors = BurnoutRiskFactors(
                workout_frequency=workout_frequency,
                sleep_hours=sleep_hours,
                stress_level=stress_level,
                recovery_time=recovery_time,
                performance_trend=performance_trend,
                age=user_profile.get('age', 30),
                gender=user_profile.get('gender', 'not_specified'),
                experience_years=user_profile.get('experience_years', 5),
                training_intensity=user_profile.get('training_intensity', 'moderate')
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(risk_factors)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Predict time to burnout
            time_to_burnout = self._predict_time_to_burnout(risk_factors)
            
            # Calculate survival probability
            survival_probability = self._calculate_survival_probability(risk_factors)
            
            # Identify specific risk factors
            identified_risk_factors = self._identify_risk_factors(risk_factors)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_factors, risk_level)
            
            # Generate survival curve data
            survival_curve_data = self._generate_survival_curve_data(risk_factors)
            
            # Store user data
            self._store_user_data(user_id, risk_factors, risk_score, risk_level)
            
            return BurnoutAnalysis(
                risk_score=risk_score,
                risk_level=risk_level,
                time_to_burnout=time_to_burnout,
                survival_probability=survival_probability,
                risk_factors=identified_risk_factors,
                recommendations=recommendations,
                survival_curve_data=survival_curve_data
            )
            
        except Exception as e:
            logger.error(f"Risk analysis error: {str(e)}")
            raise
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """Get user's profile data"""
        if user_id not in self.user_data:
            # Default profile
            self.user_data[user_id] = {
                'age': 30,
                'gender': 'not_specified',
                'experience_years': 5,
                'training_intensity': 'moderate',
                'risk_history': [],
                'last_analysis': None
            }
        
        return self.user_data[user_id]
    
    def _calculate_risk_score(self, risk_factors: BurnoutRiskFactors) -> float:
        """Calculate burnout risk score (0-100)"""
        score = 0.0
        
        # Workout frequency (0-25 points)
        if risk_factors.workout_frequency >= 6:
            score += 25
        elif risk_factors.workout_frequency >= 4:
            score += 15
        elif risk_factors.workout_frequency >= 2:
            score += 5
        
        # Sleep hours (0-20 points)
        if risk_factors.sleep_hours < 6:
            score += 20
        elif risk_factors.sleep_hours < 7:
            score += 15
        elif risk_factors.sleep_hours < 8:
            score += 10
        
        # Stress level (0-25 points)
        score += risk_factors.stress_level * 2.5
        
        # Recovery time (0-15 points)
        if risk_factors.recovery_time <= 1:
            score += 15
        elif risk_factors.recovery_time <= 2:
            score += 10
        elif risk_factors.recovery_time <= 3:
            score += 5
        
        # Performance trend (0-15 points)
        if risk_factors.performance_trend == 'declining':
            score += 15
        elif risk_factors.performance_trend == 'stable':
            score += 8
        elif risk_factors.performance_trend == 'improving':
            score += 2
        
        return min(100.0, score)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < self.risk_thresholds['low']:
            return 'low'
        elif risk_score < self.risk_thresholds['medium']:
            return 'medium'
        elif risk_score < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _predict_time_to_burnout(self, risk_factors: BurnoutRiskFactors) -> Optional[float]:
        """Predict time to burnout using Cox PH model"""
        try:
            if self.cph_model is None:
                return None
            
            # Create feature vector for prediction
            features = pd.DataFrame([{
                'workout_frequency': risk_factors.workout_frequency,
                'sleep_hours': risk_factors.sleep_hours,
                'stress_level': risk_factors.stress_level,
                'recovery_time': risk_factors.recovery_time,
                'age': risk_factors.age,
                'experience_years': risk_factors.experience_years
            }])
            
            # Predict survival time
            predicted_time = self.cph_model.predict_expectation(features)
            
            return float(predicted_time.iloc[0]) if not pd.isna(predicted_time.iloc[0]) else None
            
        except Exception as e:
            logger.error(f"Time to burnout prediction error: {e}")
            return None
    
    def _calculate_survival_probability(self, risk_factors: BurnoutRiskFactors) -> float:
        """Calculate survival probability at 1 year"""
        try:
            if self.cph_model is None:
                return 0.7  # Default probability
            
            # Create feature vector
            features = pd.DataFrame([{
                'workout_frequency': risk_factors.workout_frequency,
                'sleep_hours': risk_factors.sleep_hours,
                'stress_level': risk_factors.stress_level,
                'recovery_time': risk_factors.recovery_time,
                'age': risk_factors.age,
                'experience_years': risk_factors.experience_years
            }])
            
            # Predict survival probability at 365 days
            survival_prob = self.cph_model.predict_survival_function(features, times=[365])
            
            return float(survival_prob.iloc[0, 0]) if not pd.isna(survival_prob.iloc[0, 0]) else 0.7
            
        except Exception as e:
            logger.error(f"Survival probability calculation error: {e}")
            return 0.7
    
    def _identify_risk_factors(self, risk_factors: BurnoutRiskFactors) -> List[str]:
        """Identify specific risk factors"""
        identified_factors = []
        
        if risk_factors.workout_frequency >= 6:
            identified_factors.append("excessive_workout_frequency")
        
        if risk_factors.sleep_hours < 7:
            identified_factors.append("insufficient_sleep")
        
        if risk_factors.stress_level >= 8:
            identified_factors.append("high_stress_level")
        
        if risk_factors.recovery_time <= 2:
            identified_factors.append("insufficient_recovery_time")
        
        if risk_factors.performance_trend == 'declining':
            identified_factors.append("declining_performance")
        
        if risk_factors.experience_years < 2:
            identified_factors.append("low_experience_level")
        
        return identified_factors
    
    def _generate_recommendations(self, risk_factors: BurnoutRiskFactors, risk_level: str) -> List[str]:
        """Generate personalized recommendations to prevent burnout"""
        recommendations = []
        
        # Risk level based recommendations
        if risk_level == 'critical':
            recommendations.append("Consider taking a complete break from training for 1-2 weeks")
            recommendations.append("Consult with a sports psychologist or coach")
        elif risk_level == 'high':
            recommendations.append("Reduce training intensity by 30-50%")
            recommendations.append("Increase recovery days between workouts")
        
        # Specific factor based recommendations
        if risk_factors.workout_frequency >= 6:
            recommendations.append(f"Reduce workout frequency to {max(3, risk_factors.workout_frequency - 2)} times per week")
        
        if risk_factors.sleep_hours < 7:
            recommendations.append(f"Aim for at least {8 - risk_factors.sleep_hours:.1f} more hours of sleep per night")
        
        if risk_factors.stress_level >= 8:
            recommendations.append("Implement stress management techniques (meditation, yoga, deep breathing)")
        
        if risk_factors.recovery_time <= 2:
            recommendations.append("Increase recovery time to at least 3-4 days between intense sessions")
        
        if risk_factors.performance_trend == 'declining':
            recommendations.append("Focus on technique and form rather than intensity")
            recommendations.append("Consider deloading for 1-2 weeks")
        
        # General recommendations
        recommendations.append("Listen to your body and don't ignore warning signs")
        recommendations.append("Maintain a training log to track patterns")
        
        return recommendations
    
    def _generate_survival_curve_data(self, risk_factors: BurnoutRiskFactors) -> Dict[str, List[float]]:
        """Generate survival curve data for visualization"""
        try:
            if self.cph_model is None:
                return self._generate_mock_survival_data()
            
            # Create feature vector
            features = pd.DataFrame([{
                'workout_frequency': risk_factors.workout_frequency,
                'sleep_hours': risk_factors.sleep_hours,
                'stress_level': risk_factors.stress_level,
                'recovery_time': risk_factors.recovery_time,
                'age': risk_factors.age,
                'experience_years': risk_factors.experience_years
            }])
            
            # Generate survival curve
            times = np.arange(0, 730, 30)  # 0 to 730 days, every 30 days
            survival_function = self.cph_model.predict_survival_function(features, times=times)
            
            return {
                'times': times.tolist(),
                'survival_probabilities': survival_function.iloc[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Survival curve generation error: {e}")
            return self._generate_mock_survival_data()
    
    def _generate_mock_survival_data(self) -> Dict[str, List[float]]:
        """Generate mock survival curve data"""
        times = list(range(0, 730, 30))  # 0 to 730 days
        # Mock exponential decay survival curve
        survival_probs = [np.exp(-0.001 * t) for t in times]
        
        return {
            'times': times,
            'survival_probabilities': survival_probs
        }
    
    def _store_user_data(self, user_id: str, risk_factors: BurnoutRiskFactors, 
                        risk_score: float, risk_level: str):
        """Store user risk analysis data"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {'risk_history': []}
        
        risk_data = {
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'workout_frequency': risk_factors.workout_frequency,
            'sleep_hours': risk_factors.sleep_hours,
            'stress_level': risk_factors.stress_level,
            'recovery_time': risk_factors.recovery_time,
            'performance_trend': risk_factors.performance_trend
        }
        
        self.user_data[user_id]['risk_history'].append(risk_data)
        self.user_data[user_id]['last_analysis'] = risk_data
    
    async def generate_survival_curve(self, user_id: str) -> Dict:
        """Generate survival curve for user"""
        if user_id not in self.user_data or not self.user_data[user_id].get('last_analysis'):
            return {"error": "No analysis data available for user"}
        
        # Get last analysis
        last_analysis = self.user_data[user_id]['last_analysis']
        
        # Recreate risk factors
        risk_factors = BurnoutRiskFactors(
            workout_frequency=last_analysis['workout_frequency'],
            sleep_hours=last_analysis['sleep_hours'],
            stress_level=last_analysis['stress_level'],
            recovery_time=last_analysis['recovery_time'],
            performance_trend=last_analysis['performance_trend'],
            age=self.user_data[user_id].get('age', 30),
            gender=self.user_data[user_id].get('gender', 'not_specified'),
            experience_years=self.user_data[user_id].get('experience_years', 5),
            training_intensity=self.user_data[user_id].get('training_intensity', 'moderate')
        )
        
        survival_data = self._generate_survival_curve_data(risk_factors)
        
        return {
            'user_id': user_id,
            'risk_score': last_analysis['risk_score'],
            'risk_level': last_analysis['risk_level'],
            'survival_curve': survival_data,
            'timestamp': last_analysis['timestamp']
        }
    
    async def get_recommendations(self, user_id: str) -> Dict:
        """Get personalized recommendations to prevent burnout"""
        if user_id not in self.user_data or not self.user_data[user_id].get('last_analysis'):
            return {"error": "No analysis data available for user"}
        
        last_analysis = self.user_data[user_id]['last_analysis']
        
        # Recreate risk factors
        risk_factors = BurnoutRiskFactors(
            workout_frequency=last_analysis['workout_frequency'],
            sleep_hours=last_analysis['sleep_hours'],
            stress_level=last_analysis['stress_level'],
            recovery_time=last_analysis['recovery_time'],
            performance_trend=last_analysis['performance_trend'],
            age=self.user_data[user_id].get('age', 30),
            gender=self.user_data[user_id].get('gender', 'not_specified'),
            experience_years=self.user_data[user_id].get('experience_years', 5),
            training_intensity=self.user_data[user_id].get('training_intensity', 'moderate')
        )
        
        recommendations = self._generate_recommendations(risk_factors, last_analysis['risk_level'])
        
        # Get trend analysis
        risk_history = self.user_data[user_id].get('risk_history', [])
        trend = self._analyze_risk_trend(risk_history)
        
        return {
            'user_id': user_id,
            'current_risk_level': last_analysis['risk_level'],
            'current_risk_score': last_analysis['risk_score'],
            'recommendations': recommendations,
            'risk_trend': trend,
            'analysis_count': len(risk_history)
        }
    
    def _analyze_risk_trend(self, risk_history: List[Dict]) -> str:
        """Analyze trend in risk scores over time"""
        if len(risk_history) < 2:
            return "insufficient_data"
        
        # Get last 3 analyses
        recent_analyses = risk_history[-3:]
        scores = [analysis['risk_score'] for analysis in recent_analyses]
        
        # Calculate trend
        if len(scores) >= 2:
            slope = (scores[-1] - scores[0]) / len(scores)
            
            if slope > 5:
                return "increasing"
            elif slope < -5:
                return "decreasing"
            else:
                return "stable"
        
        return "stable" 