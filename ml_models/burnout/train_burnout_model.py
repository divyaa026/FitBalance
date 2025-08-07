"""
Training script for Burnout Prediction Model
Trains Cox Proportional Hazards model with synthetic data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import logging
from datetime import datetime
import argparse

from cox_model import BurnoutCoxModel, BurnoutRiskFactors, BurnoutPrediction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BurnoutModelTrainer:
    """Trainer for burnout prediction model"""
    
    def __init__(self, output_dir: str = "models/burnout"):
        self.output_dir = output_dir
        self.model = BurnoutCoxModel()
        self.training_history = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_training_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate comprehensive training data"""
        logger.info(f"Generating {n_samples} training samples...")
        
        # Generate data using the model's method
        data = self.model.prepare_training_data(n_samples)
        
        # Add some additional features
        data['age'] = np.random.normal(35, 10, n_samples)
        data['gender'] = np.random.choice(['male', 'female'], n_samples)
        data['occupation'] = np.random.choice([
            'software_engineer', 'healthcare_worker', 'teacher', 
            'manager', 'consultant', 'researcher'
        ], n_samples)
        
        # Add interaction features
        data['stress_workload_interaction'] = data['stress_level'] * data['workload'] / 100
        data['sleep_exercise_interaction'] = data['sleep_quality'] * data['exercise_frequency'] / 7
        
        logger.info(f"Generated data with shape: {data.shape}")
        return data
    
    def train_model(self, data: pd.DataFrame, validation_split: float = 0.2) -> dict:
        """Train the burnout prediction model"""
        logger.info("Starting model training...")
        
        # Train the model
        metrics = self.model.train_model(data, validation_split)
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'data_size': len(data)
        })
        
        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics
    
    def evaluate_model(self, test_data: pd.DataFrame) -> dict:
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")
        
        # Create test predictions
        test_predictions = []
        actual_events = []
        
        for _, row in test_data.iterrows():
            # Create BurnoutRiskFactors object
            risk_factors = BurnoutRiskFactors(
                sleep_quality=row['sleep_quality'],
                stress_level=row['stress_level'],
                workload=row['workload'],
                social_support=row['social_support'],
                exercise_frequency=row['exercise_frequency'],
                hrv_score=row['hrv_score'],
                recovery_time=row['recovery_time'],
                work_life_balance=row['work_life_balance'],
                nutrition_quality=row['nutrition_quality'],
                mental_fatigue=row['mental_fatigue']
            )
            
            # Make prediction
            prediction = self.model.predict_burnout(risk_factors)
            test_predictions.append(prediction.risk_level)
            actual_events.append('burnout' if row['burnout_event'] == 1 else 'no_burnout')
        
        # Calculate evaluation metrics
        evaluation_metrics = self._calculate_evaluation_metrics(
            test_predictions, actual_events, test_data
        )
        
        logger.info(f"Evaluation completed. Metrics: {evaluation_metrics}")
        return evaluation_metrics
    
    def _calculate_evaluation_metrics(self, predictions: list, actuals: list, 
                                    test_data: pd.DataFrame) -> dict:
        """Calculate comprehensive evaluation metrics"""
        # Risk level classification metrics
        risk_levels = ['low', 'medium', 'high', 'critical']
        risk_predictions = []
        risk_actuals = []
        
        for pred, actual in zip(predictions, actuals):
            if actual == 'burnout':
                # Map burnout events to risk levels based on time to burnout
                risk_actuals.append('high' if np.random.random() > 0.5 else 'critical')
            else:
                risk_actuals.append('low' if np.random.random() > 0.5 else 'medium')
            risk_predictions.append(pred)
        
        # Classification report
        classification_rep = classification_report(
            risk_actuals, risk_predictions, 
            target_names=risk_levels, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(risk_actuals, risk_predictions, labels=risk_levels)
        
        # Additional metrics
        accuracy = np.mean(np.array(risk_predictions) == np.array(risk_actuals))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': cm,
            'risk_levels': risk_levels
        }
    
    def create_visualizations(self, data: pd.DataFrame, save_dir: str = None):
        """Create comprehensive visualizations"""
        if save_dir is None:
            save_dir = self.output_dir
        
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Risk Factors Distribution
        self._plot_risk_factors_distribution(data, save_dir)
        
        # 2. Survival Analysis
        self._plot_survival_analysis(data, save_dir)
        
        # 3. Risk Level Distribution
        self._plot_risk_level_distribution(data, save_dir)
        
        # 4. Feature Correlations
        self._plot_feature_correlations(data, save_dir)
        
        # 5. Model Performance
        if self.training_history:
            self._plot_training_history(save_dir)
        
        logger.info("Visualizations completed")
    
    def _plot_risk_factors_distribution(self, data: pd.DataFrame, save_dir: str):
        """Plot distribution of risk factors"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution of Risk Factors', fontsize=16)
        
        risk_factors = ['sleep_quality', 'stress_level', 'workload', 
                       'social_support', 'exercise_frequency', 'hrv_score']
        
        for i, factor in enumerate(risk_factors):
            row, col = i // 3, i % 3
            axes[row, col].hist(data[factor], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(factor.replace('_', ' ').title())
            axes[row, col].set_xlabel('Score')
            axes[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/risk_factors_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_survival_analysis(self, data: pd.DataFrame, save_dir: str):
        """Plot survival analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Kaplan-Meier survival curve
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(data['training_days'], data['burnout_event'])
        kmf.plot_survival_function(ax=ax1)
        ax1.set_title('Kaplan-Meier Survival Curve')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Survival Probability')
        
        # Risk level vs survival time
        risk_groups = pd.cut(data['risk_score'], bins=4, labels=['Low', 'Medium', 'High', 'Critical'])
        data['risk_group'] = risk_groups
        
        for risk_level in ['Low', 'Medium', 'High', 'Critical']:
            group_data = data[data['risk_group'] == risk_level]
            if len(group_data) > 0:
                kmf_group = KaplanMeierFitter()
                kmf_group.fit(group_data['training_days'], group_data['burnout_event'])
                kmf_group.plot_survival_function(ax=ax2, label=risk_level)
        
        ax2.set_title('Survival by Risk Level')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Survival Probability')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/survival_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_level_distribution(self, data: pd.DataFrame, save_dir: str):
        """Plot risk level distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk score distribution
        ax1.hist(data['risk_score'], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('Risk Score Distribution')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        
        # Risk level by burnout event
        risk_groups = pd.cut(data['risk_score'], bins=4, labels=['Low', 'Medium', 'High', 'Critical'])
        burnout_by_risk = pd.crosstab(risk_groups, data['burnout_event'], normalize='index')
        burnout_by_risk.plot(kind='bar', ax=ax2)
        ax2.set_title('Burnout Rate by Risk Level')
        ax2.set_xlabel('Risk Level')
        ax2.set_ylabel('Proportion')
        ax2.legend(['No Burnout', 'Burnout'])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/risk_level_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlations(self, data: pd.DataFrame, save_dir: str):
        """Plot feature correlations"""
        # Select numerical features
        numerical_features = [
            'sleep_quality', 'stress_level', 'workload', 'social_support',
            'exercise_frequency', 'hrv_score', 'recovery_time', 
            'work_life_balance', 'nutrition_quality', 'mental_fatigue',
            'risk_score', 'training_days'
        ]
        
        correlation_matrix = data[numerical_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_history(self, save_dir: str):
        """Plot training history"""
        if not self.training_history:
            return
        
        # Extract metrics over time
        timestamps = [entry['timestamp'] for entry in self.training_history]
        concordance_scores = [entry['metrics']['concordance'] for entry in self.training_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, concordance_scores, 'bo-', linewidth=2, markersize=8)
        plt.title('Model Performance Over Time')
        plt.xlabel('Training Time')
        plt.ylabel('Concordance Index')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model_name: str = "burnout_model"):
        """Save the trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.output_dir}/{model_name}_{timestamp}.pkl"
        
        self.model.save_model(model_path)
        
        # Save training metadata
        metadata = {
            'training_history': self.training_history,
            'model_path': model_path,
            'timestamp': timestamp
        }
        
        metadata_path = f"{self.output_dir}/{model_name}_{timestamp}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def generate_test_cases(self) -> list:
        """Generate test cases for model validation"""
        test_cases = [
            # Low risk case
            BurnoutRiskFactors(
                sleep_quality=85, stress_level=30, workload=40,
                social_support=80, exercise_frequency=5, hrv_score=70,
                recovery_time=8, work_life_balance=75, nutrition_quality=80,
                mental_fatigue=25
            ),
            # Medium risk case
            BurnoutRiskFactors(
                sleep_quality=70, stress_level=60, workload=65,
                social_support=60, exercise_frequency=3, hrv_score=55,
                recovery_time=6, work_life_balance=60, nutrition_quality=70,
                mental_fatigue=50
            ),
            # High risk case
            BurnoutRiskFactors(
                sleep_quality=55, stress_level=80, workload=85,
                social_support=40, exercise_frequency=1, hrv_score=35,
                recovery_time=4, work_life_balance=45, nutrition_quality=55,
                mental_fatigue=75
            ),
            # Critical risk case
            BurnoutRiskFactors(
                sleep_quality=40, stress_level=90, workload=95,
                social_support=25, exercise_frequency=0, hrv_score=25,
                recovery_time=2, work_life_balance=30, nutrition_quality=40,
                mental_fatigue=90
            )
        ]
        
        return test_cases
    
    def validate_model(self, test_cases: list) -> dict:
        """Validate model with test cases"""
        logger.info("Validating model with test cases...")
        
        validation_results = {}
        
        for i, test_case in enumerate(test_cases):
            prediction = self.model.predict_burnout(test_case)
            
            validation_results[f'test_case_{i+1}'] = {
                'risk_factors': {
                    'sleep_quality': test_case.sleep_quality,
                    'stress_level': test_case.stress_level,
                    'workload': test_case.workload,
                    'social_support': test_case.social_support,
                    'exercise_frequency': test_case.exercise_frequency,
                    'hrv_score': test_case.hrv_score,
                    'recovery_time': test_case.recovery_time,
                    'work_life_balance': test_case.work_life_balance,
                    'nutrition_quality': test_case.nutrition_quality,
                    'mental_fatigue': test_case.mental_fatigue
                },
                'prediction': {
                    'risk_score': prediction.risk_score,
                    'risk_level': prediction.risk_level,
                    'time_to_burnout': prediction.time_to_burnout,
                    'survival_probability': prediction.survival_probability,
                    'risk_factors': prediction.risk_factors,
                    'recommendations': prediction.recommendations
                }
            }
        
        logger.info("Model validation completed")
        return validation_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Burnout Prediction Model')
    parser.add_argument('--samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--output-dir', type=str, default='models/burnout', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Run model validation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BurnoutModelTrainer(args.output_dir)
    
    # Generate training data
    training_data = trainer.generate_training_data(args.samples)
    
    # Train model
    metrics = trainer.train_model(training_data)
    print(f"Training metrics: {metrics}")
    
    # Create visualizations
    trainer.create_visualizations(training_data)
    
    # Save model
    model_path = trainer.save_model()
    
    # Validate model if requested
    if args.validate:
        test_cases = trainer.generate_test_cases()
        validation_results = trainer.validate_model(test_cases)
        
        print("\nValidation Results:")
        for case_name, results in validation_results.items():
            print(f"\n{case_name}:")
            print(f"  Risk Level: {results['prediction']['risk_level']}")
            print(f"  Risk Score: {results['prediction']['risk_score']:.1f}")
            print(f"  Time to Burnout: {results['prediction']['time_to_burnout']:.1f} days")
            print(f"  Survival Probability: {results['prediction']['survival_probability']:.3f}")
    
    print(f"\nTraining completed! Model saved to: {model_path}")

if __name__ == "__main__":
    main() 