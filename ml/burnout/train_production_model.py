"""
Production Cox Proportional Hazards and ML Model Training for Burnout Prediction
Uses comprehensive longitudinal synthetic dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

class BurnoutModelTrainer:
    """Train production burnout prediction models"""
    
    def __init__(self, data_dir: str = "datasets/burnout"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("ml_models/burnout")
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load longitudinal and aggregated datasets"""
        
        longitudinal_path = self.data_dir / "burnout_longitudinal_dataset.csv"
        aggregated_path = self.data_dir / "burnout_aggregated_dataset.csv"
        
        if not longitudinal_path.exists():
            print("⚠️  Dataset not found. Generating...")
            import subprocess
            subprocess.run(["python", "datasets/generate_burnout_dataset.py"])
        
        longitudinal_df = pd.read_csv(longitudinal_path)
        aggregated_df = pd.read_csv(aggregated_path)
        
        print(f"✅ Loaded longitudinal data: {len(longitudinal_df):,} records")
        print(f"✅ Loaded aggregated data: {len(aggregated_df)} users")
        
        return longitudinal_df, aggregated_df
    
    def prepare_cox_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Cox PH model"""
        
        # Encode categorical variables
        categorical_cols = ['gender', 'occupation', 'archetype']
        
        df_encoded = df.copy()
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        # Select features for Cox model
        feature_cols = [
            'age', 'gender', 'occupation',
            'avg_sleep_quality', 'avg_stress_level', 'avg_workload',
            'avg_social_support', 'avg_exercise_frequency', 'avg_hrv_score',
            'avg_recovery_time', 'avg_work_life_balance', 'avg_nutrition_quality',
            'avg_mental_fatigue', 'std_sleep_quality', 'std_stress_level',
            'std_workload', 'sleep_trend', 'stress_trend', 'performance_trend'
        ]
        
        # Create Cox dataframe
        cox_df = df_encoded[feature_cols + ['days_to_burnout', 'burnout_event']].copy()
        cox_df = cox_df.rename(columns={'days_to_burnout': 'T', 'burnout_event': 'E'})
        
        return cox_df, feature_cols
    
    def train_cox_model(self, cox_df: pd.DataFrame, feature_cols: list):
        """Train Cox Proportional Hazards model"""
        
        print("\n🔬 Training Cox Proportional Hazards Model...")
        
        # Split data
        train_df, test_df = train_test_split(cox_df, test_size=0.2, random_state=42)
        
        # Train Cox PH model
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(train_df, duration_col='T', event_col='E')
        
        # Print summary
        print("\n📊 Cox Model Summary:")
        print(cph.summary[['coef', 'exp(coef)', 'p']])
        
        # Evaluate on test set
        train_concordance = concordance_index(
            train_df['T'], -cph.predict_partial_hazard(train_df), train_df['E']
        )
        test_concordance = concordance_index(
            test_df['T'], -cph.predict_partial_hazard(test_df), test_df['E']
        )
        
        print(f"\n✅ Train C-index: {train_concordance:.4f}")
        print(f"✅ Test C-index: {test_concordance:.4f}")
        
        # Save model
        model_path = self.models_dir / "cox_ph_model.pkl"
        joblib.dump(cph, model_path)
        print(f"💾 Saved Cox model to {model_path}")
        
        # Plot top hazard ratios
        self.plot_hazard_ratios(cph)
        
        return cph, test_concordance
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML models"""
        
        # Encode categorical variables
        df_encoded = df.copy()
        categorical_cols = ['gender', 'occupation', 'archetype']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        # Select features
        feature_cols = [
            'age', 'gender', 'occupation',
            'avg_sleep_quality', 'avg_stress_level', 'avg_workload',
            'avg_social_support', 'avg_exercise_frequency', 'avg_hrv_score',
            'avg_recovery_time', 'avg_work_life_balance', 'avg_nutrition_quality',
            'avg_mental_fatigue', 'std_sleep_quality', 'std_stress_level',
            'std_workload', 'sleep_trend', 'stress_trend', 'performance_trend'
        ]
        
        X = df_encoded[feature_cols].values
        y = df_encoded['burnout_event'].values
        
        return X, y, feature_cols
    
    def train_ml_models(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        """Train ML classification models"""
        
        print("\n🤖 Training Machine Learning Models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        scaler_path = self.models_dir / "feature_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")
        
        results = {}
        
        # Train Random Forest
        print("\n  Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        
        results['random_forest'] = {
            'model': rf_model,
            'auc': rf_auc,
            'predictions': rf_pred_proba
        }
        
        print(f"    ✅ Random Forest AUC: {rf_auc:.4f}")
        
        # Train Gradient Boosting
        print("\n  Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        gb_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_pred_proba)
        
        results['gradient_boosting'] = {
            'model': gb_model,
            'auc': gb_auc,
            'predictions': gb_pred_proba
        }
        
        print(f"    ✅ Gradient Boosting AUC: {gb_auc:.4f}")
        
        # Save models
        rf_path = self.models_dir / "random_forest_model.pkl"
        gb_path = self.models_dir / "gradient_boosting_model.pkl"
        
        joblib.dump(rf_model, rf_path)
        joblib.dump(gb_model, gb_path)
        
        print(f"💾 Saved Random Forest to {rf_path}")
        print(f"💾 Saved Gradient Boosting to {gb_path}")
        
        # Plot feature importances
        self.plot_feature_importances(rf_model, feature_names, "Random Forest")
        self.plot_feature_importances(gb_model, feature_names, "Gradient Boosting")
        
        # Classification report
        print("\n📊 Classification Report (Random Forest):")
        rf_pred = rf_model.predict(X_test_scaled)
        print(classification_report(y_test, rf_pred, target_names=['No Burnout', 'Burnout']))
        
        return results, X_test_scaled, y_test
    
    def plot_hazard_ratios(self, cph):
        """Plot hazard ratios from Cox model"""
        
        plt.figure(figsize=(10, 8))
        
        # Get top 10 features by absolute coefficient
        summary = cph.summary.copy()
        summary['abs_coef'] = summary['coef'].abs()
        top_features = summary.nlargest(10, 'abs_coef')
        
        # Plot
        plt.barh(range(len(top_features)), top_features['exp(coef)'])
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Hazard Ratio (exp(coef))')
        plt.title('Top 10 Hazard Ratios - Cox PH Model')
        plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plot_path = self.models_dir / "cox_hazard_ratios.png"
        plt.savefig(plot_path)
        print(f"📊 Saved hazard ratios plot to {plot_path}")
        plt.close()
    
    def plot_feature_importances(self, model, feature_names, model_name):
        """Plot feature importances"""
        
        plt.figure(figsize=(10, 8))
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.tight_layout()
        
        plot_path = self.models_dir / f"{model_name.lower().replace(' ', '_')}_importances.png"
        plt.savefig(plot_path)
        print(f"📊 Saved feature importances to {plot_path}")
        plt.close()
    
    def train_all(self):
        """Train all burnout prediction models"""
        
        print("🔥 Training Production Burnout Prediction Models")
        print("=" * 60)
        
        # Load data
        longitudinal_df, aggregated_df = self.load_data()
        
        # Train Cox PH model
        cox_data, cox_features = self.prepare_cox_data(aggregated_df)
        cox_model, cox_concordance = self.train_cox_model(cox_data, cox_features)
        
        # Train ML models
        X, y, ml_features = self.prepare_ml_data(aggregated_df)
        ml_results, X_test, y_test = self.train_ml_models(X, y, ml_features)
        
        # Save label encoders
        encoders_path = self.models_dir / "label_encoders.pkl"
        joblib.dump(self.label_encoders, encoders_path)
        print(f"\n💾 Saved label encoders to {encoders_path}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 Training Complete!")
        print("=" * 60)
        print(f"Cox PH Model C-index: {cox_concordance:.4f}")
        print(f"Random Forest AUC: {ml_results['random_forest']['auc']:.4f}")
        print(f"Gradient Boosting AUC: {ml_results['gradient_boosting']['auc']:.4f}")
        print(f"\n📁 Models saved to: {self.models_dir}")

def main():
    """Train all burnout models"""
    
    trainer = BurnoutModelTrainer()
    trainer.train_all()

if __name__ == "__main__":
    main()
