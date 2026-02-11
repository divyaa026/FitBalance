"""
Production-Ready Burnout Dataset Generator with Longitudinal Data
Generates comprehensive synthetic user data for Cox PH and ML models
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

class BurnoutDatasetGenerator:
    """Generate realistic synthetic burnout prediction data"""
    
    def __init__(self, output_dir: str = "datasets/burnout"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # User archetypes
        self.archetypes = {
            'balanced': {
                'sleep_quality': (70, 85),
                'stress_level': (20, 40),
                'workload': (40, 60),
                'social_support': (70, 90),
                'exercise_frequency': (4, 6),
                'burnout_prob': 0.05
            },
            'overworked': {
                'sleep_quality': (50, 65),
                'stress_level': (60, 80),
                'workload': (75, 95),
                'social_support': (40, 60),
                'exercise_frequency': (1, 3),
                'burnout_prob': 0.45
            },
            'recovering': {
                'sleep_quality': (65, 80),
                'stress_level': (40, 60),
                'workload': (50, 70),
                'social_support': (60, 80),
                'exercise_frequency': (3, 5),
                'burnout_prob': 0.15
            },
            'high_risk': {
                'sleep_quality': (30, 50),
                'stress_level': (75, 95),
                'workload': (80, 100),
                'social_support': (20, 40),
                'exercise_frequency': (0, 2),
                'burnout_prob': 0.75
            },
            'athlete': {
                'sleep_quality': (75, 90),
                'stress_level': (30, 50),
                'workload': (60, 80),
                'social_support': (70, 85),
                'exercise_frequency': (5, 7),
                'burnout_prob': 0.10
            }
        }
    
    def generate_user_profile(self, archetype: str, user_id: int) -> Dict:
        """Generate a realistic user profile"""
        
        profile = self.archetypes[archetype]
        
        # Generate base characteristics
        age = int(np.random.normal(35, 10))
        age = np.clip(age, 18, 65)
        
        gender = np.random.choice(['male', 'female', 'other'], p=[0.48, 0.48, 0.04])
        
        occupation = np.random.choice([
            'software_engineer', 'healthcare_worker', 'teacher', 
            'manager', 'consultant', 'researcher', 'athlete',
            'entrepreneur', 'sales', 'creative'
        ])
        
        # Generate initial metrics
        return {
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'occupation': occupation,
            'archetype': archetype,
            'start_date': datetime.now() - timedelta(days=np.random.randint(180, 730))
        }
    
    def generate_daily_metrics(self, profile: Dict, day: int, 
                               prev_metrics: Dict = None) -> Dict:
        """Generate daily metrics with temporal correlation"""
        
        archetype_params = self.archetypes[profile['archetype']]
        
        # Base values from archetype
        sleep_range = archetype_params['sleep_quality']
        stress_range = archetype_params['stress_level']
        workload_range = archetype_params['workload']
        support_range = archetype_params['social_support']
        exercise_range = archetype_params['exercise_frequency']
        
        # Add temporal correlation (metrics evolve slowly)
        if prev_metrics is not None:
            # 80% correlation with previous day
            sleep_quality = 0.8 * prev_metrics['sleep_quality'] + 0.2 * np.random.uniform(*sleep_range)
            stress_level = 0.8 * prev_metrics['stress_level'] + 0.2 * np.random.uniform(*stress_range)
            workload = 0.8 * prev_metrics['workload'] + 0.2 * np.random.uniform(*workload_range)
            social_support = 0.8 * prev_metrics['social_support'] + 0.2 * np.random.uniform(*support_range)
        else:
            sleep_quality = np.random.uniform(*sleep_range)
            stress_level = np.random.uniform(*stress_range)
            workload = np.random.uniform(*workload_range)
            social_support = np.random.uniform(*support_range)
        
        # Exercise frequency (discrete)
        exercise_frequency = np.random.randint(*exercise_range)
        
        # Derived metrics
        hrv_score = 100 - stress_level * 0.5 - (100 - sleep_quality) * 0.3
        hrv_score = np.clip(hrv_score + np.random.normal(0, 5), 20, 100)
        
        recovery_time = 10 - (sleep_quality / 15) + (stress_level / 20)
        recovery_time = np.clip(recovery_time + np.random.normal(0, 0.5), 2, 14)
        
        work_life_balance = 100 - workload * 0.6 + social_support * 0.4
        work_life_balance = np.clip(work_life_balance + np.random.normal(0, 5), 0, 100)
        
        nutrition_quality = sleep_quality * 0.4 + (100 - stress_level) * 0.3 + exercise_frequency * 5
        nutrition_quality = np.clip(nutrition_quality + np.random.normal(0, 5), 20, 100)
        
        mental_fatigue = stress_level * 0.6 + workload * 0.3 - social_support * 0.1
        mental_fatigue = np.clip(mental_fatigue + np.random.normal(0, 5), 0, 100)
        
        # Performance metrics
        performance_score = (sleep_quality + (100 - stress_level) + (100 - mental_fatigue)) / 3
        performance_score = np.clip(performance_score + np.random.normal(0, 5), 0, 100)
        
        # Mood score
        mood_score = (social_support + (100 - stress_level) + work_life_balance) / 3
        mood_score = np.clip(mood_score + np.random.normal(0, 5), 0, 100)
        
        return {
            'day': day,
            'sleep_quality': round(sleep_quality, 2),
            'stress_level': round(stress_level, 2),
            'workload': round(workload, 2),
            'social_support': round(social_support, 2),
            'exercise_frequency': exercise_frequency,
            'hrv_score': round(hrv_score, 2),
            'recovery_time': round(recovery_time, 2),
            'work_life_balance': round(work_life_balance, 2),
            'nutrition_quality': round(nutrition_quality, 2),
            'mental_fatigue': round(mental_fatigue, 2),
            'performance_score': round(performance_score, 2),
            'mood_score': round(mood_score, 2)
        }
    
    def calculate_burnout_risk(self, metrics_history: List[Dict]) -> float:
        """Calculate burnout risk based on historical metrics"""
        
        if len(metrics_history) < 7:
            return 0.0
        
        # Get recent 30 days
        recent = metrics_history[-30:]
        
        # Calculate risk factors
        avg_sleep = np.mean([m['sleep_quality'] for m in recent])
        avg_stress = np.mean([m['stress_level'] for m in recent])
        avg_workload = np.mean([m['workload'] for m in recent])
        avg_support = np.mean([m['social_support'] for m in recent])
        avg_exercise = np.mean([m['exercise_frequency'] for m in recent])
        avg_mental_fatigue = np.mean([m['mental_fatigue'] for m in recent])
        avg_recovery = np.mean([m['recovery_time'] for m in recent])
        
        # Weighted risk score
        risk_score = (
            (100 - avg_sleep) * 0.20 +
            avg_stress * 0.25 +
            avg_workload * 0.20 +
            (100 - avg_support) * 0.10 +
            (7 - avg_exercise) * 5 * 0.10 +
            avg_mental_fatigue * 0.10 +
            (avg_recovery - 5) * 10 * 0.05
        )
        
        # Check for deteriorating trends
        if len(metrics_history) >= 60:
            recent_30 = metrics_history[-30:]
            previous_30 = metrics_history[-60:-30]
            
            sleep_trend = np.mean([m['sleep_quality'] for m in recent_30]) - np.mean([m['sleep_quality'] for m in previous_30])
            stress_trend = np.mean([m['stress_level'] for m in recent_30]) - np.mean([m['stress_level'] for m in previous_30])
            
            if sleep_trend < -10:  # Sleep declining
                risk_score += 10
            if stress_trend > 10:  # Stress increasing
                risk_score += 10
        
        return np.clip(risk_score, 0, 100)
    
    def determine_burnout_event(self, metrics_history: List[Dict], 
                                archetype_burnout_prob: float) -> Tuple[bool, int]:
        """Determine if and when burnout occurs"""
        
        if len(metrics_history) < 30:
            return False, 0
        
        # Calculate cumulative risk
        risk_scores = []
        for i in range(30, len(metrics_history)):
            risk = self.calculate_burnout_risk(metrics_history[:i+1])
            risk_scores.append(risk)
        
        # Burnout occurs if risk sustained high + random chance
        for day_idx, risk in enumerate(risk_scores):
            if risk > 70:  # High risk threshold
                # Probability increases with sustained high risk
                sustained_days = sum(1 for r in risk_scores[max(0, day_idx-14):day_idx+1] if r > 70)
                burnout_chance = archetype_burnout_prob * (1 + sustained_days / 14)
                
                if np.random.random() < burnout_chance:
                    return True, day_idx + 30  # Return day of burnout
        
        return False, len(metrics_history)
    
    def generate_user_longitudinal_data(self, profile: Dict, num_days: int = 365) -> pd.DataFrame:
        """Generate longitudinal data for a single user"""
        
        metrics_history = []
        prev_metrics = None
        
        for day in range(num_days):
            daily_metrics = self.generate_daily_metrics(profile, day, prev_metrics)
            
            # Add user info
            daily_metrics.update({
                'user_id': profile['user_id'],
                'age': profile['age'],
                'gender': profile['gender'],
                'occupation': profile['occupation'],
                'archetype': profile['archetype'],
                'date': (profile['start_date'] + timedelta(days=day)).strftime('%Y-%m-%d')
            })
            
            metrics_history.append(daily_metrics)
            prev_metrics = daily_metrics
        
        # Determine burnout
        burnout_occurred, burnout_day = self.determine_burnout_event(
            metrics_history, 
            self.archetypes[profile['archetype']]['burnout_prob']
        )
        
        # Add burnout information to each record
        for i, metrics in enumerate(metrics_history):
            metrics['burnout_event'] = 1 if (burnout_occurred and i >= burnout_day) else 0
            metrics['days_to_burnout'] = max(0, burnout_day - i) if burnout_occurred else num_days - i
            metrics['burnout_occurred'] = 1 if burnout_occurred else 0
        
        return pd.DataFrame(metrics_history)
    
    def generate_dataset(self, num_users: int = 1000, days_per_user: int = 365) -> pd.DataFrame:
        """Generate complete burnout prediction dataset"""
        
        print(f"Generating burnout dataset for {num_users} users over {days_per_user} days...")
        
        all_data = []
        
        for user_id in range(num_users):
            # Select archetype with realistic distribution
            archetype = np.random.choice(
                list(self.archetypes.keys()),
                p=[0.35, 0.25, 0.20, 0.10, 0.10]  # More balanced, fewer high-risk
            )
            
            # Generate user profile
            profile = self.generate_user_profile(archetype, user_id)
            
            # Generate longitudinal data
            user_data = self.generate_user_longitudinal_data(profile, days_per_user)
            all_data.append(user_data)
            
            if (user_id + 1) % 100 == 0:
                print(f"  Generated data for {user_id + 1}/{num_users} users")
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        
        # Save dataset
        csv_path = f"{self.output_dir}/burnout_longitudinal_dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Dataset saved to {csv_path}")
        print(f"   Total records: {len(df):,}")
        print(f"   Total users: {num_users}")
        
        # Generate aggregated dataset for model training
        self.generate_aggregated_dataset(df)
        
        return df
    
    def generate_aggregated_dataset(self, df: pd.DataFrame):
        """Generate aggregated dataset with summary statistics per user"""
        
        print("\nGenerating aggregated dataset for model training...")
        
        # Group by user and calculate statistics
        user_stats = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            
            # Get last observation for survival analysis
            last_record = user_data.iloc[-1]
            
            # Calculate statistics
            stats = {
                'user_id': user_id,
                'age': last_record['age'],
                'gender': last_record['gender'],
                'occupation': last_record['occupation'],
                'archetype': last_record['archetype'],
                
                # Average metrics
                'avg_sleep_quality': user_data['sleep_quality'].mean(),
                'avg_stress_level': user_data['stress_level'].mean(),
                'avg_workload': user_data['workload'].mean(),
                'avg_social_support': user_data['social_support'].mean(),
                'avg_exercise_frequency': user_data['exercise_frequency'].mean(),
                'avg_hrv_score': user_data['hrv_score'].mean(),
                'avg_recovery_time': user_data['recovery_time'].mean(),
                'avg_work_life_balance': user_data['work_life_balance'].mean(),
                'avg_nutrition_quality': user_data['nutrition_quality'].mean(),
                'avg_mental_fatigue': user_data['mental_fatigue'].mean(),
                
                # Variability metrics
                'std_sleep_quality': user_data['sleep_quality'].std(),
                'std_stress_level': user_data['stress_level'].std(),
                'std_workload': user_data['workload'].std(),
                
                # Trend metrics (last 30 days vs first 30 days)
                'sleep_trend': user_data.tail(30)['sleep_quality'].mean() - user_data.head(30)['sleep_quality'].mean(),
                'stress_trend': user_data.tail(30)['stress_level'].mean() - user_data.head(30)['stress_level'].mean(),
                'performance_trend': user_data.tail(30)['performance_score'].mean() - user_data.head(30)['performance_score'].mean(),
                
                # Outcome
                'training_days': len(user_data),
                'burnout_event': last_record['burnout_occurred'],
                'days_to_burnout': user_data[user_data['burnout_event'] == 1].iloc[0]['day'] if last_record['burnout_occurred'] == 1 else len(user_data)
            }
            
            user_stats.append(stats)
        
        # Create aggregated dataframe
        agg_df = pd.DataFrame(user_stats)
        
        # Save aggregated dataset
        agg_path = f"{self.output_dir}/burnout_aggregated_dataset.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"✅ Aggregated dataset saved to {agg_path}")
        print(f"   Total users: {len(agg_df)}")
        print(f"   Burnout rate: {agg_df['burnout_event'].mean():.1%}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        print(f"   Average training days: {agg_df['training_days'].mean():.0f}")
        print(f"   Burnout events: {agg_df['burnout_event'].sum()}")
        print(f"   No burnout: {(agg_df['burnout_event'] == 0).sum()}")
        print("\n   By archetype:")
        print(agg_df.groupby('archetype')['burnout_event'].agg(['count', 'sum', 'mean']))
        
        return agg_df

def main():
    """Generate burnout training dataset"""
    
    generator = BurnoutDatasetGenerator()
    
    # Generate dataset with 1000 users over 365 days each
    dataset = generator.generate_dataset(num_users=1000, days_per_user=365)
    
    print("\n✅ Burnout dataset generation complete!")
    print(f"📁 Location: {generator.output_dir}")
    print(f"📊 Total records: {len(dataset):,}")

if __name__ == "__main__":
    main()
