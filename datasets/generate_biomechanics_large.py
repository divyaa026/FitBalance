import pandas as pd
import numpy as np
import json
import os
import gc
from datetime import datetime
from tqdm import tqdm
import psutil

class OptimizedBiomechanicsDataGenerator:
    def __init__(self, chunk_size=1000):
        """
        Initialize with optimized settings for large-scale generation
        
        Args:
            chunk_size (int): Number of users to process in each chunk
        """
        self.chunk_size = chunk_size
        self.survey_insights = {
            'age_groups': {'teen': 15, 'young_adult': 65, 'adult': 25, 'senior': 5},
            'common_exercises': ['squat', 'deadlift', 'bench_press', 'overhead_press', 'lunge'],
            'fitness_levels': ['beginner', 'intermediate', 'advanced'],
            'common_errors': {
                'squat': ['knee_valgus', 'back_rounding', 'heels_lifting', 'depth_insufficient'],
                'deadlift': ['back_rounding', 'hip_rise_early', 'bar_path_forward'],
                'bench_press': ['elbow_flare', 'asymmetry', 'range_limited']
            }
        }
        
        # Pre-compute probability arrays for efficiency
        self._precompute_distributions()
    
    def _precompute_distributions(self):
        """Pre-compute probability distributions for faster generation"""
        self.age_values = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        self.age_probs = [0.1, 0.3, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
        self.fitness_levels = ['beginner', 'intermediate', 'advanced']
        self.fitness_probs = [0.4, 0.4, 0.2]
        self.injury_probs = [0.3, 0.7]  # [True, False]
        
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def generate_user_profiles_chunk(self, start_idx, chunk_size):
        """Generate a chunk of user profiles efficiently"""
        user_data = []
        
        # Pre-generate random values in batches for efficiency
        ages = np.random.choice(self.age_values, size=chunk_size, p=self.age_probs)
        heights = np.clip(np.random.normal(170, 10, chunk_size), 140, 200)
        weights = np.clip(np.random.normal(70, 15, chunk_size), 40, 150)
        fitness_levels = np.random.choice(self.fitness_levels, size=chunk_size, p=self.fitness_probs)
        experience_months = np.maximum(1, np.random.exponential(24, chunk_size))
        injuries = np.random.choice([True, False], size=chunk_size, p=self.injury_probs)
        
        for i in range(chunk_size):
            user = {
                'user_id': f'user_{start_idx + i:06d}',
                'age': int(ages[i]),
                'height_cm': float(heights[i]),
                'weight_kg': float(weights[i]),
                'fitness_level': fitness_levels[i],
                'exercise_experience_months': float(experience_months[i]),
                'previous_injuries': bool(injuries[i])
            }
            user_data.append(user)
        
        return user_data

    def _get_quality_probabilities(self, base_score):
        """Calculate probabilities for form quality based on user profile"""
        if base_score >= 0.7:
            return [0.6, 0.3, 0.08, 0.02]  # excellent, good, poor, dangerous
        elif base_score >= 0.5:
            return [0.3, 0.5, 0.15, 0.05]
        elif base_score >= 0.3:
            return [0.1, 0.4, 0.4, 0.1]
        else:
            return [0.05, 0.25, 0.5, 0.2]

    def _get_error_multiplier(self, form_quality):
        """Determine how much error to introduce based on form quality"""
        multipliers = {
            'excellent': 0.1,
            'good': 0.3,
            'poor': 0.7,
            'dangerous': 1.5
        }
        return multipliers[form_quality]

    def _get_exercise_base_position(self, exercise, joint, frame, total_frames):
        """Get base joint positions for different exercises"""
        progress = frame / total_frames
        
        if exercise == 'squat':
            if joint in ['knee_left', 'knee_right']:
                y = 0.5 + 0.3 * np.sin(progress * np.pi)
                return 0, y, 0
            elif joint == 'hip':
                return 0, 0.7 + 0.2 * np.sin(progress * np.pi), 0
            else:
                return np.random.normal(0, 0.1), np.random.normal(0.5, 0.2), np.random.normal(0, 0.1)
        
        elif exercise == 'deadlift':
            if joint in ['hip', 'spine']:
                y = 0.8 - 0.3 * np.sin(progress * np.pi)
                return 0, y, 0
            else:
                return np.random.normal(0, 0.1), np.random.normal(0.6, 0.2), np.random.normal(0, 0.1)
        
        return np.random.normal(0, 0.2), np.random.normal(0.5, 0.3), np.random.normal(0, 0.2)

    def _generate_joint_coordinates(self, exercise, form_quality, frame, total_frames):
        """Generate realistic 3D joint coordinates based on exercise and form quality"""
        joints = ['hip', 'knee_left', 'knee_right', 'ankle_left', 'ankle_right', 
                  'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right',
                  'wrist_left', 'wrist_right', 'spine', 'neck', 'head']
        
        coordinates = {}
        error_multiplier = self._get_error_multiplier(form_quality)
        
        for joint in joints:
            base_x, base_y, base_z = self._get_exercise_base_position(exercise, joint, frame, total_frames)
            
            noise_x = np.random.normal(0, 0.1 * error_multiplier)
            noise_y = np.random.normal(0, 0.15 * error_multiplier)
            noise_z = np.random.normal(0, 0.08 * error_multiplier)
            
            coordinates[joint] = {
                'x': float(base_x + noise_x),
                'y': float(base_y + noise_y), 
                'z': float(base_z + noise_z)
            }
        
        return coordinates

    def _generate_frame_data(self, exercise, form_quality):
        """Generate synthetic 3D joint coordinates"""
        frames = []
        num_frames = np.random.randint(10, 30)
        
        for frame in range(num_frames):
            frame_data = {
                'frame_number': frame,
                'joint_coordinates': self._generate_joint_coordinates(exercise, form_quality, frame, num_frames),
                'timestamp_ms': frame * 100
            }
            frames.append(frame_data)
        
        return frames

    def _generate_joint_angles(self, exercise, form_quality):
        """Generate joint angle data"""
        angles = {}
        
        if exercise == 'squat':
            angles['knee_flexion'] = float(np.random.normal(90, 20 if form_quality == 'excellent' else 40))
            angles['hip_flexion'] = float(np.random.normal(45, 15 if form_quality == 'excellent' else 30))
            angles['ankle_dorsiflexion'] = float(np.random.normal(35, 10 if form_quality == 'excellent' else 25))
            
        elif exercise == 'deadlift':
            angles['hip_hinge'] = float(np.random.normal(60, 15 if form_quality == 'excellent' else 35))
            angles['knee_flexion'] = float(np.random.normal(40, 10 if form_quality == 'excellent' else 25))
            angles['back_angle'] = float(np.random.normal(20, 5 if form_quality == 'excellent' else 20))
        
        elif exercise == 'bench_press':
            angles['elbow_flexion'] = float(np.random.normal(75, 10 if form_quality == 'excellent' else 25))
            angles['shoulder_flexion'] = float(np.random.normal(45, 8 if form_quality == 'excellent' else 20))
            
        return angles

    def _generate_errors(self, exercise, form_quality):
        """Generate specific form errors based on exercise and quality"""
        if form_quality == 'excellent':
            return []
        
        error_library = {
            'squat': ['knee_valgus', 'back_rounding', 'heels_lifting', 'depth_insufficient', 'knee_hyperextension'],
            'deadlift': ['back_rounding', 'hip_rise_early', 'bar_path_forward', 'shoulders_behind_bar'],
            'bench_press': ['elbow_flare', 'asymmetry', 'range_limited', 'shoulder_shrug'],
            'overhead_press': ['back_arching', 'head_forward', 'elbow_flare', 'range_limited'],
            'lunge': ['knee_over_toe', 'torso_lean', 'hip_instability', 'depth_insufficient']
        }
        
        available_errors = error_library.get(exercise, [])
        if form_quality == 'good':
            num_errors = np.random.randint(1, 3)
        elif form_quality == 'poor':
            num_errors = np.random.randint(2, 4)
        else:  # dangerous
            num_errors = np.random.randint(3, min(5, len(available_errors) + 1))
        
        return list(np.random.choice(available_errors, min(num_errors, len(available_errors)), replace=False))

    def generate_exercise_sequences_chunk(self, user_chunk, sequences_per_user=3):
        """Generate exercise sequences for a chunk of users"""
        exercises = ['squat', 'deadlift', 'bench_press', 'overhead_press', 'lunge']
        form_quality_options = ['excellent', 'good', 'poor', 'dangerous']
        
        exercise_data = []
        
        for user in user_chunk:
            for seq_num in range(sequences_per_user):
                exercise = np.random.choice(exercises)
                
                # Determine form quality based on user profile
                base_quality = 0.7 if user['fitness_level'] == 'advanced' else 0.4
                if user['previous_injuries']:
                    base_quality -= 0.1
                if user['exercise_experience_months'] > 24:
                    base_quality += 0.2
                    
                form_quality = np.random.choice(form_quality_options, 
                                              p=self._get_quality_probabilities(base_quality))
                
                # Generate sequence data
                sequence = {
                    'sequence_id': f"{user['user_id']}_seq_{seq_num}",
                    'user_id': user['user_id'],
                    'exercise_type': exercise,
                    'frames_data': self._generate_frame_data(exercise, form_quality),
                    'joint_angles': self._generate_joint_angles(exercise, form_quality),
                    'form_quality': form_quality,
                    'specific_errors': self._generate_errors(exercise, form_quality),
                    'injury_risk_score': float(np.random.beta(2, 5) if form_quality == 'dangerous' else np.random.beta(5, 2)),
                    'timestamp': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}",
                    'equipment_used': np.random.choice(['barbell', 'dumbbell', 'bodyweight']),
                    'repetition_count': np.random.randint(5, 15)
                }
                exercise_data.append(sequence)
        
        return exercise_data

    def save_chunk_to_files(self, chunk_data, chunk_num, is_first_chunk=False):
        """Save chunk data to files efficiently"""
        
        # Save to JSON file (append mode)
        json_filename = 'biomechanics_dataset_large.json'
        if is_first_chunk:
            # Create new file with metadata and start array
            with open(json_filename, 'w') as f:
                f.write('{\n  "metadata": {\n')
                f.write(f'    "dataset_name": "FitBalance_Biomechanics_Dataset_Large",\n')
                f.write(f'    "generation_date": "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",\n')
                f.write(f'    "data_schema_version": "1.0",\n')
                f.write(f'    "exercises_covered": ["squat", "deadlift", "bench_press", "overhead_press", "lunge"]\n')
                f.write('  },\n  "exercise_sequences": [\n')
        
        # Append chunk data to JSON
        with open(json_filename, 'a') as f:
            for i, seq in enumerate(chunk_data):
                if not is_first_chunk or i > 0:
                    f.write(',\n')
                json.dump(seq, f, indent=4)
        
        # Save to CSV file (append mode)
        csv_filename = 'biomechanics_dataset_large.csv'
        flat_data = []
        for seq in chunk_data:
            flat_record = {
                'sequence_id': seq['sequence_id'],
                'user_id': seq['user_id'],
                'exercise_type': seq['exercise_type'],
                'form_quality': seq['form_quality'],
                'specific_errors': ', '.join(seq['specific_errors']),
                'injury_risk_score': seq['injury_risk_score'],
                'repetition_count': seq['repetition_count'],
                'equipment_used': seq['equipment_used']
            }
            flat_data.append(flat_record)
        
        df_chunk = pd.DataFrame(flat_data)
        if is_first_chunk:
            df_chunk.to_csv(csv_filename, index=False, mode='w')
        else:
            df_chunk.to_csv(csv_filename, index=False, mode='a', header=False)
        
        # Clear memory
        del flat_data, df_chunk
        gc.collect()

def create_large_biomechanics_dataset(total_users=50000, sequences_per_user=3, chunk_size=1000):
    """
    Main function to create large-scale biomechanics dataset with chunked processing
    
    Args:
        total_users (int): Total number of users to generate (default: 50,000)
        sequences_per_user (int): Number of exercise sequences per user (default: 3)
        chunk_size (int): Number of users to process in each chunk (default: 1,000)
    """
    
    print("🚀 FitBalance Large-Scale Biomechanics Dataset Generation")
    print("=" * 60)
    print(f"📊 Target: {total_users:,} users, {total_users * sequences_per_user:,} sequences")
    print(f"🔧 Chunk size: {chunk_size:,} users per chunk")
    print(f"💾 Memory usage at start: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    print()
    
    generator = OptimizedBiomechanicsDataGenerator(chunk_size=chunk_size)
    
    # Calculate number of chunks
    num_chunks = (total_users + chunk_size - 1) // chunk_size
    
    # Initialize counters
    total_sequences_generated = 0
    
    # Clean up old files
    for filename in ['biomechanics_dataset_large.json', 'biomechanics_dataset_large.csv']:
        if os.path.exists(filename):
            os.remove(filename)
    
    # Process chunks with progress tracking
    with tqdm(total=num_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for chunk_idx in range(num_chunks):
            start_time = datetime.now()
            
            # Calculate chunk boundaries
            start_user_idx = chunk_idx * chunk_size
            current_chunk_size = min(chunk_size, total_users - start_user_idx)
            
            # Generate user profiles for this chunk
            user_chunk = generator.generate_user_profiles_chunk(start_user_idx, current_chunk_size)
            
            # Generate exercise sequences for this chunk
            exercise_sequences = generator.generate_exercise_sequences_chunk(
                user_chunk, sequences_per_user=sequences_per_user
            )
            
            # Save chunk to files
            is_first_chunk = (chunk_idx == 0)
            generator.save_chunk_to_files(exercise_sequences, chunk_idx, is_first_chunk)
            
            # Update counters
            total_sequences_generated += len(exercise_sequences)
            
            # Memory management
            del user_chunk, exercise_sequences
            gc.collect()
            
            # Progress update
            end_time = datetime.now()
            chunk_duration = (end_time - start_time).total_seconds()
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            pbar.set_postfix({
                'users': f"{start_user_idx + current_chunk_size:,}",
                'sequences': f"{total_sequences_generated:,}",
                'mem_mb': f"{memory_usage:.1f}",
                'time_s': f"{chunk_duration:.1f}"
            })
            pbar.update(1)
    
    # Finalize JSON file
    with open('biomechanics_dataset_large.json', 'a') as f:
        f.write('\n  ]\n}')
    
    # Final statistics
    print("\n✅ Dataset Generation Complete!")
    print("=" * 60)
    print(f"📊 Final Statistics:")
    print(f"   • Total users: {total_users:,}")
    print(f"   • Total sequences: {total_sequences_generated:,}")
    print(f"   • Sequences per user: {sequences_per_user}")
    print(f"   • Chunks processed: {num_chunks}")
    print(f"   • Final memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    
    # File size information
    if os.path.exists('biomechanics_dataset_large.json'):
        json_size = os.path.getsize('biomechanics_dataset_large.json') / 1024 / 1024
        print(f"   • JSON file size: {json_size:.1f} MB")
    
    if os.path.exists('biomechanics_dataset_large.csv'):
        csv_size = os.path.getsize('biomechanics_dataset_large.csv') / 1024 / 1024
        print(f"   • CSV file size: {csv_size:.1f} MB")
    
    print(f"\n📁 Output files:")
    print(f"   • biomechanics_dataset_large.json - Complete dataset with 3D coordinates")
    print(f"   • biomechanics_dataset_large.csv - Flattened dataset for analysis")
    
    return {
        'total_users': total_users,
        'total_sequences': total_sequences_generated,
        'chunks_processed': num_chunks
    }

# Run the large-scale dataset creation
if __name__ == "__main__":
    # Generate large dataset: 50,000 users, 3 sequences each = 150,000 total sequences
    result = create_large_biomechanics_dataset(
        total_users=50000,
        sequences_per_user=3,
        chunk_size=1000  # Process 1,000 users at a time
    )