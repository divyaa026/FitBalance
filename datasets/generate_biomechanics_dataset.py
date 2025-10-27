"""
Production-Ready Biomechanics Dataset Generator
Generates comprehensive synthetic training data for GNN-LSTM models
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import cv2
from typing import List, Dict, Tuple
import mediapipe as mp

class BiomechanicsDatasetGenerator:
    """Generate synthetic biomechanics training data"""
    
    def __init__(self, output_dir: str = "datasets/biomechanics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        
        # MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Exercise types
        self.exercises = ['squat', 'deadlift', 'bench_press', 'overhead_press', 'lunge']
        
        # Form quality levels
        self.form_qualities = ['excellent', 'good', 'fair', 'poor']
        
    def generate_pose_sequence(self, exercise: str, form_quality: str, 
                               frames: int = 30) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic pose sequence for an exercise"""
        
        # Base joint angles for each exercise type
        if exercise == 'squat':
            base_angles = self._generate_squat_sequence(frames, form_quality)
        elif exercise == 'deadlift':
            base_angles = self._generate_deadlift_sequence(frames, form_quality)
        elif exercise == 'bench_press':
            base_angles = self._generate_bench_press_sequence(frames, form_quality)
        elif exercise == 'overhead_press':
            base_angles = self._generate_overhead_press_sequence(frames, form_quality)
        else:  # lunge
            base_angles = self._generate_lunge_sequence(frames, form_quality)
        
        # Convert angles to 3D landmarks
        landmarks_sequence = []
        for frame_angles in base_angles:
            landmarks = self._angles_to_landmarks(frame_angles, exercise)
            landmarks_sequence.append(landmarks)
        
        # Generate ground truth labels
        labels = self._generate_labels(exercise, form_quality, base_angles)
        
        return np.array(landmarks_sequence), labels
    
    def _generate_squat_sequence(self, frames: int, quality: str) -> List[Dict]:
        """Generate squat movement sequence"""
        sequence = []
        
        # Squat phases: standing -> descending -> bottom -> ascending -> standing
        phase_frames = frames // 4
        
        for i in range(frames):
            phase = i // phase_frames
            progress = (i % phase_frames) / phase_frames
            
            angles = {
                'hip_angle': 180,
                'knee_angle': 180,
                'ankle_angle': 90,
                'back_angle': 0,
                'shoulder_angle': 45
            }
            
            if phase == 0:  # Standing
                pass  # Keep default angles
            elif phase == 1:  # Descending
                angles['hip_angle'] = 180 - (90 * progress)
                angles['knee_angle'] = 180 - (90 * progress)
                angles['ankle_angle'] = 90 - (20 * progress)
            elif phase == 2:  # Bottom
                angles['hip_angle'] = 90
                angles['knee_angle'] = 90
                angles['ankle_angle'] = 70
            else:  # Ascending
                angles['hip_angle'] = 90 + (90 * progress)
                angles['knee_angle'] = 90 + (90 * progress)
                angles['ankle_angle'] = 70 + (20 * progress)
            
            # Add form quality variations
            if quality == 'poor':
                angles['back_angle'] += np.random.uniform(10, 30)  # Excessive forward lean
                angles['knee_angle'] += np.random.uniform(-15, -5)  # Knees cave in
            elif quality == 'fair':
                angles['back_angle'] += np.random.uniform(5, 15)
                angles['knee_angle'] += np.random.uniform(-10, 0)
            elif quality == 'good':
                angles['back_angle'] += np.random.uniform(0, 5)
            
            # Add natural variation
            for key in angles:
                angles[key] += np.random.normal(0, 2)
            
            sequence.append(angles)
        
        return sequence
    
    def _generate_deadlift_sequence(self, frames: int, quality: str) -> List[Dict]:
        """Generate deadlift movement sequence"""
        sequence = []
        phase_frames = frames // 3
        
        for i in range(frames):
            phase = i // phase_frames
            progress = (i % phase_frames) / phase_frames
            
            angles = {
                'hip_angle': 45,
                'knee_angle': 90,
                'ankle_angle': 90,
                'back_angle': 0,
                'shoulder_angle': 90
            }
            
            if phase == 0:  # Starting position
                pass
            elif phase == 1:  # Lifting
                angles['hip_angle'] = 45 + (135 * progress)
                angles['knee_angle'] = 90 + (90 * progress)
            else:  # Lowering
                angles['hip_angle'] = 180 - (135 * progress)
                angles['knee_angle'] = 180 - (90 * progress)
            
            # Add form quality variations
            if quality == 'poor':
                angles['back_angle'] += np.random.uniform(20, 40)  # Rounded back
            elif quality == 'fair':
                angles['back_angle'] += np.random.uniform(10, 20)
            elif quality == 'good':
                angles['back_angle'] += np.random.uniform(0, 10)
            
            for key in angles:
                angles[key] += np.random.normal(0, 2)
            
            sequence.append(angles)
        
        return sequence
    
    def _generate_bench_press_sequence(self, frames: int, quality: str) -> List[Dict]:
        """Generate bench press movement sequence"""
        sequence = []
        phase_frames = frames // 2
        
        for i in range(frames):
            phase = i // phase_frames
            progress = (i % phase_frames) / phase_frames
            
            angles = {
                'shoulder_angle': 90,
                'elbow_angle': 90,
                'wrist_angle': 180,
                'back_angle': 0,
                'hip_angle': 90
            }
            
            if phase == 0:  # Lowering
                angles['elbow_angle'] = 90 - (45 * progress)
            else:  # Pressing
                angles['elbow_angle'] = 45 + (45 * progress)
            
            # Form quality variations
            if quality == 'poor':
                angles['back_angle'] += np.random.uniform(15, 30)  # Arched back
                angles['elbow_angle'] += np.random.uniform(-20, -10)  # Flared elbows
            elif quality == 'fair':
                angles['back_angle'] += np.random.uniform(5, 15)
                angles['elbow_angle'] += np.random.uniform(-10, 0)
            
            for key in angles:
                angles[key] += np.random.normal(0, 2)
            
            sequence.append(angles)
        
        return sequence
    
    def _generate_overhead_press_sequence(self, frames: int, quality: str) -> List[Dict]:
        """Generate overhead press movement sequence"""
        sequence = []
        phase_frames = frames // 2
        
        for i in range(frames):
            phase = i // phase_frames
            progress = (i % phase_frames) / phase_frames
            
            angles = {
                'shoulder_angle': 90,
                'elbow_angle': 90,
                'wrist_angle': 180,
                'back_angle': 0,
                'hip_angle': 180
            }
            
            if phase == 0:  # Pressing
                angles['shoulder_angle'] = 90 + (90 * progress)
                angles['elbow_angle'] = 90 + (90 * progress)
            else:  # Lowering
                angles['shoulder_angle'] = 180 - (90 * progress)
                angles['elbow_angle'] = 180 - (90 * progress)
            
            if quality == 'poor':
                angles['back_angle'] += np.random.uniform(15, 30)  # Excessive lean
            elif quality == 'fair':
                angles['back_angle'] += np.random.uniform(5, 15)
            
            for key in angles:
                angles[key] += np.random.normal(0, 2)
            
            sequence.append(angles)
        
        return sequence
    
    def _generate_lunge_sequence(self, frames: int, quality: str) -> List[Dict]:
        """Generate lunge movement sequence"""
        sequence = []
        phase_frames = frames // 4
        
        for i in range(frames):
            phase = i // phase_frames
            progress = (i % phase_frames) / phase_frames
            
            angles = {
                'hip_angle_front': 180,
                'knee_angle_front': 180,
                'hip_angle_back': 180,
                'knee_angle_back': 180,
                'back_angle': 0
            }
            
            if phase == 0:  # Standing
                pass
            elif phase == 1:  # Stepping forward
                angles['hip_angle_front'] = 180 - (90 * progress)
                angles['knee_angle_front'] = 180 - (90 * progress)
                angles['knee_angle_back'] = 180 - (90 * progress)
            elif phase == 2:  # Bottom
                angles['hip_angle_front'] = 90
                angles['knee_angle_front'] = 90
                angles['knee_angle_back'] = 90
            else:  # Returning
                angles['hip_angle_front'] = 90 + (90 * progress)
                angles['knee_angle_front'] = 90 + (90 * progress)
                angles['knee_angle_back'] = 90 + (90 * progress)
            
            if quality == 'poor':
                angles['back_angle'] += np.random.uniform(15, 30)
                angles['knee_angle_front'] += np.random.uniform(-20, -10)  # Knee past toes
            elif quality == 'fair':
                angles['back_angle'] += np.random.uniform(5, 15)
            
            for key in angles:
                angles[key] += np.random.normal(0, 2)
            
            sequence.append(angles)
        
        return sequence
    
    def _angles_to_landmarks(self, angles: Dict, exercise: str) -> np.ndarray:
        """Convert joint angles to 3D landmark positions (33 MediaPipe landmarks)"""
        
        landmarks = np.zeros((33, 3))
        
        # Generate realistic landmark positions based on angles
        # This is a simplified model - real implementation would use inverse kinematics
        
        # Torso landmarks (0-10)
        landmarks[0] = [0.5, 0.1, 0]  # Nose
        landmarks[11] = [0.4, 0.5, 0]  # Left shoulder
        landmarks[12] = [0.6, 0.5, 0]  # Right shoulder
        landmarks[23] = [0.45, 0.8, 0]  # Left hip
        landmarks[24] = [0.55, 0.8, 0]  # Right hip
        
        # Calculate arm positions based on shoulder angles
        if 'shoulder_angle' in angles:
            angle_rad = np.radians(angles['shoulder_angle'])
            landmarks[13] = [0.4, 0.5 + 0.2 * np.sin(angle_rad), 0.2 * np.cos(angle_rad)]  # Left elbow
            landmarks[14] = [0.6, 0.5 + 0.2 * np.sin(angle_rad), 0.2 * np.cos(angle_rad)]  # Right elbow
        
        # Calculate leg positions based on hip and knee angles
        if 'hip_angle' in angles and 'knee_angle' in angles:
            hip_rad = np.radians(180 - angles['hip_angle'])
            knee_rad = np.radians(180 - angles['knee_angle'])
            
            landmarks[25] = [0.45, 0.8 + 0.3 * np.sin(hip_rad), 0]  # Left knee
            landmarks[26] = [0.55, 0.8 + 0.3 * np.sin(hip_rad), 0]  # Right knee
            landmarks[27] = [0.45, 0.8 + 0.3 * np.sin(hip_rad) + 0.3 * np.sin(knee_rad), 0]  # Left ankle
            landmarks[28] = [0.55, 0.8 + 0.3 * np.sin(hip_rad) + 0.3 * np.sin(knee_rad), 0]  # Right ankle
        
        # Add noise for realism
        landmarks += np.random.normal(0, 0.01, landmarks.shape)
        
        return landmarks
    
    def _generate_labels(self, exercise: str, quality: str, sequence: List[Dict]) -> Dict:
        """Generate ground truth labels for training"""
        
        # Calculate form score based on quality
        quality_scores = {
            'excellent': np.random.uniform(90, 100),
            'good': np.random.uniform(75, 89),
            'fair': np.random.uniform(60, 74),
            'poor': np.random.uniform(30, 59)
        }
        
        form_score = quality_scores[quality]
        
        # Identify risk factors based on quality
        risk_factors = []
        if quality == 'poor':
            risk_factors = ['excessive_forward_lean', 'knee_valgus', 'insufficient_depth']
        elif quality == 'fair':
            risk_factors = ['moderate_forward_lean', 'minor_form_deviation']
        
        # Generate recommendations
        recommendations = []
        if 'excessive_forward_lean' in risk_factors:
            recommendations.append("Keep your chest up and maintain a neutral spine")
        if 'knee_valgus' in risk_factors:
            recommendations.append("Push your knees outward to prevent inward collapse")
        if 'insufficient_depth' in risk_factors:
            recommendations.append("Descend until thighs are parallel to the ground")
        
        # Extract joint angles from sequence
        joint_angles = {}
        for key in sequence[0].keys():
            joint_angles[key] = [frame[key] for frame in sequence]
        
        return {
            'exercise_type': exercise,
            'form_score': form_score,
            'form_quality': quality,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'joint_angles': joint_angles,
            'num_frames': len(sequence)
        }
    
    def generate_dataset(self, samples_per_exercise: int = 200, frames_per_sample: int = 30) -> pd.DataFrame:
        """Generate complete biomechanics dataset"""
        
        print(f"Generating biomechanics dataset with {samples_per_exercise * len(self.exercises)} samples...")
        
        all_samples = []
        sample_id = 0
        
        for exercise in self.exercises:
            print(f"  Generating {exercise} samples...")
            
            for quality in self.form_qualities:
                num_samples = samples_per_exercise // len(self.form_qualities)
                
                for i in range(num_samples):
                    # Generate pose sequence
                    landmarks_seq, labels = self.generate_pose_sequence(
                        exercise, quality, frames_per_sample
                    )
                    
                    # Save landmarks and labels
                    sample_data = {
                        'sample_id': sample_id,
                        'exercise_type': exercise,
                        'form_quality': quality,
                        'form_score': labels['form_score'],
                        'num_frames': frames_per_sample,
                        'risk_factors': ','.join(labels['risk_factors']),
                        'recommendations': '|'.join(labels['recommendations'])
                    }
                    
                    # Add average joint angles
                    for joint, angles in labels['joint_angles'].items():
                        sample_data[f'{joint}_mean'] = np.mean(angles)
                        sample_data[f'{joint}_std'] = np.std(angles)
                    
                    # Save landmarks to file
                    landmarks_file = f"{self.output_dir}/annotations/sample_{sample_id:06d}.npy"
                    np.save(landmarks_file, landmarks_seq)
                    sample_data['landmarks_file'] = landmarks_file
                    
                    # Save labels to JSON
                    labels_file = f"{self.output_dir}/annotations/sample_{sample_id:06d}.json"
                    with open(labels_file, 'w') as f:
                        json.dump(labels, f, indent=2)
                    sample_data['labels_file'] = labels_file
                    
                    all_samples.append(sample_data)
                    sample_id += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"    Generated {i + 1}/{num_samples} {quality} samples")
        
        # Create DataFrame
        df = pd.DataFrame(all_samples)
        
        # Save dataset
        csv_path = f"{self.output_dir}/biomechanics_dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Dataset saved to {csv_path}")
        print(f"   Total samples: {len(df)}")
        print(f"   Exercises: {', '.join(self.exercises)}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        print(df.groupby(['exercise_type', 'form_quality']).size())
        
        return df

def main():
    """Generate biomechanics training dataset"""
    
    generator = BiomechanicsDatasetGenerator()
    
    # Generate dataset with 1000 samples (200 per exercise)
    dataset = generator.generate_dataset(samples_per_exercise=200, frames_per_sample=30)
    
    print("\n✅ Biomechanics dataset generation complete!")
    print(f"📁 Location: {generator.output_dir}")
    print(f"📊 Total samples: {len(dataset)}")

if __name__ == "__main__":
    main()
