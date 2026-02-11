"""
Preprocess Public Fitness Datasets
Convert downloaded videos to training format compatible with GNN-LSTM model
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FitnessDatasetPreprocessor:
    """Preprocess public fitness datasets for training"""
    
    def __init__(self, input_dir: str = "datasets/biomechanics_real",
                 output_dir: str = "datasets/biomechanics_processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise mapping
        self.exercise_mapping = {
            'squat': 'squat',
            'squats': 'squat',
            'deadlift': 'deadlift',
            'deadlifts': 'deadlift',
            'bench_press': 'bench_press',
            'bench': 'bench_press',
            'overhead_press': 'overhead_press',
            'ohp': 'overhead_press',
            'lunge': 'lunge',
            'lunges': 'lunge',
            'pushup': 'pushup',
            'push_up': 'pushup'
        }
        
        self.processed_count = 0
        self.failed_count = 0
    
    def process_all_datasets(self):
        """Process all downloaded datasets"""
        print(" Starting Dataset Preprocessing")
        print("=" * 70)
        
        # Find all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(self.input_dir.rglob(ext))
        
        logger.info(f"Found {len(video_files)} video files")
        
        if not video_files:
            logger.error("No video files found!")
            logger.info(f"Please ensure videos are in: {self.input_dir}")
            return
        
        # Process each video
        all_sequences = []
        metadata = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                sequences, video_metadata = self.process_video(video_path)
                
                if sequences:
                    all_sequences.extend(sequences)
                    metadata.append(video_metadata)
                    self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {video_path.name}: {e}")
                self.failed_count += 1
        
        # Save processed data
        if all_sequences:
            self.save_processed_data(all_sequences, metadata)
        
        # Print summary
        self.print_summary()
    
    def process_video(self, video_path: Path) -> Tuple[List[Dict], Dict]:
        """Process a single video file"""
        
        # Extract metadata from path
        exercise_type = self._extract_exercise_type(video_path)
        form_quality = self._extract_form_quality(video_path)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract pose landmarks for each frame
        frame_landmarks = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.mp_pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract 33 landmarks (x, y, z, visibility)
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                
                frame_landmarks.append({
                    'frame': frame_idx,
                    'landmarks': landmarks,
                    'timestamp': frame_idx / fps
                })
            
            frame_idx += 1
        
        cap.release()
        
        # Create sequences (60-frame windows with 30-frame stride)
        sequences = []
        window_size = 60
        stride = 30
        
        for start_idx in range(0, len(frame_landmarks) - window_size, stride):
            sequence_frames = frame_landmarks[start_idx:start_idx + window_size]
            
            # Extract landmark array
            landmark_array = np.array([f['landmarks'] for f in sequence_frames])
            
            sequences.append({
                'video_path': str(video_path),
                'exercise_type': exercise_type,
                'form_quality': form_quality,
                'start_frame': start_idx,
                'end_frame': start_idx + window_size,
                'landmarks': landmark_array,
                'sequence_length': window_size
            })
        
        # Metadata
        video_metadata = {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'exercise_type': exercise_type,
            'form_quality': form_quality,
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0,
            'sequences_extracted': len(sequences),
            'valid_pose_frames': len(frame_landmarks)
        }
        
        return sequences, video_metadata
    
    def _extract_exercise_type(self, video_path: Path) -> str:
        """Extract exercise type from video path or filename"""
        path_str = str(video_path).lower()
        
        for key, exercise in self.exercise_mapping.items():
            if key in path_str:
                return exercise
        
        # Check parent directory names
        for parent in video_path.parents:
            parent_name = parent.name.lower()
            for key, exercise in self.exercise_mapping.items():
                if key in parent_name:
                    return exercise
        
        return 'unknown'
    
    def _extract_form_quality(self, video_path: Path) -> str:
        """Extract form quality from video path or filename"""
        path_str = str(video_path).lower()
        
        quality_keywords = {
            'excellent': ['excellent', 'perfect', 'ideal', 'pro'],
            'good': ['good', 'correct', 'proper'],
            'fair': ['fair', 'ok', 'moderate', 'average'],
            'poor': ['poor', 'bad', 'wrong', 'incorrect', 'mistake']
        }
        
        for quality, keywords in quality_keywords.items():
            if any(kw in path_str for kw in keywords):
                return quality
        
        return 'unknown'
    
    def save_processed_data(self, sequences: List[Dict], metadata: List[Dict]):
        """Save processed data in training format"""
        logger.info("Saving processed data...")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.output_dir / 'video_metadata.csv', index=False)
        logger.info(f"Saved metadata for {len(metadata)} videos")
        
        # Save sequences
        sequences_data = []
        for idx, seq in enumerate(sequences):
            # Save landmark array
            landmark_file = self.output_dir / 'landmark_sequences' / f'sequence_{idx}.npy'
            landmark_file.parent.mkdir(exist_ok=True)
            np.save(landmark_file, seq['landmarks'])
            
            # Add to sequences metadata
            sequences_data.append({
                'sequence_id': idx,
                'video_path': seq['video_path'],
                'exercise_type': seq['exercise_type'],
                'form_quality': seq['form_quality'],
                'start_frame': seq['start_frame'],
                'end_frame': seq['end_frame'],
                'sequence_length': seq['sequence_length'],
                'landmark_file': str(landmark_file.relative_to(self.output_dir))
            })
        
        # Save sequences metadata
        sequences_df = pd.DataFrame(sequences_data)
        sequences_df.to_csv(self.output_dir / 'sequences_metadata.csv', index=False)
        logger.info(f"Saved {len(sequences)} sequences")
        
        # Save distribution statistics
        self._save_statistics(sequences_df)
    
    def _save_statistics(self, sequences_df: pd.DataFrame):
        """Save dataset statistics"""
        stats = {
            'total_sequences': len(sequences_df),
            'total_videos': self.processed_count,
            'failed_videos': self.failed_count,
            'exercise_distribution': sequences_df['exercise_type'].value_counts().to_dict(),
            'quality_distribution': sequences_df['form_quality'].value_counts().to_dict(),
            'sequences_per_video': len(sequences_df) / max(self.processed_count, 1)
        }
        
        with open(self.output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset statistics:")
        logger.info(f"  Total sequences: {stats['total_sequences']}")
        logger.info(f"  Exercise distribution: {stats['exercise_distribution']}")
        logger.info(f"  Quality distribution: {stats['quality_distribution']}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 70)
        print(" Preprocessing Summary")
        print("=" * 70)
        print(f" Successfully processed: {self.processed_count} videos")
        print(f" Failed: {self.failed_count} videos")
        print(f" Output directory: {self.output_dir}")
        print("=" * 70)
        
        if self.processed_count > 0:
            print("\n Next Steps:")
            print("1. Review processed data in:", self.output_dir)
            print("2. Combine with synthetic data:")
            print("   python data/combine_datasets.py")
            print("3. Fine-tune model:")
            print("   python ml/biomechanics/fine_tune_model.py")

def main():
    """Main function"""
    print("""
========================================================================
                                                                    
        Public Dataset Preprocessing                          
                                                                    
  Converting downloaded videos to training format using MediaPipe   
                                                                    
========================================================================
    """)
    
    preprocessor = FitnessDatasetPreprocessor()
    preprocessor.process_all_datasets()

if __name__ == "__main__":
    main()
