# -*- coding: utf-8 -*-
"""
Public Fitness Dataset Integration and Download
Integrates multiple public datasets for biomechanics training
"""

import os
import requests
import zipfile
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gdown
from typing import List, Dict
import pandas as pd

class FitnessDatasetDownloader:
    """Download and integrate public fitness datasets"""
    
    def __init__(self, output_dir: str = "datasets/biomechanics_real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset sources
        self.datasets = {
            'fit_aqva': {
                'name': 'Fitness-AQA Dataset',
                'description': 'Action Quality Assessment for Fitness',
                'url': 'https://drive.google.com/uc?id=1Xs6mTnJLDnUJnm8bWhX5RgIlDxK-x8iE',
                'video_count': 500,
                'exercises': ['squat', 'pushup']
            },
            'countix': {
                'name': 'Countix Repetition Counting Dataset',
                'description': 'Exercise repetition counting videos',
                'url': 'https://drive.google.com/uc?id=1xK8cK-q9eWgUZBnqFwfKJdLECDNKQsGh',
                'video_count': 390,
                'exercises': ['squat', 'pushup', 'pullup', 'situp']
            },
            'repnet': {
                'name': 'RepNet Dataset',
                'description': 'Repetitive action counting',
                'url': 'https://sites.google.com/view/repnet',
                'video_count': 250,
                'exercises': ['squat', 'bench_press', 'deadlift']
            },
            'home_workout': {
                'name': 'Home Workout Dataset',
                'description': 'Home-based exercises',
                'url': 'https://drive.google.com/uc?id=1K3DzpHgL8FGJ6MeZ6P7C8w9vQqN8tYkL',
                'video_count': 300,
                'exercises': ['squat', 'lunge', 'pushup', 'plank']
            }
        }
    
    def download_all_datasets(self):
        """Download all available datasets"""
        print(" Starting Public Fitness Dataset Download")
        print("=" * 70)
        
        for dataset_id, dataset_info in self.datasets.items():
            print(f"\n Downloading: {dataset_info['name']}")
            print(f"   Description: {dataset_info['description']}")
            print(f"   Videos: {dataset_info['video_count']}")
            print(f"   Exercises: {', '.join(dataset_info['exercises'])}")
            
            try:
                self.download_dataset(dataset_id, dataset_info)
            except Exception as e:
                print(f"     Failed to download {dataset_info['name']}: {e}")
                print(f"   Continuing with other datasets...")
        
        print("\n Dataset download complete!")
        self.generate_summary()
    
    def download_dataset(self, dataset_id: str, dataset_info: Dict):
        """Download a specific dataset"""
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / 'videos').exists():
            print(f"    Already downloaded, skipping...")
            return
        
        # Download based on dataset type
        if 'google.com/uc' in dataset_info['url']:
            self._download_from_google_drive(dataset_info['url'], dataset_dir)
        else:
            print(f"     Manual download required from: {dataset_info['url']}")
            self._create_download_instructions(dataset_id, dataset_info)
    
    def _download_from_google_drive(self, url: str, output_dir: Path):
        """Download from Google Drive"""
        try:
            print(f"   Downloading from Google Drive...")
            output_zip = output_dir / "dataset.zip"
            
            # Use gdown for Google Drive downloads
            gdown.download(url, str(output_zip), quiet=False)
            
            # Extract
            print(f"   Extracting...")
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove zip
            output_zip.unlink()
            
            print(f"    Downloaded and extracted")
            
        except Exception as e:
            print(f"    Download failed: {e}")
            raise
    
    def _create_download_instructions(self, dataset_id: str, dataset_info: Dict):
        """Create manual download instructions"""
        instructions_file = self.output_dir / dataset_id / "DOWNLOAD_INSTRUCTIONS.txt"
        instructions_file.parent.mkdir(exist_ok=True)
        
        instructions = f"""
Manual Download Required for {dataset_info['name']}
{'=' * 70}

Dataset: {dataset_info['name']}
Description: {dataset_info['description']}
Video Count: {dataset_info['video_count']}
Exercises: {', '.join(dataset_info['exercises'])}

Instructions:
1. Visit: {dataset_info['url']}
2. Download the dataset
3. Extract to: {instructions_file.parent / 'videos'}
4. Run the preprocessing script: python datasets/preprocess_public_datasets.py

After manual download, your directory structure should be:
{instructions_file.parent}/
 videos/
    exercise1/
       video001.mp4
       video002.mp4
       ...
    exercise2/
        ...
 labels.json (if available)
"""
        
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"     Created download instructions: {instructions_file}")
    
    def download_sample_videos_from_youtube(self):
        """Download sample exercise videos from YouTube (using yt-dlp)"""
        print("\n Downloading Sample Videos from YouTube...")
        
        # Exercise form video IDs (example high-quality form videos)
        sample_videos = {
            'squat': [
                'https://www.youtube.com/watch?v=ultWZbUMPL8',  # Squat form
                'https://www.youtube.com/watch?v=YaXPRqUwItQ',  # Deep squat
            ],
            'deadlift': [
                'https://www.youtube.com/watch?v=op9kVnSso6Q',  # Deadlift form
                'https://www.youtube.com/watch?v=XxWcirHIwVo',  # Deadlift tutorial
            ],
            'bench_press': [
                'https://www.youtube.com/watch?v=rT7DgCr-3pg',  # Bench press
                'https://www.youtube.com/watch?v=BYKScL2sgCs',  # Bench technique
            ],
            'overhead_press': [
                'https://www.youtube.com/watch?v=2yjwXTZQDDI',  # OHP form
            ],
            'lunge': [
                'https://www.youtube.com/watch?v=QOVaHwm-Q6U',  # Lunge tutorial
            ]
        }
        
        try:
            import yt_dlp
            
            for exercise, urls in sample_videos.items():
                exercise_dir = self.output_dir / 'youtube_samples' / exercise
                exercise_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\n  Downloading {exercise} videos...")
                
                for idx, url in enumerate(urls):
                    try:
                        ydl_opts = {
                            'format': 'best[height<=720]',
                            'outtmpl': str(exercise_dir / f'{exercise}_{idx+1}.%(ext)s'),
                            'quiet': True,
                            'no_warnings': True,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])
                        
                        print(f"     Downloaded {exercise}_{idx+1}")
                    except Exception as e:
                        print(f"      Failed to download {url}: {e}")
            
            print("\n   YouTube sample videos downloaded")
            
        except ImportError:
            print("\n    yt-dlp not installed. Install with: pip install yt-dlp")
            print("  Skipping YouTube downloads...")
    
    def download_fitclic_dataset(self):
        """Download FitClic dataset - 5000+ annotated fitness videos"""
        print("\n Downloading FitClic Dataset (5000+ videos)...")
        
        fitclic_dir = self.output_dir / 'fitclic'
        fitclic_dir.mkdir(exist_ok=True)
        
        # FitClic dataset annotations URL
        annotations_url = "https://github.com/fitzlab-ml/fitclic/raw/main/annotations.json"
        
        try:
            print("  Downloading annotations...")
            response = requests.get(annotations_url, timeout=30)
            annotations = response.json()
            
            # Save annotations
            with open(fitclic_dir / 'annotations.json', 'w') as f:
                json.dump(annotations, f, indent=2)
            
            print(f"   Downloaded {len(annotations)} video annotations")
            
            # Create download script for videos
            self._create_fitclic_download_script(fitclic_dir, annotations)
            
        except Exception as e:
            print(f"    Failed to download FitClic: {e}")
            print(f"  Visit: https://github.com/fitzlab-ml/fitclic for manual download")
    
    def _create_fitclic_download_script(self, output_dir: Path, annotations: dict):
        """Create script to download FitClic videos"""
        script = """#!/usr/bin/env python3
\"\"\"
FitClic Video Downloader
Downloads videos from FitClic dataset annotations
\"\"\"

import json
import subprocess
from pathlib import Path
from tqdm import tqdm

def download_fitclic_videos():
    with open('annotations.json', 'r') as f:
        annotations = json.load(f)
    
    videos_dir = Path('videos')
    videos_dir.mkdir(exist_ok=True)
    
    for video_id, data in tqdm(annotations.items(), desc="Downloading videos"):
        if 'url' not in data:
            continue
        
        url = data['url']
        exercise = data.get('exercise', 'unknown')
        
        # Create exercise directory
        exercise_dir = videos_dir / exercise
        exercise_dir.mkdir(exist_ok=True)
        
        # Download video
        output_file = exercise_dir / f"{video_id}.mp4"
        
        if output_file.exists():
            continue
        
        try:
            subprocess.run([
                'yt-dlp',
                '-f', 'best[height<=720]',
                '-o', str(output_file),
                url
            ], check=True, capture_output=True)
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")

if __name__ == '__main__':
    download_fitclic_videos()
"""
        
        script_path = output_dir / 'download_videos.py'
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"   Created download script: {script_path}")
        print(f"  Run with: cd {output_dir} && python download_videos.py")
    
    def generate_summary(self):
        """Generate summary of downloaded datasets"""
        print("\n" + "=" * 70)
        print(" Dataset Download Summary")
        print("=" * 70)
        
        total_videos = 0
        exercises_covered = set()
        
        summary_data = []
        
        for dataset_id, dataset_info in self.datasets.items():
            dataset_dir = self.output_dir / dataset_id
            
            if dataset_dir.exists():
                status = " Downloaded" if (dataset_dir / 'videos').exists() else " Pending"
            else:
                status = " Not downloaded"
            
            summary_data.append({
                'Dataset': dataset_info['name'],
                'Status': status,
                'Videos': dataset_info['video_count'],
                'Exercises': ', '.join(dataset_info['exercises'])
            })
            
            if status == " Downloaded":
                total_videos += dataset_info['video_count']
                exercises_covered.update(dataset_info['exercises'])
        
        # Print summary table
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        print(f"\n Total Videos Available: {total_videos:,}")
        print(f"  Exercises Covered: {', '.join(sorted(exercises_covered))}")
        
        # Save summary
        summary_file = self.output_dir / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'total_videos': total_videos,
                'exercises': sorted(list(exercises_covered)),
                'datasets': summary_data,
                'download_date': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        print(f"\n Summary saved to: {summary_file}")
        
        # Next steps
        print("\n" + "=" * 70)
        print(" Next Steps:")
        print("=" * 70)
        print("1. Review download instructions in each dataset folder")
        print("2. Manually download datasets that require it")
        print("3. Run preprocessing: python data/preprocess_public_datasets.py")
        print("4. Fine-tune model: python ml/biomechanics/fine_tune_model.py")
        print("=" * 70)

def main():
    """Main function"""
    print("""
========================================================================
                                                                    
        Public Fitness Dataset Integration                    
                                                                    
  Downloading and integrating public fitness datasets to achieve    
  90%+ accuracy on biomechanics analysis                           
                                                                    
========================================================================
    """)
    
    downloader = FitnessDatasetDownloader()
    
    # Download all datasets
    downloader.download_all_datasets()
    
    # Download YouTube samples
    print("\n" + "=" * 70)
    response = input("Download sample videos from YouTube? (requires yt-dlp) [y/N]: ")
    if response.lower() == 'y':
        downloader.download_sample_videos_from_youtube()
    
    # Download FitClic
    print("\n" + "=" * 70)
    response = input("Download FitClic dataset annotations? (5000+ videos) [y/N]: ")
    if response.lower() == 'y':
        downloader.download_fitclic_dataset()
    
    print("\n All downloads initiated!")

if __name__ == "__main__":
    main()
