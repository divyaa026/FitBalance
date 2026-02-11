#!/usr/bin/env python3
"""
Full-scale biomechanics dataset generation: 50,000 users
This will generate the complete dataset for production use
"""

import sys
import os
import time
sys.path.append('.')

from generate_biomechanics_large import create_large_biomechanics_dataset

def run_full_scale_generation():
    """Generate the full 50,000 user dataset"""
    print("🚀 FitBalance Production Dataset Generation")
    print("=" * 60)
    print("📊 Target: 50,000 users, 150,000 sequences")
    print("⏱️  Estimated time: 15-20 minutes")
    print("💾 Expected output size: ~12-15 GB JSON, ~20 MB CSV")
    print()
    
    print("🏁 Starting full-scale generation...")
    start_time = time.time()
    
    # Clean up test files first
    test_files = ['biomechanics_dataset_large.json', 'biomechanics_dataset_large.csv']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Removed old test file: {file}")
    
    # Generate the full dataset
    result = create_large_biomechanics_dataset(
        total_users=50000,     # Full 50K users
        sequences_per_user=3,   # 3 sequences each = 150K total
        chunk_size=1000        # Process 1,000 users at a time for efficiency
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 PRODUCTION DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successfully generated {result['total_users']:,} users")
    print(f"✅ Successfully generated {result['total_sequences']:,} sequences") 
    print(f"✅ Processed {result['chunks_processed']} chunks")
    print(f"📈 Generation rate: {result['total_sequences']/(total_time/60):.0f} sequences/minute")
    
    # Final validation
    import pandas as pd
    if os.path.exists('biomechanics_dataset_large.csv'):
        print(f"\n📊 Final Dataset Validation:")
        df = pd.read_csv('biomechanics_dataset_large.csv')
        print(f"   • CSV rows: {len(df):,}")
        print(f"   • Unique users: {df['user_id'].nunique():,}")
        print(f"   • Exercise types: {', '.join(df['exercise_type'].unique())}")
        
        print(f"\n🎯 Form Quality Distribution:")
        for quality, count in df['form_quality'].value_counts().items():
            print(f"   • {quality}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\n🏋️ Exercise Distribution:")
        for exercise, count in df['exercise_type'].value_counts().items():
            print(f"   • {exercise}: {count:,}")
        
        # File sizes
        json_size = os.path.getsize('biomechanics_dataset_large.json') / 1024 / 1024
        csv_size = os.path.getsize('biomechanics_dataset_large.csv') / 1024 / 1024
        
        print(f"\n📁 Output Files:")
        print(f"   • biomechanics_dataset_large.json: {json_size:.1f} MB")
        print(f"   • biomechanics_dataset_large.csv: {csv_size:.1f} MB")
        
        print(f"\n🎉 Production dataset ready for ML training!")
        print(f"💡 Use the CSV for quick analysis and JSON for full 3D coordinate data")

if __name__ == "__main__":
    run_full_scale_generation()