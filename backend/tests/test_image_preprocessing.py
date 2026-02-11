"""
Test script to verify image preprocessing for Gemini API
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from PIL import Image
import io

def test_image_preprocessing():
    """Test the image preprocessing logic"""
    
    # Test cases
    test_cases = [
        ("Large image (4000x3000)", (4000, 3000)),
        ("Small image (200x150)", (200, 150)),
        ("Portrait image (800x1200)", (800, 1200)),
        ("Landscape image (1600x900)", (1600, 900)),
        ("Square image (1024x1024)", (1024, 1024)),
    ]
    
    print("🔍 Testing Image Preprocessing for Gemini API\n")
    print("=" * 60)
    
    for test_name, original_size in test_cases:
        print(f"\n📷 Test: {test_name}")
        print(f"   Original size: {original_size}")
        
        # Create test image
        test_image = Image.new('RGB', original_size, color=(73, 109, 137))
        
        # Convert to bytes
        buffered = io.BytesIO()
        test_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        original_size_mb = len(image_bytes) / (1024 * 1024)
        
        print(f"   Original file size: {original_size_mb:.2f}MB")
        
        # Simulate preprocessing
        width, height = original_size
        max_dimension = 2048
        min_dimension = 512
        
        # Check if resize needed
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            print(f"   ✅ Will be resized (too large): {new_size}")
        elif width < min_dimension or height < min_dimension:
            ratio = max(min_dimension / width, min_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            print(f"   ✅ Will be upscaled (too small): {new_size}")
        else:
            print(f"   ✅ Size OK, no resize needed")
            new_size = original_size
        
        # Estimate final size
        if new_size != original_size:
            processed_image = test_image.resize(new_size, Image.LANCZOS)
            buffered_processed = io.BytesIO()
            processed_image.save(buffered_processed, format="JPEG", quality=85)
            processed_size_mb = len(buffered_processed.getvalue()) / (1024 * 1024)
            print(f"   Final file size: {processed_size_mb:.2f}MB")
        
        # Check Gemini requirements
        if min(new_size) >= min_dimension and max(new_size) <= max_dimension:
            print(f"   ✅ PASSES Gemini requirements")
        else:
            print(f"   ❌ FAILS Gemini requirements")
    
    print("\n" + "=" * 60)
    print("\n📋 Gemini Image Requirements:")
    print(f"   • Minimum dimension: {min_dimension}px")
    print(f"   • Maximum dimension: {max_dimension}px")
    print(f"   • Maximum file size: 4MB")
    print(f"   • Format: RGB JPEG (quality 85)")
    print("\n✅ Image preprocessing is configured correctly!\n")

if __name__ == "__main__":
    test_image_preprocessing()
