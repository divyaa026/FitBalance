#!/usr/bin/env python3
"""
Quick test script to verify the nutrition API fix
"""
import requests
import io
from PIL import Image

def create_test_image():
    """Create a simple test image for API testing"""
    # Create a simple RGB image
    img = Image.new('RGB', (300, 300), color='red')
    
    # Save to bytes buffer
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_nutrition_api():
    """Test the nutrition analyze-meal endpoint"""
    
    # Create test image
    test_image = create_test_image()
    
    # Prepare form data
    files = {
        'meal_photo': ('test_meal.jpg', test_image, 'image/jpeg')
    }
    
    data = {
        'user_id': 'test_user',
        'dietary_restrictions': '[]'
    }
    
    try:
        print("Testing nutrition API endpoint...")
        response = requests.post(
            'http://localhost:8000/nutrition/analyze-meal',
            files=files,
            data=data,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: API endpoint working correctly!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_nutrition_api()