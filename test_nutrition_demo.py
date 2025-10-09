"""
Simple test script for the nutrition demo system
"""

import requests
import json
import tempfile
from PIL import Image, ImageDraw
import io
import os

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_123"

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw simple food-like shapes
    draw.ellipse([50, 50, 100, 100], fill='brown')  # Protein
    draw.rectangle([120, 60, 170, 110], fill='green')  # Vegetable
    draw.ellipse([60, 140, 110, 190], fill='yellow')  # Grain
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name, 'JPEG')
    return temp_file.name

def test_health_check():
    """Test the health check endpoint"""
    print("🧪 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_meal_analysis():
    """Test meal photo analysis"""
    print("\n🍽️ Testing meal analysis...")
    try:
        # Create test image
        test_image_path = create_test_image()
        
        with open(test_image_path, 'rb') as f:
            files = {'meal_photo': ('test_meal.jpg', f, 'image/jpeg')}
            data = {'user_id': TEST_USER_ID}
            
            response = requests.post(f"{BASE_URL}/nutrition/analyze-meal", files=files, data=data)
        
        # Clean up
        os.unlink(test_image_path)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Meal analysis passed")
            print(f"   Total protein: {result['total_protein']}g")
            print(f"   Target protein: {result['target_protein']}g")
            print(f"   Meal quality score: {result['meal_quality_score']}")
            print(f"   Detected foods: {len(result['detected_foods'])}")
            print(f"   Recommendations: {len(result['recommendations'])}")
            return True
        else:
            print(f"❌ Meal analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Meal analysis error: {str(e)}")
        return False

def test_recommendations():
    """Test nutrition recommendations"""
    print("\n📊 Testing nutrition recommendations...")
    try:
        response = requests.get(f"{BASE_URL}/nutrition/recommendations/{TEST_USER_ID}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Recommendations passed")
            print(f"   Recommended protein: {result['recommended_protein']}g")
            print(f"   Baseline protein: {result['baseline_protein']}g")
            print(f"   Sleep factor: {result['optimization_factors']['sleep_factor']}")
            print(f"   Activity factor: {result['optimization_factors']['activity_factor']}")
            print(f"   Recommendations: {len(result['recommendations'])}")
            return True
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Recommendations error: {str(e)}")
        return False

def test_health_data_update():
    """Test health data update"""
    print("\n💓 Testing health data update...")
    try:
        health_data = {
            "sleep_quality": 7.5,
            "activity_level": 8.0,
            "stress_level": 4.0,
            "hrv": 55.0
        }
        
        response = requests.post(f"{BASE_URL}/nutrition/health-data/{TEST_USER_ID}", json=health_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health data update passed")
            print(f"   Status: {result['status']}")
            print(f"   Updated fields: {result['updated_fields']}")
            return True
        else:
            print(f"❌ Health data update failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health data update error: {str(e)}")
        return False

def test_food_database():
    """Test food database endpoint"""
    print("\n🥗 Testing food database...")
    try:
        response = requests.get(f"{BASE_URL}/nutrition/food-database")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Food database passed")
            print(f"   Total foods: {result['total_foods']}")
            print(f"   Categories: {result['categories']}")
            return True
        else:
            print(f"❌ Food database failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Food database error: {str(e)}")
        return False

def test_demo_explanation():
    """Test demo explanation endpoint"""
    print("\n🧠 Testing demo explanation...")
    try:
        response = requests.get(f"{BASE_URL}/nutrition/demo-explanation/{TEST_USER_ID}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Demo explanation passed")
            print(f"   Confidence score: {result['confidence_score']}")
            print(f"   Predicted value: {result['predicted_value']}g")
            print(f"   Feature importance items: {len(result['feature_importance'])}")
            return True
        else:
            print(f"❌ Demo explanation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Demo explanation error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🎯 FitBalance Nutrition Demo - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Meal Analysis", test_meal_analysis),
        ("Recommendations", test_recommendations),
        ("Health Data Update", test_health_data_update),
        ("Food Database", test_food_database),
        ("Demo Explanation", test_demo_explanation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The nutrition demo system is working correctly.")
        print("\n📖 Try the interactive API docs at: http://localhost:8000/docs")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the server logs for details.")

if __name__ == "__main__":
    print("⚠️  Make sure the server is running first:")
    print("   python start_nutrition_demo.py")
    print("\nStarting tests in 3 seconds...")
    
    import time
    time.sleep(3)
    
    main()