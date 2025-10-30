"""
Test all API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_nutrition_analyze():
    """Test meal analysis endpoint"""
    print("\n[TEST] Testing POST /nutrition/analyze...")
    
    # Use a test image (you'll need to provide one)
    try:
        with open("test_images/paneer_tikka.jpg", "rb") as f:
            files = {"meal_photo": f}
            data = {"user_id": "123", "dietary_restrictions": []}
            
            response = requests.post(f"{BASE_URL}/nutrition/analyze", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[SUCCESS] Analysis successful!")
                print(f"   Detected foods: {len(result['detected_foods'])}")
                print(f"   Total protein: {result['total_protein']}g")
                print(f"   Total calories: {result['total_calories']}")
            else:
                print(f"[ERROR] Failed: {response.status_code}")
                print(response.text)
    except FileNotFoundError:
        print("[INFO] Test image not found, skipping...")

def test_nutrition_history():
    """Test history endpoint"""
    print("\n[TEST] Testing GET /nutrition/history/123...")
    
    response = requests.get(f"{BASE_URL}/nutrition/history/123?days=7")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] History retrieved!")
        print(f"   Meals found: {result['meal_count']}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_nutrition_stats():
    """Test stats endpoint"""
    print("\n[TEST] Testing GET /nutrition/stats/123...")
    
    response = requests.get(f"{BASE_URL}/nutrition/stats/123?days=7")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Stats retrieved!")
        if 'total_meals' in result:
            print(f"   Total meals: {result['total_meals']}")
            print(f"   Total protein: {result['total_protein']}g")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_food_database():
    """Test food database endpoint"""
    print("\n[TEST] Testing GET /nutrition/foods...")
    
    response = requests.get(f"{BASE_URL}/nutrition/foods")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Food database retrieved!")
        print(f"   Total foods: {result['count']}")
        print(f"   Sample foods: {result['foods'][:5]}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_burnout_analyze():
    """Test burnout analysis"""
    print("\n[TEST] Testing POST /burnout/analyze...")
    
    params = {
        "user_id": "123",
        "workout_frequency": 5,
        "sleep_hours": 7.0,
        "stress_level": 6,
        "recovery_time": 1,
        "performance_trend": "stable"
    }
    
    response = requests.post(f"{BASE_URL}/burnout/analyze", params=params)
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Burnout analysis successful!")
        print(f"   Risk score: {result['risk_score']}")
        print(f"   Risk level: {result['risk_level']}")
        print(f"   Time to burnout: {result.get('time_to_burnout', 'N/A')} days")
    else:
        print(f"[ERROR] Failed: {response.status_code}")
        print(response.text)

def test_burnout_survival_curve():
    """Test burnout survival curve"""
    print("\n[TEST] Testing GET /burnout/survival-curve/123...")
    
    response = requests.get(f"{BASE_URL}/burnout/survival-curve/123")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Survival curve retrieved!")
        print(f"   User ID: {result.get('user_id', 'N/A')}")
        print(f"   Risk score: {result.get('risk_score', 'N/A')}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_burnout_recommendations():
    """Test burnout recommendations"""
    print("\n[TEST] Testing GET /burnout/recommendations/123...")
    
    response = requests.get(f"{BASE_URL}/burnout/recommendations/123")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Recommendations retrieved!")
        print(f"   Recommendations count: {len(result.get('recommendations', []))}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_biomechanics_analyze():
    """Test biomechanics analysis"""
    print("\n[TEST] Testing POST /biomechanics/analyze...")
    
    # Create a dummy video file for testing
    try:
        with open("test_video.mp4", "rb") as f:
            files = {"video_file": f}
            data = {"exercise_type": "squat", "user_id": "123"}
            
            response = requests.post(f"{BASE_URL}/biomechanics/analyze", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[SUCCESS] Biomechanics analysis successful!")
                print(f"   Form score: {result.get('form_score', 'N/A')}")
                print(f"   Joint angles: {len(result.get('joint_angles', []))}")
            else:
                print(f"[ERROR] Failed: {response.status_code}")
    except FileNotFoundError:
        print("[INFO] Test video not found, skipping...")

def test_biomechanics_heatmap():
    """Test biomechanics heatmap"""
    print("\n[TEST] Testing GET /biomechanics/heatmap/123...")
    
    response = requests.get(f"{BASE_URL}/biomechanics/heatmap/123?exercise_type=squat")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Heatmap retrieved!")
        print(f"   User ID: {result.get('user_id', 'N/A')}")
        print(f"   Exercise: {result.get('exercise', 'N/A')}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_root_endpoint():
    """Test root endpoint"""
    print("\n[TEST] Testing GET /...")
    
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Root endpoint working!")
        print(f"   Message: {result.get('message', 'N/A')}")
        print(f"   Modules: {result.get('modules', [])}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_health_endpoint():
    """Test health endpoint"""
    print("\n[TEST] Testing GET /health...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Health endpoint working!")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Modules: {result.get('modules', {})}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

def test_nutrition_recommendations():
    """Test nutrition recommendations"""
    print("\n[TEST] Testing GET /nutrition/recommendations/123...")
    
    response = requests.get(f"{BASE_URL}/nutrition/recommendations/123?target_protein=120&activity_level=moderate")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[SUCCESS] Nutrition recommendations retrieved!")
        print(f"   Target protein: {result.get('target_protein', 'N/A')}")
        print(f"   Activity level: {result.get('activity_level', 'N/A')}")
        print(f"   Recommendations count: {len(result.get('recommendations', []))}")
    else:
        print(f"[ERROR] Failed: {response.status_code}")

if __name__ == "__main__":
    print("[TEST] Testing FitBalance API Endpoints")
    print("=" * 60)
    
    # Test all endpoints
    test_root_endpoint()
    test_health_endpoint()
    test_food_database()
    test_nutrition_history()
    test_nutrition_stats()
    test_nutrition_recommendations()
    test_burnout_analyze()
    test_burnout_survival_curve()
    test_burnout_recommendations()
    test_biomechanics_heatmap()
    # test_nutrition_analyze()  # Uncomment when you have test images
    # test_biomechanics_analyze()  # Uncomment when you have test videos
    
    print("\n" + "=" * 60)
    print("[SUCCESS] API testing complete!")
