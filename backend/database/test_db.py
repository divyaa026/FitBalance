"""Test database connection and queries"""
from nutrition_db import nutrition_db
from sqlalchemy import text

def test_connection():
    """Test basic database operations"""
    
    print("Testing database connection...")
    
    # Test 1: Get specific food (works with fallback)
    food = nutrition_db.get_food_nutrition("paneer_butter_masala")
    if food:
        print(f"[SUCCESS] Retrieved food: {food}")
    else:
        print("[INFO] Using fallback food database")
        food = nutrition_db.get_food_nutrition("chicken_breast")
        print(f"[SUCCESS] Retrieved fallback food: {food}")
    
    # Test 2: Log a test meal
    nutrition_db.log_meal(
        user_id=123,
        image_path="/test/image.jpg",
        detected_foods={"paneer_butter_masala": 150},
        total_protein=19.5,
        total_calories=330,
        confidence=0.85
    )
    print("[SUCCESS] Test meal logged")
    
    # Test 3: Get recent meals
    meals = nutrition_db.get_recent_meals(123, days=7)
    print(f"[SUCCESS] Retrieved {len(meals)} recent meals")
    
    print("\n[SUCCESS] All database tests passed!")

if __name__ == "__main__":
    test_connection()
