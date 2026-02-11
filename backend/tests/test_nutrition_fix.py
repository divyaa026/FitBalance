"""
Quick test to verify nutrition values are showing correctly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from modules.nutrition import ProteinOptimizer
from database.nutrition_db import nutrition_db

# Test the database lookup
print("=" * 60)
print("Testing Nutrition Database Lookup")
print("=" * 60)

test_foods = ["lean_beef", "spinach_sauteed", "quinoa_cooked", "brown_rice", "chicken_breast"]

for food_name in test_foods:
    nutrition = nutrition_db.get_food_nutrition(food_name)
    if nutrition:
        print(f"\n✅ {food_name}:")
        print(f"   Protein: {nutrition['protein']}g per 100g")
        print(f"   Calories: {nutrition['calories']} per 100g")
        print(f"   Carbs: {nutrition['carbs']}g")
        print(f"   Fat: {nutrition['fat']}g")
    else:
        print(f"\n❌ {food_name}: NO DATA FOUND")

print("\n" + "=" * 60)
print("Testing Mock Food Detection")
print("=" * 60)

# Test mock detection
optimizer = ProteinOptimizer()
# Create a dummy image file
import tempfile
from PIL import Image

with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
    img = Image.new('RGB', (100, 100), color='white')
    img.save(tmp.name)
    tmp_path = tmp.name

mock_foods = optimizer._mock_food_detection(tmp_path)

print(f"\nDetected {len(mock_foods)} foods:")
for food in mock_foods:
    print(f"\n  {food.name}")
    print(f"    Confidence: {food.confidence}")
    print(f"    Protein: {food.protein_content}g")
    print(f"    Calories: {food.calories}")
    print(f"    Serving: {food.serving_size}")

# Cleanup
os.unlink(tmp_path)

print("\n" + "=" * 60)
print("✅ Test Complete!")
print("=" * 60)
