"""
Step 5: Create Indian Food Nutrition Database
Creates a comprehensive nutrition database for detected foods
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive Indian food nutrition data (per 100g)
INDIAN_FOOD_NUTRITION = {
    # Breads & Rotis
    "chapati": {"protein": 3.1, "calories": 71, "carbs": 15.8, "fat": 0.4, "fiber": 2.0},
    "naan": {"protein": 8.7, "calories": 262, "carbs": 45.0, "fat": 5.1, "fiber": 2.2},
    "paratha": {"protein": 6.2, "calories": 289, "carbs": 35.0, "fat": 13.0, "fiber": 3.0},
    "roti": {"protein": 3.1, "calories": 71, "carbs": 15.8, "fat": 0.4, "fiber": 2.0},
    "puri": {"protein": 5.2, "calories": 315, "carbs": 45.0, "fat": 12.8, "fiber": 1.5},
    
    # Rice Dishes
    "rice": {"protein": 2.7, "calories": 130, "carbs": 28.2, "fat": 0.3, "fiber": 0.4},
    "biryani": {"protein": 8.5, "calories": 165, "carbs": 22.0, "fat": 4.5, "fiber": 1.5},
    "pulao": {"protein": 4.2, "calories": 142, "carbs": 25.0, "fat": 3.0, "fiber": 1.2},
    "jeera_rice": {"protein": 2.9, "calories": 138, "carbs": 28.5, "fat": 1.2, "fiber": 0.6},
    
    # Dal & Lentils
    "dal_tadka": {"protein": 9.0, "calories": 105, "carbs": 18.0, "fat": 0.8, "fiber": 7.9},
    "dal_makhani": {"protein": 8.5, "calories": 142, "carbs": 15.0, "fat": 5.5, "fiber": 6.5},
    "sambar": {"protein": 3.5, "calories": 55, "carbs": 9.0, "fat": 1.2, "fiber": 2.8},
    "rajma": {"protein": 8.7, "calories": 127, "carbs": 22.8, "fat": 0.5, "fiber": 6.4},
    "chana_masala": {"protein": 8.9, "calories": 164, "carbs": 27.4, "fat": 2.6, "fiber": 7.6},
    
    # South Indian
    "dosa": {"protein": 2.5, "calories": 120, "carbs": 22.0, "fat": 2.5, "fiber": 1.5},
    "idli": {"protein": 2.0, "calories": 39, "carbs": 8.0, "fat": 0.1, "fiber": 0.3},
    "vada": {"protein": 4.5, "calories": 135, "carbs": 15.0, "fat": 6.5, "fiber": 2.5},
    "uttapam": {"protein": 3.2, "calories": 90, "carbs": 16.0, "fat": 1.5, "fiber": 1.8},
    "upma": {"protein": 3.8, "calories": 96, "carbs": 19.0, "fat": 1.2, "fiber": 2.0},
    
    # Vegetables (Sabzi)
    "aloo_gobi": {"protein": 2.5, "calories": 65, "carbs": 12.0, "fat": 1.5, "fiber": 3.0},
    "palak_paneer": {"protein": 11.0, "calories": 125, "carbs": 6.0, "fat": 8.0, "fiber": 2.5},
    "paneer_butter_masala": {"protein": 14.2, "calories": 185, "carbs": 8.5, "fat": 12.0, "fiber": 1.8},
    "bhindi_masala": {"protein": 1.9, "calories": 40, "carbs": 7.0, "fat": 0.2, "fiber": 3.2},
    "baingan_bharta": {"protein": 1.2, "calories": 65, "carbs": 9.5, "fat": 2.8, "fiber": 3.4},
    
    # Snacks
    "samosa": {"protein": 5.0, "calories": 262, "carbs": 27.0, "fat": 15.0, "fiber": 2.0},
    "pakora": {"protein": 3.5, "calories": 180, "carbs": 18.0, "fat": 10.0, "fiber": 2.5},
    "kachori": {"protein": 6.2, "calories": 275, "carbs": 32.0, "fat": 13.0, "fiber": 3.0},
    "dhokla": {"protein": 3.2, "calories": 160, "carbs": 28.0, "fat": 3.5, "fiber": 2.0},
    
    # Sweets
    "gulab_jamun": {"protein": 4.5, "calories": 375, "carbs": 65.0, "fat": 12.0, "fiber": 0.5},
    "rasgulla": {"protein": 3.5, "calories": 186, "carbs": 38.0, "fat": 1.5, "fiber": 0.0},
    "jalebi": {"protein": 1.8, "calories": 445, "carbs": 75.0, "fat": 15.0, "fiber": 0.2},
    "ladoo": {"protein": 5.5, "calories": 420, "carbs": 55.0, "fat": 18.0, "fiber": 2.5},
    "kheer": {"protein": 3.8, "calories": 140, "carbs": 22.0, "fat": 4.5, "fiber": 0.3},
    
    # Beverages
    "chai": {"protein": 1.6, "calories": 50, "carbs": 8.0, "fat": 1.5, "fiber": 0.0},
    "lassi": {"protein": 3.2, "calories": 65, "carbs": 11.0, "fat": 1.5, "fiber": 0.0},
    
    # Protein Sources
    "paneer": {"protein": 18.3, "calories": 265, "carbs": 1.2, "fat": 20.8, "fiber": 0.0},
    "chicken_curry": {"protein": 25.0, "calories": 165, "carbs": 5.0, "fat": 7.0, "fiber": 1.0},
    "egg_curry": {"protein": 12.6, "calories": 155, "carbs": 4.5, "fat": 11.5, "fiber": 0.8},
}

def create_nutrition_database():
    """Create comprehensive nutrition database"""
    
    logger.info("\n" + "="*60)
    logger.info("CREATING INDIAN FOOD NUTRITION DATABASE")
    logger.info("="*60 + "\n")
    
    # Add serving size (standard portion)
    nutrition_db = {}
    for food_name, nutrition in INDIAN_FOOD_NUTRITION.items():
        # Determine typical serving size based on food type
        if any(x in food_name for x in ['chapati', 'roti', 'naan', 'paratha', 'puri']):
            serving_g = 50  # 1 piece
        elif any(x in food_name for x in ['dosa', 'idli', 'uttapam']):
            serving_g = 60  # 1 piece
        elif 'vada' in food_name:
            serving_g = 40  # 1 piece
        elif any(x in food_name for x in ['samosa', 'kachori']):
            serving_g = 80  # 1 piece
        elif 'rice' in food_name or 'biryani' in food_name or 'pulao' in food_name:
            serving_g = 150  # 1 bowl
        elif 'dal' in food_name or 'sambar' in food_name or 'rajma' in food_name or 'chana' in food_name:
            serving_g = 150  # 1 bowl
        elif 'curry' in food_name or 'masala' in food_name or 'paneer' in food_name:
            serving_g = 100  # 1 serving
        elif any(x in food_name for x in ['gulab_jamun', 'rasgulla']):
            serving_g = 50  # 1 piece
        elif any(x in food_name for x in ['jalebi', 'ladoo']):
            serving_g = 30  # 1 piece
        elif 'kheer' in food_name:
            serving_g = 100  # 1 bowl
        elif any(x in food_name for x in ['chai', 'lassi']):
            serving_g = 150  # 1 cup
        else:
            serving_g = 100  # default
        
        # Calculate per-serving nutrition
        nutrition_db[food_name] = {
            "protein_per_100g": nutrition["protein"],
            "calories_per_100g": nutrition["calories"],
            "carbs_per_100g": nutrition.get("carbs", 0),
            "fat_per_100g": nutrition.get("fat", 0),
            "fiber_per_100g": nutrition.get("fiber", 0),
            "serving_size_g": serving_g,
            "protein_per_serving": round(nutrition["protein"] * serving_g / 100, 1),
            "calories_per_serving": round(nutrition["calories"] * serving_g / 100, 0),
            "carbs_per_serving": round(nutrition.get("carbs", 0) * serving_g / 100, 1),
            "fat_per_serving": round(nutrition.get("fat", 0) * serving_g / 100, 1),
            "fiber_per_serving": round(nutrition.get("fiber", 0) * serving_g / 100, 1),
            "food_category": _get_category(food_name),
            "is_vegetarian": _is_vegetarian(food_name),
            "is_vegan": _is_vegan(food_name),
        }
    
    # Save database
    output_path = Path("nutrition/indian_food_nutrition_db.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(nutrition_db, f, indent=2)
    
    logger.info(f"✓ Created nutrition database: {output_path}")
    logger.info(f"✓ Total foods: {len(nutrition_db)}")
    
    # Create Python module
    create_python_module(nutrition_db)
    
    # Print stats
    print_stats(nutrition_db)
    
    return nutrition_db

def _get_category(food_name):
    """Categorize food"""
    categories = {
        "bread": ["chapati", "roti", "naan", "paratha", "puri"],
        "rice": ["rice", "biryani", "pulao", "jeera_rice"],
        "lentils": ["dal", "sambar", "rajma", "chana"],
        "south_indian": ["dosa", "idli", "vada", "uttapam", "upma"],
        "vegetables": ["aloo", "palak", "paneer", "bhindi", "baingan", "gobi"],
        "snacks": ["samosa", "pakora", "kachori", "dhokla"],
        "sweets": ["gulab", "rasgulla", "jalebi", "ladoo", "kheer"],
        "beverages": ["chai", "lassi"],
        "protein": ["chicken", "egg", "paneer"],
    }
    
    for category, keywords in categories.items():
        if any(keyword in food_name for keyword in keywords):
            return category
    return "other"

def _is_vegetarian(food_name):
    """Check if vegetarian"""
    non_veg = ["chicken", "mutton", "fish", "prawn", "egg"]
    return not any(x in food_name for x in non_veg)

def _is_vegan(food_name):
    """Check if vegan"""
    non_vegan = ["paneer", "ghee", "butter", "milk", "curd", "yogurt", "egg", "chicken", "mutton", "fish"]
    return not any(x in food_name for x in non_vegan)

def create_python_module(nutrition_db):
    """Create Python module for easy import"""
    
    code = f'''"""
Indian Food Nutrition Database
Auto-generated by step5_create_nutrition_db.py
"""

import json
from pathlib import Path
from typing import Dict, Optional

class IndianFoodNutritionDB:
    """Nutrition database for Indian foods"""
    
    def __init__(self):
        db_path = Path(__file__).parent / "indian_food_nutrition_db.json"
        with open(db_path, 'r') as f:
            self.db = json.load(f)
    
    def get_nutrition(self, food_name: str) -> Optional[Dict]:
        """Get nutrition info for a food"""
        # Normalize name
        food_name = food_name.lower().replace(' ', '_')
        return self.db.get(food_name)
    
    def search(self, query: str) -> list:
        """Search for foods matching query"""
        query = query.lower()
        matches = []
        for food_name, nutrition in self.db.items():
            if query in food_name:
                matches.append((food_name, nutrition))
        return matches
    
    def get_high_protein_foods(self, min_protein_per_serving: float = 10) -> list:
        """Get foods with high protein"""
        high_protein = []
        for food_name, nutrition in self.db.items():
            if nutrition['protein_per_serving'] >= min_protein_per_serving:
                high_protein.append((
                    food_name, 
                    nutrition['protein_per_serving'],
                    nutrition['calories_per_serving']
                ))
        return sorted(high_protein, key=lambda x: x[1], reverse=True)
    
    def get_by_category(self, category: str) -> list:
        """Get foods by category"""
        foods = []
        for food_name, nutrition in self.db.items():
            if nutrition['food_category'] == category:
                foods.append((food_name, nutrition))
        return foods
    
    def get_vegetarian_foods(self) -> list:
        """Get vegetarian foods"""
        return [(name, info) for name, info in self.db.items() if info['is_vegetarian']]
    
    def get_vegan_foods(self) -> list:
        """Get vegan foods"""
        return [(name, info) for name, info in self.db.items() if info['is_vegan']]

# Singleton instance
_db_instance = None

def get_nutrition_db() -> IndianFoodNutritionDB:
    """Get singleton nutrition database"""
    global _db_instance
    if _db_instance is None:
        _db_instance = IndianFoodNutritionDB()
    return _db_instance
'''
    
    module_path = Path("nutrition/indian_food_nutrition.py")
    with open(module_path, 'w') as f:
        f.write(code)
    
    logger.info(f"✓ Created Python module: {module_path}")

def print_stats(nutrition_db):
    """Print database statistics"""
    
    logger.info("\n" + "="*60)
    logger.info("DATABASE STATISTICS")
    logger.info("="*60)
    
    # Count by category
    categories = {}
    for food_name, nutrition in nutrition_db.items():
        cat = nutrition['food_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    logger.info("\n📊 Foods by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count}")
    
    # High protein foods
    high_protein = [(name, info['protein_per_serving']) 
                    for name, info in nutrition_db.items()]
    high_protein.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("\n💪 Top 10 high-protein foods:")
    for i, (name, protein) in enumerate(high_protein[:10], 1):
        logger.info(f"  {i}. {name}: {protein}g protein/serving")
    
    # Vegetarian/Vegan
    veg_count = sum(1 for info in nutrition_db.values() if info['is_vegetarian'])
    vegan_count = sum(1 for info in nutrition_db.values() if info['is_vegan'])
    
    logger.info(f"\n🌱 Vegetarian: {veg_count}/{len(nutrition_db)}")
    logger.info(f"🌱 Vegan: {vegan_count}/{len(nutrition_db)}")
    
    logger.info("\n" + "="*60)

def main():
    """Main function"""
    nutrition_db = create_nutrition_database()
    
    logger.info("\n✅ Nutrition database created!")
    logger.info("\n📝 Usage example:")
    logger.info("""
    from nutrition.indian_food_nutrition import get_nutrition_db
    
    db = get_nutrition_db()
    nutrition = db.get_nutrition("dosa")
    high_protein = db.get_high_protein_foods(min_protein_per_serving=10)
    veg_foods = db.get_vegetarian_foods()
    """)

if __name__ == "__main__":
    main()
