"""
Initialize FitBalance database with all tables and seed data
"""
from nutrition_db import nutrition_db, Base, FoodItems, User
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize complete database"""
    
    try:
        # Create all tables
        logger.info("[INFO] Creating database tables...")
        Base.metadata.create_all(bind=nutrition_db.engine)
        logger.info("[SUCCESS] Tables created successfully")
        
        # Populate food database with Indian foods
        logger.info("[INFO] Populating Indian food database...")
        populate_indian_foods()
        logger.info("[SUCCESS] Indian food database populated")
        
        # Create test user
        create_test_users()
        logger.info("[SUCCESS] Test users created")
        
        logger.info("\n[SUCCESS] Database initialization complete!")
        
    except Exception as e:
        logger.error(f"[ERROR] Database initialization failed: {e}")
        raise

def populate_indian_foods():
    """Populate with 80 Indian foods from trained model"""
    
    indian_foods = [
        # Format: (name, protein/100g, carbs/100g, fat/100g, calories/100g, category)
        ("aloo_gobi", 2.5, 15.0, 0.5, 75, "vegetable"),
        ("aloo_matar", 3.0, 18.0, 0.8, 90, "vegetable"),
        ("aloo_methi", 3.5, 14.0, 1.0, 80, "vegetable"),
        ("aloo_shimla_mirch", 2.0, 16.0, 0.6, 78, "vegetable"),
        ("aloo_tikki", 4.5, 25.0, 8.0, 180, "snack"),
        ("biryani", 6.0, 45.0, 8.0, 280, "rice"),
        ("butter_naan", 8.0, 50.0, 12.0, 320, "bread"),
        ("cham_cham", 5.0, 55.0, 10.0, 310, "sweet"),
        ("chana_masala", 8.5, 27.0, 4.0, 175, "curry"),
        ("chapati", 3.5, 18.0, 1.0, 100, "bread"),
        ("chole_bhature", 7.0, 48.0, 15.0, 350, "combo"),
        ("dal_makhani", 9.0, 18.0, 6.0, 165, "dal"),
        ("dal_tadka", 8.0, 20.0, 3.0, 140, "dal"),
        ("dosa", 7.0, 42.0, 1.5, 200, "breakfast"),
        ("gulab_jamun", 4.0, 60.0, 12.0, 350, "sweet"),
        ("idli", 3.5, 22.0, 0.5, 105, "breakfast"),
        ("jalebi", 2.0, 65.0, 15.0, 400, "sweet"),
        ("kadai_paneer", 12.0, 8.0, 15.0, 210, "curry"),
        ("kadhi_pakora", 5.0, 15.0, 8.0, 145, "curry"),
        ("kofta", 10.0, 12.0, 18.0, 250, "curry"),
        ("masala_dosa", 8.0, 45.0, 3.0, 230, "breakfast"),
        ("momos", 7.0, 28.0, 4.0, 170, "snack"),
        ("palak_paneer", 14.0, 7.0, 12.0, 190, "curry"),
        ("paneer_butter_masala", 13.0, 9.0, 16.0, 220, "curry"),
        ("paneer_tikka", 15.0, 5.0, 14.0, 200, "starter"),
        ("pav_bhaji", 5.0, 35.0, 10.0, 240, "street_food"),
        ("poha", 3.0, 25.0, 2.0, 130, "breakfast"),
        ("rasgulla", 6.0, 45.0, 0.5, 210, "sweet"),
        ("samosa", 6.0, 30.0, 12.0, 250, "snack"),
        ("tandoori_chicken", 28.0, 2.0, 8.0, 190, "chicken"),
        # Add more 50 foods based on Person 1's dataset...
        ("butter_chicken", 20.0, 8.0, 15.0, 250, "chicken"),
        ("rogan_josh", 18.0, 10.0, 12.0, 220, "curry"),
        ("tikka_masala", 22.0, 9.0, 14.0, 240, "chicken"),
        ("vindaloo", 15.0, 12.0, 10.0, 200, "curry"),
        ("korma", 12.0, 15.0, 20.0, 280, "curry"),
        ("naan", 8.0, 48.0, 8.0, 290, "bread"),
        ("roti", 3.0, 18.0, 0.5, 90, "bread"),
        ("paratha", 4.5, 24.0, 12.0, 230, "bread"),
        ("puri", 5.0, 28.0, 15.0, 260, "bread"),
        ("bhindi_masala", 2.0, 8.0, 3.0, 65, "vegetable"),
        ("baingan_bharta", 1.5, 10.0, 4.0, 80, "vegetable"),
        ("malai_kofta", 8.0, 15.0, 18.0, 250, "curry"),
        ("rajma", 9.0, 25.0, 0.5, 140, "curry"),
        ("chole", 8.5, 27.0, 4.0, 175, "curry"),
        ("aloo_paratha", 5.0, 30.0, 10.0, 220, "bread"),
        ("pani_puri", 3.0, 18.0, 2.0, 100, "street_food"),
        ("bhel_puri", 4.0, 22.0, 3.0, 130, "street_food"),
        ("vada_pav", 6.0, 35.0, 12.0, 260, "street_food"),
        ("dahi_vada", 5.0, 20.0, 8.0, 160, "snack"),
        ("khichdi", 6.0, 35.0, 2.0, 180, "rice"),
        ("pulao", 5.0, 40.0, 6.0, 220, "rice"),
        ("fried_rice", 6.0, 45.0, 8.0, 270, "rice"),
        ("chicken_biryani", 15.0, 50.0, 12.0, 360, "rice"),
        ("mutton_biryani", 18.0, 48.0, 15.0, 390, "rice"),
        ("fish_curry", 20.0, 8.0, 10.0, 200, "seafood"),
        ("prawn_curry", 18.0, 6.0, 12.0, 190, "seafood"),
        ("kebab", 25.0, 3.0, 15.0, 250, "starter"),
        ("seekh_kebab", 22.0, 4.0, 18.0, 270, "starter"),
        ("chicken_tikka", 27.0, 3.0, 10.0, 210, "starter"),
        ("lassi", 3.5, 12.0, 3.0, 85, "beverage"),
        ("kulfi", 5.0, 30.0, 8.0, 200, "dessert"),
        ("kheer", 4.0, 35.0, 6.0, 210, "dessert"),
        ("halwa", 3.0, 40.0, 15.0, 300, "dessert"),
        ("barfi", 6.0, 50.0, 12.0, 320, "dessert"),
        ("ladoo", 4.5, 55.0, 10.0, 310, "dessert"),
        ("mysore_pak", 3.0, 48.0, 18.0, 350, "dessert"),
        ("peda", 5.5, 52.0, 9.0, 295, "dessert"),
        ("sandesh", 6.0, 45.0, 8.0, 270, "dessert"),
        ("rasmalai", 7.0, 40.0, 10.0, 280, "dessert"),
        ("gajar_halwa", 3.5, 38.0, 12.0, 270, "dessert"),
        ("upma", 4.0, 28.0, 3.0, 150, "breakfast"),
        ("medu_vada", 5.0, 22.0, 8.0, 170, "breakfast"),
        ("uttapam", 6.0, 35.0, 2.0, 180, "breakfast"),
        ("pesarattu", 8.0, 30.0, 2.5, 170, "breakfast"),
        ("appam", 2.0, 25.0, 1.0, 115, "breakfast"),
        ("puttu", 3.5, 30.0, 1.5, 145, "breakfast"),
        ("dhokla", 4.0, 20.0, 2.0, 110, "snack"),
        ("khandvi", 5.0, 18.0, 3.0, 115, "snack"),
        ("pakora", 4.5, 22.0, 10.0, 190, "snack"),
        ("bonda", 4.0, 24.0, 8.0, 170, "snack"),
    ]
    
    session = nutrition_db.SessionLocal()
    
    for food_data in indian_foods:
        food = FoodItems(
            food_name=food_data[0],
            protein_per_100g=food_data[1],
            carbs_per_100g=food_data[2],
            fats_per_100g=food_data[3],
            calories_per_100g=food_data[4],
            food_category=food_data[5]
        )
        session.merge(food)  # Use merge to avoid duplicates
    
    session.commit()
    session.close()

def create_test_users():
    """Create test users for development"""
    session = nutrition_db.SessionLocal()
    
    test_users = [
        User(user_id=1, age=28, weight_kg=70, height_cm=175, fitness_goal="muscle_gain"),
        User(user_id=123, age=25, weight_kg=65, height_cm=168, fitness_goal="fat_loss"),
    ]
    
    for user in test_users:
        session.merge(user)
    
    session.commit()
    session.close()

if __name__ == "__main__":
    init_database()

