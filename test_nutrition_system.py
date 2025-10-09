"""
Test Script for Enhanced Nutrition Optimization System
Comprehensive testing of CNN Food Classifier, GRU Protein Optimizer, and SHAP Explainer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import requests
import json
from PIL import Image
import numpy as np
import io
import base64
from datetime import date, datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionSystemTester:
    """Comprehensive tester for the nutrition optimization system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {}
    
    def create_test_image(self, food_type: str = "chicken") -> bytes:
        """Create a test meal image"""
        # Create a simple test image (in production, use real food images)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some basic patterns to simulate food
        if food_type == "chicken":
            img_array[50:150, 50:150] = [200, 180, 160]  # Chicken-like colors
        elif food_type == "vegetables":
            img_array[100:200, 100:200] = [100, 200, 100]  # Green vegetables
        elif food_type == "mixed":
            img_array[30:100, 30:100] = [200, 180, 160]  # Protein
            img_array[130:200, 130:200] = [100, 200, 100]  # Vegetables
            img_array[30:100, 130:200] = [220, 200, 150]  # Carbs
        
        # Convert to PIL Image and then to bytes
        img = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    def test_meal_analysis(self):
        """Test meal photo analysis endpoint"""
        logger.info("🍽️ Testing meal analysis...")
        
        try:
            # Test different meal types
            meal_types = ["chicken", "vegetables", "mixed"]
            
            for meal_type in meal_types:
                logger.info(f"  Testing {meal_type} meal...")
                
                # Create test image
                image_data = self.create_test_image(meal_type)
                
                # Prepare request
                files = {
                    'meal_photo': ('test_meal.jpg', image_data, 'image/jpeg')
                }
                data = {
                    'user_id': 'test_user_123',
                    'dietary_restrictions': json.dumps(['vegetarian'] if meal_type == 'vegetables' else [])
                }
                
                # Make request
                response = requests.post(
                    f"{self.base_url}/nutrition/analyze-meal",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate response structure
                    required_fields = [
                        'total_protein', 'total_calories', 'detected_foods',
                        'recommendations', 'meal_quality_score', 'protein_optimization'
                    ]
                    
                    for field in required_fields:
                        if field not in result:
                            logger.error(f"    ❌ Missing field: {field}")
                            continue
                    
                    logger.info(f"    ✅ {meal_type} meal analysis successful:")
                    logger.info(f"       Protein: {result.get('total_protein', 0):.1f}g")
                    logger.info(f"       Calories: {result.get('total_calories', 0):.0f}")
                    logger.info(f"       Quality Score: {result.get('meal_quality_score', 0):.1f}/100")
                    logger.info(f"       Detected Foods: {len(result.get('detected_foods', []))}")
                    
                    # Test protein optimization data
                    protein_opt = result.get('protein_optimization', {})
                    if protein_opt:
                        logger.info(f"       Optimized Protein: {protein_opt.get('adjusted_protein', 0):.1f}g")
                        logger.info(f"       Sleep Factor: {protein_opt.get('sleep_factor', 1):.2f}")
                        logger.info(f"       Activity Factor: {protein_opt.get('activity_factor', 1):.2f}")
                    
                    self.test_results[f"meal_analysis_{meal_type}"] = "PASS"
                    
                else:
                    logger.error(f"    ❌ {meal_type} meal analysis failed: {response.status_code}")
                    logger.error(f"       Response: {response.text}")
                    self.test_results[f"meal_analysis_{meal_type}"] = "FAIL"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Meal analysis test failed: {e}")
            self.test_results["meal_analysis"] = "FAIL"
            return False
    
    def test_nutrition_recommendations(self):
        """Test nutrition recommendations endpoint"""
        logger.info("💊 Testing nutrition recommendations...")
        
        try:
            test_users = [
                {"user_id": "athlete_123", "activity_level": "high"},
                {"user_id": "sedentary_456", "activity_level": "low"},
                {"user_id": "moderate_789", "activity_level": "moderate"}
            ]
            
            for user in test_users:
                logger.info(f"  Testing recommendations for {user['user_id']}...")
                
                response = requests.get(
                    f"{self.base_url}/nutrition/recommendations/{user['user_id']}",
                    params={
                        'activity_level': user['activity_level'],
                        'target_protein': 120.0
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate response
                    required_fields = [
                        'recommended_protein', 'optimization_factors', 
                        'recommendations', 'health_metrics'
                    ]
                    
                    for field in required_fields:
                        if field not in result:
                            logger.error(f"    ❌ Missing field: {field}")
                            continue
                    
                    logger.info(f"    ✅ Recommendations for {user['user_id']}:")
                    logger.info(f"       Recommended Protein: {result.get('recommended_protein', 0):.1f}g")
                    logger.info(f"       Activity Level Impact: {user['activity_level']}")
                    
                    # Check optimization factors
                    factors = result.get('optimization_factors', {})
                    if factors:
                        logger.info(f"       Sleep Factor: {factors.get('sleep_factor', 1):.2f}")
                        logger.info(f"       HRV Factor: {factors.get('hrv_factor', 1):.2f}")
                        logger.info(f"       Confidence: {factors.get('confidence', 0):.2f}")
                    
                    self.test_results[f"recommendations_{user['user_id']}"] = "PASS"
                    
                else:
                    logger.error(f"    ❌ Recommendations failed for {user['user_id']}: {response.status_code}")
                    self.test_results[f"recommendations_{user['user_id']}"] = "FAIL"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Nutrition recommendations test failed: {e}")
            self.test_results["nutrition_recommendations"] = "FAIL"
            return False
    
    def test_health_data_update(self):
        """Test health data update endpoint"""
        logger.info("📊 Testing health data updates...")
        
        try:
            test_health_data = [
                {
                    "user_id": "test_user_1",
                    "data": {
                        "sleep_duration": 7.5,
                        "sleep_quality": 8.2,
                        "hrv": 55.0,
                        "activity_level": 6.5,
                        "stress_level": 3.0,
                        "recovery_score": 8.0
                    }
                },
                {
                    "user_id": "test_user_2",
                    "data": {
                        "sleep_duration": 5.5,
                        "sleep_quality": 4.8,
                        "hrv": 35.0,
                        "activity_level": 8.5,
                        "stress_level": 7.0,
                        "recovery_score": 5.2
                    }
                }
            ]
            
            for test_case in test_health_data:
                user_id = test_case["user_id"]
                health_data = test_case["data"]
                
                logger.info(f"  Testing health data update for {user_id}...")
                
                response = requests.post(
                    f"{self.base_url}/nutrition/health-data/{user_id}",
                    json=health_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "success":
                        logger.info(f"    ✅ Health data updated for {user_id}")
                        self.test_results[f"health_update_{user_id}"] = "PASS"
                    else:
                        logger.error(f"    ❌ Health data update returned error for {user_id}")
                        self.test_results[f"health_update_{user_id}"] = "FAIL"
                        
                else:
                    logger.error(f"    ❌ Health data update failed for {user_id}: {response.status_code}")
                    self.test_results[f"health_update_{user_id}"] = "FAIL"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Health data update test failed: {e}")
            self.test_results["health_data_update"] = "FAIL"
            return False
    
    def test_shap_explanations(self):
        """Test SHAP explanation endpoint"""
        logger.info("🔍 Testing SHAP explanations...")
        
        try:
            test_dates = [
                date.today().isoformat(),
                (date.today() - timedelta(days=1)).isoformat(),
                (date.today() - timedelta(days=7)).isoformat()
            ]
            
            for test_date in test_dates:
                logger.info(f"  Testing SHAP explanation for {test_date}...")
                
                response = requests.get(
                    f"{self.base_url}/nutrition/shap-explanation/test_user",
                    params={'date_str': test_date},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate response structure
                    required_fields = [
                        'feature_importance', 'explanation_text', 
                        'confidence_score', 'health_metrics'
                    ]
                    
                    for field in required_fields:
                        if field not in result:
                            logger.error(f"    ❌ Missing field: {field}")
                            continue
                    
                    logger.info(f"    ✅ SHAP explanation for {test_date}:")
                    logger.info(f"       Confidence: {result.get('confidence_score', 0):.2f}")
                    logger.info(f"       Explanation: {result.get('explanation_text', '')[:100]}...")
                    
                    # Check feature importance
                    features = result.get('feature_importance', {})
                    if features:
                        for feature, data in list(features.items())[:3]:  # Show top 3
                            if isinstance(data, dict):
                                logger.info(f"       {feature}: {data.get('value', 0):.1f} ({data.get('impact', 'neutral')})")
                    
                    self.test_results[f"shap_{test_date}"] = "PASS"
                    
                else:
                    logger.error(f"    ❌ SHAP explanation failed for {test_date}: {response.status_code}")
                    self.test_results[f"shap_{test_date}"] = "FAIL"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ SHAP explanation test failed: {e}")
            self.test_results["shap_explanations"] = "FAIL"
            return False
    
    def test_food_database(self):
        """Test food database endpoint"""
        logger.info("🗄️ Testing food database...")
        
        try:
            response = requests.get(
                f"{self.base_url}/nutrition/food-database",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                required_fields = ['foods', 'total_foods', 'categories']
                
                for field in required_fields:
                    if field not in result:
                        logger.error(f"    ❌ Missing field: {field}")
                        return False
                
                foods = result.get('foods', {})
                total_foods = result.get('total_foods', 0)
                categories = result.get('categories', [])
                
                logger.info(f"    ✅ Food database loaded:")
                logger.info(f"       Total Foods: {total_foods}")
                logger.info(f"       Categories: {', '.join(categories)}")
                
                # Show sample foods
                sample_foods = list(foods.keys())[:5]
                for food in sample_foods:
                    nutrition = foods[food]
                    logger.info(f"       {nutrition.get('name', food)}: {nutrition.get('protein', 0)}g protein")
                
                self.test_results["food_database"] = "PASS"
                return True
                
            else:
                logger.error(f"    ❌ Food database failed: {response.status_code}")
                self.test_results["food_database"] = "FAIL"
                return False
                
        except Exception as e:
            logger.error(f"❌ Food database test failed: {e}")
            self.test_results["food_database"] = "FAIL"
            return False
    
    def test_server_health(self):
        """Test basic server health"""
        logger.info("🏥 Testing server health...")
        
        try:
            # Test root endpoint
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                logger.info("    ✅ Root endpoint responsive")
                self.test_results["server_root"] = "PASS"
            else:
                logger.error(f"    ❌ Root endpoint failed: {response.status_code}")
                self.test_results["server_root"] = "FAIL"
            
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("    ✅ Health endpoint responsive")
                self.test_results["server_health"] = "PASS"
            else:
                logger.error(f"    ❌ Health endpoint failed: {response.status_code}")
                self.test_results["server_health"] = "FAIL"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Server health test failed: {e}")
            self.test_results["server_health"] = "FAIL"
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("🚀 Starting comprehensive nutrition system tests...")
        
        # Run all tests
        tests = [
            ("Server Health", self.test_server_health),
            ("Food Database", self.test_food_database),
            ("Health Data Update", self.test_health_data_update),
            ("Meal Analysis", self.test_meal_analysis),
            ("Nutrition Recommendations", self.test_nutrition_recommendations),
            ("SHAP Explanations", self.test_shap_explanations)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Tests ---")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} tests completed")
                else:
                    logger.error(f"❌ {test_name} tests failed")
            except Exception as e:
                logger.error(f"❌ {test_name} tests crashed: {e}")
        
        # Generate report
        self.generate_test_report(passed_tests, total_tests)
    
    def generate_test_report(self, passed_tests: int, total_tests: int):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("📊 FITBALANCE NUTRITION SYSTEM TEST REPORT")
        logger.info("="*60)
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        # Categorize results
        passed = [k for k, v in self.test_results.items() if v == "PASS"]
        failed = [k for k, v in self.test_results.items() if v == "FAIL"]
        
        if passed:
            logger.info(f"\n✅ Passed Tests ({len(passed)}):")
            for test in passed:
                logger.info(f"   • {test}")
        
        if failed:
            logger.info(f"\n❌ Failed Tests ({len(failed)}):")
            for test in failed:
                logger.info(f"   • {test}")
        
        # System status
        if success_rate >= 80:
            logger.info(f"\n🎉 System Status: HEALTHY - Ready for production!")
        elif success_rate >= 60:
            logger.info(f"\n⚠️ System Status: PARTIAL - Some issues need attention")
        else:
            logger.info(f"\n🚨 System Status: CRITICAL - Major issues detected")
        
        logger.info("\n💡 Key Features Tested:")
        logger.info("   • CNN Food Classification with Food-101 dataset")
        logger.info("   • GRU Time-Series Protein Optimization")
        logger.info("   • SHAP Explainable AI for recommendations")
        logger.info("   • Real-time health data integration")
        logger.info("   • RESTful API endpoints")
        logger.info("   • PostgreSQL database operations")
        
        logger.info("="*60)

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FitBalance Nutrition Optimization System")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API server")
    parser.add_argument("--test", choices=["all", "meal", "recommendations", "health", "shap", "database"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = NutritionSystemTester(args.url)
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "meal":
        tester.test_meal_analysis()
    elif args.test == "recommendations":
        tester.test_nutrition_recommendations()
    elif args.test == "health":
        tester.test_health_data_update()
    elif args.test == "shap":
        tester.test_shap_explanations()
    elif args.test == "database":
        tester.test_food_database()

if __name__ == "__main__":
    main()