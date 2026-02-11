# FitBalance Backend Documentation

## Architecture

### Tech Stack
- **Framework:** FastAPI (Python 3.12)
- **Database:** PostgreSQL 15 (with fallback to in-memory storage)
- **ORM:** SQLAlchemy
- **ML Models:** TensorFlow, PyTorch, lifelines
- **Image Processing:** OpenCV, PIL
- **API Integration:** Google Gemini Vision API (optional)

### Project Structure
```
backend/
├── api/
│   └── main.py              # FastAPI app and endpoints
├── modules/
│   ├── nutrition.py         # Nutrition analysis module
│   ├── biomechanics.py      # Exercise form analysis
│   └── burnout.py           # Burnout prediction
├── database/
│   ├── nutrition_db.py      # Database ORM models
│   └── init_database.py     # Database initialization
├── integrations/
│   └── health_apis.py       # Health tracking API integrations
└── utils/
    └── error_handlers.py    # Error handling utilities
```

## API Endpoints

### Nutrition Endpoints

#### POST /nutrition/analyze
Analyze meal photo for nutrition content.

**Request:**
```bash
curl -X POST "http://localhost:8000/nutrition/analyze" \
  -F "meal_photo=@image.jpg" \
  -F "user_id=123" \
  -F "dietary_restrictions=[]"
```

**Response:**
```json
{
  "total_protein": 25.5,
  "total_calories": 450,
  "total_carbs": 45.0,
  "total_fat": 15.0,
  "detected_foods": [
    {
      "name": "paneer_tikka",
      "confidence": 0.92,
      "protein_content": 15.0,
      "calories": 200,
      "serving_size": "100g"
    }
  ],
  "protein_deficit": 5.0,
  "recommendations": [
    "Add more protein to meet your daily target",
    "Try adding: chicken breast, salmon, eggs"
  ],
  "meal_quality_score": 85.5,
  "nutritional_balance": {
    "protein": 25.5,
    "calories": 450,
    "carbs": 45.0,
    "fat": 15.0
  }
}
```

#### GET /nutrition/history/{user_id}
Get user's meal history.

**Request:**
```bash
curl "http://localhost:8000/nutrition/history/123?days=7"
```

**Response:**
```json
{
  "user_id": 123,
  "days": 7,
  "meal_count": 5,
  "meals": [
    {
      "timestamp": "2024-01-15T12:30:00",
      "detected_foods": {"paneer_tikka": 150},
      "total_protein": 22.5,
      "total_calories": 300
    }
  ]
}
```

#### GET /nutrition/stats/{user_id}
Get aggregated nutrition statistics.

**Request:**
```bash
curl "http://localhost:8000/nutrition/stats/123?days=7"
```

**Response:**
```json
{
  "user_id": 123,
  "days": 7,
  "total_meals": 5,
  "total_protein": 112.5,
  "total_calories": 1500.0,
  "avg_protein_per_meal": 22.5,
  "avg_calories_per_meal": 300.0
}
```

#### GET /nutrition/foods
List all foods in database.

**Request:**
```bash
curl "http://localhost:8000/nutrition/foods"
```

**Response:**
```json
{
  "count": 80,
  "foods": [
    {"name": "aloo_gobi", "category": "vegetable"},
    {"name": "biryani", "category": "rice"},
    {"name": "paneer_butter_masala", "category": "curry"}
  ]
}
```

#### GET /nutrition/recommendations/{user_id}
Get personalized nutrition recommendations.

**Request:**
```bash
curl "http://localhost:8000/nutrition/recommendations/123?target_protein=120&activity_level=moderate"
```

### Burnout Endpoints

#### POST /burnout/analyze
Analyze burnout risk.

**Request:**
```bash
curl -X POST "http://localhost:8000/burnout/analyze?user_id=123&workout_frequency=5&sleep_hours=7.0&stress_level=6&recovery_time=2&performance_trend=stable"
```

**Response:**
```json
{
  "risk_score": 45.5,
  "risk_level": "medium",
  "time_to_burnout": 450,
  "survival_probability": 0.75,
  "risk_factors": [
    "excessive_workout_frequency",
    "insufficient_sleep"
  ],
  "recommendations": [
    "Reduce training intensity by 30-50%",
    "Aim for at least 1.0 more hours of sleep per night",
    "Increase recovery time to at least 3-4 days between intense sessions"
  ],
  "survival_curve_data": {
    "times": [0, 30, 60, 90, 120],
    "survival_probabilities": [1.0, 0.95, 0.85, 0.75, 0.65]
  }
}
```

#### GET /burnout/survival-curve/{user_id}
Get survival curve data.

**Request:**
```bash
curl "http://localhost:8000/burnout/survival-curve/123"
```

#### GET /burnout/recommendations/{user_id}
Get personalized recommendations to prevent burnout.

**Request:**
```bash
curl "http://localhost:8000/burnout/recommendations/123"
```

### Biomechanics Endpoints

#### POST /biomechanics/analyze
Analyze exercise form from video.

**Request:**
```bash
curl -X POST "http://localhost:8000/biomechanics/analyze" \
  -F "video_file=@exercise_video.mp4" \
  -F "exercise_type=squat" \
  -F "user_id=123"
```

**Response:**
```json
{
  "form_score": 85.5,
  "joint_angles": [
    {"joint_name": "knee", "angle": 95.2, "is_abnormal": true},
    {"joint_name": "hip", "angle": 85.1, "is_abnormal": false}
  ],
  "torques": [
    {"joint_name": "knee", "torque_magnitude": 45.2, "risk_level": "medium"}
  ],
  "recommendations": [
    "Keep knees aligned with toes",
    "Maintain proper hip hinge"
  ],
  "exercise_type": "squat",
  "user_id": "123"
}
```

#### GET /biomechanics/heatmap/{user_id}
Get torque heatmap for user's exercise.

**Request:**
```bash
curl "http://localhost:8000/biomechanics/heatmap/123?exercise_type=squat"
```

## Database Schema

### Tables

**users**
- user_id (PK) - Integer
- age - Integer
- weight_kg - Float
- height_cm - Float
- fitness_goal - String(50)
- created_at - DateTime

**food_items**
- food_id (PK) - Integer
- food_name - String(100), unique
- protein_per_100g - Float
- carbs_per_100g - Float
- fats_per_100g - Float
- calories_per_100g - Float
- food_category - String(50)

**meal_logs**
- meal_id (PK) - Integer
- user_id (FK) - Integer
- meal_image_path - String(255)
- detected_foods - JSON
- total_protein - Float
- total_calories - Float
- confidence_score - Float
- meal_timestamp - DateTime

**health_metrics**
- id (PK) - Integer
- user_id (FK) - Integer
- date - Date
- sleep_duration - Float
- sleep_quality - Float
- hrv - Float
- activity_level - Float
- stress_level - Integer
- created_at - DateTime

**protein_recommendations**
- rec_id (PK) - Integer
- user_id (FK) - Integer
- date - Date
- baseline_protein - Float
- adjusted_protein - Float
- sleep_factor - Float
- hrv_factor - Float
- activity_factor - Float
- shap_explanation - JSON
- created_at - DateTime

## Setup & Running

### Install Dependencies
```bash
pip install -r requirements.txt
pip install psycopg2-binary python-multipart opencv-python
```

### Configure Environment
Create `.env`:
```env
NUTRITION_DATABASE_URL=postgresql://postgres:fitbalance123@localhost:5432/fitbalance_nutrition
GEMINI_API_KEY=your_key_here
```

### Initialize Database
```bash
python backend/database/init_database.py
```

### Run Server
```bash
uvicorn backend.api.main:app --reload --port 8000
```

## Testing

```bash
# Run test suite
python backend/test_api.py

# Manual API testing
curl http://localhost:8000/
curl http://localhost:8000/health
```

## ML Models

### Burnout Prediction Model
- **Type:** Cox Proportional Hazards
- **Performance:** C-index = 0.842
- **Features:** 15 features including sleep, stress, workout frequency, HRV
- **Location:** `ml_models/burnout/cox_model.pkl`

### Nutrition Analysis Model
- **Type:** CNN-GRU with Gemini Vision API fallback
- **Features:** Image-based food detection
- **Database:** 80+ Indian foods with nutritional data

## Performance

- **API Response Time (p95):** < 500ms
- **Database Query Time:** < 100ms
- **ML Inference Time:** 200-500ms per request
- **Image Processing:** 1-3 seconds per meal analysis

## Error Handling

The system includes comprehensive error handling:

1. **Database Connection Failures:** Falls back to in-memory storage
2. **ML Model Loading Errors:** Uses mock implementations
3. **API Integration Failures:** Graceful degradation
4. **Image Validation:** Rejects non-food images
5. **Input Validation:** Comprehensive parameter checking

## Security Considerations

- [ ] Use environment variables for secrets
- [ ] Implement rate limiting
- [ ] Add authentication (TODO)
- [ ] Enable CORS properly
- [ ] Validate all inputs
- [ ] Sanitize file uploads

## Troubleshooting

### Database Connection Failed
- Verify PostgreSQL is running
- Check DATABASE_URL in .env
- Test connection: `psql $DATABASE_URL`

### Model Loading Errors
- Verify model files exist in ml_models/
- Check Python path configuration
- Review model file permissions

### API Returns 500 Errors
- Check logs in terminal
- Verify all dependencies installed
- Test individual modules

### Image Processing Issues
- Verify OpenCV installation
- Check image file formats
- Review file permissions

## Next Steps

1. Implement user authentication
2. Add API rate limiting
3. Set up Redis caching
4. Implement WebSocket for real-time updates
5. Add comprehensive logging
6. Set up monitoring (Prometheus/Grafana)
7. Deploy to production environment

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Use type hints for all functions
5. Handle errors gracefully with proper logging

