# Gemini AI Nutrition Analysis Guide

## How It Works

The FitBalance nutrition system uses **Google's Gemini 2.5 Flash Vision API** to analyze meal photos and provide detailed nutritional information.

## What Gemini Does

### 1. Food Detection
When you upload a meal photo, Gemini:
- **Identifies all visible foods** in the image
- **Breaks down complex dishes** into individual ingredients (e.g., a salad → lettuce, tomatoes, chicken, dressing)
- **Recognizes preparation methods** (grilled, steamed, fried, etc.)
- **Estimates portion sizes** in grams based on visual cues

### 2. Nutritional Analysis
For each detected food item, Gemini provides:
- **Protein content** (grams)
- **Total calories**
- **Carbohydrates** (grams)
- **Fats** (grams)
- **Fiber** (grams)
- **Confidence score** (0.0 - 1.0) indicating detection certainty

### 3. Intelligent Portion Estimation
Gemini estimates realistic portion sizes:
- Typical chicken breast: 150-200g
- Side of vegetables: 80-120g
- Grain portions: 100-150g

### 4. Quality Assessment
The system also provides:
- **Meal type classification** (breakfast/lunch/dinner/snack)
- **Overall quality rating** (high/medium/low)
- **Personalized recommendations** based on your fitness goals

## Current Training

Gemini is pre-trained on:
- ✅ Extensive food image datasets
- ✅ Nutritional databases (USDA, international food databases)
- ✅ Visual portion estimation
- ✅ Multi-cultural cuisine recognition

### Enhanced Prompt Engineering
We've optimized Gemini with a detailed prompt that:
1. Requests structured JSON output with specific nutrition fields
2. Instructs it to calculate totals based on estimated portions
3. Asks for confidence scores on each detection
4. Requires detailed breakdown of complex dishes
5. Normalizes food names (using underscores, common identifiers)

## Expected Output Format

```json
{
  "detected_foods": [
    {
      "name": "grilled_chicken_breast",
      "confidence": 0.92,
      "estimated_grams": 180,
      "protein_per_100g": 31.0,
      "calories_per_100g": 165,
      "total_protein": 55.8,
      "total_calories": 297,
      "preparation_method": "grilled",
      "visual_description": "grilled chicken breast, appears well-cooked"
    },
    {
      "name": "steamed_broccoli",
      "confidence": 0.88,
      "estimated_grams": 100,
      "protein_per_100g": 2.8,
      "calories_per_100g": 34,
      "total_protein": 2.8,
      "total_calories": 34,
      "preparation_method": "steamed"
    }
  ],
  "meal_summary": {
    "total_items": 2,
    "meal_type": "lunch",
    "overall_quality": "high",
    "notes": "Well-balanced meal with lean protein and vegetables"
  }
}
```

## Fallback System

If Gemini detection fails or returns incomplete data:
1. **Database lookup** - checks comprehensive nutrition database
2. **Fuzzy matching** - tries to match similar food names
3. **Category-based estimation** - estimates based on food type (protein, vegetable, carb, etc.)

## Improving Accuracy

### For Best Results:
✅ **Good lighting** - ensure the meal is well-lit
✅ **Clear angle** - take photo from directly above or slight angle
✅ **Single plate** - focus on one plate/meal at a time
✅ **Minimal clutter** - remove unnecessary items from frame
✅ **Standard dishware** - helps with portion estimation

### What Can Affect Accuracy:
⚠️ Dark/blurry images
⚠️ Mixed dishes (casseroles, stews) - harder to separate ingredients
⚠️ Heavily sauced/covered foods
⚠️ Non-food items in frame
⚠️ Unusual plating or tiny portions

## Testing & Validation

The system logs detailed information for each analysis:
- Raw Gemini response
- Parsed nutrition data
- Confidence scores
- Fallback database matches

Check backend logs for detailed analysis output.

## API Configuration

The Gemini API key is configured in `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

Current model: **gemini-2.5-flash** (optimized for speed and accuracy)

## Future Enhancements

Planned improvements:
- [ ] User feedback loop to refine estimates
- [ ] Custom portion size calibration per user
- [ ] Meal history tracking for better recommendations
- [ ] Integration with fitness wearables for activity-based targets
- [ ] Support for recipe/meal prep batch analysis

---

## Quick Start

1. **Upload a meal photo** via the nutrition page
2. **Wait 2-3 seconds** for Gemini analysis
3. **Review detected foods** and nutritional breakdown
4. **Check confidence scores** - higher = more accurate
5. **Get personalized recommendations** based on your goals

The system continuously improves as Google updates Gemini's training data.
