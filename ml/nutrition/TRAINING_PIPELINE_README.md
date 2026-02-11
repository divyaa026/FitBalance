# 🍛 Indian Food Classification - Complete Training Pipeline

## 📋 Overview

Complete ML pipeline to train an Indian food classifier from your existing dataset of 80 food categories with ~50 images each.

**Quick Stats:**
- 📊 Current Dataset: 80 classes × 50 images = **4,000 images**
- 🎯 After Augmentation: 80 classes × ~300 images = **24,000 images**  
- 🏆 Expected Accuracy: **85-90%** (after training)
- ⏱️ Training Time: **1-2 hours** (GPU) or **2-3 hours** (CPU)

---

## 🚀 Quick Start (Option A - MVP)

### Step 1: Run Complete Pipeline

```powershell
# From project root
python ml_models/nutrition/run_complete_pipeline.py
```

This will:
1. ✨ Augment your 4,000 images → 24,000 images (30 min)
2. 🧠 Train EfficientNetB3 model with transfer learning (1-2 hours)
3. 📊 Evaluate model and generate metrics (10 min)

**Total Time: 1.5 - 3 hours**

### Step 2: Integrate with Backend

```powershell
# Create integration files
python ml_models/nutrition/step4_integrate_backend.py

# Create nutrition database
python ml_models/nutrition/step5_create_nutrition_db.py
```

### Step 3: Update Backend Code

Follow instructions in `ml_models/nutrition/INTEGRATION_GUIDE.md`

---

## 📂 Pipeline Structure

```
ml_models/nutrition/
├── step1_augment_data.py          # Data augmentation (4K → 24K images)
├── step2_train_model.py           # EfficientNetB3 training
├── step3_evaluate_model.py        # Model evaluation & metrics
├── step4_integrate_backend.py     # Backend integration helper
├── step5_create_nutrition_db.py   # Nutrition database creation
├── run_complete_pipeline.py       # Master orchestrator
├── trained_classifier.py          # Auto-generated model wrapper
└── INTEGRATION_GUIDE.md           # Backend integration instructions
```

---

## 🔬 Detailed Steps

### Step 1: Data Augmentation

**Script:** `step1_augment_data.py`

**What it does:**
- Takes your 4,000 original images
- Applies 5 augmentations per image:
  - ✅ Rotation (±20°)
  - ✅ Horizontal/Vertical shift (20%)
  - ✅ Zoom (80-120%)
  - ✅ Brightness (70-130%)
  - ✅ Horizontal flip
- Outputs: **24,000 augmented images**

**Run individually:**
```powershell
python ml_models/nutrition/step1_augment_data.py
```

**Output:**
```
nutrition/augmented_indian_food_dataset/
├── class_1/
│   ├── original_1.jpg
│   ├── aug_1_0.jpg
│   ├── aug_1_1.jpg
│   └── ...
├── class_2/
└── ...
```

**Time:** ~30 minutes

---

### Step 2: Model Training

**Script:** `step2_train_model.py`

**Architecture:**
- **Base Model:** EfficientNetB3 (pre-trained on ImageNet)
- **Custom Head:** Global Average Pooling → Dense(256) → Dropout(0.5) → Dense(80)
- **Input Size:** 224×224×3

**Training Strategy:**

**Phase 1: Train Custom Layers (30 epochs)**
- Freeze base model (EfficientNetB3)
- Train only custom head
- Learning Rate: 1e-3
- Optimizer: Adam

**Phase 2: Fine-tune All Layers (20 epochs)**
- Unfreeze entire model
- Fine-tune all weights
- Learning Rate: 1e-5 (lower for stability)
- Optimizer: Adam

**Callbacks:**
- ✅ **ModelCheckpoint** - Save best model based on validation accuracy
- ✅ **EarlyStopping** - Stop if no improvement for 5 epochs
- ✅ **ReduceLROnPlateau** - Reduce LR if validation accuracy plateaus

**Run individually:**
```powershell
python ml_models/nutrition/step2_train_model.py
```

**Output:**
```
ml_models/nutrition/models/indian_food_YYYYMMDD_HHMMSS/
├── best_model.keras           # Best model (highest val accuracy)
├── final_model.keras          # Final model after all epochs
├── class_names.json           # List of 80 food classes
├── config.json                # Training configuration
└── training_history.json      # Loss/accuracy per epoch
```

**Time:** 1-2 hours (GPU) or 2-3 hours (CPU)

**Expected Accuracy:**
- Training: 90-95%
- Validation: 85-90%
- Test: 85-90%

---

### Step 3: Model Evaluation

**Script:** `step3_evaluate_model.py`

**Metrics:**
- ✅ **Overall Accuracy**
- ✅ **Top-5 Accuracy**
- ✅ **Per-Class Accuracy**
- ✅ **Confusion Matrix**
- ✅ **Classification Report** (precision, recall, F1-score)

**Visualizations:**
- 📊 Confusion matrix heatmap
- 📈 Training history plots
- 🖼️ Random sample predictions

**Run individually:**
```powershell
python ml_models/nutrition/step3_evaluate_model.py
```

**Output:**
```
ml_models/nutrition/models/indian_food_YYYYMMDD_HHMMSS/
├── evaluation_report.csv      # Per-class metrics
├── confusion_matrix.png       # Visual confusion matrix
└── evaluation_results.json    # JSON summary
```

**Time:** ~10 minutes

---

## 🔗 Backend Integration

### Step 4: Integration Setup

**Script:** `step4_integrate_backend.py`

**What it does:**
- Creates `trained_classifier.py` - wrapper for your model
- Generates `INTEGRATION_GUIDE.md` - step-by-step integration

**Run:**
```powershell
python ml_models/nutrition/step4_integrate_backend.py
```

**Manual Integration (Option 1):**

Edit `backend/modules/nutrition.py`:

```python
# Add at top
import sys
from pathlib import Path
ml_models_path = Path(__file__).parent.parent.parent / "ml_models"
sys.path.insert(0, str(ml_models_path))

from nutrition.trained_classifier import TrainedFoodClassifier

# In ProteinOptimizer.__init__
self.trained_classifier = TrainedFoodClassifier()

# Replace _detect_foods method
def _detect_foods(self, image_path: str) -> List[FoodItem]:
    # Try trained model first
    if self.trained_classifier.is_loaded():
        predictions = self.trained_classifier.predict(image_path, top_k=3)
        detected_foods = []
        for food_name, confidence in predictions:
            nutrition = self._get_nutrition_for_food(food_name)
            food_item = FoodItem(
                name=food_name,
                confidence=confidence,
                protein_content=nutrition['protein'],
                # ... other fields
            )
            detected_foods.append(food_item)
        if detected_foods:
            return detected_foods
    
    # Fallback to Gemini
    return self._detect_foods_with_gemini(image_path)
```

---

### Step 5: Nutrition Database

**Script:** `step5_create_nutrition_db.py`

**What it does:**
- Creates comprehensive nutrition database for 35+ Indian foods
- Includes: protein, calories, carbs, fat, fiber (per 100g and per serving)
- Categorizes foods, marks vegetarian/vegan

**Run:**
```powershell
python ml_models/nutrition/step5_create_nutrition_db.py
```

**Output:**
```
nutrition/
├── indian_food_nutrition_db.json  # JSON database
└── indian_food_nutrition.py       # Python module
```

**Usage:**
```python
from nutrition.indian_food_nutrition import get_nutrition_db

db = get_nutrition_db()
nutrition = db.get_nutrition("dosa")
# {'protein_per_serving': 1.5, 'calories_per_serving': 72, ...}

high_protein = db.get_high_protein_foods(min_protein_per_serving=10)
veg_foods = db.get_vegetarian_foods()
```

---

## 📊 Expected Results

### Model Performance

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | 90-95% |
| Validation Accuracy | 85-90% |
| Test Accuracy | 85-90% |
| Top-5 Accuracy | 95-98% |

### Best Performing Classes
(Typically foods with distinct visual features)
- Dosa
- Idli
- Naan
- Biryani
- Samosa
- Gulab Jamun

### Challenging Classes
(Foods that look similar)
- Chapati vs Roti
- Different types of Dal
- Similar curries

---

## 🐛 Troubleshooting

### Issue: Low Accuracy (<80%)

**Solutions:**
1. Run more augmentation (increase to 10 augmentations/image)
2. Train for more epochs (increase Phase 1 to 40, Phase 2 to 30)
3. Use EfficientNetB4 (larger model)
4. Check for mislabeled images

### Issue: Overfitting (Train >> Val accuracy)

**Solutions:**
1. Increase dropout (0.5 → 0.6)
2. Add more augmentation
3. Early stopping is working - use best_model.keras
4. Reduce model complexity

### Issue: Out of Memory (GPU)

**Solutions:**
1. Reduce batch size (32 → 16 or 8)
2. Use EfficientNetB0 (smaller model)
3. Train on CPU (slower but works)

### Issue: Slow Training

**Solutions:**
1. Reduce epochs for testing (Phase 1: 10, Phase 2: 5)
2. Use smaller validation split (0.2 → 0.1)
3. Reduce augmentations (5 → 3)

---

## 📈 Next Steps (After MVP)

### Phase 2: Scale to 150 Classes (Option D)

**Missing 70 Indian Foods:**

**Critical Breakfast Items:**
- Poha, Kanda Poha, Rava Upma, Vermicelli Upma, Pesarattu

**Snacks:**
- Bhel Puri, Pani Puri, Dahi Puri, Sev Puri, Ragda Pattice

**Main Dishes:**
- Chole Bhature, Pav Bhaji, Misal Pav, Vada Pav, Aloo Paratha

**Regional Specialties:**
- Mysore Dosa, Rava Dosa, Set Dosa, Masala Dosa variants

**Action Plan:**

1. **Web Scraping** (3-4 days)
   ```powershell
   python ml_models/nutrition/web_scraper.py --foods missing_foods.txt
   ```
   - Target: 200-300 images per class
   - Sources: Google Images, Flickr, Unsplash

2. **Manual Collection** (1-2 days)
   - Restaurant menus
   - Food delivery apps
   - YouTube cooking videos (frame extraction)

3. **Synthetic Generation** (1 day)
   ```powershell
   python ml_models/nutrition/synthetic_generator.py --min-images 300
   ```
   - Use DALL-E/Stable Diffusion for rare foods
   - Target: Fill gaps to reach 300 images/class

4. **Retrain** (2-3 hours)
   ```powershell
   python ml_models/nutrition/run_complete_pipeline.py --classes 150
   ```
   - Expected accuracy: **90-93%** on 150 classes

---

## 🎯 Success Metrics

### MVP Success (Current - 80 classes)
- ✅ Model accuracy ≥85%
- ✅ Integration with backend working
- ✅ Can detect top Indian foods (dal, rice, roti, dosa, idli)
- ✅ Nutrition data available for detected foods

### Full Success (Phase 2 - 150 classes)
- ✅ Model accuracy ≥90%
- ✅ Covers 95% of common Indian meals
- ✅ Regional cuisine coverage (North, South, East, West)
- ✅ Breakfast, lunch, dinner, snacks all covered

---

## 📝 File Outputs Summary

### After Complete Pipeline

```
ml_models/nutrition/
└── models/
    └── indian_food_YYYYMMDD_HHMMSS/
        ├── best_model.keras              # Use this for production
        ├── final_model.keras
        ├── class_names.json
        ├── config.json
        ├── training_history.json
        ├── evaluation_report.csv
        ├── confusion_matrix.png
        └── evaluation_results.json

nutrition/
├── augmented_indian_food_dataset/       # 24,000 images
├── indian_food_nutrition_db.json
└── indian_food_nutrition.py

ml_models/nutrition/
├── trained_classifier.py                # Auto-generated
└── INTEGRATION_GUIDE.md                 # Auto-generated
```

---

## 🔧 Configuration

### Modify Training Parameters

Edit `step2_train_model.py`:

```python
# Line ~20-30
IMG_SIZE = 224           # Change to 299 for EfficientNetB4
BATCH_SIZE = 32          # Reduce if OOM (16, 8)
EPOCHS_PHASE1 = 30       # Initial training
EPOCHS_PHASE2 = 20       # Fine-tuning
LEARNING_RATE_1 = 1e-3   # Phase 1 LR
LEARNING_RATE_2 = 1e-5   # Phase 2 LR
DROPOUT = 0.5            # Increase if overfitting
```

### Modify Augmentation

Edit `step1_augment_data.py`:

```python
# Line ~30-40
datagen = ImageDataGenerator(
    rotation_range=20,           # ± degrees
    width_shift_range=0.2,       # % of width
    height_shift_range=0.2,      # % of height
    zoom_range=0.2,              # 80-120%
    brightness_range=[0.7, 1.3], # 70-130%
    horizontal_flip=True,
    fill_mode='nearest'
)

augmentations_per_image = 5      # Increase for more data
```

---

## 💡 Tips & Best Practices

### Training
1. ✅ Always use `best_model.keras` (highest validation accuracy)
2. ✅ Monitor training logs - watch for overfitting
3. ✅ Start with fewer epochs for testing, then scale up
4. ✅ Use GPU if available (10x faster)

### Data Quality
1. ✅ Check for mislabeled images before training
2. ✅ Remove duplicate/corrupted images
3. ✅ Ensure similar image sizes (Gemini API may have variations)
4. ✅ Verify class balance (~equal images per class)

### Integration
1. ✅ Test with Gemini fallback (handles unknown foods)
2. ✅ Add confidence threshold (e.g., only accept >0.5 confidence)
3. ✅ Log predictions for monitoring
4. ✅ Create nutrition database for all 80 classes

---

## 🎓 Learning Resources

### Transfer Learning
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Food Classification
- [Food-101 Dataset](https://www.kaggle.com/dansbecker/food-101)
- [Indian Food Classification](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)

### Model Optimization
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/gpu_performance_analysis)
- [Keras Callbacks](https://keras.io/api/callbacks/)

---

## 📞 Support

### Common Questions

**Q: How long does training take?**
A: 1-2 hours on GPU (NVIDIA RTX 3060+), 2-3 hours on CPU (modern i7/i9)

**Q: What if I don't have a GPU?**
A: Training works on CPU, just takes longer. Reduce batch size if needed.

**Q: Can I add more food classes later?**
A: Yes! Add images to dataset, retrain with `run_complete_pipeline.py`

**Q: What accuracy should I expect?**
A: 85-90% on 80 classes, 90-93% after scaling to 150 classes

**Q: How do I fix low accuracy?**
A: 1) More training epochs, 2) More augmentations, 3) Larger model, 4) More data

---

## ✅ Checklist

### Before Training
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset exists at `nutrition/indian_food_dataset/`
- [ ] ~4,000 images across 80 classes
- [ ] GPU available (optional but recommended)

### After Training
- [ ] Model accuracy ≥85%
- [ ] `best_model.keras` exists
- [ ] Integration with backend complete
- [ ] Nutrition database created
- [ ] Tested with real food images
- [ ] Gemini fallback working

---

## 🎉 Let's Train!

```powershell
# Ready? Let's go!
python ml_models/nutrition/run_complete_pipeline.py
```

**Expected Timeline:**
- ⏱️ Augmentation: 30 min
- ⏱️ Training: 1-2 hours
- ⏱️ Evaluation: 10 min
- ⏱️ Integration: 30 min

**Total: 2-3 hours to production-ready model! 🚀**

---

*Generated by FitBalance ML Pipeline - Option A (Quick Start MVP)*
