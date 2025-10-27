# ⚡ Quick Start - Train All Models

## 🚀 One Command to Rule Them All

```powershell
# 1. Activate environment
.\fitbalance_env\Scripts\Activate.ps1

# 2. Train everything (30-60 minutes)
python train_all_production_models.py
```

That's it! This will:
- ✅ Load the pre-generated datasets (1,000 biomechanics + 365,000 burnout records)
- 🏋️ Train GNN-LSTM model (50 epochs)
- 🏋️ Train Cox PH, Random Forest, Gradient Boosting models
- 💾 Save all trained models
- 📊 Generate evaluation metrics and visualizations

---

## 🎯 Individual Components

### If you want to train models separately:

```powershell
# Biomechanics GNN-LSTM (20-30 min)
python ml_models\biomechanics\train_production_model.py

# Burnout models (10-15 min)
python ml_models\burnout\train_production_model.py
```

### If you need to regenerate datasets:

```powershell
# Regenerate biomechanics dataset
python datasets\generate_biomechanics_dataset.py

# Regenerate burnout dataset
python datasets\generate_burnout_dataset.py
```

---

## 🔑 Setup Gemini API (Optional but Recommended)

```powershell
# Set your API key
$env:GEMINI_API_KEY="your-gemini-api-key-here"

# Test integration
python integrations\gemini_integration.py
```

Get your free API key at: https://makersuite.google.com/app/apikey

---

## 📊 After Training

### Check model files exist:

```powershell
# Biomechanics model
ls ml_models\biomechanics\gnn_lstm_best.pth

# Burnout models
ls ml_models\burnout\*.pkl
```

### Test model loading:

```python
# Python REPL
import torch
import joblib

# Load biomechanics model
model = torch.load('ml_models/biomechanics/gnn_lstm_best.pth')
print("✅ Biomechanics model loaded")

# Load burnout models
cox = joblib.load('ml_models/burnout/cox_ph_model.pkl')
rf = joblib.load('ml_models/burnout/random_forest_model.pkl')
print("✅ Burnout models loaded")
```

---

## 🐛 Troubleshooting

### Out of memory during training?
```powershell
# Edit train scripts to reduce batch size
# biomechanics: Line 444, change batch_size=16 to batch_size=8
# burnout: Uses entire dataset (no batching needed)
```

### CUDA errors?
```python
# Models automatically use CPU if CUDA unavailable
# Check with: python -c "import torch; print(torch.cuda.is_available())"
```

### Package errors?
```powershell
pip install --upgrade torch torchvision mediapipe scikit-learn lifelines
```

---

## ✅ You're Done When:

- [ ] `python train_all_production_models.py` completes successfully
- [ ] All model files exist in `ml_models/` subdirectories
- [ ] No errors in terminal output
- [ ] Ready to push to GitHub!

---

## 🚀 Push to GitHub

```powershell
git add .
git commit -m "feat: Add production ML models with comprehensive datasets

- Biomechanics: GNN-LSTM with 1,000 training sequences
- Burnout: Cox PH + RF + GB with 365,000 longitudinal records
- Gemini AI integration for personalized recommendations
- Master training pipeline for reproducibility"

git push origin main
```

**Note**: If models are large, consider using Git LFS or adding to `.gitignore` and hosting on cloud storage.

---

That's it! Your team can now start their work. Refer to:
- Person 2: `docs/PERSON_2_BACKEND_BURNOUT_TASKS.md`
- Person 3: `docs/PERSON_3_FRONTEND_TASKS.md`
- Person 4: `docs/PERSON_4_DEVOPS_TASKS.md`

🎉 Great work!
