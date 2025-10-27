# Indian Food Dataset Analysis & Synthetic Data Generation Plan

## 📊 Current Dataset Analysis

### **What You Have:**
- ✅ **80 food categories** with images
- ✅ **~50 images per category** (estimated 4,000 total images)
- ✅ Covers traditional Indian dishes from all regions

### **Categories Present:**
- North Indian: butter_chicken, naan, dal_makhani, paneer_butter_masala, chapati
- South Indian: kuzhi_paniyaram, unni_appam, phirni
- Desserts: gulab_jamun, jalebi, rasgulla, gajar_ka_halwa, mysore_pak
- Street Food: kachori, aloo_tikki, bhatura
- Regional: maach_jhol (Bengali), litti_chokha (Bihari)

---

## ❌ Critical Missing Foods (High Priority for Indian Diet)

### **1. Breakfast Items** (Most Common!)
- ❌ Dosa (all varieties: masala, plain, rava, set)
- ❌ Idli
- ❌ Vada (medu vada, masala vada)
- ❌ Upma
- ❌ Uttapam
- ❌ Pongal
- ❌ Appam
- ❌ Puttu
- ❌ Paratha (aloo, gobi, methi)

### **2. Daily Staples**
- ❌ Sambar
- ❌ Rasam
- ❌ Curd Rice
- ❌ Lemon Rice
- ❌ Tamarind Rice
- ❌ Pulao
- ❌ Jeera Rice
- ❌ Plain Rice with Dal

### **3. Main Course (Common)** 
- ❌ Fish Curry
- ❌ Prawn Masala
- ❌ Egg Curry
- ❌ Mixed Veg Curry
- ❌ Aloo Curry
- ❌ Rajma
- ❌ Chole
- ❌ Pav Bhaji

### **4. Snacks & Street Food**
- ❌ Samosa (CRITICAL - very common!)
- ❌ Pakora/Bhajji
- ❌ Pani Puri/Golgappa
- ❌ Bhel Puri
- ❌ Sev Puri
- ❌ Dahi Puri
- ❌ Vada Pav
- ❌ Misal Pav
- ❌ Pav Bhaji

### **5. Breads**
- ❌ Roti (plain)
- ❌ Puri
- ❌ Bhakri
- ❌ Kulcha
- ❌ Rumali Roti

### **6. Accompaniments**
- ❌ Raita
- ❌ Pickle/Achar
- ❌ Papad
- ❌ Chutney (coconut, mint, tomato)

### **7. Beverages**
- ❌ Chai/Tea
- ❌ Coffee
- ❌ Buttermilk/Chaas
- ❌ Jaljeera
- ❌ Nimbu Pani

---

## 🎯 Recommended Dataset Size

### **For Good CNN Accuracy:**
```
Minimum: 100-200 images per class × 150 classes = 15,000-30,000 images
Recommended: 300-500 images per class × 150 classes = 45,000-75,000 images
Optimal: 1,000+ images per class × 150 classes = 150,000+ images
```

### **Your Target:**
```
Current: 80 classes × 50 images = 4,000 images
Add Missing: 70 new classes × 300 images = 21,000 images
Current Dataset Augmentation: 80 × 250 more = 20,000 images
---
TOTAL TARGET: 150 classes × 300 images = 45,000 images
```

---

## 🤖 Synthetic Data Generation Strategy

### **Option 1: Use Stable Diffusion (BEST for Food)** ✅ RECOMMENDED
```python
# Using Stable Diffusion XL with Food LoRA
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)

# Generate with food-specific prompts
prompt = """
High-quality photo of {food_name}, traditional Indian cuisine,
served on a white plate, natural lighting, appetizing, 
restaurant quality, professional food photography, 
detailed texture, vibrant colors, top view
"""

# Generate 300 variations per food
for i in range(300):
    image = pipe(
        prompt=prompt.format(food_name="masala dosa"),
        negative_prompt="cartoon, drawing, low quality, blurry",
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    image.save(f"dosa_{i}.jpg")
```

**Advantages:**
- ✅ High quality, realistic images
- ✅ Full control over variations
- ✅ Can generate unlimited images
- ✅ Fast (~5-10 seconds per image on GPU)

**Requirements:**
- GPU with 8GB+ VRAM (or use Google Colab free GPU)
- ~20GB storage for model
- Python packages: diffusers, transformers, torch

---

### **Option 2: Use AI Image Generation APIs**
```python
# OpenAI DALL-E 3
import openai

response = openai.Image.create(
    prompt="Professional photo of masala dosa on banana leaf",
    n=1,
    size="1024x1024"
)

# Cost: ~$0.04 per image
# 45,000 images × $0.04 = $1,800 (EXPENSIVE!)
```

**Alternatives (Cheaper):**
- Stability AI API: ~$0.002 per image ($90 for 45k images)
- Replicate.com: Pay per generation
- Midjourney: $30/month unlimited (best value!)

---

### **Option 3: Web Scraping + Data Augmentation** ✅ FREE
```python
# 1. Scrape images from food websites
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

sources = [
    "https://www.zomato.com/",
    "https://www.swiggy.com/",
    "Google Images",
    "Indian food blogs",
    "Recipe websites"
]

# 2. Data Augmentation (multiply existing images)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Generate 5 variations from each image
for img in original_images:
    for i in range(5):
        augmented = datagen.random_transform(img)
        save_image(augmented)

# Result: 50 images × 6 (original + 5 augmented) = 300 images per class
```

**Advantages:**
- ✅ FREE
- ✅ Real food images
- ✅ Quick to implement
- ✅ No GPU needed

---

### **Option 4: Hybrid Approach** ✅ RECOMMENDED
```
1. Current Dataset (80 classes × 50 images) = 4,000 base images
2. Augment 6x (rotation, flip, brightness) = 24,000 images
3. Scrape missing foods (70 classes × 100 images) = 7,000 images
4. Augment scraped 3x = 21,000 images
5. Generate 30 synthetic per missing class (70 × 30) = 2,100 images
---
TOTAL: 55,100 images (150 classes × ~370 avg images)
```

---

## 📸 How to Get Images

### **Method 1: Web Scraping (Fastest)**
```python
# google_images_download (deprecated but works with selenium)
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
arguments = {
    "keywords": "masala dosa indian food",
    "limit": 100,
    "print_urls": True,
    "format": "jpg",
    "size": "medium"
}
response.download(arguments)
```

### **Method 2: Bing Image Search API** (Better)
```python
import requests

subscription_key = "YOUR_KEY"
search_url = "https://api.bing.microsoft.com/v7.0/images/search"

headers = {"Ocp-Apim-Subscription-Key": subscription_key}
params = {
    "q": "masala dosa indian food",
    "count": 100,
    "imageType": "photo"
}

response = requests.get(search_url, headers=headers, params=params)
images = response.json()["value"]

for img in images:
    download_image(img["contentUrl"])
```

**Cost:** Free tier: 1,000 images/month

### **Method 3: Flickr API** (Best Quality)
```python
import flickrapi

flickr = flickrapi.FlickrAPI(api_key, api_secret)
photos = flickr.photos.search(
    text='masala dosa',
    per_page=100,
    extras='url_o,url_l'
)

# Download high-res images
for photo in photos['photos']['photo']:
    download_image(photo.get('url_l'))
```

### **Method 4: Food Dataset Repositories**
- Kaggle: Search "Indian food" datasets
- Roboflow: Pre-labeled food datasets
- GitHub: Indian food recognition datasets

---

## 🏋️ Training Strategy

### **Phase 1: Transfer Learning (Fast, Good Accuracy)**
```python
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

# Load pre-trained model
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers
base_model.trainable = False

# Add custom head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(150, activation='softmax')  # 150 food classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
```

**Expected Accuracy:** 85-92% on test set

### **Phase 2: Fine-Tuning (Better Accuracy)**
```python
# Unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Retrain with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data
)
```

**Expected Accuracy:** 90-95% on test set

---

## 📈 Accuracy Calculation

```python
# 1. Split data
train_split = 0.7  # 70% training
val_split = 0.15   # 15% validation
test_split = 0.15  # 15% testing

# 2. Evaluate
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 3. Per-class accuracy
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_labels

print(classification_report(y_true, y_pred_classes, 
                           target_names=food_class_names))

# 4. Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(cm, food_class_names)

# 5. Top-k accuracy
top5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(
    y_true, y_pred, k=5
)
```

---

## 🎯 Action Plan (What I Can Help With)

### **Option A: Quick Start (Augmentation Only)** - 1 hour
1. Augment existing 4,000 images to 24,000
2. Train on current 80 classes
3. Get baseline accuracy
4. ✅ Fast, uses what you have

### **Option B: Web Scraping** - 4-6 hours  
1. Scrape 100 images for 70 missing foods = 7,000 images
2. Augment to 21,000 images
3. Combine with current dataset = 45,000 total
4. Train on 150 classes
5. ✅ Free, good quality

### **Option C: Stable Diffusion** - 8-12 hours
1. Set up Stable Diffusion on Colab
2. Generate 300 images per missing class = 21,000 images
3. Combine with current dataset = 45,000+ total
4. Train on 150 classes
5. ✅ Best quality, full control

### **Option D: Hybrid (RECOMMENDED)** - 6-8 hours
1. Augment current 6x = 24,000 images
2. Scrape 50 images per missing class = 3,500 images
3. Augment scraped 3x = 10,500 images
4. Generate 50 synthetic per critical missing food (20 foods) = 1,000 images
5. TOTAL: ~40,000 images, 150 classes
6. ✅ Best balance of quality, cost, and time

---

## 💰 Cost Comparison

| Method | Cost | Time | Quality | Total Images |
|--------|------|------|---------|--------------|
| Augmentation Only | $0 | 1 hour | Good | 24,000 |
| Web Scraping | $0 | 6 hours | Excellent | 45,000 |
| Stable Diffusion (Local) | $0* | 12 hours | Very Good | 45,000+ |
| Stable Diffusion (Colab) | $10/month | 8 hours | Very Good | 45,000+ |
| DALL-E 3 | $1,800 | 2 hours | Excellent | 45,000 |
| Hybrid | $0-10 | 8 hours | Excellent | 40,000+ |

*Requires GPU

---

## 🚀 Which Option Do You Want?

**A)** Quick start with augmentation (get results in 1 hour)
**B)** Web scraping + augmentation (best free option)
**C)** Stable Diffusion generation (highest quality synthetic)
**D)** Hybrid approach (RECOMMENDED - balanced)

Let me know and I'll write the complete code for your chosen approach!
