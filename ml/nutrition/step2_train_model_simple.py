"""
Step 2: Training Script (Simplified & Robust)
Trains MobileNetV2 model on augmented Indian food dataset
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 20
DATASET_DIR = "nutrition/augmented_indian_food_dataset"

def create_model(num_classes):
    """Create a simple, robust model"""
    logger.info("Building model with MobileNetV2 backbone...")
    
    # Load MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create model using Sequential API (more stable)
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def load_datasets():
    """Load training and validation datasets"""
    logger.info("Loading datasets...")
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        horizontal_flip=True
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    logger.info(f"✓ Loaded {num_classes} food classes")
    logger.info(f"✓ Training samples: {train_generator.samples}")
    logger.info(f"✓ Validation samples: {val_generator.samples}")
    
    return train_generator, val_generator, class_names

def train_model():
    """Main training function"""
    
    logger.info("\n" + "="*60)
    logger.info("INDIAN FOOD CLASSIFIER TRAINING")
    logger.info("="*60)
    logger.info(f"Dataset: {DATASET_DIR}")
    logger.info(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Phase 1 epochs: {PHASE1_EPOCHS}")
    logger.info(f"Phase 2 epochs: {PHASE2_EPOCHS}")
    logger.info("="*60 + "\n")
    
    # Load data
    train_gen, val_gen, class_names = load_datasets()
    num_classes = len(class_names)
    
    # Create model
    model, base_model = create_model(num_classes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    logger.info(f"✓ Model built with {num_classes} classes")
    logger.info(f"✓ Total parameters: {model.count_params():,}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"ml_models/nutrition/models/indian_food_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Output directory: {output_dir}\n")
    
    # ===== PHASE 1: Train custom layers =====
    logger.info("="*60)
    logger.info("PHASE 1: Training custom layers (frozen base)")
    logger.info("="*60 + "\n")
    
    callbacks_phase1 = [
        ModelCheckpoint(
            str(output_dir / "phase1_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    logger.info("\n✅ Phase 1 complete!")
    logger.info(f"Best validation accuracy: {max(history1.history['val_accuracy']):.4f}\n")
    
    # ===== PHASE 2: Fine-tune entire model =====
    logger.info("="*60)
    logger.info("PHASE 2: Fine-tuning entire model")
    logger.info("="*60 + "\n")
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    logger.info(f"✓ Unfroze base model, now training {model.count_params():,} parameters\n")
    
    callbacks_phase2 = [
        ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    logger.info("\n✅ Phase 2 complete!")
    logger.info(f"Best validation accuracy: {max(history2.history['val_accuracy']):.4f}\n")
    
    # Save final model and metadata
    model.save(output_dir / "final_model.keras")
    
    with open(output_dir / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    config = {
        "model_type": "MobileNetV2",
        "img_size": IMG_SIZE,
        "num_classes": num_classes,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "training_date": datetime.now().isoformat()
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save training history
    history_combined = {
        "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()}
    }
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history_combined, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\n📁 Model saved to: {output_dir}")
    logger.info(f"\n📊 Final Results:")
    logger.info(f"  Training accuracy: {history2.history['accuracy'][-1]:.4f}")
    logger.info(f"  Validation accuracy: {history2.history['val_accuracy'][-1]:.4f}")
    logger.info(f"  Top-5 accuracy: {history2.history['top5_accuracy'][-1]:.4f}")
    logger.info(f"\n🎯 Best model: {output_dir / 'best_model.keras'}")
    logger.info(f"📝 Class names: {output_dir / 'class_names.json'}")
    logger.info("")

if __name__ == "__main__":
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            logger.info(f"  {gpu}")
    else:
        logger.info("ℹ No GPU found, training on CPU")
    
    logger.info("")
    
    # Train
    train_model()
