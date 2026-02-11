"""
Step 2: Training Script
Trains EfficientNetB3 model on augmented Indian food dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianFoodClassifier:
    def __init__(self, dataset_dir, img_size=224, batch_size=32):
        """
        Args:
            dataset_dir: Path to augmented dataset
            img_size: Image size (224 for EfficientNetB3)
            batch_size: Batch size for training
        """
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def load_data(self, validation_split=0.2):
        """Load and prepare datasets"""
        logger.info("Loading datasets...")
        
        # Training data generator (with additional augmentation)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            # Additional augmentation during training
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1]
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            str(self.dataset_dir),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.val_generator = val_datagen.flow_from_directory(
            str(self.dataset_dir),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        logger.info(f"✓ Loaded {self.num_classes} food classes")
        logger.info(f"✓ Training samples: {self.train_generator.samples}")
        logger.info(f"✓ Validation samples: {self.val_generator.samples}")
        
        return self.train_generator, self.val_generator
    
    def build_model(self):
        """Build EfficientNetB3 model with custom head"""
        logger.info("Building model...")
        
        try:
            # Try EfficientNetB3 first
            logger.info("Attempting to load EfficientNetB3 with ImageNet weights...")
            base_model = EfficientNetB3(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_size, self.img_size, 3),
                pooling=None
            )
            logger.info("✓ Successfully loaded EfficientNetB3")
        except Exception as e:
            logger.warning(f"EfficientNetB3 failed: {e}")
            logger.info("Falling back to MobileNetV2...")
            # Fallback: Use MobileNetV2 (more stable)
            base_model = MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_size, self.img_size, 3),
                pooling=None
            )
            logger.info("✓ Successfully loaded MobileNetV2")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom head
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
        logger.info(f"✓ Model built with {self.num_classes} output classes")
        logger.info(f"✓ Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, epochs=30, fine_tune_epochs=20):
        """Train model in two phases"""
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"ml_models/nutrition/models/indian_food_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                str(output_dir / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
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
            ),
            TensorBoard(
                log_dir=str(output_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        # Phase 1: Train only the top layers
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Training custom layers (frozen base)")
        logger.info("="*60)
        
        history1 = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune the entire model
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Fine-tuning entire model")
        logger.info("="*60)
        
        # Unfreeze base model
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
        logger.info(f"✓ Unfroze base model for fine-tuning")
        
        # Continue training
        history2 = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=fine_tune_epochs,
            initial_epoch=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'phase1': history1.history,
            'phase2': history2.history
        }
        
        # Save final model
        final_model_path = output_dir / 'final_model.keras'
        self.model.save(final_model_path)
        logger.info(f"✓ Saved final model to {final_model_path}")
        
        # Save class names
        class_names_path = output_dir / 'class_names.json'
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        logger.info(f"✓ Saved class names to {class_names_path}")
        
        # Save training config
        config = {
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'phase1_epochs': epochs,
            'phase2_epochs': fine_tune_epochs,
            'total_epochs': epochs + fine_tune_epochs,
            'timestamp': timestamp
        }
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\n✅ Training complete! Model saved to {output_dir}")
        
        return self.history, output_dir
    
    def evaluate(self):
        """Evaluate model on validation set"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATING MODEL")
        logger.info("="*60)
        
        results = self.model.evaluate(self.val_generator, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        logger.info(f"\n📊 Validation Results:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy']*100:.2f}%")
        
        return metrics


def main():
    """Main training pipeline"""
    
    # Configuration
    DATASET_DIR = "nutrition/augmented_indian_food_dataset"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    PHASE1_EPOCHS = 30
    PHASE2_EPOCHS = 20
    
    logger.info("="*60)
    logger.info("INDIAN FOOD CLASSIFIER TRAINING")
    logger.info("="*60)
    logger.info(f"Dataset: {DATASET_DIR}")
    logger.info(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Phase 1 epochs: {PHASE1_EPOCHS}")
    logger.info(f"Phase 2 epochs: {PHASE2_EPOCHS}")
    logger.info("="*60 + "\n")
    
    # Initialize classifier
    classifier = IndianFoodClassifier(
        dataset_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load data
    classifier.load_data(validation_split=0.2)
    
    # Build model
    classifier.build_model()
    
    # Train model
    history, output_dir = classifier.train(
        epochs=PHASE1_EPOCHS,
        fine_tune_epochs=PHASE2_EPOCHS
    )
    
    # Evaluate model
    metrics = classifier.evaluate()
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"📁 Model directory: {output_dir}")
    logger.info(f"📊 Final accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"🎯 Top-5 accuracy: {metrics['top5_accuracy']*100:.2f}%")
    logger.info("="*60)
    
    logger.info("\n📝 Next steps:")
    logger.info("1. Run step3_evaluate_model.py to see detailed metrics")
    logger.info("2. Update backend/modules/nutrition.py to use the trained model")
    logger.info("3. Test the model with real food images!")


if __name__ == "__main__":
    # Enable GPU memory growth (if using GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✓ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU config error: {e}")
    else:
        logger.info("ℹ No GPU found, training on CPU")
    
    main()
