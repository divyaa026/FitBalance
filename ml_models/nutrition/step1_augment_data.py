"""
Step 1: Data Augmentation Script
Augments existing 4,000 images to 24,000 images using various transformations
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAugmenter:
    def __init__(self, source_dir, output_dir, augmentations_per_image=5):
        """
        Args:
            source_dir: Path to Indian Food Images/Indian Food Images
            output_dir: Path to save augmented images
            augmentations_per_image: Number of augmented versions per original image
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.augmentations_per_image = augmentations_per_image
        
        # Create augmentation generator
        self.datagen = ImageDataGenerator(
            rotation_range=20,           # Rotate ±20 degrees
            width_shift_range=0.2,       # Shift horizontally by 20%
            height_shift_range=0.2,      # Shift vertically by 20%
            shear_range=0.15,            # Shear transformation
            zoom_range=0.2,              # Zoom in/out by 20%
            horizontal_flip=True,        # Random horizontal flip
            brightness_range=[0.7, 1.3], # Brightness adjustment
            fill_mode='nearest'          # Fill empty pixels
        )
        
        self.stats = {
            'total_original': 0,
            'total_augmented': 0,
            'classes_processed': 0
        }
    
    def augment_class(self, class_name):
        """Augment all images in a single food class"""
        class_source_dir = self.source_dir / class_name
        class_output_dir = self.output_dir / class_name
        
        # Create output directory
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(class_source_dir.glob('*.jpg')) + \
                     list(class_source_dir.glob('*.jpeg')) + \
                     list(class_source_dir.glob('*.png'))
        
        if not image_files:
            logger.warning(f"No images found in {class_name}")
            return 0
        
        augmented_count = 0
        
        for img_path in tqdm(image_files, desc=f"Augmenting {class_name}", leave=False):
            try:
                # Load image
                img = load_img(str(img_path), target_size=(224, 224))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                
                # Save original image
                original_output_path = class_output_dir / f"{img_path.stem}_original.jpg"
                img.save(original_output_path, quality=95)
                augmented_count += 1
                
                # Generate augmented images
                i = 0
                for batch in self.datagen.flow(x, batch_size=1, save_prefix='aug', save_format='jpg'):
                    if i >= self.augmentations_per_image:
                        break
                    
                    # Convert to PIL Image
                    aug_img = Image.fromarray(batch[0].astype('uint8'))
                    
                    # Save augmented image
                    aug_output_path = class_output_dir / f"{img_path.stem}_aug_{i}.jpg"
                    aug_img.save(aug_output_path, quality=95)
                    
                    augmented_count += 1
                    i += 1
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        self.stats['total_original'] += len(image_files)
        self.stats['total_augmented'] += augmented_count
        
        return augmented_count
    
    def augment_all(self):
        """Augment all food classes"""
        # Get all class directories
        class_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(class_dirs)} food classes")
        logger.info(f"Will generate {self.augmentations_per_image + 1} images per original")
        logger.info(f"Output directory: {self.output_dir}")
        
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            count = self.augment_class(class_dir.name)
            self.stats['classes_processed'] += 1
            logger.info(f"✓ {class_dir.name}: {count} images generated")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("AUGMENTATION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Classes processed: {self.stats['classes_processed']}")
        logger.info(f"Original images: {self.stats['total_original']}")
        logger.info(f"Total images (with augmentations): {self.stats['total_augmented']}")
        logger.info(f"Expansion factor: {self.stats['total_augmented'] / max(self.stats['total_original'], 1):.1f}x")
        logger.info("="*60)
        
        return self.stats


def main():
    """Main execution"""
    # Configuration
    SOURCE_DIR = "nutrition/Indian Food Images/Indian Food Images"
    OUTPUT_DIR = "nutrition/augmented_indian_food_dataset"
    AUGMENTATIONS_PER_IMAGE = 5  # Will create 6x total (1 original + 5 augmented)
    
    logger.info("Starting data augmentation...")
    logger.info(f"Source: {SOURCE_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # Create augmenter
    augmenter = DataAugmenter(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        augmentations_per_image=AUGMENTATIONS_PER_IMAGE
    )
    
    # Run augmentation
    stats = augmenter.augment_all()
    
    logger.info("\n✅ Augmentation complete! Ready for training.")
    logger.info(f"📁 Dataset location: {OUTPUT_DIR}")
    logger.info(f"📊 Total images: {stats['total_augmented']}")
    logger.info(f"🎯 Food classes: {stats['classes_processed']}")


if __name__ == "__main__":
    main()
