#!/usr/bin/env python3
"""
Data Augmentation Script for Weapon Detection
Increases training data diversity to improve model accuracy
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataAugmentor:
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define augmentation pipeline
        self.augmentations = A.Compose([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomFog(p=0.1),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
        ], p=1.0)
    
    def augment_dataset(self, multiplier=3):
        """Augment the dataset by creating multiple variations of each image"""
        
        images_dir = os.path.join(self.source_dir, 'images')
        labels_dir = os.path.join(self.source_dir, 'labels')
        
        output_images_dir = os.path.join(self.output_dir, 'images')
        output_labels_dir = os.path.join(self.output_dir, 'labels')
        
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        image_files = list(Path(images_dir).glob('*.jpg'))
        print(f"Found {len(image_files)} images to augment")
        
        total_created = 0
        
        for img_path in image_files:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Load corresponding label
            label_path = os.path.join(labels_dir, img_path.stem + '.txt')
            
            # Copy original files first
            cv2.imwrite(os.path.join(output_images_dir, img_path.name), image)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_content = f.read()
                with open(os.path.join(output_labels_dir, img_path.stem + '.txt'), 'w') as f:
                    f.write(label_content)
            
            # Create augmented versions
            for i in range(multiplier):
                # Apply augmentation
                augmented = self.augmentations(image=image)
                aug_image = augmented['image']
                
                # Save augmented image
                aug_name = f"{img_path.stem}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(output_images_dir, aug_name), aug_image)
                
                # Copy label for augmented image
                if os.path.exists(label_path):
                    aug_label_name = f"{img_path.stem}_aug_{i}.txt"
                    with open(os.path.join(output_labels_dir, aug_label_name), 'w') as f:
                        f.write(label_content)
                
                total_created += 1
        
        print(f"✅ Created {total_created} augmented images")
        print(f"Total dataset size: {len(image_files) + total_created} images")

def main():
    """Main function to run data augmentation"""
    
    source_dataset = "test_images/weapon-detection.v4i.yolov5pytorch/train"
    output_dataset = "augmented_weapon_dataset"
    
    print("🎨 Starting data augmentation...")
    
    augmentor = DataAugmentor(source_dataset, output_dataset)
    augmentor.augment_dataset(multiplier=2)  # Create 2x more data
    
    print("✅ Data augmentation completed!")
    print(f"Augmented dataset saved to: {output_dataset}")

if __name__ == "__main__":
    main()
