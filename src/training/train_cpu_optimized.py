#!/usr/bin/env python3
"""
CPU-Optimized YOLO Model Retraining for Higher Accuracy
This script will retrain your weapon detection model optimized for CPU training
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def train_cpu_optimized_model():
    """Train YOLO model with parameters optimized for CPU training"""
    
    print("🖥️  Starting CPU-optimized model training...")
    print("⚠️  Note: CPU training will be slower but still effective")
    
    # Use YOLOv8 nano for faster CPU training
    model = YOLO('yolov8n.pt')  # Nano model for CPU efficiency
    
    # CPU-optimized training configuration
    training_config = {
        'data': 'test_images/weapon-detection.v4i.yolov5pytorch/data.yaml',
        'epochs': 50,            # Reduced epochs for CPU training
        'batch': 4,              # Smaller batch size for CPU
        'imgsz': 416,            # Smaller image size for faster training
        'patience': 15,          # Patience for early stopping
        'save_period': 5,        # Save checkpoint every 5 epochs
        'workers': 2,            # Fewer workers for CPU
        'device': 'cpu',         # Force CPU usage
        
        # Learning rate and optimizer
        'optimizer': 'AdamW',    # Good optimizer for small datasets
        'lr0': 0.001,           # Learning rate optimized for CPU
        'lrf': 0.01,            # Final learning rate fraction
        'momentum': 0.937,       # Momentum
        'weight_decay': 0.0005,  # L2 regularization
        'warmup_epochs': 3,      # Warmup epochs
        
        # Reduced data augmentation for faster training
        'hsv_h': 0.01,          # Light hue augmentation
        'hsv_s': 0.5,           # Saturation augmentation
        'hsv_v': 0.3,           # Value/brightness augmentation
        'degrees': 10.0,        # Rotation degrees
        'translate': 0.1,       # Translation fraction
        'scale': 0.8,           # Scale range
        'shear': 1.0,           # Shear degrees
        'perspective': 0.0,     # No perspective (CPU intensive)
        'flipud': 0.0,          # No vertical flip
        'fliplr': 0.5,          # Horizontal flip probability
        'mosaic': 0.8,          # Reduced mosaic augmentation
        'mixup': 0.1,           # Light mixup augmentation
        'copy_paste': 0.1,      # Light copy-paste augmentation
        
        # Loss function weights
        'box': 7.5,             # Box regression loss weight
        'cls': 0.5,             # Classification loss weight
        'dfl': 1.5,             # Distribution focal loss weight
        
        # CPU-optimized settings
        'cos_lr': True,         # Cosine learning rate scheduler
        'close_mosaic': 10,     # Disable mosaic in last N epochs
        'amp': False,           # Disable AMP for CPU
        'fraction': 1.0,        # Use full dataset
        'multi_scale': False,   # Disable multi-scale for CPU
        'plots': True,          # Generate training plots
        'val': True,            # Validate during training
        'save_json': True,      # Save results in JSON
    }
    
    print("📊 CPU Training Configuration:")
    key_params = ['epochs', 'batch', 'imgsz', 'optimizer', 'lr0', 'device']
    for param in key_params:
        print(f"   {param}: {training_config[param]}")
    
    print("\n🎯 Expected Results (CPU Training):")
    print("   - Current F1-Score: 18%")
    print("   - Target F1-Score: 40-60% (CPU limitations)")
    print("   - Better precision and recall")
    print("   - Reduced false positives")
    
    print(f"\n⏱️  Estimated training time: 1-2 hours on CPU")
    print("   (Progress will be shown during training)")
    
    # Start training
    print("\n🚀 Starting CPU training...")
    try:
        results = model.train(**training_config)
        
        print("\n🎉 Training completed successfully!")
        print("📁 Results saved in: runs/detect/train/")
        print("🏆 Best model: runs/detect/train/weights/best.pt")
        print("📊 Training metrics: runs/detect/train/results.png")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def create_model_comparison_script():
    """Create script to compare old vs new model"""
    
    script_content = '''#!/usr/bin/env python3
"""
Compare Old vs New Weapon Detection Models
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import time

def compare_models():
    """Compare performance of old and new models"""
    
    print("🔄 COMPARING OLD VS NEW MODELS")
    print("="*50)
    
    try:
        old_model = YOLO("best.pt")
        new_model = YOLO("runs/detect/train/weights/best.pt")
        print("✅ Both models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Test on sample images
    test_dir = "test_images/weapon-detection.v4i.yolov5pytorch/train/images"
    test_images = list(Path(test_dir).glob("*.jpg"))[:20]  # Test 20 images
    
    print(f"\\n🧪 Testing on {len(test_images)} images...")
    
    old_detections = 0
    new_detections = 0
    old_confidence_sum = 0
    new_confidence_sum = 0
    
    for img_path in test_images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Test old model
        old_results = old_model(image, conf=0.3, verbose=False)
        old_count = len(old_results[0].boxes) if old_results[0].boxes is not None else 0
        old_detections += old_count
        
        if old_results[0].boxes is not None:
            for box in old_results[0].boxes:
                old_confidence_sum += float(box.conf[0])
        
        # Test new model
        new_results = new_model(image, conf=0.3, verbose=False)
        new_count = len(new_results[0].boxes) if new_results[0].boxes is not None else 0
        new_detections += new_count
        
        if new_results[0].boxes is not None:
            for box in new_results[0].boxes:
                new_confidence_sum += float(box.conf[0])
    
    # Calculate averages
    old_avg_conf = old_confidence_sum / old_detections if old_detections > 0 else 0
    new_avg_conf = new_confidence_sum / new_detections if new_detections > 0 else 0
    
    print(f"\\n📊 COMPARISON RESULTS:")
    print("-"*30)
    print(f"{'Metric':<20} {'Old Model':<12} {'New Model':<12}")
    print("-"*30)
    print(f"{'Total Detections':<20} {old_detections:<12} {new_detections:<12}")
    print(f"{'Avg Confidence':<20} {old_avg_conf:<12.3f} {new_avg_conf:<12.3f}")
    print(f"{'Detections/Image':<20} {old_detections/len(test_images):<12.2f} {new_detections/len(test_images):<12.2f}")
    
    # Recommendation
    if new_avg_conf > old_avg_conf and new_detections >= old_detections * 0.8:
        print("\\n🎉 NEW MODEL IS BETTER!")
        print("✅ Higher confidence and good detection rate")
        print("💡 Recommended: Replace your current model")
        print("   Command: cp runs/detect/train/weights/best.pt best_improved.pt")
    elif new_detections > old_detections * 1.2:
        print("\\n🎯 NEW MODEL DETECTS MORE WEAPONS")
        print("⚠️  But check for false positives in real usage")
        print("💡 Consider using new model with higher confidence threshold")
    else:
        print("\\n📈 MODELS HAVE SIMILAR PERFORMANCE")
        print("💡 Try more training epochs or different parameters")

if __name__ == "__main__":
    compare_models()
'''
    
    with open("compare_models.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created model comparison script: compare_models.py")

def main():
    """Main function for CPU-optimized training"""
    
    print("🖥️  CPU-OPTIMIZED WEAPON DETECTION TRAINING")
    print("="*60)
    print("Goal: Increase accuracy from 18% to 40-60% (CPU optimized)")
    print()
    
    # Check data
    dataset_path = "test_images/weapon-detection.v4i.yolov5pytorch"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        print("❌ data.yaml not found, creating one...")
        
        data_config = {
            'path': os.path.abspath(dataset_path),
            'train': 'train/images',
            'val': 'train/images',
            'names': {
                0: 'Handgun',
                1: 'Knife', 
                2: 'Rifle',
                3: 'Sword'
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        print("✅ Created data.yaml file")
    
    # Count images
    train_images = list(Path(dataset_path, "train", "images").glob("*.jpg"))
    print(f"📋 Training images available: {len(train_images)}")
    
    if len(train_images) == 0:
        print("❌ No training images found!")
        return
    
    # Create comparison script
    create_model_comparison_script()
    
    # Ask for confirmation
    print(f"\\n💻 TRAINING DETAILS:")
    print("   - Model: YOLOv8n (optimized for CPU)")
    print("   - Device: CPU")
    print("   - Epochs: 50")
    print("   - Batch size: 4")
    print("   - Image size: 416x416")
    print("   - Estimated time: 1-2 hours")
    
    response = input("\\n🤔 Start CPU training now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        results = train_cpu_optimized_model()
        
        if results:
            print("\\n🎉 SUCCESS! CPU Training completed!")
            print("\\n📋 Next steps:")
            print("1. Run: python compare_models.py")
            print("2. Check if new model is better")
            print("3. Test with your surveillance system")
            print("4. Consider GPU training for even better results")
            
            print("\\n📈 Expected Improvements:")
            print("   - Better precision (fewer false positives)")
            print("   - Higher confidence scores")
            print("   - More consistent detections")
            print("   - F1-Score: 40-60% (vs current 18%)")
    else:
        print("\\n⏸️  Training skipped")
        print("💡 Run this script anytime to improve your model")

if __name__ == "__main__":
    main()
