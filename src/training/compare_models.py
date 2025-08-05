#!/usr/bin/env python3
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
    
    print(f"\n🧪 Testing on {len(test_images)} images...")
    
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
    
    print(f"\n📊 COMPARISON RESULTS:")
    print("-"*30)
    print(f"{'Metric':<20} {'Old Model':<12} {'New Model':<12}")
    print("-"*30)
    print(f"{'Total Detections':<20} {old_detections:<12} {new_detections:<12}")
    print(f"{'Avg Confidence':<20} {old_avg_conf:<12.3f} {new_avg_conf:<12.3f}")
    print(f"{'Detections/Image':<20} {old_detections/len(test_images):<12.2f} {new_detections/len(test_images):<12.2f}")
    
    # Recommendation
    if new_avg_conf > old_avg_conf and new_detections >= old_detections * 0.8:
        print("\n🎉 NEW MODEL IS BETTER!")
        print("✅ Higher confidence and good detection rate")
        print("💡 Recommended: Replace your current model")
        print("   Command: cp runs/detect/train/weights/best.pt best_improved.pt")
    elif new_detections > old_detections * 1.2:
        print("\n🎯 NEW MODEL DETECTS MORE WEAPONS")
        print("⚠️  But check for false positives in real usage")
        print("💡 Consider using new model with higher confidence threshold")
    else:
        print("\n📈 MODELS HAVE SIMILAR PERFORMANCE")
        print("💡 Try more training epochs or different parameters")

if __name__ == "__main__":
    compare_models()
