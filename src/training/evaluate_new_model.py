#!/usr/bin/env python3
"""
Evaluate Weapon Detection Model
Get accuracy, F1 score, and mAP50 metrics
"""

from ultralytics import YOLO
import json

def evaluate_model():
    """Evaluate the trained weapon detection model"""
    
    print("📊 Model Evaluation")
    print("="*40)
    
    # Load your trained model
    try:
        model = YOLO("../../data/models/best.pt")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test on validation data
    print("\n🧪 Running validation...")
    
    try:
        # Validate the model
        results = model.val(data="../../data/datasets/test_images/weapon-detection.v4i.yolov5pytorch/data.yaml")
        
        print("\n📈 Model Performance Metrics:")
        print("="*50)
        
        # Extract key metrics
        map50 = results.box.map50 if hasattr(results.box, 'map50') else 0
        map50_95 = results.box.map if hasattr(results.box, 'map') else 0
        precision = results.box.mp if hasattr(results.box, 'mp') else 0
        recall = results.box.mr if hasattr(results.box, 'mr') else 0
        f1_score = results.box.f1 if hasattr(results.box, 'f1') else 0
        
        print(f"🎯 mAP@0.5:      {map50:.3f} ({map50*100:.1f}%)")
        print(f"🎯 mAP@0.5:0.95:  {map50_95:.3f} ({map50_95*100:.1f}%)")
        print(f"🎯 Precision:     {precision:.3f} ({precision*100:.1f}%)")
        print(f"🎯 Recall:        {recall:.3f} ({recall*100:.1f}%)")
        print(f"🎯 F1-Score:      {f1_score:.3f} ({f1_score*100:.1f}%)")
        
        # Class-wise performance
        print(f"\n📋 Per-Class Performance (mAP@0.5):")
        print("-"*40)
        classes = ['Handgun', 'Knife', 'Rifle', 'Sword']
        if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
            for i, class_name in enumerate(classes):
                if i < len(results.box.ap50):
                    class_map50 = results.box.ap50[i]
                    print(f"{class_name:<12}: {class_map50:.3f} ({class_map50*100:.1f}%)")
        
        # Summary
        print(f"\n🏆 SUMMARY:")
        print("="*30)
        if map50 >= 0.8:
            print("🟢 EXCELLENT - Your model has high accuracy!")
        elif map50 >= 0.6:
            print("🟡 GOOD - Model performs well, room for improvement")
        elif map50 >= 0.4:
            print("🟠 FAIR - Consider more training or data augmentation")
        else:
            print("🔴 NEEDS IMPROVEMENT - Requires significant training")
            
        print(f"\n📊 Key Metrics for Reference:")
        print(f"   • Accuracy (mAP@0.5): {map50*100:.1f}%")
        print(f"   • F1-Score: {f1_score*100:.1f}%")
        print(f"   • Precision: {precision*100:.1f}%")
        print(f"   • Recall: {recall*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        print("💡 Make sure your dataset paths are correct")

if __name__ == "__main__":
    evaluate_model()
