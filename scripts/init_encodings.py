#!/usr/bin/env python3
"""
Initialize Face Encodings
========================
Create an initial encodings.pickle file for the face recognition system
"""

import pickle
import os

def create_initial_encodings():
    """Create an empty encodings.pickle file if it doesn't exist"""
    
    config_dir = "config"
    encodings_file = os.path.join(config_dir, "encodings.pickle")
    
    # Create config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    if not os.path.exists(encodings_file):
        print(f"🔧 Creating initial encodings file: {encodings_file}")
        
        # Create empty encodings structure
        initial_data = {
            "encodings": [],
            "names": []
        }
        
        with open(encodings_file, "wb") as f:
            pickle.dump(initial_data, f)
        
        print(f"✅ Created empty encodings file")
        print("📝 To add faces:")
        print("   1. Use the web interface at /register_face")
        print("   2. Or run: python src/utils/create_face_dataset.py")
        print()
    else:
        print(f"✅ Encodings file already exists: {encodings_file}")
        
        # Show current status
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
        
        num_faces = len(data.get("encodings", []))
        names = data.get("names", [])
        print(f"📊 Current encodings: {num_faces} faces")
        if names:
            print(f"🏷️  Known people: {', '.join(set(names))}")

if __name__ == "__main__":
    create_initial_encodings()
