import os
import cv2
import face_recognition
import pickle
import numpy as np
from multiprocessing import shared_memory

encodings_file = "encodings.pickle"
if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
else:
    known_encodings = []
    known_names = []

def face_recognition_process(shm_name, shape, output_queue, cam_id):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    
    while True:
        frame = frame_buffer.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)
        
        for encoding, (top, right, bottom, left) in zip(encodings, boxes):
            matches = face_recognition.compare_faces(known_encodings, encoding,          tolerance=0.5)
            name = "Unknown"
            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    recognized_name = known_names[i]
                    counts[recognized_name] = counts.get(recognized_name, 0) + 1
                name = max(counts, key=counts.get)
            
            output_queue.put({
                "cam_id": cam_id,
                "name": name,
                "bbox": (left, top, right, bottom),
                "detection_type": "face"
            })


if __name__ == "__main__":
    print("Run main.py to start the system.")

