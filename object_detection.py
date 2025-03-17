import cv2
from ultralytics import YOLO
from multiprocessing import shared_memory
import numpy as np

def object_detection_process(shm_name, shape, output_queue, cam_id):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    model = YOLO('yolov8n.pt')  # Ensure the model file is available
    
    while True:
        frame = frame_buffer.copy()
        results = model.predict(frame, imgsz=320, verbose=False)
        detected_objects = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                label = model.names[int(box.cls[0])]
                confidence = box.conf[0]
                detected_objects.append({
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": (x1, y1, x2, y2)
                })
        if detected_objects:
            output_queue.put({"cam_id": cam_id, "detections": detected_objects})

if __name__ == "__main__":
    print("Run main.py to start the system.")

