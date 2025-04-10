# import cv2
# from ultralytics import YOLO
# from multiprocessing import shared_memory
# import numpy as np

# def object_detection_process(shm_name, shape, output_queue, cam_id):
#     shared_mem = shared_memory.SharedMemory(name=shm_name)
#     frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
#     model = YOLO('yolov8n.pt')  # Ensure the model file is available
    
#     while True:
#         frame = frame_buffer.copy()
#         results = model.predict(frame, imgsz=320, verbose=False)
#         detected_objects = []
#         for result in results:
#             boxes = result.boxes.cpu().numpy()
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].astype(int)
#                 label = model.names[int(box.cls[0])]
#                 confidence = box.conf[0]
#                 detected_objects.append({
#                     "label": label,
#                     "confidence": float(confidence),
#                     "bbox": (x1, y1, x2, y2)
#                 })
#         if detected_objects:
#             output_queue.put({"cam_id": cam_id, "detections": detected_objects})

# if __name__ == "__main__":
#     print("Run main.py to start the system.")


# import cv2
# from ultralytics import YOLO
# from multiprocessing import shared_memory
# import numpy as np

# def object_detection_process(shm_name, shape, output_queue, cam_id):
#     shared_mem = shared_memory.SharedMemory(name=shm_name)
#     frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
#     model = YOLO('yolov8n.pt')  # Ensure the model file is available

#     while True:
#         # ✅ Fix: Explicitly copy frame and convert to BGR
#         frame = frame_buffer.copy()
        
#         if frame is None or frame.shape != shape:
#             print(f"[ERROR] Camera {cam_id}: Frame shape mismatch. Skipping detection.")
#             continue  # Avoid processing corrupt frames
        
#         # ✅ Fix: Ensure correct color format
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         results = model.predict(frame, imgsz=320, verbose=False)
#         detected_objects = []
        
#         for result in results:
#             boxes = result.boxes.cpu().numpy()
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].astype(int)
#                 label = model.names[int(box.cls[0])]
#                 confidence = float(box.conf[0])

#                 detected_objects.append({
#                     "label": label,
#                     "confidence": confidence,
#                     "bbox": (x1, y1, x2, y2)
#                 })

#                 # ✅ Fix: Draw bounding boxes on the image
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         if detected_objects:
#             output_queue.put({"cam_id": cam_id, "detections": detected_objects})
            
#             # ✅ Fix: Save the processed frame with detections
#             cv2.imwrite(f"object_detection_cam{cam_id}.jpg", frame)
#             print(f"[INFO] Object detection frame saved: object_detection_cam{cam_id}.jpg")

# if __name__ == "__main__":
#     print("Run main.py to start the system.")



import cv2
from ultralytics import YOLO
from multiprocessing import shared_memory
import numpy as np
import time
import time

def object_detection_process(shm_name, shape, output_queue, cam_id):
    """
    Continuously reads frames from shared memory, runs YOLO object detection,
    and outputs detections via the output_queue. Also draws bounding boxes and
    saves the processed frame for debugging.
    
    Args:
        shm_name (str): Name of the shared memory block.
        shape (tuple): Expected frame shape (height, width, channels).
        output_queue (Queue): Multiprocessing queue to output detections.
        cam_id (int): Identifier for the camera.
    """
    # Access shared memory and create a NumPy array view
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    
    # Initialize YOLO model (ensure 'yolov8n.pt' is available)
    model = YOLO('best.pt')
    """
    Continuously reads frames from shared memory, runs YOLO object detection,
    and outputs detections via the output_queue. Also draws bounding boxes and
    saves the processed frame for debugging.
    
    Args:
        shm_name (str): Name of the shared memory block.
        shape (tuple): Expected frame shape (height, width, channels).
        output_queue (Queue): Multiprocessing queue to output detections.
        cam_id (int): Identifier for the camera.
    """
    # Access shared memory and create a NumPy array view
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    
    # Initialize YOLO model (ensure 'yolov8n.pt' is available)
    model = YOLO('yolov8n.pt')
    
    while True:
        # Add a short sleep to mitigate race conditions with the video capture process.
        time.sleep(0.01)
        
        # Make a deep copy of the frame from shared memory.
        frame = np.copy(frame_buffer)
        
        # Validate the frame: check if it's None, has the expected shape, and is non-empty.
        if frame is None or frame.shape != shape or np.all(frame == 0):
            print(f"[ERROR] Camera {cam_id}: Invalid or empty frame. Skipping detection.")
            continue
        
        # Debug: Print mean pixel value to check if the frame is updating.
        print(f"[DEBUG] Camera {cam_id}: Frame mean pixel value: {frame.mean():.2f}")
        
        # Convert the frame from RGB to BGR (if your shared memory stores RGB)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Camera {cam_id}: Failed to convert color space: {e}")
            continue
        
        # Run YOLO object detection
        try:
            results = model.predict(frame, imgsz=320, verbose=False)
        except Exception as e:
            print(f"[ERROR] YOLO prediction failed for camera {cam_id}: {e}")
            continue
        
        # Add a short sleep to mitigate race conditions with the video capture process.
        time.sleep(0.01)
        
        # Make a deep copy of the frame from shared memory.
        frame = np.copy(frame_buffer)
        
        # Validate the frame: check if it's None, has the expected shape, and is non-empty.
        if frame is None or frame.shape != shape or np.all(frame == 0):
            print(f"[ERROR] Camera {cam_id}: Invalid or empty frame. Skipping detection.")
            continue
        
        # Debug: Print mean pixel value to check if the frame is updating.
        print(f"[DEBUG] Camera {cam_id}: Frame mean pixel value: {frame.mean():.2f}")
        
        # Convert the frame from RGB to BGR (if your shared memory stores RGB)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Camera {cam_id}: Failed to convert color space: {e}")
            continue
        
        # Run YOLO object detection
        try:
            results = model.predict(frame, imgsz=320, verbose=False)
        except Exception as e:
            print(f"[ERROR] YOLO prediction failed for camera {cam_id}: {e}")
            continue
        
        detected_objects = []
        for result in results:
            try:
                boxes = result.boxes.cpu().numpy()
            except Exception as e:
                print(f"[ERROR] Camera {cam_id}: Failed to get bounding boxes: {e}")
                continue
            try:
                boxes = result.boxes.cpu().numpy()
            except Exception as e:
                print(f"[ERROR] Camera {cam_id}: Failed to get bounding boxes: {e}")
                continue
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                except Exception as e:
                    print(f"[ERROR] Camera {cam_id}: Error parsing bounding box: {e}")
                    continue
                try:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                except Exception as e:
                    print(f"[ERROR] Camera {cam_id}: Error parsing bounding box: {e}")
                    continue
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                confidence = float(box.conf[0])
                detected_objects.append({
                    "label": label,
                    "confidence": confidence,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })
                # Draw bounding box and label on the frame for debugging
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
                # Draw bounding box and label on the frame for debugging
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if detected_objects:
            output_queue.put({"cam_id": cam_id, "detections": detected_objects})
            # Save the processed frame with detections for debugging
            saved_filename = f"object_detection_cam{cam_id}.jpg"
            cv2.imwrite(saved_filename, frame)
            print(f"[INFO] Object detection frame saved: {saved_filename}")
            # Save the processed frame with detections for debugging
            saved_filename = f"object_detection_cam{cam_id}.jpg"
            cv2.imwrite(saved_filename, frame)
            print(f"[INFO] Object detection frame saved: {saved_filename}")

if __name__ == "__main__":
    print("Run main.py to start the system.")
