# import cv2
# import numpy as np
# from multiprocessing import shared_memory

# def motion_detection_process(shm_name, shape, motion_queue, cam_id):
#     shared_mem = shared_memory.SharedMemory(name=shm_name)
#     frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
#     bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)
    
#     while True:
#         frame = frame_buffer.copy()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         fg_mask = bg_subtractor.apply(gray)
#         motion_score = cv2.countNonZero(fg_mask)
#         if motion_score > 100:  # Adjust threshold as needed.
#             alert_data = {
#                 "cam_id": cam_id,
#                 "message": f"Motion detected with score {motion_score}",
#                 "severity": "medium",
#                 "detection_type": "motion"
#             }
#             motion_queue.put(alert_data)

# if __name__ == "__main__":
#     print("Run main.py to start the system.")


import cv2
import numpy as np
import os
import time
from multiprocessing import shared_memory

# 🔹 Motion Detection Process
def motion_detection_process(shm_name, shape, motion_queue, cam_id):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)

    while True:
        frame = frame_buffer.copy()

        # ✅ Ensure the frame is valid before processing
        if frame is None or frame.size == 0:
            print(f"[ERROR] Camera {cam_id}: Invalid frame received.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        motion_score = cv2.countNonZero(fg_mask)

        if motion_score > 100:  # Adjust threshold if needed
            image_path = save_motion_frame(frame, cam_id)

            if image_path:
                alert_data = {
                    "cam_id": cam_id,
                    "message": f"Motion detected with score {motion_score}",
                    "severity": "medium",
                    "detection_type": "motion",
                    "image_path": image_path  # ✅ Attach the captured image
                }
                motion_queue.put(alert_data)
            else:
                print(f"[ERROR] Camera {cam_id}: Failed to save motion frame.")

# 🔹 Save Motion Frame (✅ FIXED)
def save_motion_frame(frame, cam_id):
    image_path = f"motion_alert_cam{cam_id}.jpg"
    
    # ✅ Save and validate the image
    cv2.imwrite(image_path, frame)
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        print(f"[INFO] Motion frame saved: {image_path}")
        return image_path
    else:
        print(f"[ERROR] Failed to save motion image for Camera {cam_id}.")
        return None

if __name__ == "__main__":
    print("Run main.py to start the system.")

