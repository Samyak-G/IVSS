import cv2
import numpy as np
from multiprocessing import shared_memory
import time

def motion_detection_process(shm_name, shape, motion_queue, cam_id):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)
    
    while True:
        # Optional: small delay to help synchronization with video capture
        time.sleep(0.01)
        
        frame = frame_buffer.copy()
        
        # Check if the frame is too dark (likely incomplete)
        if np.mean(frame) < 10:
            print(f"[DEBUG] Camera {cam_id}: Frame mean {np.mean(frame):.2f} too low. Skipping frame.")
            continue
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"[ERROR] Camera {cam_id}: Color conversion failed: {e}")
            continue
        
        fg_mask = bg_subtractor.apply(gray)
        motion_score = cv2.countNonZero(fg_mask)
        
        # Debug print: log motion score
        print(f"[DEBUG] Camera {cam_id}: Motion score: {motion_score}")
        
        if motion_score > 100:  # Adjust threshold as needed.
            alert_data = {
                "cam_id": cam_id,
                "message": f"Motion detected with score {motion_score}",
                "severity": "medium",
                "detection_type": "motion"
            }
            motion_queue.put(alert_data)

if __name__ == "__main__":
    print("Run main.py to start the system.")
