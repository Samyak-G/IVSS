import cv2
import numpy as np
from multiprocessing import shared_memory

def motion_detection_process(shm_name, shape, motion_queue, cam_id):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)
    
    while True:
        frame = frame_buffer.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        motion_score = cv2.countNonZero(fg_mask)
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

