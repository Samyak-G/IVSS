import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os
import json

from video_capture import video_capture_process
from motion_detection import motion_detection_process
from object_detection import object_detection_process
from face_recognition_module import face_recognition_process
from alert_module import alert_process

FRAME_SHAPE = (240, 320, 3)  # (height, width, channels)

def load_camera_settings():
    if os.path.exists("camera_settings.json"):
        with open("camera_settings.json", "r") as f:
            data = json.load(f)
            # If settings are stored as list of strings (legacy), convert them.
            if data and isinstance(data[0], str):
                return [{"source": src, "detections": ["motion", "object", "face"]} for src in data]
            return data
    else:
        # Default to local webcam with all detections enabled.
        return [{"source": "0", "detections": ["motion", "object", "face"]}]

if __name__ == "__main__":
    camera_settings = load_camera_settings()  # List of dicts: {"source": "...", "detections": [...]}
    num_cameras = len(camera_settings)
    
    shared_mem_list = []
    processes = []
    
    # Global queues for alerts from all cameras
    object_queue = mp.Queue()
    face_queue = mp.Queue()
    motion_queue = mp.Queue()
    
    for i, cam_config in enumerate(camera_settings):
        source = cam_config["source"]
        detections = cam_config.get("detections", [])
        shm_name = f"video_frame_shm_{i}"
        shared_mem = shared_memory.SharedMemory(create=True, size=np.prod(FRAME_SHAPE), name=shm_name)
        shared_mem_list.append(shared_mem)
        
        # Always start video capture.
        processes.append(mp.Process(target=video_capture_process, args=(shm_name, FRAME_SHAPE, source, i)))
        
        # Start detection processes based on user-selected detections.
        if "motion" in detections:
            processes.append(mp.Process(target=motion_detection_process, args=(shm_name, FRAME_SHAPE, motion_queue, i)))
        if "object" in detections:
            processes.append(mp.Process(target=object_detection_process, args=(shm_name, FRAME_SHAPE, object_queue, i)))
        if "face" in detections:
            processes.append(mp.Process(target=face_recognition_process, args=(shm_name, FRAME_SHAPE, face_queue, i)))
    
    processes.append(mp.Process(target=alert_process, args=(object_queue, face_queue, motion_queue)))
    
    try:
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    finally:
        for shm in shared_mem_list:
            shm.close()
            shm.unlink()

