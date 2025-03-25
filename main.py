import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os
import json
import time
import threading

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

# Global detection queues (for all cameras)
object_queue = mp.Queue()
face_queue = mp.Queue()
motion_queue = mp.Queue()

def spawn_detection_processes(camera_settings, detection_processes):
    """Spawns detection processes based on camera_settings and updates detection_processes dict.
       detection_processes is a dict with keys (cam_index, detection_type).
    """
    for i, cam_config in enumerate(camera_settings):
        shm_name = f"video_frame_shm_{i}"
        detections = cam_config.get("detections", [])
        # For each detection type, if enabled and not already running, spawn it.
        for det in ["motion", "object", "face"]:
            key = (i, det)
            if det in detections and key not in detection_processes:
                if det == "motion":
                    p = mp.Process(target=motion_detection_process, args=(shm_name, FRAME_SHAPE, motion_queue, i))
                elif det == "object":
                    p = mp.Process(target=object_detection_process, args=(shm_name, FRAME_SHAPE, object_queue, i))
                elif det == "face":
                    p = mp.Process(target=face_recognition_process, args=(shm_name, FRAME_SHAPE, face_queue, i))
                p.start()
                detection_processes[key] = p
    # No return needed; detection_processes is updated in place.

def kill_detection_processes_for_camera(camera_index, detection_processes):
    """Terminates all detection processes for a given camera index."""
    keys_to_remove = [key for key in detection_processes if key[0] == camera_index]
    for key in keys_to_remove:
        p = detection_processes[key]
        p.terminate()
        p.join()
        del detection_processes[key]

def update_detection_processes(new_settings, old_settings, detection_processes):
    """Compare new and old settings and update detection processes accordingly."""
    # Handle cameras that remain (by index) in new_settings:
    min_len = min(len(new_settings), len(old_settings))
    for i in range(min_len):
        new_dets = set(new_settings[i].get("detections", []))
        old_dets = set(old_settings[i].get("detections", []))
        shm_name = f"video_frame_shm_{i}"
        # For each detection type, if newly enabled, spawn; if disabled, kill.
        for det in ["motion", "object", "face"]:
            key = (i, det)
            if det in new_dets and det not in old_dets:
                # Spawn the new detection process
                if det == "motion":
                    p = mp.Process(target=motion_detection_process, args=(shm_name, FRAME_SHAPE, motion_queue, i))
                elif det == "object":
                    p = mp.Process(target=object_detection_process, args=(shm_name, FRAME_SHAPE, object_queue, i))
                elif det == "face":
                    p = mp.Process(target=face_recognition_process, args=(shm_name, FRAME_SHAPE, face_queue, i))
                p.start()
                detection_processes[key] = p
            elif det not in new_dets and det in old_dets:
                # Terminate the detection process if it exists
                if key in detection_processes:
                    p = detection_processes[key]
                    p.terminate()
                    p.join()
                    del detection_processes[key]
    # For any new cameras added:
    if len(new_settings) > len(old_settings):
        for i in range(len(old_settings), len(new_settings)):
            shm_name = f"video_frame_shm_{i}"
            for det in new_settings[i].get("detections", []):
                key = (i, det)
                if det == "motion":
                    p = mp.Process(target=motion_detection_process, args=(shm_name, FRAME_SHAPE, motion_queue, i))
                elif det == "object":
                    p = mp.Process(target=object_detection_process, args=(shm_name, FRAME_SHAPE, object_queue, i))
                elif det == "face":
                    p = mp.Process(target=face_recognition_process, args=(shm_name, FRAME_SHAPE, face_queue, i))
                p.start()
                detection_processes[key] = p
    # For cameras removed in new settings:
    if len(new_settings) < len(old_settings):
        for i in range(len(new_settings), len(old_settings)):
            kill_detection_processes_for_camera(i, detection_processes)

def watch_settings_changes(detection_processes, old_settings):
    """Watches the camera_settings.json file for changes and updates detection processes."""
    settings_file = "camera_settings.json"
    last_mtime = os.path.getmtime(settings_file) if os.path.exists(settings_file) else None
    while True:
        time.sleep(2)
        if os.path.exists(settings_file):
            mtime = os.path.getmtime(settings_file)
            if mtime != last_mtime:
                print("[INFO] Settings file changed. Updating detection processes...")
                new_settings = load_camera_settings()
                update_detection_processes(new_settings, old_settings, detection_processes)
                # Update old_settings for next comparison
                old_settings.clear()
                old_settings.extend(new_settings)
                last_mtime = mtime

if __name__ == "__main__":
    # Load initial camera settings
    camera_settings = load_camera_settings()  # List of dicts: {"source": "...", "detections": [...]}
    num_cameras = len(camera_settings)
    
    # Create shared memory for each camera
    shared_mem_list = []
    for i in range(num_cameras):
        shm_name = f"video_frame_shm_{i}"
        shared_mem = shared_memory.SharedMemory(create=True, size=np.prod(FRAME_SHAPE), name=shm_name)
        shared_mem_list.append(shared_mem)
    
    processes = []  # List for video capture and alert process
    detection_processes = {}  # Dictionary for detection processes; keys: (camera_index, detection_type)
    
    # Global queues (already created above)
    
    # Spawn video capture processes (always running) for each camera.
    for i, cam_config in enumerate(camera_settings):
        shm_name = f"video_frame_shm_{i}"
        source = cam_config["source"]
        p = mp.Process(target=video_capture_process, args=(shm_name, FRAME_SHAPE, source, i))
        p.start()
        processes.append(p)
    
    # Spawn detection processes based on initial settings.
    spawn_detection_processes(camera_settings, detection_processes)
    
    # Spawn the alert process (runs once, reading from global queues)
    alert_p = mp.Process(target=alert_process, args=(object_queue, face_queue, motion_queue))
    alert_p.start()
    processes.append(alert_p)
    
    # Start a background thread to watch for settings changes.
    # We'll use a list (old_settings) so that it can be updated in-place.
    old_settings = camera_settings.copy()
    watcher_thread = threading.Thread(target=watch_settings_changes, args=(detection_processes, old_settings), daemon=True)
    watcher_thread.start()
    
    try:
        for p in processes:
            p.join()
        # Also join on all detection processes (if they ever finish)
        for p in detection_processes.values():
            p.join()
    finally:
        # Cleanup: close and unlink shared memory blocks.
        for shm in shared_mem_list:
            shm.close()
            shm.unlink()
