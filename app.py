from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import numpy as np
from multiprocessing import shared_memory
import sqlite3
import os
import json
import face_recognition
import pickle

app = Flask(__name__)

FRAME_SHAPE = (240, 320, 3)

def gen_frames(shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        print("Shared memory block not found. Is the backend running?")
        return
    frame_buffer = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    while True:
        frame = frame_buffer.copy()
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    shm.close()

def load_camera_settings():
    if os.path.exists("camera_settings.json"):
        with open("camera_settings.json", "r") as f:
            data = json.load(f)
            if data and isinstance(data[0], str):
                return [{"source": src, "detections": ["motion", "object", "face"]} for src in data]
            return data
    else:
        return [{"source": "0", "detections": ["motion", "object", "face"]}]

@app.route('/')
def dashboard():
    camera_settings = load_camera_settings()
    return render_template('dashboard.html', camera_ids=camera_settings)

@app.route('/live_monitoring')
def live_monitoring():
    camera_settings = load_camera_settings()
    return render_template('live_monitoring.html', camera_ids=camera_settings)

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/settings')
def settings():
    cameras = load_camera_settings()
    cameras_json = json.dumps(cameras)
    return render_template('settings.html', cameras_json=cameras_json)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    shm_name = f"video_frame_shm_{cam_id}"
    return Response(gen_frames(shm_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/dashboard_stats')
def dashboard_stats():
    data = {
        "active_cameras": len(load_camera_settings()),
        "motion_detected": 0,
        "security_alerts": 0,
        "storage_usage": 80
    }
    return jsonify(data)

@app.route('/api/alerts')
def api_alerts():
    # Modified query to include the camera field for each alert.
    conn = sqlite3.connect('alerts.db')
    c = conn.cursor()
    c.execute("SELECT camera, location, time, message, severity FROM alerts ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    alerts_list = []
    for row in rows:
        alerts_list.append({
            "camera": row[0],
            "location": row[1],
            "time": row[2],
            "message": row[3],
            "severity": row[4]
        })
    return jsonify(alerts_list)

@app.route('/api/save_camera_settings', methods=['POST'])
def save_camera_settings():
    data = request.get_json()
    cameras = data.get("cameras", [])
    # Ensure the first camera is always present and its source is not empty.
    if len(cameras) == 0 or not cameras[0].get("source"):
        cameras.insert(0, {"source": "0", "detections": ["motion", "object", "face"]})
    with open("camera_settings.json", "w") as f:
        json.dump(cameras, f)
    return jsonify({"status": "success", "cameras": cameras})


# -----------------------------------------------
# New Web-Based Face Registration Module
# -----------------------------------------------

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = 'dataset'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_encodings(dataset_dir="dataset", encodings_file="encodings.pickle"):
    """
    Processes all images in the dataset folder, computes face encodings,
    and saves them in encodings_file.
    """
    image_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if allowed_file(file):
                image_paths.append(os.path.join(root, file))
    
    known_encodings = []
    known_names = []
    
    for image_path in image_paths:
        # Assume the person's name is the folder name
        name = os.path.basename(os.path.dirname(image_path))
        image = cv2.imread(image_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)
    
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_file, "wb") as f:
        pickle.dump(data, f)
    return len(known_encodings)

@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    message = ""
    if request.method == 'POST':
        person_name = request.form.get("person_name")
        if not person_name:
            message = "Please enter the person’s name."
            return render_template("register_face.html", message=message)
        
        if 'face_image' not in request.files:
            message = "No file part in the request."
            return render_template("register_face.html", message=message)
        
        file = request.files["face_image"]
        if file.filename == "":
            message = "No file selected."
            return render_template("register_face.html", message=message)
        
        if file and allowed_file(file.filename):
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            # Create a subfolder for the person
            person_dir = os.path.join(app.config['UPLOAD_FOLDER'], person_name)
            os.makedirs(person_dir, exist_ok=True)
            filepath = os.path.join(person_dir, filename)
            file.save(filepath)
            
            # Update encodings based on the new image(s)
            count = update_encodings()
            message = f"Face image saved and encodings updated. Total encodings: {count}"
        else:
            message = "Invalid file type. Allowed types: png, jpg, jpeg."
    return render_template("register_face.html", message=message)

if __name__ == '__main__':
    app.run(debug=True)

