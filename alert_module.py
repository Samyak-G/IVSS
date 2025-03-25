import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import plyer
import sqlite3
from datetime import datetime
import json
import os
import time

def load_camera_settings():
    """Loads user-configured camera settings from file."""
    if os.path.exists("camera_settings.json"):
        with open("camera_settings.json", "r") as f:
            data = json.load(f)
            # Handle legacy settings if necessary.
            if data and isinstance(data[0], str):
                return [{"source": src, "detections": ["motion", "object", "face"]} for src in data]
            return data
    else:
        # Default to local webcam with all detections enabled.
        return [{"source": "0", "detections": ["motion", "object", "face"]}]

def send_email_notification(subject, message):
    sender_email = "your-email@gmail.com"       # Replace with your email
    receiver_email = "recipient-email@example.com"  # Replace with recipient's email
    password = "your-password"                   # Replace with your email password
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

def send_local_notification(title, message):
    plyer.notification.notify(
        title=title,
        message=message,
        app_name='Alert System',
        timeout=10
    )

def store_alert(camera, location, message, severity):
    db_path = os.path.join(os.getcwd(), 'alerts.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera TEXT,
        location TEXT,
        time TEXT,
        message TEXT,
        severity TEXT
    )
    ''')
    alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO alerts (camera, location, time, message, severity) VALUES (?, ?, ?, ?, ?)",
              (camera, location, alert_time, message, severity))
    conn.commit()
    conn.close()
    print(f"[DEBUG] Stored alert: {camera}, {location}, {alert_time}, {message}, {severity}")


def alert_process(object_queue, face_queue, motion_queue):
    """
    Processes alert events from the detection queues.
    Only generates alerts for detection types enabled in the user settings.
    """
    while True:
        # Reload settings to catch any changes.
        camera_settings = load_camera_settings()
        
        # Process motion alerts.
        if not motion_queue.empty():
            motion_alert = motion_queue.get()
            cam_id = motion_alert.get("cam_id")
            # New field indicating type of detection
            detection_type = motion_alert.get("detection_type", "motion")
            if cam_id is not None and cam_id < len(camera_settings):
                detections_enabled = camera_settings[cam_id].get("detections", [])
                if detection_type in detections_enabled:
                    store_alert(f"Camera {cam_id}", "Motion Detection",
                                motion_alert.get('message', 'Motion detected'),
                                motion_alert.get('severity', 'medium'))
                    send_email_notification("Motion Detected", motion_alert.get('message', 'Motion detected.'))
                    send_local_notification("Motion Detected", motion_alert.get('message', 'Motion detected.'))
        
        # Process object detection alerts.
        if not object_queue.empty():
            obj_event = object_queue.get()
            cam_id = obj_event.get("cam_id")
            detection_type = obj_event.get("detection_type", "object")
            if cam_id is not None and cam_id < len(camera_settings):
                detections_enabled = camera_settings[cam_id].get("detections", [])
                if detection_type in detections_enabled:
                    detections = obj_event.get("detections", [])
                    if any(obj.get('label') == 'person' for obj in detections):
                        store_alert(f"Camera {cam_id}", "Object Detection",
                                    "A person has been detected.", "high")
                        send_email_notification("Person Detected", "A person has been detected.")
                        send_local_notification("Person Detected", "A person has been detected.")
                        
        # P        # Process face recognition alerts.
        if not face_queue.empty():
            face_event = face_queue.get()
            cam_id = face_event.get("cam_id")
            detection_type = face_event.get("detection_type", "face")
            if cam_id is not None and cam_id < len(camera_settings):
                detections_enabled = camera_settings[cam_id].get("detections", [])
                if detection_type in detections_enabled:
                    # Alert for any face detection regardless of recognition result.
                    detected_name = face_event.get('name', 'Unknown')
                    alert_message = f"Face detected: {detected_name}"
                    store_alert(f"Camera {cam_id}", "Face Recognition", alert_message, "high")
                    send_email_notification("Face Detected", alert_message)
                    send_local_notification("Face Detected", alert_message)

        
        # Avoid busy looping.
        time.sleep(0.05)

