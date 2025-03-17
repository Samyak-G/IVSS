# IVSS - Integrated Video Surveillance System

IVSS is a multi-camera integrated video surveillance system built in Python. It leverages real-time motion detection, object detection (via YOLO), and face recognition to monitor and alert on security events. The system is designed to be modular and scalable, with a web-based dashboard for live monitoring, alerts, analytics, settings, and face registration.

## Features

- **Multi-Camera Support:**  
  Configure and manage multiple camera sources.

- **Real-Time Detection:**  
  - **Motion Detection:** Uses OpenCV’s background subtraction to detect movement.
  - **Object Detection:** Utilizes YOLO (via the ultralytics library) to detect objects (e.g., persons).
  - **Face Recognition:** Detects and recognizes faces using the `face_recognition` library.

- **Alert System:**  
  Generates alerts based on user-selected detection types. Alerts are sent via email and local notifications, and stored in an SQLite database.

- **Web-Based Interface:**  
  A Flask web application provides:
  - **Dashboard:** Live camera feeds.
  - **Live Monitoring:** Multi-camera real-time feed.
  - **Alerts & Notifications:** View historical alerts.
  - **Analytics:** System performance and detection statistics.
  - **System Settings:** Configure camera sources and select detections (with options to delete additional camera sources, while keeping the first as default).
  - **Face Registration:** A web-based form to register new faces and update face encodings.


## Installation

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <repository-folder>

## Set Up a Virtual Environment

Create a virtual environment to manage dependencies:
```bash
python -m venv venv
```
Activate it :
```bash
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```


