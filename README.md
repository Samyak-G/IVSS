# 🔐 Intelligent Video Survillence System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-red.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://ultralytics.com/)

A comprehensive AI-powered security monitoring system featuring real-time face recognition, motion detection, and object/weapon detection with 86.7% accuracy. Built with a modular architecture for scalability and easy customization.

## ✨ Features

### 🎥 **Multi-Camera Support**  
- Configure and manage multiple camera sources simultaneously
- Support for IP cameras, USB cameras, and video files
- Real-time streaming with adaptive quality

### 🤖 **Advanced AI Detection**  
- **Motion Detection:** OpenCV-based background subtraction for movement detection
- **Object Detection:** YOLO v8 integration for person, weapon, and object detection
- **Face Recognition:** State-of-the-art face recognition with encoding-based matching
- **Weapon Detection:** Specialized model with 86.7% accuracy for security threats

### 🚨 **Smart Alert System**  
- Real-time email notifications and local alerts
- SQLite database for alert history and analytics
- Configurable alert thresholds and detection sensitivity
- Multi-channel notification support

### 🌐 **Web Interface**  
- **Live Dashboard:** Real-time camera feeds and system status
- **Multi-Camera Monitoring:** Simultaneous feed management
- **Alert Management:** Historical alerts with filtering and search
- **Analytics Dashboard:** Detection statistics and system performance
- **Settings Panel:** Camera configuration and detection preferences
- **Face Registration:** Web-based interface for adding authorized personnel

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/IVSS.git
   cd IVSS
   ```

2. **Set Up Virtual Environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your configuration
   nano .env  # or use your preferred editor
   ```

5. **Initialize Database:**
   ```bash
   python scripts/init_encodings.py
   ```

## 🔧 Configuration

### Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env` and update the following:

```env
# Database
SQLALCHEMY_DATABASE_URI="sqlite:///security_monitoring.db"

# Email Notifications
MAIL_USERNAME="your-email@gmail.com"
MAIL_PASSWORD="your-app-password"
ADMIN_EMAIL="admin@example.com"

# Detection Thresholds
MOTION_THRESHOLD=0.5
FACE_RECOGNITION_THRESHOLD=0.6
OBJECT_DETECTION_CONFIDENCE=0.5
```

### Camera Setup

1. **USB Cameras:** Set `DEFAULT_CAMERA_INDEX=0` (or appropriate index)
2. **IP Cameras:** Use RTSP URL format: `rtsp://username:password@ip:port/path`
3. **Video Files:** Provide absolute path to video file

## 🎯 Usage

### Command Line Interface

```bash
# Start the main security monitoring system
python main.py

# Launch web interface
python src/web/app.py

# Train custom models (optional)
python src/training/train_cpu_optimized.py
```

### Web Interface

1. Start the web application:
   ```bash
   python src/web/app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Configure cameras and detection settings through the web interface

## 📁 Project Structure

```
security-monitoring-system/
├── src/                    # Source code
│   ├── core/              # Core detection modules
│   ├── detection/         # Detection algorithms
│   ├── web/               # Flask web application
│   ├── training/          # Model training scripts
│   └── utils/             # Utility functions
├── data/                  # Data storage
│   ├── models/            # AI models (not in repo)
│   ├── alerts/            # Alert logs and database
│   └── datasets/          # Training datasets (not in repo)
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── documentation/         # Project documentation
└── requirements.txt       # Python dependencies
```

## 🔧 API Reference

### Detection Modules

```python
from src.detection.motion_detection import MotionDetector
from src.detection.face_recognition_module import FaceRecognitionModule
from src.detection.object_detection import ObjectDetector

# Initialize detectors
motion_detector = MotionDetector()
face_detector = FaceRecognitionModule()
object_detector = ObjectDetector()
```

### Alert System

```python
from src.core.alert_module import AlertModule

# Send alert
alert_module = AlertModule()
alert_module.send_alert("Motion detected", "motion")
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 🚀 Deployment

### Local Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "src.web.app:app"
```

## 📊 Performance Metrics

- **Face Recognition Accuracy:** 95%+
- **Weapon Detection Accuracy:** 86.7%
- **Motion Detection Sensitivity:** Configurable (default: 0.5)
- **Real-time Processing:** Up to 30 FPS on modern hardware
- **Multi-camera Support:** Up to 8 simultaneous feeds

## 🛠️ Troubleshooting

### Common Issues

1. **Camera Not Detected:**
   - Check camera permissions
   - Verify camera index in configuration
   - Test with different camera indices (0, 1, 2...)

2. **High CPU Usage:**
   - Reduce frame resolution in camera settings
   - Adjust detection frequency
   - Use GPU acceleration if available

3. **Email Notifications Not Working:**
   - Verify SMTP settings in `.env`
   - Check Gmail app passwords for Gmail SMTP
   - Ensure firewall allows SMTP traffic

### Logs

Check application logs in:
- `data/alerts/alerts_log.txt`
- Console output for real-time debugging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


