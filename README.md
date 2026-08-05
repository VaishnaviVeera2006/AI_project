# AI Face Recognition Attendance System

## 📌 Overview
This project is an AI-based Face Recognition Attendance System developed using Python and OpenCV.

It captures student faces, trains a recognition model, identifies students through a webcam, and automatically marks attendance in a CSV file.

## 🚀 Features

- Face Registration
- Face Detection
- Face Recognition
- Automatic Attendance
- CSV Attendance Report
- Real-time Webcam Detection

## 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Pillow

## Project Structure

```
AI_PROJECT/
├── dataset/
├── trainer/
├── attendance/
├── face_register.py
├── train_model.py
├── face_recognition.py
├── face_detection.py
└── haarcascade_frontalface_default.xml
```

## How to Run

### 1. Install requirements

```
pip install -r requirements.txt
```

### 2. Register Face

```
python face_register.py
```

### 3. Train Model

```
python train_model.py
```

### 4. Start Attendance

```
python face_recognition.py
```

Attendance will be saved automatically in:

```
attendance/attendance.csv
```

## Author

**Vaishnavi Veera**