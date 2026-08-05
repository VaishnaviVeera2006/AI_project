#  AI Face Recognition Attendance System

An AI-based Face Recognition Attendance System developed using Python and OpenCV.

##  Overview

This project captures student faces, trains a face recognition model, recognizes students through a webcam, and automatically records attendance in a CSV file.

##  Features

- Face Registration
- Face Detection
- Face Recognition
- Automatic Attendance
- CSV Attendance Report
- Easy to Use Interface

##  Technologies Used

- Python
- OpenCV
- NumPy
- Pandas

##  Project Structure

```
AI_project/
│
├── dataset/
├── trainer/
├── attendance/
├── face_register.py
├── train_model.py
├── face_recognition.py
├── face_detection.py
├── requirements.txt
└── README.md
```

## ▶ Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶ How to Run

### Register a Face

```bash
python face_register.py
```

### Train the Model

```bash
python train_model.py
```

### Start Face Recognition

```bash
python face_recognition.py
```

Attendance will be saved in:

```
attendance/attendance.csv
```

##  Author

**Vaishnavi Veera**