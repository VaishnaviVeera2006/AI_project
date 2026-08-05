import cv2
import os
import csv
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Read student names
dataset_path = "dataset"
students = sorted(os.listdir(dataset_path))

# Create attendance folder if it doesn't exist
if not os.path.exists("attendance"):
    os.makedirs("attendance")

attendance_file = "attendance/attendance.csv"

# Create attendance file with header
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70 and id < len(students):
            name = students[id]

            if name not in marked:
                now = datetime.now()
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    ])
                marked.add(name)

            text = name
        else:
            text = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()