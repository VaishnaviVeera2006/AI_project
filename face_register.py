import cv2
import os

# Ask for student's name
name = input("Enter Student Name: ")

# Create folder for the student
dataset_path = "dataset"
student_path = os.path.join(dataset_path, name)

if not os.path.exists(student_path):
    os.makedirs(student_path)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

count = 0

print("Looking for face...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        count += 1

        face = gray[y:y+h, x:x+w]

        filename = os.path.join(student_path, f"{count}.jpg")

        cv2.imwrite(filename, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Images: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nRegistration Complete!")
print(f"50 images saved in: {student_path}")