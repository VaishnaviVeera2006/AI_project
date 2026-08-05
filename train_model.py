import cv2
import os
from PIL import Image
import numpy as np

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "dataset"
faces = []
ids = []
label_map = {}

current_id = 0

# Read all student folders
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = Image.open(image_path).convert('L')
        image_np = np.array(img, 'uint8')

        faces.append(image_np)
        ids.append(current_id)

    current_id += 1

# Train the model
recognizer.train(faces, np.array(ids))

# Save trained model
if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save("trainer/trainer.yml")

print("Model trained successfully!")
print(f"Total students: {current_id}")