import cv2
import os
import numpy as np

DATASET_PATH = "dataset"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
labels = []

for label_name in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label_name)

    if not os.path.isdir(label_path):
        continue

    label = int(label_name)  # 🔥 ONLY 0,1,2,3...

    for file in os.listdir(label_path):
        if not file.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(label)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

print("✅ Training completed")
print("Labels used:", set(labels))
