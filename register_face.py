import torch
import cv2
import pickle
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# =========================
# CONFIG
# =========================
NAME = "Laxmi Joshi"                 # Change name when registering a new person
DB_FILE = "face_db.pkl"
CAMERA_INDEX = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# MODELS
# =========================
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.7]
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# =========================
# LOAD OR CREATE DATABASE
# =========================
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

if NAME not in face_db:
    face_db[NAME] = []   # multiple embeddings per person

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
print("📸 Look at the camera")
print("👉 Press S to save face")
print("👉 Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # DETECT FACES
    boxes, probs = mtcnn.detect(pil_img)

    faces = mtcnn(pil_img)  # tensors

    if boxes is not None and faces is not None:
        for i, box in enumerate(boxes):
            if probs[i] < 0.90:
                continue

            x1, y1, x2, y2 = map(int, box)

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, NAME, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(frame, "Press S to Save | Q to Quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # SAVE FACE
    # =========================
    if key == ord('s') and faces is not None:
        for face in faces:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face)

            face_db[NAME].append(embedding.cpu())

        with open(DB_FILE, "wb") as f:
            pickle.dump(face_db, f)

        print(f"✅ Saved {len(faces)} face(s) for {NAME}")
        print(f"📦 Total stored for {NAME}: {len(face_db[NAME])}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
