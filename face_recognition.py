import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np


# Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# Load Models (PRETRAINED)

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device
)

resnet = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)


# Load Image

img_path = "test.jpg"   # <-- put your image here
img = Image.open(img_path).convert('RGB')


# Detect & Align Face

face = mtcnn(img)

if face is None:
    print("No face detected")
    exit()


# Generate Face Embedding

face = face.unsqueeze(0).to(device)

with torch.no_grad():
    embedding = resnet(face)

print("Embedding shape:", embedding.shape)
print("Embedding vector (first 10 values):")
print(embedding[0][:10])
