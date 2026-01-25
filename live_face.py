import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# Device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -----------------------------
# Models
# -----------------------------
mtcnn = MTCNN(
    keep_all=True,
    device=device
)

resnet = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# -----------------------------
# Transform for FaceNet
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("✅ Webcam started | Press Q to quit")

# -----------------------------
# Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # -----------------------------
    # SAFE FACE DETECTION
    # -----------------------------
    try:
        boxes, probs = mtcnn.detect(pil_img)
    except Exception:
        boxes, probs = None, None

    if boxes is not None and probs is not None:
        for box, prob in zip(boxes, probs):

            if prob is None or prob < 0.90:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face {prob:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -----------------------------
            # Crop & preprocess face
            # -----------------------------
            face_img = pil_img.crop((x1, y1, x2, y2))

            face_tensor = preprocess(face_img).unsqueeze(0).to(device)

            # -----------------------------
            # Generate embedding
            # -----------------------------
            with torch.no_grad():
                embedding = resnet(face_tensor)

            # embedding shape: [1, 512]
            # print(embedding.shape)

    # -----------------------------
    # Show Webcam
    # -----------------------------
    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
