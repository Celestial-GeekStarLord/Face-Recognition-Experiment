import torch
import cv2
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# =========================
# CONFIG
# =========================
DB_FILE = "face_db.pkl"
THRESHOLD = 0.6
CAMERA_INDEX = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# MODELS
# =========================
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# LOAD DATABASE
# =========================
with open(DB_FILE, "rb") as f:
    face_db = pickle.load(f)

print(f"📦 Loaded {len(face_db)} registered people")

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
print("🧠 Face recognition started | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil_img)

    if boxes is not None and probs is not None:
        for box, prob in zip(boxes, probs):

            if prob < 0.9:
                continue

            x1, y1, x2, y2 = map(int, box)
            face_img = pil_img.crop((x1, y1, x2, y2))
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = resnet(face_tensor)

            name = "Unknown"
            best_score = 0.0

            # 🔥 FIX: iterate over LIST of embeddings
            for known_name, emb_list in face_db.items():
                for known_emb in emb_list:
                    score = F.cosine_similarity(
                        emb,
                        known_emb.to(device)
                    ).item()

                    if score > best_score:
                        best_score = score
                        name = known_name

            if best_score < THRESHOLD:
                name = "Unknown"

            # =========================
            # DRAW
            # =========================
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{name} ({best_score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
