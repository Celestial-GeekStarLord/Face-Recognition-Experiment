import torch
import cv2
import pickle
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms

# CONFIG
NAME = "Name"  
DB_FILE = "face_db.pkl"
CAMERA_INDEX = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# MODELS - EXACT SAME AS LIVE RECOGNITION
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    post_process=False  # Critical: same as live script
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# SAME PREPROCESSING AS LIVE RECOGNITION
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# LOAD OR CREATE DATABASE
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
    print(f"📦 Loaded existing database with {len(face_db)} people")
else:
    face_db = {}
    print("📦 Creating new database")

if NAME not in face_db:
    face_db[NAME] = []
    print(f"➕ Added {NAME} to database")
else:
    print(f"ℹ️ {NAME} already has {len(face_db[NAME])} embedding(s)")

# CAMERA
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("\n📸 Look at the camera")
print("👉 Press S to save face (get different angles!)")
print("👉 Press Q to quit\n")

saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read from camera")
        break

    # Convert frame - SAME AS LIVE RECOGNITION
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Detect faces
    try:
        boxes, probs = mtcnn.detect(pil_img)
    except Exception as e:
        print(f"⚠️ Detection error: {e}")
        boxes, probs = None, None

    faces_to_save = []

    if boxes is not None and probs is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.90:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, NAME, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Store face info for saving
            faces_to_save.append((x1, y1, x2, y2))

    # Display instructions
    cv2.putText(frame, f"Saved: {saved_count} | Press S to Save | Q to Quit",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    # SAVE FACE - SAME PROCESSING AS LIVE RECOGNITION
    if key == ord('s') and faces_to_save:
        for (x1, y1, x2, y2) in faces_to_save:
            # Extract face - EXACT SAME AS LIVE RECOGNITION
            face_img = pil_img.crop((x1, y1, x2, y2))
            
            # Preprocess - EXACT SAME AS LIVE RECOGNITION
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)
            
            # Get embedding - EXACT SAME AS LIVE RECOGNITION
            with torch.no_grad():
                embedding = resnet(face_tensor)
            
            # Store on CPU
            face_db[NAME].append(embedding.cpu())
            saved_count += 1

        # Save database
        with open(DB_FILE, "wb") as f:
            pickle.dump(face_db, f)

        print(f"✅ Saved {len(faces_to_save)} face(s) for {NAME}")
        print(f"📦 Total stored for {NAME}: {len(face_db[NAME])}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Registration complete!")
print(f"📊 Final database:")
for name, embeddings in face_db.items():
    print(f"   - {name}: {len(embeddings)} embedding(s)")